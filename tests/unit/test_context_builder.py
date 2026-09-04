from brd_knowledge.context import ContextBuilder
from brd_knowledge.schemas.context import ContextBuildRequest
from brd_knowledge.schemas.retrieval import RetrievedChunk


def retrieved_chunk(
    rank: int,
    *,
    chunk_id: str | None = None,
    text: str = "The system shall retain an audit trail.",
    content_type: str = "prose",
) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        cosine_distance=0.1 * rank,
        similarity=1.0 - (0.1 * rank),
        chunk_id=chunk_id or f"chunk-{rank}",
        document_id="doc-1",
        section_id="section-1",
        section_title="Audit",
        section_path=["Controls", "Audit"],
        content_type=content_type,
        text=text,
        page_start=rank,
        page_end=rank + 1,
        source_block_ids=[f"block-{rank}"],
        source_table_ids=[f"table-{rank}"] if content_type == "table" else [],
        provenance=[
            {
                "document_id": "doc-1",
                "page_number": rank,
                "block_id": f"block-{rank}",
                "reading_order_index": rank,
            }
        ],
        quality_notes=["native text fallback"] if content_type == "table" else [],
    )


def test_preserves_retrieval_order_and_traceability_for_prose_and_tables() -> None:
    chunks = [
        retrieved_chunk(1),
        retrieved_chunk(2, text="Header A | Header B\nValue A | Value B", content_type="table"),
    ]

    result = ContextBuilder().build(
        ContextBuildRequest(retrieved_chunks=chunks, max_characters=10_000)
    )

    assert [item.evidence_id for item in result.evidence] == ["E1", "E2"]
    assert [item.chunk_id for item in result.evidence] == ["chunk-1", "chunk-2"]
    assert result.evidence[0].source_block_ids == ["block-1"]
    assert result.evidence[0].provenance[0].reading_order_index == 1
    assert result.evidence[1].source_table_ids == ["table-2"]
    assert result.evidence[1].quality_notes == ["native text fallback"]
    assert "[E1]" in result.rendered_text
    assert "Section: Controls > Audit" in result.rendered_text
    assert "Pages: 2-3" in result.rendered_text
    assert "Header A | Header B\nValue A | Value B" in result.rendered_text


def test_excludes_empty_and_duplicate_chunks_without_renumbering_evidence() -> None:
    chunks = [
        retrieved_chunk(1),
        retrieved_chunk(2, text="   "),
        retrieved_chunk(3, chunk_id="chunk-1", text="duplicate"),
        retrieved_chunk(4),
    ]

    result = ContextBuilder().build(
        ContextBuildRequest(retrieved_chunks=chunks, max_characters=10_000)
    )

    assert [item.chunk_id for item in result.evidence] == ["chunk-1", "chunk-4"]
    assert [item.evidence_id for item in result.evidence] == ["E1", "E2"]
    assert [(item.retrieval_rank, item.reason) for item in result.exclusions] == [
        (2, "empty"),
        (3, "duplicate"),
    ]


def test_prefix_packing_reports_later_chunks_that_would_fit_unused_budget() -> None:
    first = retrieved_chunk(1, text="first")
    oversized = retrieved_chunk(2, text="x" * 1_000)
    small = retrieved_chunk(3, text="small")
    builder = ContextBuilder()
    unbounded = builder.build(
        ContextBuildRequest(
            retrieved_chunks=[first, oversized, small],
            max_characters=10_000,
        )
    )
    budget = (
        unbounded.evidence[0].rendered_characters
        + 2
        + unbounded.evidence[2].rendered_characters
    )

    result = builder.build(
        ContextBuildRequest(
            retrieved_chunks=[first, oversized, small],
            max_characters=budget,
        )
    )

    assert [item.chunk_id for item in result.evidence] == ["chunk-1"]
    assert [item.reason for item in result.exclusions] == [
        "budget_exceeded",
        "budget_exceeded",
    ]
    assert result.exclusions[0].would_fit_remaining_budget is False
    assert result.exclusions[1].would_fit_remaining_budget is True
    assert result.exclusions[1].remaining_characters == result.unused_characters
    assert result.used_characters == len(result.rendered_text)
    assert result.unused_characters == result.max_characters - result.used_characters


def test_budget_never_truncates_evidence_and_repeated_builds_are_identical() -> None:
    chunks = [retrieved_chunk(1, text="complete evidence"), retrieved_chunk(2)]
    builder = ContextBuilder()
    full = builder.build(
        ContextBuildRequest(retrieved_chunks=chunks, max_characters=10_000)
    )
    exact_budget = full.evidence[0].rendered_characters
    request = ContextBuildRequest(retrieved_chunks=chunks, max_characters=exact_budget)

    first = builder.build(request)
    second = builder.build(request)

    assert first == second
    assert first.evidence[0].text == "complete evidence"
    assert first.rendered_text.endswith("complete evidence")
    assert "chunk-2" not in first.rendered_text
    assert first.exclusions[0].reason == "budget_exceeded"
