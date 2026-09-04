from __future__ import annotations

from brd_knowledge.schemas.context import (
    ConstructedContext,
    ContextBuildRequest,
    ContextEvidence,
    ContextExclusion,
)
from brd_knowledge.schemas.retrieval import RetrievedChunk
from brd_knowledge.schemas.source import SourceReference

EVIDENCE_SEPARATOR = "\n\n"


class ContextBuilder:
    def build(self, request: ContextBuildRequest) -> ConstructedContext:
        candidates, exclusions = self._filter_candidates(request.retrieved_chunks)
        evidence: list[ContextEvidence] = []
        rendered_items: list[str] = []
        budget_stopped = False

        for chunk in candidates:
            evidence_id = f"E{len(evidence) + 1}"
            rendered = self._render_chunk(evidence_id, chunk)
            separator_characters = len(EVIDENCE_SEPARATOR) if rendered_items else 0
            required_characters = separator_characters + len(rendered)
            used_characters = sum(len(item) for item in rendered_items) + (
                len(EVIDENCE_SEPARATOR) * max(0, len(rendered_items) - 1)
            )
            remaining_characters = request.max_characters - used_characters

            if budget_stopped or required_characters > remaining_characters:
                budget_stopped = True
                exclusions.append(
                    ContextExclusion(
                        chunk_id=chunk.chunk_id,
                        retrieval_rank=chunk.rank,
                        reason="budget_exceeded",
                        rendered_characters=len(rendered),
                        required_characters=required_characters,
                        remaining_characters=remaining_characters,
                        would_fit_remaining_budget=(
                            required_characters <= remaining_characters
                        ),
                    )
                )
                continue

            rendered_items.append(rendered)
            evidence.append(self._to_evidence(evidence_id, chunk, len(rendered)))

        rendered_text = EVIDENCE_SEPARATOR.join(rendered_items)
        used_characters = len(rendered_text)
        return ConstructedContext(
            evidence=evidence,
            rendered_text=rendered_text,
            max_characters=request.max_characters,
            used_characters=used_characters,
            unused_characters=request.max_characters - used_characters,
            exclusions=sorted(exclusions, key=lambda item: item.retrieval_rank),
        )

    def _filter_candidates(
        self,
        chunks: list[RetrievedChunk],
    ) -> tuple[list[RetrievedChunk], list[ContextExclusion]]:
        candidates: list[RetrievedChunk] = []
        exclusions: list[ContextExclusion] = []
        seen_chunk_ids: set[str] = set()
        for chunk in chunks:
            if not chunk.text.strip():
                exclusions.append(
                    ContextExclusion(
                        chunk_id=chunk.chunk_id,
                        retrieval_rank=chunk.rank,
                        reason="empty",
                    )
                )
                continue
            if chunk.chunk_id in seen_chunk_ids:
                exclusions.append(
                    ContextExclusion(
                        chunk_id=chunk.chunk_id,
                        retrieval_rank=chunk.rank,
                        reason="duplicate",
                    )
                )
                continue
            seen_chunk_ids.add(chunk.chunk_id)
            candidates.append(chunk)
        return candidates, exclusions

    def _to_evidence(
        self,
        evidence_id: str,
        chunk: RetrievedChunk,
        rendered_characters: int,
    ) -> ContextEvidence:
        return ContextEvidence(
            evidence_id=evidence_id,
            retrieval_rank=chunk.rank,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            section_id=chunk.section_id,
            section_title=chunk.section_title,
            section_path=chunk.section_path,
            content_type=chunk.content_type,
            text=chunk.text,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            source_block_ids=chunk.source_block_ids,
            source_table_ids=chunk.source_table_ids,
            provenance=[SourceReference.model_validate(item) for item in chunk.provenance],
            quality_notes=chunk.quality_notes,
            similarity=chunk.similarity,
            rendered_characters=rendered_characters,
        )

    def _render_chunk(self, evidence_id: str, chunk: RetrievedChunk) -> str:
        pages = (
            str(chunk.page_start)
            if chunk.page_start == chunk.page_end
            else f"{chunk.page_start}-{chunk.page_end}"
        )
        section = " > ".join(chunk.section_path) if chunk.section_path else "NONE"
        lines = [
            f"[{evidence_id}]",
            f"Document: {chunk.document_id}",
            f"Section: {section}",
            f"Pages: {pages}",
            f"Content type: {chunk.content_type}",
            f"Chunk ID: {chunk.chunk_id}",
        ]
        if chunk.quality_notes:
            lines.append(f"Quality notes: {'; '.join(chunk.quality_notes)}")
        lines.extend(["", chunk.text])
        return "\n".join(lines)
