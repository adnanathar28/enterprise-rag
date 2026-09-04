import re

from brd_knowledge.core.exceptions import InvalidCitationError
from brd_knowledge.schemas.context import ConstructedContext, ContextEvidence
from brd_knowledge.schemas.generation import ModelAnswerPayload, ResolvedCitation

CITATION_TOKEN_PATTERN = re.compile(r"\[([A-Za-z][A-Za-z0-9_-]*)\]")


def validate_and_resolve_citations(
    payload: ModelAnswerPayload,
    context: ConstructedContext,
) -> tuple[list[str], list[ResolvedCitation]]:
    evidence_by_id = {item.evidence_id: item for item in context.evidence}
    inline_ids = _deduplicate(CITATION_TOKEN_PATTERN.findall(payload.answer_text))
    declared_ids = _deduplicate(payload.cited_evidence_ids)
    unknown_ids = [
        evidence_id
        for evidence_id in _deduplicate([*inline_ids, *declared_ids])
        if evidence_id not in evidence_by_id
    ]
    if unknown_ids:
        raise InvalidCitationError(
            f"Generated answer cites unknown evidence IDs: {', '.join(unknown_ids)}"
        )
    if inline_ids != declared_ids:
        raise InvalidCitationError(
            "Inline citations must match cited_evidence_ids in first-use order."
        )
    if not payload.insufficient_evidence and not inline_ids:
        raise InvalidCitationError("A sufficient answer must cite at least one evidence item.")
    return inline_ids, [_resolve(evidence_by_id[evidence_id]) for evidence_id in inline_ids]


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _resolve(evidence: ContextEvidence) -> ResolvedCitation:
    return ResolvedCitation(
        evidence_id=evidence.evidence_id,
        document_id=evidence.document_id,
        chunk_id=evidence.chunk_id,
        section_id=evidence.section_id,
        section_path=evidence.section_path,
        page_start=evidence.page_start,
        page_end=evidence.page_end,
        source_block_ids=evidence.source_block_ids,
        source_table_ids=evidence.source_table_ids,
        provenance=evidence.provenance,
        quality_notes=evidence.quality_notes,
    )
