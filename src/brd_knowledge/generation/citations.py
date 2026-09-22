import re

from brd_knowledge.core.exceptions import InvalidCitationError
from brd_knowledge.core.generation_diagnostics import log_generation_failure
from brd_knowledge.schemas.context import ConstructedContext, ContextEvidence
from brd_knowledge.schemas.generation import ModelAnswerPayload, ResolvedCitation

BRACKETED_TEXT_PATTERN = re.compile(r"\[(\s*[A-Za-z][^\[\]]*)\]")
CITATION_GROUP_PATTERN = re.compile(
    r"\s*([A-Za-z][A-Za-z0-9_-]*(?:\s*,\s*[A-Za-z][A-Za-z0-9_-]*)*)\s*"
)


def validate_and_resolve_citations(
    payload: ModelAnswerPayload,
    context: ConstructedContext,
) -> tuple[str, list[str], list[ResolvedCitation]]:
    evidence_by_id = {item.evidence_id: item for item in context.evidence}
    try:
        normalized_answer_text, inline_ids = _normalize_inline_citations(payload.answer_text)
    except InvalidCitationError as exc:
        log_generation_failure(
            "inline_citation_parsing", exc, payload=payload.model_dump(mode="json")
        )
        raise
    inline_ids = _deduplicate(inline_ids)
    declared_ids = _deduplicate(payload.cited_evidence_ids)
    unknown_ids = [
        evidence_id
        for evidence_id in _deduplicate([*inline_ids, *declared_ids])
        if evidence_id not in evidence_by_id
    ]
    if unknown_ids:
        _reject(
            "evidence_id_validation",
            f"Generated answer cites unknown evidence IDs: {', '.join(unknown_ids)}",
            payload,
        )
    if inline_ids != declared_ids:
        _reject(
            "cited_evidence_ids_consistency",
            "Inline citations must match cited_evidence_ids in first-use order.",
            payload,
        )
    if not payload.insufficient_evidence and not inline_ids:
        _reject(
            "cited_evidence_ids_consistency",
            "A sufficient answer must cite at least one evidence item.",
            payload,
        )
    return (
        normalized_answer_text,
        inline_ids,
        [_resolve(evidence_by_id[evidence_id]) for evidence_id in inline_ids],
    )


def _normalize_inline_citations(answer_text: str) -> tuple[str, list[str]]:
    inline_ids: list[str] = []

    def normalize_group(match: re.Match[str]) -> str:
        content = match.group(1)
        group = CITATION_GROUP_PATTERN.fullmatch(content)
        if group is None:
            raise InvalidCitationError(f"Malformed inline citation group: [{content}]")
        evidence_ids = [item.strip() for item in group.group(1).split(",")]
        inline_ids.extend(evidence_ids)
        return " ".join(f"[{evidence_id}]" for evidence_id in evidence_ids)

    normalized = BRACKETED_TEXT_PATTERN.sub(normalize_group, answer_text)
    return normalized, inline_ids


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _reject(stage: str, message: str, payload: ModelAnswerPayload) -> None:
    error = InvalidCitationError(message)
    log_generation_failure(stage, error, payload=payload.model_dump(mode="json"))
    raise error


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
