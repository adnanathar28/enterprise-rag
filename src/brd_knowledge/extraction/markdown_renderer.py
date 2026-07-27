from brd_knowledge.extraction.schemas import ExtractionSourceRef, ExtractionUnit


class ExtractionMarkdownRenderer:
    def render(self, units: list[ExtractionUnit]) -> str:
        lines = ["<!-- extraction_view_version: 1 -->", ""]
        current_page = None
        current_section = None

        for unit in units:
            if unit.page_number != current_page:
                current_page = unit.page_number
                current_section = None
                lines.extend([f"<!-- page: {unit.page_number} -->", ""])

            section_key = (unit.section_id, unit.section_title)
            if unit.section_title and section_key != current_section:
                current_section = section_key
                lines.extend(
                    [
                        self._section_marker(unit),
                        f"## {unit.section_title}",
                        "",
                    ]
                )

            lines.extend([self._unit_marker(unit), self._render_unit_text(unit), ""])

        return "\n".join(lines).rstrip() + "\n"

    def _section_marker(self, unit: ExtractionUnit) -> str:
        parts = []
        if unit.section_id is not None:
            parts.append(f"section_id: {unit.section_id}")
        parts.append(f"page: {unit.page_number}")
        return f"<!-- {' | '.join(parts)} -->"

    def _unit_marker(self, unit: ExtractionUnit) -> str:
        parts = [
            f"unit_id: {unit.unit_id}",
            f"type: {unit.content_type}",
            f"page: {unit.page_number}",
        ]
        for source_ref in unit.source_refs:
            parts.extend(self._source_ref_parts(source_ref))
        return f"<!-- {' | '.join(parts)} -->"

    def _source_ref_parts(self, source_ref: ExtractionSourceRef) -> list[str]:
        parts = []
        if source_ref.section_id is not None:
            parts.append(f"section_id: {source_ref.section_id}")
        if source_ref.block_id is not None:
            parts.append(f"block_id: {source_ref.block_id}")
        if source_ref.table_id is not None:
            parts.append(f"table_id: {source_ref.table_id}")
        if source_ref.cell_ids:
            parts.append(f"cell_ids: {','.join(source_ref.cell_ids)}")
        return parts

    def _render_unit_text(self, unit: ExtractionUnit) -> str:
        if unit.content_type == "list_item":
            return f"- {unit.text}"
        return unit.text


def render_extraction_markdown(units: list[ExtractionUnit]) -> str:
    return ExtractionMarkdownRenderer().render(units)
