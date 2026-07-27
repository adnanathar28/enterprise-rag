import importlib.util
from pathlib import Path
from types import ModuleType

from brd_knowledge.schemas.table import ParsedTable, TableCell, TableRow


def load_process_document_script() -> ModuleType:
    script_path = Path("scripts/process_document.py")
    spec = importlib.util.spec_from_file_location("process_document_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_table_markdown_expands_spanned_cells_for_readability() -> None:
    process_document_script = load_process_document_script()
    table = ParsedTable(
        table_id="table-1",
        rows=[
            TableRow(
                row_index=0,
                cells=[
                    TableCell(row_index=0, column_index=0, text="Description"),
                    TableCell(
                        row_index=0,
                        column_index=1,
                        text="Length Width Height",
                        column_span=3,
                    ),
                ],
            ),
            TableRow(
                row_index=1,
                cells=[
                    TableCell(row_index=1, column_index=0, text="Small"),
                    TableCell(row_index=1, column_index=1, text="20"),
                    TableCell(row_index=1, column_index=2, text="15"),
                    TableCell(row_index=1, column_index=3, text="12"),
                ],
            ),
        ],
        cells=[
            TableCell(row_index=0, column_index=0, text="Description"),
            TableCell(
                row_index=0,
                column_index=1,
                text="Length Width Height",
                column_span=3,
            ),
            TableCell(row_index=1, column_index=0, text="Small"),
            TableCell(row_index=1, column_index=1, text="20"),
            TableCell(row_index=1, column_index=2, text="15"),
            TableCell(row_index=1, column_index=3, text="12"),
        ],
    )

    assert process_document_script.table_to_markdown(table) == [
        "| Description | Length Width Height | Length Width Height | Length Width Height |",
        "| ----------- | ------------------- | ------------------- | ------------------- |",
        "| Small       | 20                  | 15                  | 12                  |",
    ]
