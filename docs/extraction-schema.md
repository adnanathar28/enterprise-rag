# Extraction Schema

The first normalized output shape is represented by the Pydantic schemas in `src/brd_knowledge/schemas`.

The core objects are:

- `DocumentMetadata`
- `BoundingBox`
- `TextBlock`
- `TableCell`
- `ParsedTable`
- `DocumentSection`
- `ParsedDocument`
- `SourceReference`
- `ExtractedRequirement`

`ExtractedRequirement` exists as an initial contract only. The project does not yet implement requirement extraction logic.

Every parsed object that originates from the source document should carry enough page-level provenance to trace it back to the original BRD.
