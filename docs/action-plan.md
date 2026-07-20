# Action Plan

## Current Phase

Scaffold the backend repository and define the normalized document schemas.

## Next Implementation Steps

1. Validate the JSON schema against representative synthetic BRDs.
2. Implement a thin Docling parser adapter.
3. Normalize Docling output into `ParsedDocument`.
4. Add page-level parser diagnostics.
5. Persist parsed documents in PostgreSQL.
6. Add integration tests with non-confidential sample PDFs.

## Out of Scope

- Embeddings
- Vector databases
- RAG
- Chatbots
- LangChain
- LlamaIndex
- LLM-based extraction
- Requirement deduplication or semantic comparison
