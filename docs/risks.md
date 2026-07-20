# Risks

## Parser Fidelity

BRDs often mix prose, tables, headers, footers, scanned pages and inconsistent numbering. Parser output should be inspected before relying on automated normalization.

## Provenance Loss

The first milestone depends on preserving page and source references. Any parser adapter should prefer incomplete structured output with clear provenance over cleaner text that cannot be traced.

## Confidentiality

Real BRDs may contain confidential business information. Do not commit confidential documents, parsed outputs or generated artifacts derived from confidential BRDs.

## Scope Creep

Semantic search, embeddings, RAG and LLM extraction are deliberately excluded until the normalized document representation is stable.
