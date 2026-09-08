export type ProviderName = "gemini" | "local_qwen";
export type IndexingStatus = "not_indexed" | "ready" | "needs_reindex";

export interface IndexingSummary {
  status: IndexingStatus;
  compatible_chunk_count: number;
  total_chunk_count: number;
}

export interface DocumentSummary {
  document_id: string;
  filename: string;
  original_filename: string;
  stored_filename: string;
  file_type: string;
  size_bytes: number;
  page_count: number;
  parse_status: string | null;
  parser_name: string | null;
  parser_version: string | null;
  created_at: string;
  indexing: IndexingSummary;
}

export interface ProviderCapability {
  provider: ProviderName;
  label: string;
  model: string;
  configured: boolean;
  is_default: boolean;
}

export interface ApplicationCapabilities {
  providers: ProviderCapability[];
  allowed_document_extensions: string[];
  max_upload_size_bytes: number;
}

export interface SourceReference {
  document_id: string;
  page_number: number | null;
  parser_item_id: string | null;
  section_id: string | null;
  block_id: string | null;
  table_id: string | null;
  cell_id: string | null;
  reading_order_index: number | null;
  text_excerpt: string | null;
}

export interface RetrievedChunk {
  rank: number;
  cosine_distance: number;
  similarity: number;
  chunk_id: string;
  document_id: string;
  section_id: string | null;
  section_title: string | null;
  section_path: string[];
  content_type: "prose" | "table";
  text: string;
  page_start: number;
  page_end: number;
  source_block_ids: string[];
  source_table_ids: string[];
  provenance: SourceReference[];
  quality_notes: string[];
}

export interface ContextEvidence extends Omit<RetrievedChunk, "rank" | "cosine_distance"> {
  evidence_id: string;
  retrieval_rank: number;
  rendered_characters: number;
}

export interface ContextExclusion {
  chunk_id: string;
  retrieval_rank: number;
  reason: "duplicate" | "empty" | "budget_exceeded";
  rendered_characters: number | null;
  required_characters: number | null;
  remaining_characters: number | null;
  would_fit_remaining_budget: boolean | null;
}

export interface ConstructedContext {
  evidence: ContextEvidence[];
  rendered_text: string;
  max_characters: number;
  used_characters: number;
  unused_characters: number;
  exclusions: ContextExclusion[];
}

export interface ResolvedCitation {
  evidence_id: string;
  document_id: string;
  chunk_id: string;
  section_id: string | null;
  section_path: string[];
  page_start: number;
  page_end: number;
  source_block_ids: string[];
  source_table_ids: string[];
  provenance: SourceReference[];
  quality_notes: string[];
}

export interface GenerationMetadata {
  provider: string;
  model: string;
  revision: string | null;
  max_output_tokens: number;
  prompt_tokens: number | null;
  output_tokens: number | null;
  provider_request_id: string | null;
  model_version: string | null;
  thinking_tokens: number | null;
}

export interface GroundedAnswer {
  answer_text: string;
  cited_evidence_ids: string[];
  citations: ResolvedCitation[];
  insufficient_evidence: boolean;
  metadata: GenerationMetadata;
}

export interface DocumentQuestionResponse {
  document_id: string;
  question: string;
  answer: GroundedAnswer;
  context: ConstructedContext;
  retrieved_chunks: RetrievedChunk[];
  elapsed_seconds: number;
}
