import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

import { App } from "./App";
import type {
  ApplicationCapabilities,
  DocumentQuestionResponse,
  DocumentSummary,
} from "./api/types";

const readyDocument: DocumentSummary = {
  document_id: "doc-1",
  filename: "Enterprise Controls BRD.pdf",
  original_filename: "Enterprise Controls BRD.pdf",
  stored_filename: "stored.pdf",
  file_type: "pdf",
  size_bytes: 42_000,
  page_count: 18,
  parse_status: "success",
  parser_name: "docling",
  parser_version: "2.0",
  created_at: "2026-09-08T08:00:00Z",
  indexing: {
    status: "ready",
    compatible_chunk_count: 42,
    total_chunk_count: 42,
  },
};

const capabilities: ApplicationCapabilities = {
  providers: [
    {
      provider: "local_qwen",
      label: "Local Qwen",
      model: "qwen3:8b",
      configured: true,
      is_default: true,
    },
    {
      provider: "gemini",
      label: "Gemini",
      model: "gemini-3.1-flash-lite",
      configured: false,
      is_default: false,
    },
  ],
  allowed_document_extensions: [".docx", ".pdf"],
  max_upload_size_bytes: 25_000_000,
};

const groundedResponse: DocumentQuestionResponse = {
  document_id: "doc-1",
  question: "What is the audit requirement?",
  answer: {
    answer_text: "The system must retain audit records [E1].",
    cited_evidence_ids: ["E1"],
    citations: [
      {
        evidence_id: "E1",
        document_id: "doc-1",
        chunk_id: "chunk-1",
        section_id: "section-1",
        section_path: ["Controls", "Audit"],
        page_start: 7,
        page_end: 7,
        source_block_ids: ["block-1"],
        source_table_ids: [],
        provenance: [],
        quality_notes: [],
      },
    ],
    insufficient_evidence: false,
    metadata: {
      provider: "ollama",
      model: "qwen3:8b",
      revision: "revision-1",
      max_output_tokens: 1024,
      prompt_tokens: 812,
      output_tokens: 24,
      provider_request_id: null,
      model_version: "qwen3:8b",
      thinking_tokens: 0,
    },
  },
  context: {
    evidence: [
      {
        evidence_id: "E1",
        retrieval_rank: 1,
        chunk_id: "chunk-1",
        document_id: "doc-1",
        section_id: "section-1",
        section_title: "Audit",
        section_path: ["Controls", "Audit"],
        content_type: "prose",
        text: "The system shall retain complete audit records for regulated operations.",
        page_start: 7,
        page_end: 7,
        source_block_ids: ["block-1"],
        source_table_ids: [],
        provenance: [],
        quality_notes: [],
        similarity: 0.91,
        rendered_characters: 92,
      },
    ],
    rendered_text: "[E1] Audit evidence",
    max_characters: 20_000,
    used_characters: 92,
    unused_characters: 19_908,
    exclusions: [],
  },
  retrieved_chunks: [
    {
      rank: 1,
      cosine_distance: 0.09,
      similarity: 0.91,
      chunk_id: "chunk-1",
      document_id: "doc-1",
      section_id: "section-1",
      section_title: "Audit",
      section_path: ["Controls", "Audit"],
      content_type: "prose",
      text: "The system shall retain complete audit records for regulated operations.",
      page_start: 7,
      page_end: 7,
      source_block_ids: ["block-1"],
      source_table_ids: [],
      provenance: [],
      quality_notes: [],
    },
  ],
  elapsed_seconds: 1.42,
};

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function mockApi(documents: DocumentSummary[], answer = groundedResponse) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    void init;
    const url = String(input);
    if (url.endsWith("/documents/")) return jsonResponse(documents);
    if (url.endsWith("/capabilities")) return jsonResponse(capabilities);
    if (url.endsWith("/documents/doc-1/questions")) return jsonResponse(answer);
    return jsonResponse({ detail: "Not found" }, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

test("asks a document-scoped question and renders authoritative evidence", async () => {
  const fetchMock = mockApi([readyDocument]);
  const user = userEvent.setup();
  render(<App />);

  expect(await screen.findByRole("heading", { name: readyDocument.filename })).toBeVisible();
  await user.type(
    screen.getByRole("textbox", { name: `Question about ${readyDocument.filename}` }),
    "What is the audit requirement?",
  );
  await user.click(screen.getByRole("button", { name: "Ask question" }));

  expect(await screen.findByRole("heading", { name: "Grounded response" })).toBeVisible();
  expect(screen.getByText("The system must retain audit records", { exact: false })).toBeVisible();
  const source = screen.getByRole("article");
  expect(within(source).getByText("Controls / Audit")).toBeVisible();
  expect(within(source).getByText("Page 7")).toBeVisible();
  expect(within(source).getByText(/complete audit records/, { selector: "p" })).toBeVisible();
  const fullEvidence = within(source).getByText(/complete audit records/, { selector: "pre" });
  expect(fullEvidence).not.toBeVisible();
  await user.click(within(source).getByText("View full evidence & provenance"));
  expect(fullEvidence).toBeVisible();
  expect(within(source).getByText("block-1")).toBeVisible();

  const questionRequest = fetchMock.mock.calls.find(([url]) =>
    String(url).endsWith("/documents/doc-1/questions"),
  );
  expect(JSON.parse(String(questionRequest?.[1]?.body))).toEqual({
    question: "What is the audit requirement?",
    provider: "local_qwen",
  });
});

test("keeps question controls disabled for an unindexed document", async () => {
  mockApi([
    {
      ...readyDocument,
      indexing: { status: "not_indexed", compatible_chunk_count: 0, total_chunk_count: 0 },
    },
  ]);
  render(<App />);

  expect(await screen.findByText(/must be indexed with the active embedding/)).toBeVisible();
  expect(screen.getByRole("button", { name: "Ask question" })).toBeDisabled();
  expect(screen.getByRole("textbox")).toBeDisabled();
});

test("example questions fill the composer without calling generation", async () => {
  const fetchMock = mockApi([readyDocument]);
  const user = userEvent.setup();
  render(<App />);
  const example = await screen.findByRole("button", { name: /Which systems need to integrate/ });
  await user.click(example);
  expect(screen.getByRole("textbox")).toHaveValue("Which systems need to integrate?");
  expect(screen.getByRole("textbox")).toHaveFocus();
  expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith("/questions"))).toBe(false);
});

test("shows the backend insufficient-evidence state without inventing sources", async () => {
  mockApi([readyDocument], {
    ...groundedResponse,
    answer: {
      ...groundedResponse.answer,
      answer_text: "The supplied evidence does not specify a retention period.",
      cited_evidence_ids: [],
      citations: [],
      insufficient_evidence: true,
    },
  });
  const user = userEvent.setup();
  render(<App />);

  const input = await screen.findByRole("textbox");
  await user.type(input, "How long are records retained?");
  await user.click(screen.getByRole("button", { name: "Ask question" }));

  expect(await screen.findByRole("heading", { name: "Insufficient evidence" })).toBeVisible();
  expect(screen.getByText("No evidence was cited for this response.")).toBeVisible();
});
