import type {
  ApplicationCapabilities,
  DocumentQuestionResponse,
  DocumentSummary,
  IngestionSummary,
  ProviderName,
} from "./types";

const API_BASE = "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    let message = `Request failed with status ${response.status}.`;
    try {
      const payload = (await response.json()) as {
        detail?: string | { message?: string };
      };
      if (typeof payload.detail === "string") message = payload.detail;
      else if (payload.detail?.message) message = payload.detail.message;
    } catch {
      // Keep the stable status-based message for non-JSON failures.
    }
    throw new ApiError(message, response.status);
  }
  return (await response.json()) as T;
}

export function listDocuments(signal?: AbortSignal): Promise<DocumentSummary[]> {
  return request<DocumentSummary[]>("/documents/", { signal });
}

export function getCapabilities(signal?: AbortSignal): Promise<ApplicationCapabilities> {
  return request<ApplicationCapabilities>("/capabilities", { signal });
}

export function getDocument(
  documentId: string,
  signal?: AbortSignal,
): Promise<DocumentSummary> {
  return request<DocumentSummary>(`/documents/${encodeURIComponent(documentId)}`, { signal });
}

export function ingestDocument(file: File, signal?: AbortSignal): Promise<IngestionSummary> {
  const body = new FormData();
  body.append("file", file);
  return request<IngestionSummary>("/documents/ingest", {
    method: "POST",
    body,
    signal,
  });
}

export function indexDocument(
  documentId: string,
  signal?: AbortSignal,
): Promise<DocumentSummary> {
  return request<DocumentSummary>(`/documents/${encodeURIComponent(documentId)}/index`, {
    method: "POST",
    signal,
  });
}

export function askDocumentQuestion(
  documentId: string,
  question: string,
  provider: ProviderName,
  signal?: AbortSignal,
): Promise<DocumentQuestionResponse> {
  return request<DocumentQuestionResponse>(
    `/documents/${encodeURIComponent(documentId)}/questions`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, provider }),
      signal,
    },
  );
}
