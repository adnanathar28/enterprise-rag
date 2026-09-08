import { useEffect, useMemo, useRef, useState } from "react";

import { ApiError, askDocumentQuestion, getCapabilities, listDocuments } from "./api/client";
import type {
  ApplicationCapabilities,
  DocumentQuestionResponse,
  DocumentSummary,
  ProviderName,
} from "./api/types";
import { AnswerPanel } from "./components/AnswerPanel";
import { DocumentSidebar } from "./components/DocumentSidebar";
import { QuestionComposer } from "./components/QuestionComposer";

interface InitialData {
  documents: DocumentSummary[];
  capabilities: ApplicationCapabilities;
}

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error && error.name === "AbortError") return "";
  return "The knowledge service could not be reached. Check that the API is running.";
}

export function App() {
  const [initialData, setInitialData] = useState<InitialData | null>(null);
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);
  const [selectedProvider, setSelectedProvider] = useState<ProviderName | null>(null);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<DocumentQuestionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const activeRequest = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    Promise.all([listDocuments(controller.signal), getCapabilities(controller.signal)])
      .then(([documents, capabilities]) => {
        if (!active) return;
        setInitialData({ documents, capabilities });
        const firstReady = documents.find((document) => document.indexing.status === "ready");
        setSelectedDocumentId(firstReady?.document_id ?? documents[0]?.document_id ?? null);
        const preferredProvider =
          capabilities.providers.find((provider) => provider.is_default && provider.configured) ??
          capabilities.providers.find((provider) => provider.configured);
        setSelectedProvider(preferredProvider?.provider ?? null);
      })
      .catch((requestError: unknown) => {
        if (!active) return;
        const message = errorMessage(requestError);
        if (message) setError(message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, []);

  useEffect(() => () => activeRequest.current?.abort(), []);

  const selectedDocument = useMemo(
    () =>
      initialData?.documents.find((document) => document.document_id === selectedDocumentId) ?? null,
    [initialData, selectedDocumentId],
  );

  function selectDocument(documentId: string) {
    activeRequest.current?.abort();
    setSelectedDocumentId(documentId);
    setQuestion("");
    setResult(null);
    setError(null);
    setSubmitting(false);
  }

  async function submitQuestion() {
    if (!selectedDocument || !selectedProvider || !question.trim()) return;
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    setSubmitting(true);
    setError(null);
    setResult(null);
    try {
      const response = await askDocumentQuestion(
        selectedDocument.document_id,
        question.trim(),
        selectedProvider,
        controller.signal,
      );
      setResult(response);
    } catch (requestError) {
      const message = errorMessage(requestError);
      if (message) setError(message);
    } finally {
      if (activeRequest.current === controller) {
        activeRequest.current = null;
        setSubmitting(false);
      }
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <span>BRD Knowledge</span>
        </div>
        <div className="workspace-label">Requirements workspace</div>
      </header>

      <div className="workspace-layout">
        <DocumentSidebar
          documents={initialData?.documents ?? []}
          selectedDocumentId={selectedDocumentId}
          loading={loading}
          onSelect={selectDocument}
        />

        <main className="main-content">
          {error && !initialData ? (
            <div className="fatal-state" role="alert">
              <p className="eyebrow">Connection error</p>
              <h1>Knowledge service unavailable</h1>
              <p>{error}</p>
            </div>
          ) : loading ? (
            <div className="workspace-loading" aria-label="Loading workspace">
              <span />
              <span />
              <span />
            </div>
          ) : !selectedDocument ? (
            <div className="empty-workspace">
              <p className="eyebrow">Document workspace</p>
              <h1>No documents available</h1>
              <p>Ingest and index a BRD before asking document-grounded questions.</p>
            </div>
          ) : (
            <div className="document-workspace">
              <header className="document-header">
                <p className="eyebrow">Document workspace</p>
                <h1>{selectedDocument.filename}</h1>
                <div className="document-metadata">
                  <span>{selectedDocument.file_type.toUpperCase()}</span>
                  <span>{selectedDocument.page_count} pages</span>
                  <span>{selectedDocument.indexing.total_chunk_count} indexed chunks</span>
                  <span>
                    Added {new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(
                      new Date(selectedDocument.created_at),
                    )}
                  </span>
                </div>
              </header>

              <QuestionComposer
                document={selectedDocument}
                providers={initialData?.capabilities.providers ?? []}
                selectedProvider={selectedProvider}
                question={question}
                submitting={submitting}
                onProviderChange={setSelectedProvider}
                onQuestionChange={setQuestion}
                onSubmit={submitQuestion}
              />

              {error && (
                <div className="query-error" role="alert">
                  <strong>Question could not be completed.</strong>
                  <span>{error}</span>
                </div>
              )}

              {submitting ? (
                <div className="answer-loading" aria-live="polite">
                  <span className="loading-rule" />
                  <div>
                    <p>Retrieving evidence and preparing a grounded answer…</p>
                    <span>This may take longer when a local model is loading.</span>
                  </div>
                </div>
              ) : result ? (
                <AnswerPanel result={result} />
              ) : (
                !error && selectedDocument.indexing.status === "ready" && (
                  <section className="answer-placeholder" aria-label="Question suggestions">
                    <h2>Start with what you need to know</h2>
                    <p>Ask about requirements, controls, integrations, or business rules.
                      Answers link back to evidence in this document.</p>
                    <div className="example-prompts">
                      {[
                        "What are the key business requirements?",
                        "Which systems need to integrate?",
                        "What controls and approvals are required?",
                      ].map((prompt) => (
                        <button key={prompt} type="button" onClick={() => {
                          setQuestion(prompt);
                          window.document.getElementById("document-question")?.focus();
                        }}>
                          {prompt}<span aria-hidden="true">↗</span>
                        </button>
                      ))}
                    </div>
                  </section>
                )
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
