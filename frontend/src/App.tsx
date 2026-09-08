import { useEffect, useMemo, useRef, useState } from "react";

import {
  ApiError,
  askDocumentQuestion,
  getCapabilities,
  getDocument,
  indexDocument,
  ingestDocument,
  listDocuments,
} from "./api/client";
import type {
  ApplicationCapabilities,
  DocumentQuestionResponse,
  DocumentSummary,
  ProviderName,
} from "./api/types";
import { AnswerPanel } from "./components/AnswerPanel";
import { DocumentSidebar } from "./components/DocumentSidebar";
import { QuestionComposer } from "./components/QuestionComposer";
import { UploadPanel, type UploadPhase } from "./components/UploadPanel";

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
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<DocumentQuestionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadPhase, setUploadPhase] = useState<UploadPhase>(null);
  const [processingFilename, setProcessingFilename] = useState<string | null>(null);
  const [preparationError, setPreparationError] = useState<string | null>(null);
  const [preparingDocumentId, setPreparingDocumentId] = useState<string | null>(null);
  const activeRequest = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    Promise.all([listDocuments(controller.signal), getCapabilities(controller.signal)])
      .then(([documents, capabilities]) => {
        if (!active) return;
        setInitialData({ documents, capabilities });
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

  function selectDocument(documentId: string | null) {
    activeRequest.current?.abort();
    activeRequest.current = null;
    setSelectedDocumentId(documentId);
    setLibraryOpen(false);
    setQuestion("");
    setResult(null);
    setError(null);
    setSubmitting(false);
    setPreparationError(null);
  }

  function upsertDocument(document: DocumentSummary) {
    setInitialData((current) => current && {
      ...current,
      documents: [document, ...current.documents.filter(
        (candidate) => candidate.document_id !== document.document_id,
      )],
    });
  }

  async function uploadDocument(file: File) {
    setUploadError(null);
    setProcessingFilename(file.name);
    setUploadPhase("uploading");
    try {
      const ingestion = await ingestDocument(file);
      setUploadPhase("indexing");
      try {
        const indexed = await indexDocument(ingestion.document_id);
        upsertDocument(indexed);
        setSelectedDocumentId(indexed.document_id);
        setUploadPhase(null);
        setProcessingFilename(null);
      } catch (indexingError) {
        const message = errorMessage(indexingError);
        let recoveredSavedDocument = false;
        try {
          const saved = await getDocument(ingestion.document_id);
          upsertDocument(saved);
          setSelectedDocumentId(saved.document_id);
          recoveredSavedDocument = true;
        } catch {
          setUploadError(message || "Search preparation failed.");
        }
        if (recoveredSavedDocument) {
          setPreparationError(message || "Search preparation failed.");
        }
        setUploadPhase(null);
        setProcessingFilename(null);
      }
    } catch (ingestionError) {
      const message = errorMessage(ingestionError);
      if (message) setUploadError(message);
      setUploadPhase(null);
    }
  }

  async function prepareDocument(documentId: string) {
    setPreparingDocumentId(documentId);
    setPreparationError(null);
    try {
      const indexed = await indexDocument(documentId);
      upsertDocument(indexed);
    } catch (indexingError) {
      const message = errorMessage(indexingError);
      if (message) setPreparationError(message);
    } finally {
      setPreparingDocumentId(null);
    }
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
      if (activeRequest.current === controller) setResult(response);
    } catch (requestError) {
      const message = errorMessage(requestError);
      if (message && activeRequest.current === controller) setError(message);
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
          <span>Knowledge</span>
        </div>
        <nav className="workspace-actions" aria-label="Workspace navigation">
          {selectedDocument && <button type="button" onClick={() => selectDocument(null)}>Upload a document</button>}
          <button type="button" aria-expanded={libraryOpen} aria-controls="previous-documents"
            onClick={() => setLibraryOpen(!libraryOpen)}>Previous documents</button>
        </nav>
      </header>

      <div className="workspace-layout">
        <div id="previous-documents" hidden={!libraryOpen}>
          <DocumentSidebar
            documents={initialData?.documents ?? []}
            selectedDocumentId={selectedDocumentId}
            loading={loading}
            onSelect={selectDocument}
          />
        </div>

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
            initialData && <UploadPanel
              capabilities={initialData.capabilities}
              phase={uploadPhase}
              processingFilename={processingFilename}
              uploadError={uploadError}
              onUpload={uploadDocument}
            />
          ) : (
            <div className="document-workspace">
              <header className="document-header">
                <p className="eyebrow">Document workspace</p>
                <h1>{selectedDocument.filename}</h1>
                <div className="document-metadata">
                  <span>{selectedDocument.file_type.toUpperCase()}</span>
                  <span>{selectedDocument.page_count} pages</span>
                  <span>{selectedDocument.indexing.status === "ready" ? "Ready for questions" : "Search preparation needed"}</span>
                  <span>
                    Added {new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(
                      new Date(selectedDocument.created_at),
                    )}
                  </span>
                </div>
              </header>

              {selectedDocument.indexing.status !== "ready" ? (
                <section className="preparation-panel" aria-labelledby="preparation-heading">
                  <h2 id="preparation-heading">Prepare this document for questions</h2>
                  <p>The document is saved. Prepare it for search before asking questions.</p>
                  {preparationError && <div className="query-error" role="alert">
                    <strong>Search preparation failed.</strong>
                    <span>{preparationError}</span>
                  </div>}
                  <button className="submit-button" type="button"
                    disabled={preparingDocumentId === selectedDocument.document_id}
                    onClick={() => prepareDocument(selectedDocument.document_id)}>
                    {preparingDocumentId === selectedDocument.document_id ? "Preparing…" : "Prepare for search"}
                  </button>
                </section>
              ) : <QuestionComposer
                document={selectedDocument}
                providers={initialData?.capabilities.providers ?? []}
                selectedProvider={selectedProvider}
                question={question}
                submitting={submitting}
                onProviderChange={setSelectedProvider}
                onQuestionChange={setQuestion}
                onSubmit={submitQuestion}
              />}

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
                  <section className="answer-placeholder" aria-label="About answers">
                    <p>Answers are grounded in evidence from the selected document.</p>
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
