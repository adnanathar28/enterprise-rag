import type { DocumentSummary } from "../api/types";

interface DocumentSidebarProps {
  documents: DocumentSummary[];
  selectedDocumentId: string | null;
  loading: boolean;
  onSelect: (documentId: string) => void;
}

function readinessLabel(document: DocumentSummary): string {
  if (document.indexing.status === "ready") return "Ready";
  if (document.indexing.status === "needs_reindex") return "Needs preparation";
  return "Needs preparation";
}

function documentMeta(document: DocumentSummary): string {
  const fileType = document.file_type.toUpperCase();
  const pages = `${document.page_count} ${document.page_count === 1 ? "page" : "pages"}`;
  return `${fileType} · ${pages}`;
}

export function DocumentSidebar({
  documents,
  selectedDocumentId,
  loading,
  onSelect,
}: DocumentSidebarProps) {
  return (
    <aside className="document-sidebar" aria-label="Document library">
      <div className="sidebar-heading">
        <h2>Previous documents</h2>
        <span className="document-count">{documents.length}</span>
      </div>

      <div className="document-list">
        {loading ? (
          <div className="sidebar-loading" aria-label="Loading documents">
            <span />
            <span />
            <span />
          </div>
        ) : documents.length === 0 ? (
          <p className="sidebar-empty">No documents have been added.</p>
        ) : (
          documents.map((document) => {
            const selected = document.document_id === selectedDocumentId;
            return (
              <button
                className="document-row"
                data-selected={selected}
                key={document.document_id}
                type="button"
                aria-pressed={selected}
                title={document.filename}
                onClick={() => onSelect(document.document_id)}
              >
                <span className="document-name">{document.filename}</span>
                <span className="document-row-meta">
                  <span className="document-subline">{documentMeta(document)}</span>
                  <span className="readiness" data-ready={document.indexing.status === "ready"}>
                    <span className="status-dot" />
                    {readinessLabel(document)}
                  </span>
                </span>
              </button>
            );
          })
        )}
      </div>
    </aside>
  );
}
