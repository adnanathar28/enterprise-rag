import { useRef, useState } from "react";

import type { ApplicationCapabilities } from "../api/types";

export type UploadPhase = "uploading" | "indexing" | null;

interface UploadPanelProps {
  capabilities: ApplicationCapabilities;
  phase: UploadPhase;
  processingFilename: string | null;
  uploadError: string | null;
  onUpload: (file: File) => void;
}

export function UploadPanel({
  capabilities,
  phase,
  processingFilename,
  uploadError,
  onUpload,
}: UploadPanelProps) {
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const pdfAllowed = capabilities.allowed_document_extensions.some((extension) => extension.toLowerCase() === ".pdf");
  const limit = capabilities.max_upload_size_bytes;

  function chooseFiles(files: File[]) {
    setError(null);
    if (files.length === 0) return;
    if (files.length !== 1) {
      setError("Choose one PDF at a time.");
      return;
    }
    const candidate = files[0];
    if (!pdfAllowed || !candidate.name.toLowerCase().endsWith(".pdf")) {
      setError("Choose a PDF document.");
    } else if (candidate.size === 0) {
      setError("This file is empty. Choose another PDF.");
    } else if (candidate.size > limit) {
      setError("This file exceeds the upload size limit.");
    } else {
      setFile(candidate);
      onUpload(candidate);
    }
  }

  const busy = phase !== null;

  return (
    <section className="upload-workspace" aria-labelledby="upload-heading">
      <p className="eyebrow">Document workspace</p>
      <h1 id="upload-heading">Upload a document</h1>
      <p className="upload-intro">Ask questions. Find answers with sources.</p>
      {busy ? (
        <div className="processing-panel" aria-live="polite">
          <span className="processing-spinner" aria-hidden="true" />
          <div>
            <h2>{phase === "uploading" ? "Uploading and parsing document" : "Preparing for search"}</h2>
            <p title={processingFilename ?? undefined}>{processingFilename}</p>
          </div>
          <ol className="processing-steps">
            <li data-state={phase === "uploading" ? "active" : "complete"}>Upload and parse</li>
            <li data-state={phase === "indexing" ? "active" : "pending"}>Prepare for search</li>
            <li data-state="pending">Ready</li>
          </ol>
        </div>
      ) : <div className="upload-dropzone" data-dragging={dragging}
        onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
        onDragLeave={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false);
        }}
        onDrop={(event) => {
          event.preventDefault(); setDragging(false);
          chooseFiles(Array.from(event.dataTransfer.files));
        }}>
        <h2>Drop a document here</h2>
        <p>or choose a file</p>
        <button type="button" className="submit-button" disabled={!pdfAllowed}
          onClick={() => input.current?.click()}>Choose a PDF</button>
        <input ref={input} className="sr-only" type="file" accept=".pdf" tabIndex={-1}
          aria-label="PDF document" disabled={!pdfAllowed}
          onChange={(event) => {
            chooseFiles(Array.from(event.target.files ?? []));
            event.target.value = "";
          }} />
        <span className="upload-limit">
          {pdfAllowed ? `PDF · Up to ${new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(limit / 1_000_000)} MB` : "PDF uploads are unavailable on this server."}
        </span>
      </div>}
      {(error || uploadError) && <div className="query-error" role="alert">
        <strong>Document could not be processed.</strong>
        <span>{error ?? uploadError}</span>
      </div>}
      {file && !busy && <div className="selected-file" role="status">
        <div><strong title={file.name}>{file.name}</strong><span>{uploadError ? "Upload failed" : busy ? "Processing" : "Selected"}</span></div>
        {!busy && <div className="selected-file-actions">
          {uploadError && <button type="button" onClick={() => onUpload(file)}>Try again</button>}
          <button type="button" onClick={() => setFile(null)}>Remove</button>
        </div>}
      </div>}
    </section>
  );
}
