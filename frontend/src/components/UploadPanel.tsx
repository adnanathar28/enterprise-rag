import { useRef, useState } from "react";

import type { ApplicationCapabilities } from "../api/types";

export function UploadPanel({ capabilities }: { capabilities: ApplicationCapabilities }) {
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
    }
  }

  return (
    <section className="upload-workspace" aria-labelledby="upload-heading">
      <p className="eyebrow">Document workspace</p>
      <h1 id="upload-heading">Upload a document</h1>
      <p className="upload-intro">Ask questions. Find answers with sources.</p>
      <div className="upload-dropzone" data-dragging={dragging}
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
      </div>
      {error && <p className="query-error" role="alert">{error}</p>}
      {file && <div className="selected-file" role="status">
        <div><strong title={file.name}>{file.name}</strong><span>Selected locally · Not uploaded</span></div>
        <button type="button" onClick={() => setFile(null)}>Remove</button>
      </div>}
      <p className="upload-stage-note">File selection is available. Upload and processing will be connected in the next slice.</p>
    </section>
  );
}
