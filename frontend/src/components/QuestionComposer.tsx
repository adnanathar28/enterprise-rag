import type { DocumentSummary, ProviderCapability, ProviderName } from "../api/types";

interface QuestionComposerProps {
  document: DocumentSummary;
  providers: ProviderCapability[];
  selectedProvider: ProviderName | null;
  question: string;
  submitting: boolean;
  onProviderChange: (provider: ProviderName) => void;
  onQuestionChange: (question: string) => void;
  onSubmit: () => void;
}

export function QuestionComposer({
  document,
  providers,
  selectedProvider,
  question,
  submitting,
  onProviderChange,
  onQuestionChange,
  onSubmit,
}: QuestionComposerProps) {
  const ready = document.indexing.status === "ready";
  const canSubmit = ready && Boolean(question.trim()) && Boolean(selectedProvider) && !submitting;

  return (
    <section className="question-section" aria-labelledby="question-heading">
      <div className="section-heading-row">
        <div>
          <p className="eyebrow">Question</p>
          <h2 id="question-heading">Ask about this document</h2>
        </div>
        <span className="keyboard-hint">⌘ ↵ to submit</span>
      </div>

      {!ready && (
        <p className="readiness-message" role="status">
          This document must be indexed with the active embedding configuration before it can be
          queried.
        </p>
      )}

      <form
        className="question-form"
        onSubmit={(event) => {
          event.preventDefault();
          if (canSubmit) onSubmit();
        }}
      >
        <label className="sr-only" htmlFor="document-question">
          Question about {document.filename}
        </label>
        <textarea
          id="document-question"
          value={question}
          disabled={!ready || submitting}
          placeholder="Ask a specific question about requirements, controls, rules, or scope…"
          rows={4}
          onChange={(event) => onQuestionChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && canSubmit) {
              event.preventDefault();
              onSubmit();
            }
          }}
        />
        <div className="composer-footer">
          <label className="provider-control">
            <span>Model provider</span>
            <select
              aria-label="Model provider"
              value={selectedProvider ?? ""}
              disabled={submitting}
              onChange={(event) => onProviderChange(event.target.value as ProviderName)}
            >
              {providers.map((provider) => (
                <option
                  key={provider.provider}
                  value={provider.provider}
                  disabled={!provider.configured}
                >
                  {provider.label} · {provider.model}
                  {!provider.configured ? " · Not configured" : ""}
                </option>
              ))}
            </select>
          </label>
          <button className="submit-button" type="submit" disabled={!canSubmit}>
            {submitting ? "Searching…" : "Ask question"}
          </button>
        </div>
      </form>
    </section>
  );
}
