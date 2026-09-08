import type {
  ContextEvidence,
  DocumentQuestionResponse,
  ResolvedCitation,
} from "../api/types";

interface AnswerPanelProps {
  result: DocumentQuestionResponse;
}

function pageLabel(pageStart: number, pageEnd: number): string {
  return pageStart === pageEnd ? `Page ${pageStart}` : `Pages ${pageStart}–${pageEnd}`;
}

function sectionLabel(citation: ResolvedCitation, evidence?: ContextEvidence): string {
  if (citation.section_path.length) return citation.section_path.join(" / ");
  if (evidence?.section_title) return evidence.section_title;
  return "Document evidence";
}

function AnswerText({ result }: AnswerPanelProps) {
  const citationIds = new Set(result.answer.citations.map((citation) => citation.evidence_id));
  const parts = result.answer.answer_text.split(/(\[E\d+\])/g);

  return (
    <p className="answer-copy">
      {parts.map((part, index) => {
        const evidenceId = /^\[(E\d+)\]$/.exec(part)?.[1];
        if (!evidenceId || !citationIds.has(evidenceId)) {
          return <span key={`${part}-${index}`}>{part}</span>;
        }
        return (
          <a className="inline-citation" href={`#evidence-${evidenceId}`} key={`${part}-${index}`}>
            {evidenceId}
          </a>
        );
      })}
    </p>
  );
}

export function AnswerPanel({ result }: AnswerPanelProps) {
  const evidenceById = new Map(
    result.context.evidence.map((evidence) => [evidence.evidence_id, evidence]),
  );

  return (
    <div className="result-stack" aria-live="polite">
      <section className="answer-section" aria-labelledby="answer-heading">
        <div className="answered-question">
          <span>Question</span>
          <p>{result.question}</p>
        </div>
        <div className="answer-heading-row">
          <div>
            <h2 id="answer-heading">
              {result.answer.insufficient_evidence ? "Insufficient evidence" : "Grounded response"}
            </h2>
          </div>
          <span className="answer-timing">{result.elapsed_seconds.toFixed(2)}s</span>
        </div>
        <AnswerText result={result} />
      </section>

      <section className="sources-section" aria-labelledby="sources-heading">
        <div className="section-heading-row">
          <div>
            <h2 id="sources-heading">Sources</h2>
          </div>
          <span className="source-count">
            {result.answer.citations.length} {result.answer.citations.length === 1 ? "source" : "sources"}
          </span>
        </div>

        {result.answer.citations.length === 0 ? (
          <p className="no-sources">No evidence was cited for this response.</p>
        ) : (
          <div className="source-list">
            {result.answer.citations.map((citation) => {
              const evidence = evidenceById.get(citation.evidence_id);
              return (
                <article
                  className="source-row"
                  id={`evidence-${citation.evidence_id}`}
                  key={citation.evidence_id}
                >
                  <div className="source-index">{citation.evidence_id}</div>
                  <div className="source-content">
                    <div className="source-title-row">
                      <h3>{sectionLabel(citation, evidence)}</h3>
                      <span>{pageLabel(citation.page_start, citation.page_end)}</span>
                    </div>
                    {evidence && <p className="source-excerpt">{evidence.text}</p>}
                    <details className="source-expansion">
                      <summary>View full evidence & provenance</summary>
                      {evidence && <pre className="evidence-text">{evidence.text}</pre>}
                      <dl className="source-metadata">
                        <div><dt>Document</dt><dd>{citation.document_id}</dd></div>
                        <div><dt>Chunk</dt><dd>{citation.chunk_id}</dd></div>
                        <div><dt>Source blocks</dt><dd>{citation.source_block_ids.join(", ") || "None"}</dd></div>
                        <div><dt>Source tables</dt><dd>{citation.source_table_ids.join(", ") || "None"}</dd></div>
                      </dl>
                      {citation.quality_notes.length > 0 && <p>{citation.quality_notes.join(" · ")}</p>}
                      <details className="provenance-details">
                        <summary>Provenance records</summary>
                        <pre className="evidence-text">{JSON.stringify(citation.provenance, null, 2)}</pre>
                      </details>
                    </details>
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </section>

      <TechnicalDetails result={result} />
    </div>
  );
}

function TechnicalDetails({ result }: AnswerPanelProps) {
  const metadata = result.answer.metadata;
  const contextUsage = Math.round(
    (result.context.used_characters / result.context.max_characters) * 100,
  );

  return (
    <details className="technical-details">
      <summary>
        <span>Retrieved evidence & technical details</span>
        <span className="technical-summary">
          {result.retrieved_chunks.length} retrieved · {result.context.evidence.length} included
        </span>
      </summary>
      <div className="technical-body">
        <dl className="metadata-grid">
          <div>
            <dt>Provider</dt>
            <dd>{metadata.provider}</dd>
          </div>
          <div>
            <dt>Model</dt>
            <dd>{metadata.model_version ?? metadata.model}</dd>
          </div>
          <div>
            <dt>Prompt tokens</dt>
            <dd>{metadata.prompt_tokens?.toLocaleString() ?? "Unavailable"}</dd>
          </div>
          <div>
            <dt>Output tokens</dt>
            <dd>{metadata.output_tokens?.toLocaleString() ?? "Unavailable"}</dd>
          </div>
          <div>
            <dt>Context budget</dt>
            <dd>
              {result.context.used_characters.toLocaleString()} /{" "}
              {result.context.max_characters.toLocaleString()} chars ({contextUsage}%)
            </dd>
          </div>
          <div>
            <dt>Request ID</dt>
            <dd className="mono">{metadata.provider_request_id ?? "Unavailable"}</dd>
          </div>
        </dl>

        <div className="retrieval-table" role="table" aria-label="Retrieved evidence">
          <div className="retrieval-header" role="row">
            <span role="columnheader">Rank</span>
            <span role="columnheader">Section</span>
            <span role="columnheader">Page</span>
            <span role="columnheader">Similarity</span>
          </div>
          {result.retrieved_chunks.map((chunk) => (
            <div className="retrieval-row" role="row" key={chunk.chunk_id}>
              <span role="cell">{chunk.rank}</span>
              <span role="cell">{chunk.section_title ?? "Untitled section"}</span>
              <span role="cell">{pageLabel(chunk.page_start, chunk.page_end)}</span>
              <span role="cell">{(chunk.similarity * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>

        <div className="retrieved-excerpts">
          {result.retrieved_chunks.map((chunk) => (
            <details key={chunk.chunk_id}>
              <summary>#{chunk.rank} · {chunk.section_title ?? "Untitled section"} · View chunk</summary>
              <pre className="evidence-text">{chunk.text}</pre>
              <p className="mono">{chunk.chunk_id}</p>
            </details>
          ))}
        </div>

        {result.context.exclusions.length > 0 && (
          <details className="exclusions-note">
            <summary>{result.context.exclusions.length} excluded from context</summary>
            <pre className="evidence-text">{JSON.stringify(result.context.exclusions, null, 2)}</pre>
          </details>
        )}
      </div>
    </details>
  );
}
