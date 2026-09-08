# BRD Knowledge

`brd-knowledge` is an enterprise RAG application for converting Business Requirement
Documents (BRDs) into a normalized, provenance-preserving knowledge workspace.

Current phase: dense retrieval, deterministic context construction, grounded answer
generation with Gemini and local Qwen adapters, and a document-centered React UI.
Parsing, normalization, structure-aware chunking, embeddings, PostgreSQL/pgvector
persistence, and retrieval evaluation are implemented. Lexical and hybrid RRF
retrieval remain experimental.

## Tech Stack

- Python 3.11
- FastAPI
- React 19, TypeScript, Vite, and Tailwind CSS
- Pydantic
- pydantic-settings
- Docling
- PyMuPDF
- pdfplumber
- python-docx
- SQLAlchemy 2
- Alembic
- PostgreSQL with psycopg
- pytest and pytest-asyncio
- ruff
- mypy

## Install

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Run Tests

```powershell
pytest
```

## Start API

```powershell
uvicorn brd_knowledge.main:app --reload
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## Upload and prepare a document

Uploads currently accept PDF files. The backend uses two separate requests:

1. `POST /documents/ingest` with multipart `file` parses and persists the document.
   Keep the returned `document_id`.
2. `POST /documents/{document_id}/index` prepares that saved document for search
   using the existing structure-aware chunker and embedding synchronization.
   It returns the existing document summary, including `indexing.status`.

Once the status is `ready`, use `POST /documents/{document_id}/questions`.
These requests wait for completion; there is no background job or progress API.
Preparing search requires the configured local GTE embedding model and database.

Indexing returns 404 for a missing document, 422 for invalid/failed/empty parsing
output, and 503 for a preparation failure. On a preparation failure, the parsed
document remains saved: retry the index endpoint without uploading again.
Repeating indexing skips unchanged embeddings. Partial parses may be indexed;
their `partial_success` parse status remains visible in the summary. Ingestion
returns 422 for a failed parse, including its document ID when output was saved.

## Frontend

The frontend opens on a PDF selection screen, with existing documents available
through **Previous documents**. File selection and drag-and-drop are local only
for now: the UI explicitly indicates that files have not been uploaded. Connecting
the upload/index APIs is the next slice. File size limits come from the backend
capabilities endpoint; this screen accepts only PDF when the backend allows it.

Previously indexed documents retain Gemini/Qwen selection, grounded questions,
authoritative sources, and collapsed retrieval and generation diagnostics.
Documents marked `not_indexed` or `needs_reindex` cannot be queried.

Keep the API running on `127.0.0.1:8000`, then start the Vite development server:

```sh
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` requests to FastAPI, so local
development does not require a permissive CORS configuration. Run frontend checks
with `npm test`, `npm run lint`, `npm run typecheck`, and `npm run build`.

## Database

Start PostgreSQL (Docker exposes port 5433 on localhost; the container uses 5432):

```powershell
docker compose up -d
```

## Confidentiality

Do not commit confidential BRD files. Use `data/samples/` only for synthetic or approved sample documents.

## One-question Gemini smoke test

Install the tested official SDK with the project dependencies:

```sh
uv pip install --python .venv/bin/python -e ".[dev]"
```

Set `GEMINI_API_KEY` in the ignored `.env` or inject it through the environment.
Use `.env.example` for configuration names; its key is only a placeholder.
Do not put keys in command arguments or commit them. Deployment should inject the
key from a secret manager. The application uses `SecretStr` and does not log it.

The script requires PostgreSQL with existing document/chunk embeddings and the
configured GTE ModernBERT revision available locally. It does not ingest documents.
Use only a document approved for transmission to the Gemini Developer API.
For example, if the IDS benchmark document is persisted and approved:

```sh
.venv/bin/python scripts/answer_question.py \
  "Approximately how many orders, trips, and delivery stops does IDS handle each day?" \
  --document-id 'IDS_BRD_V2_140526 (2).pdf' \
  --top-k 5 --max-characters 20000 --max-output-tokens 1024
```

This is a manual, potentially billable operation. Automated tests use fakes and
HTTP mock transports. No live Gemini test runs as part of pytest.

The CLI uses exact dense retrieval, unchanged `ContextBuilder` prefix packing,
`GroundedAnswerService`, Gemini, and application-side citation validation. It prints
JSON containing the answer, authoritative resolved citations, token usage,
returned model version, Gemini response ID (in `provider_request_id`), configuration,
context exclusions, and elapsed time. Output may contain confidential answer text;
handle any redirected output accordingly. The same orchestration is available at
`POST /documents/{document_id}/questions` for the frontend.

### Gemini baseline configuration

- SDK: `google-genai==2.22.0`; Developer API `v1beta`, stateless `generate_content`.
- Model: stable `gemini-3.1-flash-lite`; other model names are rejected so limits
  and thinking controls cannot silently become incorrect. No floating `latest`
  alias or automatic model fallback. Google does not publish an immutable dated
  snapshot here; `model_version` records what the service actually returned.
- Temperature: `1.0`, following Google's Gemini 3 recommendation. Thinking:
  explicitly `minimal`, which does not guarantee zero thinking. One candidate,
  JSON Schema output, no tools or automatic function calling.
- Timeout: 60 seconds per HTTP request, converted to the SDK's milliseconds.
  SDK attempts: 1 (initial attempt only); no application retries.
- Limits: 1,048,576 input tokens and 65,536 output tokens. The existing service
  conservatively requires counted input + output reserve + 128-token margin to
  fit within the input limit. This deliberately leaves extra headroom rather
  than treating Gemini's separate input and output limits as interchangeable.
- CLI output reserve: 1,024 tokens (including any thinking); the service's existing
  default remains 512. Character packing remains a separate coarse context cap.

Before generation, the adapter calls Gemini `countTokens` with the question,
evidence, system instructions, and response JSON schema in `generateContentRequest`.
SDK 2.22.0 does not expose these count fields in Developer API mode, so the adapter
uses its public `HttpOptions.extra_body` support for the documented REST shape.
HTTP mock tests compare counting and generation payloads. Live acceptance is still
to be checked by the manual smoke test. Counting sends the evidence remotely too;
empty context skips both counting and generation. Overflow prevents generation.

The adapter passes `ModelAnswerPayload.model_json_schema()` directly using
`response_json_schema` and `application/json`. No OpenAI-specific schema rewriting
is needed. Pydantic and citation validation remain authoritative. Malformed output,
blocked responses, incomplete output, and transport/provider failures surface as
typed errors; they are not converted into insufficient-evidence answers.
`output_tokens` reports answer candidate tokens; `thinking_tokens` is separate.
Provider usage can be absent; input usage then falls back to the preflight count.

Schema conformance and valid citation IDs do not establish that every claim is
supported. This slice adds neither answer evaluation nor semantic citation checks.
Quality, account access, actual latency, and live API behavior remain unmeasured.

The selected model is a low-cost baseline for bounded document questions, not a
proven BRD-quality winner. At documented standard paid text rates ($0.25/M input,
$1.50/M output including thinking), 5,000 input and 1,024 output tokens cost about
$0.00279. The newer 3.5 Flash-Lite costs more; there is no project evidence yet to
justify that increase. Confirm the key's project uses the intended paid billing
and data terms: free and paid tiers have different data-use policies. Credits
alone do not establish the applicable enterprise privacy or residency guarantees.

Official references (checked 2026-09-07):

- [Model capabilities and limits](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite)
- [Pricing and tier data use](https://ai.google.dev/gemini-api/docs/pricing)
- [Structured output](https://ai.google.dev/gemini-api/docs/structured-output)
- [Token-counting request shape](https://ai.google.dev/api/tokens)
- [Gemini 3 temperature guidance](https://ai.google.dev/gemini-api/docs/gemini-3)
- [Thinking controls](https://ai.google.dev/gemini-api/docs/generate-content/thinking)
- [Official Python SDK](https://googleapis.github.io/python-genai/)

## Local Qwen development option

Use `--provider local_qwen --model qwen3:8b` to run the same retrieval/context/
grounding pipeline locally through Ollama. Gemini is retained; `--provider gemini`
selects it explicitly. Without the flag, `LLM_PROVIDER` selects the provider and
its default remains `gemini`. Set `LLM_PROVIDER=local_qwen` in your ignored `.env`
to make repeated development runs local. There is no automatic provider fallback.
The local path requires no Gemini key.

The supported local profile is Qwen3 8B Q4_K_M, in non-thinking mode. On the inspected
M4 Mac with 16 GB RAM, Ollama 0.32.15 and `qwen3:8b` were already installed. Start
with 8,192 context tokens, 1,024 maximum output tokens, and a 180-second timeout.
Actual speed depends on prompt length, model loading, and memory pressure. The
local configuration accepts context windows from 2,048 through 40,960, but a larger
window uses more memory. Model overrides are restricted to supported profiles;
an arbitrary Ollama model must not silently use Qwen's tokenizer.

If needed on a new machine, install Ollama and download the local model:

```sh
ollama pull qwen3:8b
# Only if the Ollama app/server is not already running:
ollama serve
```

Download only the matching tokenizer and model configuration once (no second copy
of the model weights). From the repository root:

```sh
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen3-8B',
    revision='b968826d9c46dd6066d109eabc6255188de91218',
    allow_patterns=['tokenizer.json', 'tokenizer_config.json', 'config.json'],
    cache_dir='data/outputs/tokenizer_cache',
)
PY
```

These public files are cached in ignored outputs. At query time, tokenizer loading
is offline with remote code disabled. The adapter renders the official Qwen chat
template with `enable_thinking=False`, counts that exact prompt, and sends it to
Ollama `/api/generate` with `raw=true`. This avoids a second hidden chat template.
The response schema is supplied as a decoding grammar through `format`, rather
than inserted into the prompt. Existing grounding instructions and evidence are
unchanged. Ollama's returned prompt count must equal preflight; a discrepancy is
a typed failure, not an accepted potentially truncated answer.

```sh
.venv/bin/python scripts/answer_question.py \
  "Approximately how many orders, trips, and delivery stops does IDS handle each day?" \
  --document-id 'IDS_BRD_V2_140526 (2).pdf' \
  --provider local_qwen --model qwen3:8b \
  --top-k 5 --max-characters 20000 --max-output-tokens 1024
```

As with Gemini, the document must already have embeddings in PostgreSQL. Context
packing remains character-based; token preflight rejects overflow without changing
retrieval or trimming evidence. Set `LOCAL_QWEN_CONTEXT_TOKENS=16384` only if measured
requests need it and memory permits. Lowering `--max-characters` is an explicit
change to the evidence budget and affects comparison with Gemini.

Local generation uses Qwen's recommended non-thinking sampling (temperature 0.7,
top-p 0.8, top-k 20, min-p 0), seed 42, no streaming, and no retries. A fixed seed
helps repeatability but does not guarantee identical answers across runtime/model
versions. The model's installed digest is recorded in `model_version`; no request
ID is invented. The endpoint is restricted to loopback HTTP, environment proxies
and redirects are disabled, and the adapter never pulls models or contacts Gemini.
Ollama keeps the model loaded for five minutes between requests.

Structured output does not establish factual support. The existing Pydantic and
citation validation still run, and malformed/incomplete answers are errors. Use
local Qwen to exercise integration; use Gemini to assess Gemini's answer quality.
Normal automated tests mock both the tokenizer and HTTP transport and need neither
Ollama nor downloaded tokenizer files.

References:

- [Qwen3 non-thinking mode and sampling](https://huggingface.co/Qwen/Qwen3-8B)
- [Ollama raw generation and usage](https://docs.ollama.com/api/generate)
- [Ollama schema-constrained output](https://docs.ollama.com/capabilities/structured-outputs)

Local validation on this workspace: a synthetic evidence question completed through
Qwen and the grounding service with a valid resolved citation. Preflight and
Ollama both counted 224 input tokens; output was 41 tokens.

The Docker database uses `localhost:5433` to avoid conflicting with PostgreSQL
running directly on the Mac at port 5432. Set `DATABASE_URL` using `.env.example`;
Compose retains the existing `postgres_data` volume when recreating the container.
