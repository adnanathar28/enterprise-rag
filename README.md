# BRD Knowledge

`brd-knowledge` is an early-stage backend project for converting Business Requirement Documents (BRDs) into a normalized, provenance-preserving JSON representation.

Current phase: dense retrieval, deterministic context construction, and grounded
answer generation with a Gemini adapter. Parsing, normalization, structure-aware
chunking, embeddings, PostgreSQL/pgvector persistence, and retrieval evaluation
are implemented. Lexical and hybrid RRF retrieval remain experimental.

## Tech Stack

- Python 3.11
- FastAPI
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

## Database

Start PostgreSQL:

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
handle any redirected output accordingly. There is no query API route.

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
