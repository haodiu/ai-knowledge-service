# RAG Chatbot Service

A single-tenant RAG (Retrieval-Augmented Generation) chatbot backend.

- **Online path** (FastAPI + LangGraph): a user asks a question → the app plans the query,
  retrieves evidence, has an LLM draft an answer from that evidence only, checks the citations
  are real, then streams the answer back.
- **Offline path** (Celery + RabbitMQ): a document is submitted → it's chunked, embedded, and
  versioned in the background. The new version only goes live once it's fully indexed, so no one
  ever sees a half-updated knowledge base.

Both paths meet only at PostgreSQL + pgvector, the single source of truth.

## Architecture

```mermaid
flowchart TB
    subgraph online["Online path — FastAPI + LangGraph, synchronous"]
        direction TB
        A1["POST /v1/conversations/{id}/turns<br/>JWT verified, tier resolved deterministically"]
        A2["LLM #1: plan query<br/>(+ optional tool proposal)"]
        A3["deterministic: hybrid retrieval<br/>(active version + tier filtered)<br/>+ authorized tool call"]
        A4["LLM #2: grade evidence<br/>(one bounded rewrite + retry)"]
        A5["LLM #3: draft answer<br/>+ citations"]
        A6["deterministic: validate citations<br/>(set-membership against evidence)"]
        A7["SSE: status events,<br/>then ONE validated answer"]
        A1 --> A2 --> A3 --> A4 --> A5 --> A6 --> A7
    end

    subgraph offline["Offline path — Celery + RabbitMQ, async"]
        direction TB
        B1["POST /internal/ingestions<br/>service token"]
        B2["document_version(building)<br/>+ ingestion_job(queued)"]
        MQ[["RabbitMQ"]]
        B4["Celery worker:<br/>chunk + embed"]
        B5["atomic activation:<br/>old→superseded, new→active,<br/>knowledge_version += 1"]
        B1 --> B2 --> MQ --> B4 --> B5
    end

    DB[("PostgreSQL + pgvector<br/>source of truth")]
    A3 -->|read| DB
    B5 -->|write| DB

    R[("Redis<br/>rate-limiting only — never a cache")]
    A1 -.->|per-user quota check| R
```

## Quick start

```bash
cp .env.example .env                  # fill in secrets; .env is gitignored
docker compose up -d --build --wait   # api, worker, postgres, redis, rabbitmq
```

Migrations run automatically when the `api` container starts — no separate step needed.

```bash
# Ingest a document (no API key needed by default)
python -m app.ingestion.cli ingest ./documents

# Ask a question (add --fake to skip needing a real API key)
python -m app.ai.cli ask "What is the refund window?" --tier general --fake
```

Drop `--fake` and set `GEMINI_API_KEY` in `.env` to use a real model.

## Run tests

```bash
pytest -q tests/unit   # fast, no database or broker needed
pytest -q              # full suite, needs DATABASE_URL
```

## Learn more

- [`CLAUDE.md`](CLAUDE.md) — architecture, invariants, and conventions for contributors.
- [`DEMO.md`](DEMO.md) — a step-by-step demo walkthrough.
- [`docs/api-integration-guide.md`](docs/api-integration-guide.md) — the HTTP/SSE
  interface reference for the host team: auth, endpoints, SSE event format, error
  codes, and the outbound Subscription-tool contract the host must implement.
- [`docs/document-conversion-prompt.md`](docs/document-conversion-prompt.md) — an
  AI-agent prompt for converting PDF/Word/etc. into the `.md`+front-matter shape
  `app/ingestion` expects, before ingesting.

