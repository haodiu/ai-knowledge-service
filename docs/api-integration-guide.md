# API integration guide

For the host team (Spring Boot Payment & Subscription Platform) integrating with this
RAG chatbot service. This is a reference, not a walkthrough — for a scripted,
step-by-step run see [`DEMO.md`](../DEMO.md); for importable requests see
[`postman/rag-chatbot.postman_collection.json`](../postman/rag-chatbot.postman_collection.json);
for the architecture and internal invariants see [`CLAUDE.md`](../CLAUDE.md).

The system is single-tenant (no `tenant_id` anywhere) and integrates with the host in
**two directions**, shown in [`README.md`](../README.md)'s architecture diagram:

1. **Host → chatbot**: the host mints a JWT for its own authenticated user and calls
   the chat API (§3) on their behalf; the host also submits documents for indexing
   via the ingestion API (§5).
2. **Chatbot → host**: during a chat turn, the chatbot may call back into the host's
   own Subscription service (§6) to answer a subscription-related question. The host
   must implement that endpoint for the tool to be usable — it's optional; chat works
   without it, the tool just never gets offered.

Base URL, port, and TLS termination are deployment concerns not covered here (see
`docker-compose.yml`'s `API_HOST_PORT` for local dev — default `http://localhost:8000`).

## 1. Overview of endpoints

| Method & path | Auth | Purpose |
|---|---|---|
| `POST /v1/conversations` | user JWT | Create a conversation |
| `POST /v1/conversations/{id}/turns` | user JWT + rate limit | Ask a question (SSE) |
| `GET /v1/turns/{turn_id}/sources/{source_id}` | user JWT | Resolve a citation |
| `POST /internal/ingestions` | service token | Submit a document version |
| `GET /internal/ingestions/{job_id}` | service token | Poll an ingestion job |
| `GET /healthz` | none | Process liveness |
| `GET /readyz` | none | Dependency readiness |
| `GET {SUBSCRIPTION_SERVICE_BASE_URL}/subscriptions` (**host implements this**) | service token, sent by us | We call this during a turn |

## 2. Authentication

There are three distinct credentials in play. Never mix them up — they're separate
trust boundaries and none of them is interchangeable with another.

### 2.1 User JWT (host → chatbot, chat endpoints)

Verified in `app/auth/jwt.py`. **HS256 with a shared secret** — not RS256/JWKS (a
deliberate single-trusted-host decision, since this is not a public multi-issuer
scenario). The host mints one short-lived token per request/session for its own
already-authenticated user.

- Header: `Authorization: Bearer <token>`
- Signing algorithm: `HS256`, secret = `JWT_SECRET` (shared out-of-band with the
  chatbot's deployment).
- Required claims: `sub` (non-empty string — becomes `user_id`), `tier` (must be
  exactly `"general"` or `"internal"`, see §2.4), `iss` (must equal `JWT_ISSUER`),
  `aud` (must equal `JWT_AUDIENCE`), `exp`. `iat`/`nbf` are not required but are
  honored if present.
- Clock skew tolerance: `JWT_LEEWAY_SECONDS` (default `30`, configurable `0`–`300`)
  applied to `exp`/`nbf`.
- Any verification failure (bad signature, expired, wrong `iss`/`aud`, missing/invalid
  `sub` or `tier`) is collapsed into a single generic `401 {"detail": "invalid token"}`
  — the specific reason is never returned to the caller, on purpose (it's exactly the
  kind of detail an attacker would use to probe the boundary). A missing/malformed
  `Authorization` header is `401 {"detail": "missing bearer token"}`.

Minting example (Python, `PyJWT`):

```python
import jwt, time

token = jwt.encode(
    {
        "sub": "the-host-user-id",
        "tier": "general",  # or "internal"
        "iss": "payment-subscription-platform",  # must match JWT_ISSUER
        "aud": "rag-chatbot",                     # must match JWT_AUDIENCE
        "iat": int(time.time()),
        "exp": int(time.time()) + 300,
    },
    JWT_SECRET,
    algorithm="HS256",
)
```

Keep the token short-lived (minutes, not hours) — it is minted per request/session,
not stored long-term.

### 2.2 Service token (host → chatbot, ingestion endpoint)

A separate, simpler trust boundary: host-to-host, no user identity or tier involved
at all (`app/api/dependencies.py::get_service_auth`).

- Header: `X-Service-Token: <token>`
- Compared with a constant-time comparison against `INGESTION_SERVICE_TOKEN`.
- Failure: `401 {"detail": "invalid service token"}`.
- This token authorizes *document submission*, not chat — it carries no tier/user
  context, so don't try to reuse it as a stand-in for a user JWT.

### 2.3 Subscription-tool token (chatbot → host, outbound)

The reverse direction — see §6. Configured on the chatbot side as
`SUBSCRIPTION_SERVICE_TOKEN`; the host validates it, the chatbot never receives one.

### 2.4 Tier

`tier` is a flat claim on the user JWT — **not** a per-document or per-request
parameter the host can pass any other way. Two values only:

- `general` — can see `general`-tier content only.
- `internal` — can see `general` **and** `internal`-tier content.

Tier resolution and document authorization are deterministic application code on the
chatbot side; the host's only job is to put the correct value in the JWT claim.

## 3. Conversation & Turn API

### `POST /v1/conversations`

Creates a conversation owned by the caller (`auth.user_id` from the JWT).

- Auth: user JWT.
- Request body: none.
- Response `201 Created`: `{"id": "<uuid>"}`.

### `POST /v1/conversations/{conversation_id}/turns`

Ask a question in an existing conversation. Streams the answer over **SSE**
(`media_type: text/event-stream`).

- Auth: user JWT, plus rate limiting (§3.4).
- Request body:
  ```json
  { "question": "What is the refund window?" }
  ```
  `question`: 1–2000 characters (`Field(min_length=1, max_length=MAX_QUESTION_CHARS)`).
  An out-of-range value is rejected before the graph even runs (`422`, standard
  FastAPI validation) — this is a generous outer bound; the graph's own semantic
  limit can still reject a technically-in-range but nonsensical question as
  `blocked/invalid_question` (§4).
- Response headers: `X-Turn-Id: <uuid>` — set once, at the start of the stream. **Save
  this.** It's the only way to get `turn_id`, which you need to call the citation
  endpoint (§3.3); no SSE frame repeats it.
- Pre-stream failures (plain HTTP, not SSE — the stream hasn't started yet):
  - `404` — conversation doesn't exist, or exists but isn't owned by this JWT's
    `user_id`. **Both cases return the identical 404** — a distinct 403 would leak
    whether the conversation exists to a caller who doesn't own it.
  - `401` — missing/invalid JWT (see §2.1).
  - `429` — rate limited (see §3.4).
  - `422` — request body failed validation (e.g. `question` too long/empty).
- Once the stream starts, it always returns `200` — any failure past that point is
  communicated **inside** the SSE stream (§3.2), never as a raw 500 or a dropped
  connection. A genuinely unexpected server-side exception during streaming still
  surfaces as a well-formed `unavailable`/`internal_error` event, not a broken stream.

#### 3.1 SSE wire format

Every frame is standard SSE — `event: <name>\ndata: <json>\n\n` — one JSON object per
frame, no `id:` field, no periodic heartbeat/comment pings.

Sequence for one turn:

1. Zero or more **`event: status`**, `data: {"phase": "<phase>"}` — emitted live as
   the graph progresses. `phase` is one of `planning`, `retrieving`, `grading`,
   `generating`, `calling_tool`.
2. If the turn is answered, zero or more **`event: answer_chunk`**,
   `data: {"delta": "<text>"}` — progressive rendering of the answer, a few words at
   a time. Purely a UX pacing feature: concatenating every `delta` reproduces the
   final answer text exactly, and the terminal `answer` event below still carries the
   full text + citations regardless — a client that ignores `answer_chunk` entirely
   behaves identically.
3. Exactly one **terminal result event** (see §3.2 for the name/payload per outcome).
4. Always last: **`event: done`**, `data: {}`.

#### 3.2 Terminal event names and payloads

| Outcome (`TurnResult.status`) | SSE event name | Payload |
|---|---|---|
| `answered` | `answer` | `{"answer": "<text>", "citations": [{"source_id": "<uuid>", "document_title": "<str>", "version_no": <int>}, ...]}` |
| `clarification` | `clarification` | `{"clarification_question": "<text>"}` |
| `insufficient_evidence` | `insufficient_evidence` | `{"detail": "<code>"}` — see §4 |
| `blocked` | `blocked` | `{"detail": "<code>"}` — see §4 |
| `temporarily_unavailable` | `unavailable` | `{"detail": "<code>", "retry_after_seconds"?: <float>}` — see §4 |

`retry_after_seconds` is present only when the underlying failure carried one (a
`429` from the LLM provider or the subscription tool, or a rate-limited outcome) —
absent otherwise.

A **`citations`** entry gives you the pieces you already need to display a source
reference (title, version). To show the actual cited text, call §3.3.

If streaming itself throws an unexpected exception (a real bug, not a graph-modeled
outcome), you still get a well-formed terminal event: `event: unavailable`,
`data: {"detail": "internal_error"}`, then `done` — never a bare connection drop or
raw HTTP 500 mid-stream.

### `GET /v1/turns/{turn_id}/sources/{source_id}`

Resolves one citation (`source_id`, from an `answer` event's `citations` array) back
to the exact snapshot text it was drawn from.

- Auth: user JWT. `turn_id` must belong to a turn the caller owns — otherwise `404`
  (same not-found/not-owned collapsing as above).
- Response `200`: `{"source_id": "<uuid>", "document_title": "<str>", "version_no": <int>, "text_snapshot": "<str>"}`.
- Why a separate call instead of inlining the text: citations are an **immutable
  snapshot** taken at answer time, independent of whether the underlying document
  version is later deleted by retention cleanup — fetching it separately keeps the
  main answer payload small and lets old citations keep resolving indefinitely.

### 3.4 Rate limiting (turn endpoint only)

Applies **only** to `POST /v1/conversations/{id}/turns`, per `user_id` (the JWT's
`sub`), fixed 60-second window.

- Limit: `RATE_LIMIT_REQUESTS_PER_MINUTE` (default `20`).
- Over the limit: `429 {"detail": "rate limit exceeded"}`, with header
  `Retry-After: <integer seconds until the window resets>`.
- If Redis itself is unreachable: fails **open** by default
  (`RATE_LIMIT_FAIL_OPEN=true`) — the request goes through rather than blocking chat
  on a rate-limiter outage. If configured fail-closed, an outage instead returns
  `503 {"detail": "rate limiter unavailable"}`.
- No rate limiting exists on `POST /v1/conversations` (creation), the citation
  endpoint, or the ingestion endpoints.

## 4. Error & result reference

Every terminal SSE event carries a `detail` string (except `answer`/`clarification`,
which are the two "successful" shapes). `detail` is always exactly one of the values
below — nothing else is ever produced; an unrecognized internal state is normalized
to `unexpected_state` rather than left unnamed. Use this table to drive client-side
error handling/messaging.

### `insufficient_evidence` — nothing went wrong, there just isn't enough to answer from. Not retryable in the same form; rephrasing may help.

| `detail` | Meaning |
|---|---|
| `no_retrieval_needed` | The planner decided no retrieval/tool was needed, but then had nothing to answer with. |
| `no_evidence` | Retrieval and/or the subscription tool genuinely found nothing. Also where a tool `404` (not found **or** not authorized — indistinguishable by design) lands. |
| `grader_insufficient` | Evidence was retrieved but judged insufficient to answer, and no valid rewrite was possible. |
| `rewrite_rejected` | A rewritten query was proposed but rejected (blank, too long, or identical to the original). |
| `rewrite_no_new_evidence` | A rewrite was attempted but found nothing new before the retry budget ran out. |
| `generator_insufficient` | The answering step itself decided the evidence didn't support an answer. |

### `clarification` — the system needs more from the user before it can proceed. `clarification_question` carries the actual question to show.

| `detail` (informational only — status is already `clarification`) | Meaning |
|---|---|
| `planner_clarification` | The planner judged the question itself too ambiguous. |
| `generator_clarification` | The draft answer step asked for clarification. |
| `tool_ambiguous` | The subscription tool's identifier resolved to more than one record (host returned `409`). |

### `blocked` — a deterministic safety/validation gate rejected the turn. Not retryable by resending the same input; the input or the system state must change.

| `detail` | Meaning |
|---|---|
| `invalid_citations` | The drafted answer cited something outside the evidence set — blocked before it could ever be streamed (this is the citation-validation invariant; it never trusts the LLM's own citations). |
| `invalid_question` | The question itself failed validation up front. |
| `retrieval_attempts_exceeded` | Hit the hard cap on retrieval attempts for this turn. |
| `structured_output_invalid` | A model's output didn't parse into the expected schema, even after one repair attempt. |

### `temporarily_unavailable` — an infrastructure/provider failure. Safe to retry later; check `retry_after_seconds` when present.

| `detail` | Meaning | `retry_after_seconds`? |
|---|---|---|
| `timeout` | An LLM call timed out. | no |
| `rate_limited` | An LLM provider returned `429`. | yes |
| `unavailable` | An LLM provider is down or misconfigured. | no |
| `budget_exhausted` | The turn's fixed LLM-call budget was used up. | no |
| `embedding_dimension` / `embedding_invalid` | An embedding call returned an unusable vector. | no |
| `graph_timeout` | The whole turn exceeded its hard wall-clock budget (30s). | no |
| `persistence_failed` | An answer was fully validated but couldn't be durably saved — so it was **not** returned rather than being served unpersisted. | no |
| `tool_error` | Generic subscription-tool failure code (rarely seen directly). | no |
| `tool_timeout` | The subscription-tool HTTP call timed out. | no |
| `tool_rate_limited` | The host's subscription endpoint returned `429`. | yes |
| `tool_unavailable` | The subscription endpoint is down, misconfigured, or returned an unexpected status/body. | no |
| `model_error` | Generic base code; not raised directly in practice. | no |

### Never expect to see these

`internal_error`, `unexpected_state`, `graph_recursion_limit` are reserved for
genuine server-side bugs — they are never a legitimate outcome of a well-formed
request. If you observe one, that's a bug report for the chatbot team, not a client
input to correct.

## 5. Ingestion API

Host-to-host only (service token, §2.2). No rate limiting on this router.

### `POST /internal/ingestions`

Submit one document version. Content is versioned per `external_id` — resubmitting
identical content is a safe no-op; the pipeline is idempotent by design (chunk +
embed + activate is one atomic step on the chatbot side).

Request body:

```json
{
  "external_id": "refund-policy",
  "title": "Refund policy",
  "tier": "general",
  "content": "... full document text ...",
  "embedding_model": "gemini-embedding-001"
}
```

All fields required. `tier` must be `"general"` or `"internal"` (invalid value →
`422`, standard FastAPI/pydantic enum validation, before the handler runs).

Response `202 Accepted` always on success:

```json
{
  "job_id": "<uuid-or-null>",
  "status": "queued | completed | failed",
  "document_external_id": "refund-policy",
  "version_no": 3
}
```

- `status: "completed"`, `job_id: null` — the submitted content is byte-identical to
  the currently active version; nothing was queued, there's no job to poll.
- `status: "queued"`, `job_id: <uuid>` — a new version is being chunked/embedded in
  the background; poll it (below).
- `status: "failed"` — the content is identical to a version that was already
  **superseded** by something newer (i.e. this would be a content rollback), which
  this API does not support. `job_id` still points at an audit row you can poll for
  the record, but nothing was queued. This is a `202`, not a `4xx` — it's a
  meaningful terminal outcome, not a malformed request.
- `400 Bad Request` — any other rejection (e.g. tier mismatch against an existing
  document, or the document is in a non-accepting state). `detail` is a plain string.
- `401` — invalid/missing service token.

### `GET /internal/ingestions/{job_id}`

Poll a job's status.

Response `200`:

```json
{
  "job_id": "<uuid>",
  "status": "queued | processing | retrying | completed | failed | superseded",
  "document_external_id": "refund-policy",
  "version_no": 3,
  "error_code": "<string-or-null>",
  "retry_count": 0
}
```

`error_code` is set only when `status` is `failed`/`retrying`; besides
`superseded_content` (the rollback-rejection case above), it can be any normalized
provider-error code (`timeout`, `rate_limited`, `unavailable`,
`structured_output_invalid`, `embedding_dimension`, `embedding_invalid`) if the
transient/permanent failure happened during embedding.

`404` if `job_id` is unknown.

## 6. Outbound: Subscription tool contract (the host must implement this)

During a turn, the chatbot may — **at its own discretion, based on the question** —
call **out** to the host to look up a live subscription record and use it as
evidence for the answer. This is entirely optional to support: if
`SUBSCRIPTION_SERVICE_BASE_URL`/`SUBSCRIPTION_SERVICE_TOKEN` aren't configured on the
chatbot side, this tool is simply never offered — chat still works, just without
subscription-lookup capability.

### `GET {SUBSCRIPTION_SERVICE_BASE_URL}/subscriptions`

Request, sent by the chatbot:

- Query params: **exactly one** of `subscription_id` or `customer_id` (1–64 chars
  each). Neither or both present is a chatbot-side bug (never sent in practice).
- Headers:
  - `X-Service-Token: <SUBSCRIPTION_SERVICE_TOKEN>` — the host validates this.
  - `X-On-Behalf-Of: <user_id>` — the asking user's id (from their verified JWT's
    `sub`). **The raw user JWT is deliberately never forwarded** — the host receives
    only the bare user id and is the sole authority on whether that user may see the
    record. There is no channel for the model to choose *whose* subscription is
    read, only *which one of that user's* — enforcing that boundary is the host's
    job, using `X-On-Behalf-Of`.
- Timeout: `SUBSCRIPTION_TOOL_TIMEOUT_SECONDS` (chatbot-side config, default `5s`,
  max `10s`) — the host should respond well within that.

Response contract, by status code:

| Status | Meaning | Chatbot behavior |
|---|---|---|
| `200` | Found and authorized. Body below. | Used as evidence for the answer. |
| `404` | Not found **or** not authorized for this user — return the identical `404` for both; do not distinguish them. | Treated as "no evidence" — the turn proceeds as if nothing was found, never a distinguishable error. |
| `409` | The identifier resolved to more than one record. | Turn becomes a `clarification` asking the user for a more specific identifier. |
| `429` | Rate limited. `Retry-After` header (**numeric seconds only** — an HTTP-date value is not parsed and is treated as unknown). | Turn ends `temporarily_unavailable`/`tool_rate_limited`, with `retry_after_seconds` if parsed. |
| timeout / connection error | — | `temporarily_unavailable`/`tool_timeout` or `tool_unavailable`. |
| any other status, or an unparsable `200` body | — | `temporarily_unavailable`/`tool_unavailable`. |

`200` response body (extra fields are ignored, so the host may add fields freely
without breaking this integration):

```json
{
  "subscription_id": "sub_123",
  "customer_id": "cus_456",
  "status": "active",
  "plan_name": "Pro",
  "current_period_end": "2026-10-01T00:00:00Z",
  "observed_at": "2026-09-25T10:00:00Z"
}
```

Required: `subscription_id`, `customer_id`, `status`, `observed_at` (all strings —
`observed_at`/`current_period_end` are passed through as-is, not parsed as
datetimes on the chatbot side). Optional: `plan_name`, `current_period_end`.

## 7. Health checks

No auth required on either endpoint. Intended for your own load balancer/orchestrator
probes, not for end users.

### `GET /healthz`

Process liveness only — touches no dependency. Always `200 {"status": "ok"}` if the
process is up at all.

### `GET /readyz`

Checks dependencies: `postgres`, `rabbitmq` (**required** — failing either fails
readiness), and `redis` (best-effort — a Redis failure degrades but doesn't fail
readiness, since Redis in this system is rate-limiting only, never load-bearing for
correctness).

```json
{ "status": "ok | degraded | unavailable", "checks": {"postgres": "ok", "rabbitmq": "ok", "redis": "ok"} }
```

- `status: "ok"`, `200` — everything healthy.
- `status: "degraded"`, `200` — Redis down, Postgres/RabbitMQ fine. Chat and
  ingestion both still function (rate limiting fails open by default).
- `status: "unavailable"`, `503` — Postgres or RabbitMQ down. Route traffic away.

Each probe has a bounded timeout (`HEALTH_CHECK_TIMEOUT_SECONDS`, default `2s`).

## 8. Configuration reference

Env vars relevant to integration (see `app/settings.py` for the complete list,
including LLM-role config that's internal to the chatbot and not integrator-facing).

| Env var | Required? | Default | Purpose |
|---|---|---|---|
| `JWT_SECRET` | yes | — | HS256 shared secret for verifying the host-minted user JWT (§2.1). |
| `JWT_ISSUER` | yes | — | Must match the `iss` claim the host mints. |
| `JWT_AUDIENCE` | yes | — | Must match the `aud` claim the host mints. |
| `JWT_LEEWAY_SECONDS` | no | `30` | Clock-skew tolerance for `exp`/`nbf`, range `0`–`300`. |
| `INGESTION_SERVICE_TOKEN` | yes | — | Shared secret for `X-Service-Token` on `/internal/ingestions` (§2.2). |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | no | `20` | Per-user turn-creation limit (§3.4). |
| `RATE_LIMIT_FAIL_OPEN` | no | `true` | Whether a Redis outage blocks chat (`false`) or lets it through (`true`). |
| `SUBSCRIPTION_SERVICE_BASE_URL` | no | unset | Base URL of the host's Subscription service (§6). Tool is disabled if unset. |
| `SUBSCRIPTION_SERVICE_TOKEN` | no | unset | Shared secret the chatbot sends as `X-Service-Token` when calling out (§6). |
| `SUBSCRIPTION_TOOL_TIMEOUT_SECONDS` | no | `5.0` | Timeout for the outbound subscription call, max `10`. |
| `HEALTH_CHECK_TIMEOUT_SECONDS` | no | `2.0` | Per-probe timeout for `/readyz`, max `10`. |

In the local `docker-compose.yml` dev stack, `JWT_ISSUER` defaults to
`payment-subscription-platform` and `JWT_AUDIENCE` to `rag-chatbot` if unset — a
production deployment should still set these explicitly rather than rely on that
fallback.

## 9. See also

- [`DEMO.md`](../DEMO.md) — a full worked walkthrough (ingest → track job → ask →
  update → re-ask → open citation), including a copy-pasteable real-HTTP appendix.
- [`postman/rag-chatbot.postman_collection.json`](../postman/rag-chatbot.postman_collection.json) —
  importable requests for the conversation/turn/source endpoints.
- [`CLAUDE.md`](../CLAUDE.md) — architecture, invariants, and conventions, for anyone
  working on this repo's own code (not needed just to integrate against it).
