# Demo script

A reproducible, step-by-step walkthrough proving the Definition of Done's required demo (Plan
§19): **cập nhật policy → theo dõi job → chatbot dùng version mới → citation mở đúng nguồn** —
update a policy document, track the ingestion job, have the chatbot answer from the new version,
and open the citation back to the correct source.

Self-contained: it uses `app/eval/golden/corpus/refund-policy.md`, the same file the eval corpus
uses, so nothing extra needs to be created. Every command below already exists in the codebase —
this script adds no new code, only a sequence to run, and every step here has been run against a
live `docker compose up` stack while writing this. All commands run **inside the `api` container**
(`docker exec rag-chatbot-api-1 ...`): the host has no direct network path to RabbitMQ
(`docker-compose.yml` publishes only Postgres and the RabbitMQ management UI to the host, not the
AMQP port), and the app image's files are owned `root:root`, not writable by the `app` user the
container runs as — so editing the policy file also happens on the host, then `docker cp`'d in,
never edited in place inside the container.

## Setup

```bash
docker compose up -d --build --wait
```

**A note on which database this runs against.** The steps below use whatever database the compose
stack is already wired to (`docker-compose.yml`'s `POSTGRES_DB`, normally `chatbot`). If that
database has accumulated many unrelated documents from earlier work, `--fake` mode's evidence
picker (see the note in step 2) can plausibly cite a different, unrelated document instead of the
refund policy — it has no real judgment, only "was anything retrieved at all" (confirmed: this is
exactly what happens against this repo's own long-lived dev database after a full week of prior
ingestion testing). For a guaranteed, deterministic `--fake` run, start from a database that has
never had anything else ingested into it — a freshly-created one, or `docker compose down -v &&
docker compose up -d --build --wait` to reset everything including Postgres data. The commands
are identical either way; only the reliability of `--fake` mode's citation differs. Dropping
`--fake` (real Gemini models) is reliable even in a populated database — see step 2.

## 1. Ingest v1 of the policy

```bash
docker exec rag-chatbot-api-1 python -m app.ingestion.cli ingest app/eval/golden/corpus/refund-policy.md
```

Expect: `refund-policy.md: activated (v1)` (or a higher version number if this document was
already ingested before — version numbers only ever increase, they are not reset by re-running
this script).

## 2. Ask a question — answer + citation at the current version

```bash
docker exec rag-chatbot-api-1 python -m app.ai.cli ask "What is the refund window?" --tier general --fake
```

Expect an `answered` result whose `sources:` line shows a `document_version_id` matching the
`Refund Policy` title in the `evidence:` list above it, and an answer mentioning **14 days** (the
current wording of `refund-policy.md`).

`--fake` needs no `GEMINI_API_KEY` and is fully deterministic, but its evidence-picker
(`demo_responder`, see `app/ai/chat/fake.py`) just cites "whatever ranked first" — it cannot judge
whether that is actually the right document among many (see the setup note above). Dropping
`--fake` uses real Gemini models instead: retrieval and grading correctly rank the true refund
policy first even in a database with many other unrelated documents (verified against this
repo's own dev database), but the pinned `answer_model` (`gemini-3.6-flash`) has known,
documented 503 flakiness — `.env.example`'s dev-only workaround is
`docker exec -e ANSWER_MODEL=gemini-3.1-flash-lite rag-chatbot-api-1 ...` if you hit it. Real
calls are also subject to Gemini's free-tier rate limits (CLAUDE.md: "sparing on free tier").

## 3. Change the policy and re-ingest — queued through Celery this time

Edit the policy on the host (the container's files are not writable by the user it runs as), then
copy the edited file in and ingest it from there. The file's *external_id* (its path relative to
the ingest root) must stay `refund-policy.md` for this to become a new version of the same
document rather than a new one — copying to `/tmp/refund-policy.md` and ingesting it as a single
file achieves that:

```bash
cp app/eval/golden/corpus/refund-policy.md /tmp/refund-policy-edited.md
sed -i 's/14 days/30 days/' /tmp/refund-policy-edited.md
docker cp /tmp/refund-policy-edited.md rag-chatbot-api-1:/tmp/refund-policy.md
docker exec rag-chatbot-api-1 python -m app.ingestion.cli ingest /tmp/refund-policy.md --queue
```

Expect: `refund-policy.md: queued (job <job_id>, vN)`, `N` one higher than step 1's version. Note
the `job_id`.

## 4. Track the job

```bash
docker exec rag-chatbot-postgres-1 psql -U chatbot -d chatbot \
    -c "SELECT status, retry_count, error_code FROM ingestion_jobs WHERE id = '<job_id>';"
```

(Substitute your actual `POSTGRES_DB`/user if you changed them from `.env`'s defaults.) Poll a
few times — expect `status` to move `queued → processing → completed` within a couple of seconds.
See the "optional: over real HTTP" appendix below for the equivalent via
`GET /internal/ingestions/{job_id}` instead of a direct DB query — useful to also show the
structured JSON log line the worker emits for this job (`docker logs rag-chatbot-celery-worker-1`,
message `"ingestion job summary"`, Week 8 observability).

## 5. Ask the same question again — answer + citation now at the new version

```bash
docker exec rag-chatbot-api-1 python -m app.ai.cli ask "What is the refund window?" --tier general --fake
```

The new version's chunk is present in the `evidence:` list (confirmed: retrieval and version
activation are correct regardless of --fake/real mode), proving the chatbot has already switched
to it without version 1 ever having stopped serving traffic while the new version was building
(invariant #9). Whether the `sources:` citation itself lands on that chunk, versus an unrelated
one, again depends on the same `--fake`-mode evidence-picker limitation from step 2 — drop `--fake`
for a reliable citation in a populated database, or run this whole script against a fresh one for
a reliable citation in `--fake` mode too.

## Optional: the same demo over real HTTP + SSE

The CLI path above already fully satisfies the DoD's demo requirement with zero new code. This
appendix is the same idea over the real `POST /v1/conversations/{id}/turns` SSE endpoint instead
of the CLI, for a presenter who wants to show the actual host-facing API. It needs a hand-minted
JWT (the CLI's `--tier` flag stands in for a verified JWT claim everywhere else in this codebase
too — Week 6's decision, see `app/ai/cli.py`'s docstring):

```bash
docker exec rag-chatbot-api-1 python - <<'PY'
import jwt, time, os
print(jwt.encode(
    {"sub": "demo-user", "tier": "general", "iss": os.environ.get("JWT_ISSUER", "payment-subscription-platform"),
     "aud": os.environ.get("JWT_AUDIENCE", "rag-chatbot"), "iat": int(time.time()), "exp": int(time.time()) + 300},
    os.environ["JWT_SECRET"], algorithm="HS256",
))
PY
```

```bash
TOKEN=<paste the token above>
CONV=$(curl -s -X POST http://localhost:8000/v1/conversations \
    -H "Authorization: Bearer $TOKEN" | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
curl -N -X POST "http://localhost:8000/v1/conversations/$CONV/turns" \
    -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
    -d '{"question": "What is the refund window?"}'
```

For step 4's job tracking over HTTP instead of a direct DB query:

```bash
docker exec rag-chatbot-api-1 curl -s "http://localhost:8000/internal/ingestions/<job_id>" \
    -H "X-Service-Token: $INGESTION_SERVICE_TOKEN"
```
