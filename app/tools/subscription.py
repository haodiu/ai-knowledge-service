"""The subscription tool (Plan §10): a read-only, on-behalf-of HTTP call to the host's
`get_subscription` endpoint, plus turning its result into an `Evidence` entry so it flows through
the exact same build_evidence/grade/generate/validate pipeline as a retrieved chunk.

Uses plain `httpx` (a direct dev/runtime dependency), not `httpx2` (the `openai` SDK's transitive
fork used only inside `app/ai/chat/openai_compat.py`) -- this is the first outbound HTTP call in
the codebase that is not a provider SDK call, so it gets its own, unrelated client.

Authorization (invariant #1, #4): the call authenticates itself to the host with a service
credential (`X-Service-Token`, mirroring `app/api/dependencies.py::get_service_auth`'s boundary
for ingestion) and forwards only `ctx.auth.user_id` (`X-On-Behalf-Of`) -- never the raw user JWT.
`GetSubscriptionArgs` has no `user_id` field, so the model has no channel to choose *whose*
subscription is read, only *which* one; the host is the sole authority on whether that user may
see it. A 404 is returned identically whether the record does not exist or the user is not
authorized for it (Plan §10's own table) -- this client does not try to tell those apart either.
"""
import uuid
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from app.ai.errors import ConfigurationError
from app.ai.schemas import GetSubscriptionArgs
from app.retrieval.schemas import Evidence
from app.settings import Settings
from app.tools.errors import (
    ToolAmbiguous,
    ToolNotFound,
    ToolRateLimited,
    ToolTimeout,
    ToolUnavailable,
)

# Fixed constant, never derived from anything request-scoped: only its combination with a
# per-request string (see `subscription_to_evidence`) needs to be unique, not this namespace.
_TOOL_EVIDENCE_NAMESPACE = uuid.UUID("2b6e2b0a-2c1a-4c8e-9b7c-9b6a9b2b6e2b")


class SubscriptionSnapshot(BaseModel):
    """The host's response body (Plan §10: `200` + `observed_at`).

    `extra="ignore"`, not `"forbid"`: this describes a third-party HTTP response we do not own,
    not one of our own structured-output contracts -- the host may add fields over time without
    breaking this client.
    """

    model_config = ConfigDict(extra="ignore")

    subscription_id: str
    customer_id: str
    status: str
    plan_name: str | None = None
    current_period_end: str | None = None
    observed_at: str


class SubscriptionToolClient(Protocol):
    async def get_subscription(
        self, args: GetSubscriptionArgs, *, user_id: str, timeout_seconds: float
    ) -> SubscriptionSnapshot:
        """Returns the snapshot on 200, or raises a `ToolError` subclass (Plan §10's table:
        404 -> ToolNotFound, 409 -> ToolAmbiguous, 429 -> ToolRateLimited, timeout/5xx ->
        ToolTimeout/ToolUnavailable). No retry here -- same rule as `ChatModelClient`: a retry is
        an orchestration decision made once, by the caller, never hidden in the adapter."""
        ...


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None  # HTTP-date form: treat as unknown rather than guess


class HttpSubscriptionToolClient:
    def __init__(
        self,
        *,
        base_url: str,
        service_token: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = service_token
        self._client = http_client or httpx.AsyncClient()

    async def get_subscription(
        self, args: GetSubscriptionArgs, *, user_id: str, timeout_seconds: float
    ) -> SubscriptionSnapshot:
        params = (
            {"subscription_id": args.subscription_id}
            if args.subscription_id is not None
            else {"customer_id": args.customer_id}
        )
        try:
            response = await self._client.get(
                f"{self._base_url}/subscriptions",
                params=params,
                headers={"X-Service-Token": self._token, "X-On-Behalf-Of": user_id},
                timeout=timeout_seconds,
            )
        except httpx.TimeoutException:
            raise ToolTimeout("subscription host call timed out") from None
        except httpx.HTTPError as exc:
            raise ToolUnavailable(f"subscription host connection failed: {type(exc).__name__}") \
                from None

        if response.status_code == 200:
            try:
                return SubscriptionSnapshot.model_validate(response.json())
            except (ValueError, ValidationError) as exc:
                raise ToolUnavailable(f"subscription host returned an unparsable body: {exc!s}") \
                    from None
        if response.status_code == 404:
            raise ToolNotFound("subscription not found or not authorized")
        if response.status_code == 409:
            raise ToolAmbiguous("identifier did not resolve to exactly one subscription")
        if response.status_code == 429:
            raise ToolRateLimited(
                "subscription host rate limited (HTTP 429)",
                retry_after_seconds=_retry_after(response),
            )
        raise ToolUnavailable(f"subscription host returned HTTP {response.status_code}")


def subscription_to_evidence(snapshot: SubscriptionSnapshot, *, request_id: str) -> Evidence:
    """A tool result, shaped as an `Evidence` entry so it needs zero special-casing anywhere else
    (`build_evidence`, `validate_citations`, `_user_with_evidence`'s nonce-delimited untrusted-data
    block all take a plain `Sequence[Evidence]`).

    UUIDs are `uuid5`-derived from `request_id` (the turn id, unique per turn) so re-running the
    same turn is deterministic, and provably non-colliding with a real chunk id: Postgres
    `gen_random_uuid()` always produces a `uuid4` (version nibble 4), `uuid5` always fixes that
    nibble to 5 -- the two id spaces can never intersect.
    """
    doc = uuid.uuid5(_TOOL_EVIDENCE_NAMESPACE, f"{request_id}:get_subscription:document")
    ver = uuid.uuid5(_TOOL_EVIDENCE_NAMESPACE, f"{request_id}:get_subscription:version")
    chk = uuid.uuid5(_TOOL_EVIDENCE_NAMESPACE, f"{request_id}:get_subscription:chunk")
    text_lines = [
        f"subscription_id: {snapshot.subscription_id}",
        f"customer_id: {snapshot.customer_id}",
        f"status: {snapshot.status}",
    ]
    if snapshot.plan_name is not None:
        text_lines.append(f"plan_name: {snapshot.plan_name}")
    if snapshot.current_period_end is not None:
        text_lines.append(f"current_period_end: {snapshot.current_period_end}")
    text_lines.append(f"observed_at: {snapshot.observed_at}")
    return Evidence(
        chunk_id=chk,
        document_version_id=ver,
        document_id=doc,
        title=f"Subscription {snapshot.subscription_id}",
        version_no=0,
        chunk_index=0,
        text="\n".join(text_lines),
        # A high sentinel score: live subscription data answering a "subscription"/"hybrid"
        # question is always the most relevant single fact available. NOTE: this score alone does
        # NOT protect it from build_evidence's char/count cap once it moves from `new` into
        # `existing` on a later retrieve call (the bounded-rewrite loop) -- build_evidence ranks
        # ALL of `new` ahead of ALL of `existing` regardless of score. `is_tool_evidence` below is
        # what actually keeps it pinned across a rewrite; see app/graph/nodes/retrieval.py.
        score=1e9,
    )


def is_tool_evidence(e: Evidence) -> bool:
    """True for an `Evidence` synthesised by `subscription_to_evidence` above -- `uuid5` always
    sets the UUID version nibble to 5, whereas a real chunk id is Postgres `gen_random_uuid()`
    (`uuid4`, version nibble 4). A structural check, not a heuristic; see that function's
    docstring for the non-collision argument this relies on.

    `retrieve_node` (app/graph/nodes/retrieval.py) uses this to re-pin tool evidence ahead of a
    second `build_evidence` call after a rewrite, so it is never silently evicted by a full page
    of freshly retrieved chunks the way an ordinary `existing` entry could be.
    """
    return e.chunk_id.version == 5


def build_tool_client(settings: Settings) -> SubscriptionToolClient:
    """Mirrors `app/ai/registry.py::build_registry`: fail closed at construction time rather than
    on the first call. No real Payment/Subscription host exists yet, so these settings are
    optional (`app/settings.py`) -- the fake/`--fake` path is the only one that works without
    them, same precedent as `gemini_api_key`."""
    if (
        settings.subscription_service_base_url is None
        or settings.subscription_service_token is None
    ):
        raise ConfigurationError(
            "SUBSCRIPTION_SERVICE_BASE_URL and SUBSCRIPTION_SERVICE_TOKEN are required for the "
            "real subscription tool (or use the fake client)"
        )
    return HttpSubscriptionToolClient(
        base_url=settings.subscription_service_base_url,
        service_token=settings.subscription_service_token.get_secret_value(),
    )
