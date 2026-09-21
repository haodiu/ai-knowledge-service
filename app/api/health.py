import asyncio
import logging
from collections.abc import Mapping
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel

from app.api.dependencies import Probe, get_probes, get_settings_dep
from app.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# Plan §15: readiness fails when PostgreSQL/RabbitMQ are down; Redis may be degraded.
# The Redis fail-open/closed policy is a Week 6 decision (Plan §13.3).
REQUIRED = frozenset({"postgres", "rabbitmq"})


class LivenessResponse(BaseModel):
    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    checks: dict[str, Literal["ok", "fail"]]


@router.get("/healthz", response_model=LivenessResponse)
async def healthz() -> LivenessResponse:
    """Process liveness only — deliberately touches no dependency."""
    return LivenessResponse(status="ok")


async def _run(name: str, probe: Probe, timeout: float) -> tuple[str, bool]:
    try:
        await asyncio.wait_for(probe(), timeout=timeout)
    except Exception as exc:
        # Log the exception type only: messages can embed connection URLs/credentials.
        logger.warning("readiness probe failed: %s (%s)", name, type(exc).__name__)
        return name, False
    return name, True


@router.get("/readyz", response_model=ReadinessResponse)
async def readyz(
    response: Response,
    probes: Annotated[Mapping[str, Probe], Depends(get_probes)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> ReadinessResponse:
    results = dict(
        await asyncio.gather(
            *(_run(n, p, settings.health_check_timeout_seconds) for n, p in probes.items())
        )
    )
    checks: dict[str, Literal["ok", "fail"]] = {
        n: "ok" if ok else "fail" for n, ok in results.items()
    }

    if any(not results[n] for n in REQUIRED):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(status="unavailable", checks=checks)
    if not all(results.values()):
        return ReadinessResponse(status="degraded", checks=checks)
    return ReadinessResponse(status="ok", checks=checks)
