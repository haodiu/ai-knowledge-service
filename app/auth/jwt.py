"""Verify the host-minted JWT (Plan §7) into a trusted AuthorizationContext.

HS256, shared secret (Week 6 decision -- see CLAUDE.md/plan discussion; not RS256/JWKS). This is
the ONLY place `tier`/`user_id` may be read from a token; everything downstream (retrieval,
authorization, the graph) only ever sees the resulting `AuthorizationContext`, never the raw JWT.
"""
import logging

import jwt as pyjwt

from app.auth.policies import AuthorizationContext
from app.retrieval.schemas import Tier
from app.settings import Settings

_LOG = logging.getLogger(__name__)


class InvalidTokenError(Exception):
    """Verification failed for any reason. The caller (an API dependency) must map this to a
    generic 401 -- never let the specific reason (expired vs. bad signature vs. bad claim) reach
    the client; that is exactly the kind of detail an attacker would use to probe the boundary."""


def verify_token(token: str, settings: Settings) -> AuthorizationContext:
    """Verify signature/exp/nbf/iss/aud, then build an AuthorizationContext from `sub`/`tier`.

    Never accepts `user_id` or `tier` from anywhere but this verified token (CLAUDE.md invariant
    #1: authorization is deterministic application code, never the LLM, and never client input).
    """
    try:
        payload = pyjwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=["HS256"],
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
            leeway=settings.jwt_leeway_seconds,
            options={"require": ["exp", "sub", "iss", "aud"]},
        )
    except pyjwt.InvalidTokenError as exc:
        # Log the failure class only -- never the token itself (it may still be a live credential).
        _LOG.warning("JWT verification failed: %s", type(exc).__name__)
        raise InvalidTokenError("token verification failed") from exc

    user_id = payload.get("sub")
    tier_claim = payload.get("tier")
    if not isinstance(user_id, str) or not user_id:
        _LOG.warning("JWT verification failed: missing or empty 'sub' claim")
        raise InvalidTokenError("token verification failed")
    try:
        tier = Tier(tier_claim) if isinstance(tier_claim, str) else None
    except ValueError:
        tier = None
    if tier is None:
        _LOG.warning("JWT verification failed: invalid 'tier' claim %r", tier_claim)
        raise InvalidTokenError("token verification failed")

    return AuthorizationContext(user_id=user_id, tier=tier)
