"""Every rejection path for verify_token (Plan §7, CLAUDE.md invariant #1).

Written before wiring the API dependency: the 401 boundary must be right on its own, testable
without any HTTP layer.
"""
import time

import jwt
import pytest
from pydantic import SecretStr

from app.auth.jwt import InvalidTokenError, verify_token
from app.retrieval.schemas import Tier
from app.settings import Settings

SECRET = "0" * 32
ISSUER = "payment-subscription-platform"
AUDIENCE = "rag-chatbot"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        environment="test",
        database_url=SecretStr("postgresql+psycopg://u:p@localhost:1/db"),
        redis_url=SecretStr("redis://localhost:1/0"),
        rabbitmq_url=SecretStr("amqp://u:p@localhost:1//"),
        jwt_secret=SecretStr(SECRET),
        jwt_issuer=ISSUER,
        jwt_audience=AUDIENCE,
        ingestion_service_token=SecretStr("svc"),
    )


def _token(
    *,
    secret: str = SECRET,
    sub: str | None = "user-1",
    tier: str | None = "general",
    iss: str | None = ISSUER,
    aud: str | None = AUDIENCE,
    exp_delta: float = 300,
    extra_claims: dict[str, object] | None = None,
    algorithm: str = "HS256",
) -> str:
    claims: dict[str, object] = {}
    if sub is not None:
        claims["sub"] = sub
    if tier is not None:
        claims["tier"] = tier
    if iss is not None:
        claims["iss"] = iss
    if aud is not None:
        claims["aud"] = aud
    claims["exp"] = time.time() + exp_delta
    if extra_claims:
        claims.update(extra_claims)
    return jwt.encode(claims, secret, algorithm=algorithm)


def test_valid_token_yields_the_right_authorization_context(settings: Settings) -> None:
    ctx = verify_token(_token(sub="user-42", tier="internal"), settings)
    assert (ctx.user_id, ctx.tier) == ("user-42", Tier.INTERNAL)


def test_expired_token_is_rejected(settings: Settings) -> None:
    # Well beyond jwt_leeway_seconds (clock-skew tolerance), so this is unambiguously expired.
    with pytest.raises(InvalidTokenError):
        verify_token(_token(exp_delta=-3600), settings)


def test_not_yet_valid_token_is_rejected(settings: Settings) -> None:
    with pytest.raises(InvalidTokenError):
        verify_token(_token(extra_claims={"nbf": time.time() + 3600}), settings)


def test_wrong_signature_is_rejected(settings: Settings) -> None:
    with pytest.raises(InvalidTokenError):
        verify_token(_token(secret="1" * 32), settings)


def test_wrong_issuer_is_rejected(settings: Settings) -> None:
    with pytest.raises(InvalidTokenError):
        verify_token(_token(iss="someone-else"), settings)


def test_wrong_audience_is_rejected(settings: Settings) -> None:
    with pytest.raises(InvalidTokenError):
        verify_token(_token(aud="someone-else"), settings)


def test_missing_issuer_is_rejected(settings: Settings) -> None:
    with pytest.raises(InvalidTokenError):
        verify_token(_token(iss=None), settings)


def test_missing_audience_is_rejected(settings: Settings) -> None:
    with pytest.raises(InvalidTokenError):
        verify_token(_token(aud=None), settings)


def test_missing_subject_is_rejected(settings: Settings) -> None:
    with pytest.raises(InvalidTokenError):
        verify_token(_token(sub=None), settings)


def test_missing_tier_claim_is_rejected(settings: Settings) -> None:
    with pytest.raises(InvalidTokenError):
        verify_token(_token(tier=None), settings)


def test_unknown_tier_value_is_rejected(settings: Settings) -> None:
    with pytest.raises(InvalidTokenError):
        verify_token(_token(tier="superadmin"), settings)


def test_none_algorithm_is_rejected(settings: Settings) -> None:
    """The classic "alg: none" forgery attempt."""
    claims = {
        "sub": "user-1", "tier": "general", "iss": ISSUER, "aud": AUDIENCE,
        "exp": time.time() + 300,
    }
    unsigned = jwt.encode(claims, key=None, algorithm="none")
    with pytest.raises(InvalidTokenError):
        verify_token(unsigned, settings)


def test_garbage_token_is_rejected(settings: Settings) -> None:
    with pytest.raises(InvalidTokenError):
        verify_token("not-a-jwt-at-all", settings)


def test_error_message_never_leaks_verification_detail(settings: Settings) -> None:
    with pytest.raises(InvalidTokenError) as exc_info:
        verify_token(_token(exp_delta=-3600), settings)
    assert "exp" not in str(exc_info.value).lower()
    assert "expired" not in str(exc_info.value).lower()
