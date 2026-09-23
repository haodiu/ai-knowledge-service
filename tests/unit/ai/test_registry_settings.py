"""Model names are pinned settings; ONE adapter class serves every chat role/provider."""
import pytest
from pydantic import SecretStr, ValidationError

from app.ai.chat.openai_compat import OpenAICompatibleChatClient
from app.ai.embeddings.fake import FakeEmbeddingClient
from app.ai.embeddings.gemini import GeminiEmbeddingClient
from app.ai.errors import ConfigurationError
from app.ai.registry import build_fake_registry, build_registry
from app.settings import Settings


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_env_file=None` does not hide real environment variables, so a developer's local
    ANSWER_MODEL / GEMINI_API_KEY override must not leak into assertions about the defaults."""
    for name in ("PLANNER_MODEL", "GRADER_MODEL", "ANSWER_MODEL", "EMBEDDING_MODEL",
                 "PROMPT_VERSION", "CHAT_BASE_URL", "CHAT_PROVIDER", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _settings(**over: object) -> Settings:
    base: dict[str, object] = {
        "environment": "test",
        "database_url": SecretStr("postgresql+psycopg://u:p@localhost:1/db"),
        "redis_url": SecretStr("redis://localhost:1/0"),
        "rabbitmq_url": SecretStr("amqp://u:p@localhost:1//"),
        "jwt_secret": SecretStr("test-jwt-secret"),
        "jwt_issuer": "test-issuer",
        "jwt_audience": "test-audience",
        "ingestion_service_token": SecretStr("test-service-token"),
    }
    base.update(over)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def test_model_defaults_are_the_confirmed_pinned_names() -> None:
    s = _settings()
    assert (s.planner_model, s.grader_model) == ("gemini-3.1-flash-lite",) * 2
    assert s.answer_model == "gemini-3.6-flash"
    assert s.embedding_model == "gemini-embedding-001"
    assert s.prompt_version == "v1"
    assert s.chat_base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"


def test_models_prefix_is_stripped_because_the_compat_endpoint_wants_bare_names() -> None:
    s = _settings(answer_model="models/gemini-3.6-flash", planner_model="models/x")
    assert s.answer_model == "gemini-3.6-flash" and s.planner_model == "x"


@pytest.mark.parametrize("name", ["gemini-flash-latest", "gemini-flash-lite-latest", "x-latest"])
def test_floating_latest_aliases_are_rejected_to_keep_demos_reproducible(name: str) -> None:
    for field in ("planner_model", "grader_model", "answer_model", "embedding_model"):
        with pytest.raises(ValidationError):
            _settings(**{field: name})


def test_model_names_cannot_be_blank() -> None:
    with pytest.raises(ValidationError):
        _settings(planner_model="  ")


def test_real_registry_uses_one_adapter_class_for_all_roles() -> None:
    reg = build_registry(_settings(gemini_api_key=SecretStr("k")))
    assert all(
        type(c) is OpenAICompatibleChatClient for c in (reg.planner, reg.grader, reg.answer)
    )
    assert (reg.planner.model_name, reg.grader.model_name, reg.answer.model_name) == (
        "gemini-3.1-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash")
    assert reg.planner.provider == "gemini"
    assert isinstance(reg.embeddings, GeminiEmbeddingClient)


def test_real_registry_without_an_api_key_fails_closed() -> None:
    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        build_registry(_settings(gemini_api_key=None))


def test_api_key_is_a_secret_in_settings_repr() -> None:
    assert "supersecret" not in repr(_settings(gemini_api_key=SecretStr("supersecret")))


def test_fake_registry_needs_no_key() -> None:
    reg = build_fake_registry()
    assert isinstance(reg.embeddings, FakeEmbeddingClient)
