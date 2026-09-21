from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

GEMINI_OPENAI_COMPAT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _pinned_model_name(value: str) -> str:
    """Normalise a model id and refuse floating aliases.

    Google's model listing returns "models/<name>" but the OpenAI-compatible endpoint wants the
    bare name. "*-latest" aliases silently move to a newer model, which would make a demo
    irreproducible, so they are rejected rather than tolerated.
    """
    name = value.strip().removeprefix("models/")
    if not name:
        raise ValueError("model name must not be blank")
    if name.endswith("-latest"):
        raise ValueError(f"{name!r} is a floating alias; pin an exact model version")
    return name


class Settings(BaseSettings):
    """Connection URLs embed credentials, hence SecretStr. Model roles: Plan §11.4."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: Literal["local", "test", "production"] = "local"

    database_url: SecretStr  # postgresql+psycopg://user:pass@host:5432/db
    redis_url: SecretStr  # redis://host:6379/0
    rabbitmq_url: SecretStr  # amqp://user:pass@host:5672//

    health_check_timeout_seconds: float = Field(default=2.0, gt=0, le=10)

    # --- LLM roles (Plan §11.4). Exact, confirmed names; change by config, never in the graph.
    chat_provider: str = "gemini"  # label recorded in model_calls.provider
    chat_base_url: str = GEMINI_OPENAI_COMPAT_URL
    gemini_api_key: SecretStr | None = None  # optional: only --fake runs work without it
    planner_model: str = "gemini-3.1-flash-lite"
    grader_model: str = "gemini-3.1-flash-lite"
    answer_model: str = "gemini-3.6-flash"
    embedding_model: str = "gemini-embedding-001"
    prompt_version: str = "v1"
    # Per-call bound; the whole graph has GRAPH_TIMEOUT_SECONDS=30 (Plan §8.4).
    chat_timeout_seconds: float = Field(default=20.0, gt=0, le=30)
    embedding_timeout_seconds: float = Field(default=15.0, gt=0, le=30)

    @field_validator("planner_model", "grader_model", "answer_model", "embedding_model")
    @classmethod
    def _pin_model(cls, value: str) -> str:
        return _pinned_model_name(value)


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # values come from the environment
