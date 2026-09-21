from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Week 1 settings only. Connection URLs embed credentials, hence SecretStr."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: Literal["local", "test", "production"] = "local"

    database_url: SecretStr  # postgresql+psycopg://user:pass@host:5432/db
    redis_url: SecretStr  # redis://host:6379/0
    rabbitmq_url: SecretStr  # amqp://user:pass@host:5672//

    health_check_timeout_seconds: float = Field(default=2.0, gt=0, le=10)


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # values come from the environment
