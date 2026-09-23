import pytest
from pydantic import SecretStr

from app.settings import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        environment="test",
        database_url=SecretStr("postgresql+psycopg://u:p@localhost:1/db"),
        redis_url=SecretStr("redis://localhost:1/0"),
        rabbitmq_url=SecretStr("amqp://u:p@localhost:1//"),
        health_check_timeout_seconds=0.2,
        jwt_secret=SecretStr("test-jwt-secret"),
        jwt_issuer="test-issuer",
        jwt_audience="test-audience",
        ingestion_service_token=SecretStr("test-service-token"),
    )
