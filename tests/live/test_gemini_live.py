"""LIVE probes against the real Gemini API. Skipped unless GEMINI_API_KEY is in the environment.

    GEMINI_API_KEY=... pytest -q -m live tests/live

These are the only tests that can show (a) the API honours output_dimensionality=1536 and
(b) Gemini's OpenAI-compatible endpoint accepts our json_schema for all three contracts. The
mocked-transport unit tests cannot. Free tier: run sparingly, a 429 here is not a code bug.
"""
import math
import os

import pytest

from app.ai.chat.openai_compat import OpenAICompatibleChatClient
from app.ai.embeddings.gemini import GeminiEmbeddingClient
from app.ai.schemas import AnswerDraft, EvidenceGrade, QueryPlan
from app.ai.types import ModelMessage
from app.db.models import EMBEDDING_DIM
from app.settings import Settings

pytestmark = pytest.mark.live
KEY = os.environ.get("GEMINI_API_KEY")


@pytest.fixture
def s() -> Settings:
    if not KEY:
        pytest.skip("GEMINI_API_KEY not set")
    return Settings(_env_file=None, environment="test", database_url="postgresql://x/x",  # type: ignore[arg-type]
                    redis_url="redis://x", rabbitmq_url="amqp://x", jwt_secret="x",  # type: ignore[arg-type]
                    jwt_issuer="x", jwt_audience="x", ingestion_service_token="x")  # type: ignore[arg-type]


async def test_live_embedding_is_1536_dims_and_normalised(s: Settings) -> None:
    client = GeminiEmbeddingClient(api_key=KEY or "")
    resp = await client.embed(["refund policy"], model_version=s.embedding_model, kind="query")
    (vec,) = resp.vectors
    assert len(vec) == EMBEDDING_DIM
    assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-6)


async def test_live_document_and_query_task_types_give_different_vectors(s: Settings) -> None:
    client = GeminiEmbeddingClient(api_key=KEY or "")
    (q,) = (await client.embed(["refund"], model_version=s.embedding_model, kind="query")).vectors
    doc = await client.embed(["refund"], model_version=s.embedding_model, kind="document")
    (d,) = doc.vectors
    assert q != d


@pytest.mark.parametrize(
    ("model", "prompt"),
    [
        (QueryPlan, 'Plan a search for: "What is the refund window?" intent=policy.'),
        (EvidenceGrade, "Grade: question 'refund window?' evidence 'Refunds within 14 days.'"),
        (AnswerDraft, "Answer 'hello' with status insufficient_evidence and no citations."),
    ],
)
async def test_live_json_schema_is_accepted_for_each_contract(
    s: Settings, model: type, prompt: str
) -> None:
    client = OpenAICompatibleChatClient(
        provider="gemini", base_url=s.chat_base_url, api_key=KEY or "", model=s.planner_model
    )
    resp = await client.complete_structured(
        messages=[ModelMessage(role="user", content=prompt)],
        response_model=model, purpose="plan", timeout_seconds=30,
    )
    assert isinstance(resp.parsed, model)
