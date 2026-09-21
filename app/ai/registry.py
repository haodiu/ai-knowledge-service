"""ModelRegistry (Plan §11.3): one ChatModelClient per role + the EmbeddingClient.

Adding another OpenAI-compatible provider is a new entry here, not new adapter code.
"""
from dataclasses import dataclass

from app.ai.chat.base import ChatModelClient
from app.ai.chat.fake import FakeChatModelClient, demo_responder
from app.ai.chat.openai_compat import OpenAICompatibleChatClient
from app.ai.embeddings.base import EmbeddingClient
from app.ai.embeddings.fake import FakeEmbeddingClient
from app.ai.embeddings.gemini import GeminiEmbeddingClient
from app.ai.errors import ConfigurationError
from app.settings import Settings


@dataclass(frozen=True)
class ModelRegistry:
    planner: ChatModelClient
    grader: ChatModelClient
    answer: ChatModelClient
    embeddings: EmbeddingClient


def build_registry(settings: Settings) -> ModelRegistry:
    if settings.gemini_api_key is None:
        raise ConfigurationError("GEMINI_API_KEY is required for real models (or use --fake)")
    key = settings.gemini_api_key.get_secret_value()

    def chat(model: str) -> OpenAICompatibleChatClient:
        return OpenAICompatibleChatClient(
            provider=settings.chat_provider,
            base_url=settings.chat_base_url,
            api_key=key,
            model=model,
        )

    return ModelRegistry(
        planner=chat(settings.planner_model),
        grader=chat(settings.grader_model),
        answer=chat(settings.answer_model),
        embeddings=GeminiEmbeddingClient(
            api_key=key, timeout_seconds=settings.embedding_timeout_seconds
        ),
    )


def build_fake_registry() -> ModelRegistry:
    def fake(name: str) -> FakeChatModelClient:
        return FakeChatModelClient(responder=demo_responder, model_name=name)

    return ModelRegistry(
        planner=fake("fake-planner"),
        grader=fake("fake-grader"),
        answer=fake("fake-answer"),
        embeddings=FakeEmbeddingClient(),
    )
