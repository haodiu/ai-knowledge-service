"""ONE ChatModelClient for every OpenAI-compatible endpoint (Gemini today; DeepSeek/MiniMax/GLM
later are new registry entries, not new adapter code — provided they accept `json_schema`).

Structured output is forced with `response_format=json_schema`, then re-validated locally with
Pydantic: the provider's schema support is best-effort, our validation is not.

`max_retries=0` is REQUIRED, not extra caution: the openai SDK defaults to 2 and silently re-sends
on 429/5xx/timeouts. That turns one logical call into up to three requests that neither
`model_calls` nor MAX_GENERATIVE_LLM_CALLS (invariant #7) can see. Retries are an orchestration
decision (Week 4), made once, counted, and inside the graph timeout.
"""
import time
from collections.abc import Sequence
from typing import Any

import httpx2
import openai
from pydantic import BaseModel

from app.ai.chat.base import parse_structured
from app.ai.chat.schema import to_provider_schema
from app.ai.errors import (
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
    StructuredOutputError,
)
from app.ai.types import ModelMessage, ModelResponse, Purpose, Usage

_BAD_FINISH = {"length", "content_filter"}


def _retry_after(exc: openai.APIStatusError) -> float | None:
    raw = exc.response.headers.get("retry-after")
    try:
        return max(0.0, float(raw)) if raw is not None else None
    except ValueError:
        return None  # HTTP-date form: treat as unknown rather than guess


class OpenAICompatibleChatClient:
    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        http_client: httpx2.AsyncClient | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.provider = provider
        self.model_name = model
        self._temperature = temperature
        self._client = openai.AsyncOpenAI(
            api_key=api_key, base_url=base_url, max_retries=0, http_client=http_client
        )

    async def complete_structured[T: BaseModel](
        self,
        *,
        messages: Sequence[ModelMessage],
        response_model: type[T],
        purpose: Purpose,
        timeout_seconds: float,
    ) -> ModelResponse[T]:
        wire_messages: list[Any] = [{"role": m.role, "content": m.content} for m in messages]
        response_format: Any = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": to_provider_schema(response_model),
            },
        }
        started = time.perf_counter()
        try:
            completion = await self._client.chat.completions.create(
                model=self.model_name,
                messages=wire_messages,
                temperature=self._temperature,
                timeout=timeout_seconds,
                response_format=response_format,
            )
        except openai.RateLimitError as exc:
            raise ModelRateLimited(
                f"{self.provider} rate limited (HTTP 429)", retry_after_seconds=_retry_after(exc)
            ) from None
        except openai.APITimeoutError:
            raise ModelTimeout(f"{self.provider} call timed out") from None
        except openai.APIConnectionError:
            raise ModelUnavailable(f"{self.provider} connection failed") from None
        except openai.APIStatusError as exc:
            raise ModelUnavailable(f"{self.provider} returned HTTP {exc.status_code}") from None
        except openai.OpenAIError:
            raise ModelUnavailable(f"{self.provider} client error") from None
        latency_ms = int((time.perf_counter() - started) * 1000)

        if not completion.choices:
            raise StructuredOutputError("provider returned no choices")
        choice = completion.choices[0]
        if choice.finish_reason in _BAD_FINISH:
            raise StructuredOutputError(f"output not usable (finish_reason={choice.finish_reason})")
        content: Any = choice.message.content
        if not content:
            raise StructuredOutputError("provider returned empty content")

        parsed = parse_structured(content, response_model)
        usage = completion.usage
        return ModelResponse(
            parsed,
            self.provider,
            self.model_name,
            Usage(
                input_tokens=usage.prompt_tokens if usage else None,
                output_tokens=usage.completion_tokens if usage else None,
            ),
            latency_ms,
        )
