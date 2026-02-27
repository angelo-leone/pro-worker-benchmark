"""
Model interface for the Pro-Worker AI Benchmark.
Wraps litellm to provide a unified interface for calling any LLM.
Supports Ollama (local), OpenRouter (API), and direct API providers.
"""

import asyncio
import time
from dataclasses import dataclass, field

import litellm


@dataclass
class ModelConfig:
    """Configuration for a model to be evaluated."""
    id: str
    name: str
    provider: str
    api_base: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class ModelResponse:
    """Response from a model call."""
    content: str
    model_id: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


class ModelClient:
    """Unified client for calling LLMs via litellm."""

    def __init__(self, config: ModelConfig):
        self.config = config
        # Suppress litellm verbose logging
        litellm.suppress_debug_info = True

    def call(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 5,
    ) -> ModelResponse:
        """Send messages to the model and return the response.
        Retries with exponential backoff on rate limit errors."""
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        last_error = None
        for attempt in range(max_retries):
            start = time.perf_counter()
            try:
                response = litellm.completion(
                    model=self.config.id,
                    messages=full_messages,
                    temperature=temp,
                    max_tokens=tokens,
                    api_base=self.config.api_base,
                )
                latency = (time.perf_counter() - start) * 1000

                content = response.choices[0].message.content or ""
                usage = response.usage
                return ModelResponse(
                    content=content,
                    model_id=self.config.id,
                    latency_ms=latency,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                )
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if "rate" in error_str or "429" in error_str or "ratelimit" in error_str:
                    wait = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
                    time.sleep(wait)
                    continue
                # Non-rate-limit error, don't retry
                latency = (time.perf_counter() - start) * 1000
                return ModelResponse(
                    content="",
                    model_id=self.config.id,
                    latency_ms=latency,
                    error=str(e),
                )

        # All retries exhausted
        return ModelResponse(
            content="",
            model_id=self.config.id,
            latency_ms=0,
            error=f"Rate limited after {max_retries} retries: {last_error}",
        )

    async def call_async(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        """Async version of call() for concurrent execution."""
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        start = time.perf_counter()
        try:
            response = await litellm.acompletion(
                model=self.config.id,
                messages=full_messages,
                temperature=temp,
                max_tokens=tokens,
                api_base=self.config.api_base,
            )
            latency = (time.perf_counter() - start) * 1000

            content = response.choices[0].message.content or ""
            usage = response.usage
            return ModelResponse(
                content=content,
                model_id=self.config.id,
                latency_ms=latency,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return ModelResponse(
                content="",
                model_id=self.config.id,
                latency_ms=latency,
                error=str(e),
            )


def build_client_from_dict(model_dict: dict) -> ModelClient:
    """Create a ModelClient from a config dictionary (as loaded from YAML)."""
    config = ModelConfig(
        id=model_dict["id"],
        name=model_dict["name"],
        provider=model_dict["provider"],
        api_base=model_dict.get("api_base"),
        temperature=model_dict.get("temperature", 0.7),
        max_tokens=model_dict.get("max_tokens", 2048),
    )
    return ModelClient(config)
