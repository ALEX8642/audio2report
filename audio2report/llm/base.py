"""LLM provider abstraction.

All providers expose the same two-method interface so the report engine
doesn't care which backend is running.
"""
from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from audio2report.config import LLMConfig


@runtime_checkable
class AbstractLLMProvider(Protocol):
    def generate(self, prompt: str) -> str:
        """Return the complete response for *prompt* (blocking)."""
        ...

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield response tokens incrementally."""
        ...


def get_provider(config: LLMConfig) -> AbstractLLMProvider:
    """Return the correct provider instance for *config.provider*."""
    if config.provider == "ollama":
        from audio2report.llm.ollama_provider import OllamaProvider
        return OllamaProvider(config)
    elif config.provider in ("openai", "openai_compatible"):
        from audio2report.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(config)
    else:
        raise ValueError(
            f"Unknown LLM provider: {config.provider!r}. "
            "Valid options: ollama, openai"
        )
