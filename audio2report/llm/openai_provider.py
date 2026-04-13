"""OpenAI-compatible LLM provider.

Works with:
  - OpenAI API               (base_url=https://api.openai.com/v1, api_key=sk-...)
  - Local LM Studio          (base_url=http://localhost:1234/v1, api_key=lm-studio)
  - llama.cpp server         (base_url=http://localhost:8080/v1, api_key=none)
  - Ollama OpenAI compat     (base_url=http://localhost:11434/v1, api_key=ollama)

Requires: pip install 'audio2report[llm]'   (installs the openai package)
"""
from __future__ import annotations

import os
from typing import Iterator

from audio2report._log import get_logger
from audio2report.config import LLMConfig

logger = get_logger(__name__)


class OpenAIProvider:
    """
    Wraps the OpenAI Python library against any OpenAI-compatible endpoint.

    Parameters
    ----------
    config:
        LLM configuration section.  ``base_url`` is forwarded directly to the
        ``openai.OpenAI`` client.  ``api_key`` takes precedence over the
        ``OPENAI_API_KEY`` environment variable.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    def _client(self):
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "openai package is not installed.  "
                "Run: pip install 'audio2report[llm]'"
            ) from exc

        api_key = (
            self._config.api_key
            or os.environ.get("OPENAI_API_KEY")
            or "sk-no-key-required"   # works for local servers that don't check
        )
        return openai.OpenAI(
            api_key=api_key,
            base_url=self._config.base_url,
        )

    def generate(self, prompt: str) -> str:
        """Return the full response (blocking)."""
        client = self._client()
        response = client.chat.completions.create(
            model=self._config.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield response tokens as they arrive."""
        client = self._client()
        for chunk in client.chat.completions.create(
            model=self._config.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ):
            token = chunk.choices[0].delta.content
            if token:
                yield token
