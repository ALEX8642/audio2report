"""Ollama LLM provider.

Communicates with the Ollama REST API directly using Python's built-in
``urllib`` — no extra dependencies required.

Ollama must be running locally (or at the configured base_url).
Quickstart: https://ollama.com — ``ollama run llama3``
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator

from audio2report._log import get_logger
from audio2report.config import LLMConfig

logger = get_logger(__name__)


class OllamaProvider:
    """
    Wraps the Ollama ``/api/generate`` endpoint.

    Parameters
    ----------
    config:
        LLM configuration section.  ``base_url`` should point to the Ollama
        server root (default: ``http://localhost:11434``).
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._generate_url = config.base_url.rstrip("/") + "/api/generate"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _request(self, payload: dict, timeout: int = 600) -> urllib.request.Request:
        return urllib.request.Request(
            self._generate_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def _payload(self, prompt: str, stream: bool) -> dict:
        return {
            "model": self._config.model,
            "prompt": prompt,
            "stream": stream,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Return the full response for *prompt* (blocking, no streaming)."""
        req = self._request(self._payload(prompt, stream=False))
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self._generate_url}. "
                "Is Ollama running?  Try: ollama serve"
            ) from exc

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield response tokens as they arrive from Ollama."""
        req = self._request(self._payload(prompt, stream=True))
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self._generate_url}. "
                "Is Ollama running?  Try: ollama serve"
            ) from exc
