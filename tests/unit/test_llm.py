"""Unit tests for the LLM module (M4).

Covers:
- Template loading (built-in and file path)
- build_prompt() — substitution, truncation
- load_transcript_text() — JSON and TXT
- get_provider() factory — correct class returned, unknown provider raises
- OllamaProvider.generate() and .stream() — mocked urllib
- OpenAIProvider.generate() and .stream() — mocked openai client
- generate_report() — streaming and blocking paths
"""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from audio2report.config import LLMConfig
from audio2report.llm.base import AbstractLLMProvider, get_provider
from audio2report.llm.ollama_provider import OllamaProvider
from audio2report.llm.openai_provider import OpenAIProvider
from audio2report.llm.report import (
    build_prompt,
    generate_report,
    load_transcript_text,
)
from audio2report.models import RunMeta

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_config(**kwargs) -> LLMConfig:
    defaults = dict(
        enabled=True,
        provider="ollama",
        model="llama3",
        base_url="http://localhost:11434",
        prompt_template="audit_report",
        max_transcript_chars=50_000,
        stream=False,
    )
    defaults.update(kwargs)
    return LLMConfig(**defaults)


def _make_meta(**kwargs) -> RunMeta:
    defaults = dict(
        root="/tmp/test",
        prime_folders=["A", "B"],
        primes=["Alice", "Bob"],
        device="cpu",
        model="large-v3",
        language="en",
        diarize=False,
        estimated_offset_b_minus_a_sec=1.23,
        anchor_count=5,
        pair_match_count=18,
        total_segments=100,
        kept_segments=80,
        suppressed_segments=20,
    )
    defaults.update(kwargs)
    return RunMeta(**defaults)


SAMPLE_TRANSCRIPT = "[00:00:01.000 - 00:00:05.000] Alice: Hello everyone."


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

class TestLoadTemplate:
    def test_builtin_audit_report_loads(self):
        """The packaged audit_report template should load without error."""
        cfg = _make_llm_config(prompt_template="audit_report")
        # build_prompt would raise if template not found
        result = build_prompt(SAMPLE_TRANSCRIPT, cfg)
        assert "{transcript}" not in result
        assert SAMPLE_TRANSCRIPT in result

    def test_builtin_template_contains_required_sections(self):
        cfg = _make_llm_config()
        result = build_prompt(SAMPLE_TRANSCRIPT, cfg)
        for section in [
            "## Executive Summary",
            "## Participants",
            "## Key Topics Discussed",
            "## Action Items",
            "## Concerns and Flags",
            "## Transcript Quality Notes",
        ]:
            assert section in result, f"Missing section: {section}"

    def test_unknown_template_raises(self):
        cfg = _make_llm_config(prompt_template="nonexistent_template_xyz")
        with pytest.raises(ValueError, match="not found"):
            build_prompt(SAMPLE_TRANSCRIPT, cfg)

    def test_file_path_template(self, tmp_path):
        """Absolute path to a .txt file should be used directly."""
        tpl = tmp_path / "my_template.txt"
        tpl.write_text("Hello {transcript} end", encoding="utf-8")
        cfg = _make_llm_config(prompt_template=str(tpl))
        result = build_prompt(SAMPLE_TRANSCRIPT, cfg)
        assert result == f"Hello {SAMPLE_TRANSCRIPT} end"


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_transcript_substituted(self):
        cfg = _make_llm_config()
        result = build_prompt(SAMPLE_TRANSCRIPT, cfg)
        assert SAMPLE_TRANSCRIPT in result

    def test_primes_substituted_from_meta(self):
        cfg = _make_llm_config()
        meta = _make_meta(primes=["Charlie", "Dana"])
        result = build_prompt(SAMPLE_TRANSCRIPT, cfg, meta=meta)
        assert "Charlie" in result
        assert "Dana" in result

    def test_primes_unknown_when_no_meta(self):
        cfg = _make_llm_config()
        result = build_prompt(SAMPLE_TRANSCRIPT, cfg, meta=None)
        assert "unknown" in result

    def test_no_unresolved_placeholders(self):
        cfg = _make_llm_config()
        meta = _make_meta()
        result = build_prompt(SAMPLE_TRANSCRIPT, cfg, meta=meta)
        assert "{transcript}" not in result
        assert "{primes}" not in result
        assert "{duration}" not in result

    def test_transcript_truncation_at_limit(self):
        """Transcript over max_transcript_chars is truncated."""
        long_transcript = "A" * 100 + "\n" + "B" * 100
        cfg = _make_llm_config(max_transcript_chars=50)
        result = build_prompt(long_transcript, cfg)
        assert "truncated" in result

    def test_truncation_preserves_tail(self):
        """After truncation, the tail (most recent content) is kept."""
        prefix = "OLD_CONTENT\n" * 10
        suffix = "NEW_IMPORTANT_LINE"
        transcript = prefix + suffix
        cfg = _make_llm_config(max_transcript_chars=len(suffix) + 5)
        result = build_prompt(transcript, cfg)
        assert "NEW_IMPORTANT_LINE" in result

    def test_short_transcript_not_truncated(self):
        cfg = _make_llm_config(max_transcript_chars=50_000)
        result = build_prompt(SAMPLE_TRANSCRIPT, cfg)
        assert "truncated" not in result

    def test_truncation_adds_notice(self):
        long_transcript = "X\n" * 1000
        cfg = _make_llm_config(max_transcript_chars=100)
        result = build_prompt(long_transcript, cfg)
        assert "transcript truncated" in result


# ---------------------------------------------------------------------------
# load_transcript_text
# ---------------------------------------------------------------------------

class TestLoadTranscriptText:
    def test_loads_txt_file(self, tmp_path):
        txt = tmp_path / "transcript.txt"
        content = "Hello world\nSecond line"
        txt.write_text(content, encoding="utf-8")
        assert load_transcript_text(txt) == content

    def test_loads_json_canonical(self, tmp_path):
        data = {
            "segments": [
                {
                    "keep": True,
                    "global_start_sec": 1.0,
                    "global_end_sec": 3.5,
                    "speaker_final": "Alice",
                    "text": "Hello there",
                },
                {
                    "keep": False,
                    "global_start_sec": 4.0,
                    "global_end_sec": 5.0,
                    "speaker_final": "Bob",
                    "text": "Suppressed segment",
                },
            ]
        }
        jfile = tmp_path / "canonical_transcript.json"
        jfile.write_text(json.dumps(data), encoding="utf-8")
        result = load_transcript_text(jfile)
        assert "Alice" in result
        assert "Hello there" in result
        # suppressed segment must be excluded
        assert "Suppressed segment" not in result

    def test_json_formats_timestamp(self, tmp_path):
        data = {
            "segments": [
                {
                    "keep": True,
                    "global_start_sec": 65.0,
                    "global_end_sec": 70.25,
                    "speaker_final": "Bob",
                    "text": "Test line",
                }
            ]
        }
        jfile = tmp_path / "t.json"
        jfile.write_text(json.dumps(data), encoding="utf-8")
        result = load_transcript_text(jfile)
        # 65 seconds = 00:01:05
        assert "00:01:05" in result

    def test_json_unknown_speaker_label(self, tmp_path):
        data = {
            "segments": [
                {
                    "keep": True,
                    "global_start_sec": 0.0,
                    "global_end_sec": 2.0,
                    "speaker_final": None,
                    "text": "Unlabelled speech",
                }
            ]
        }
        jfile = tmp_path / "t.json"
        jfile.write_text(json.dumps(data), encoding="utf-8")
        result = load_transcript_text(jfile)
        assert "UNKNOWN" in result

    def test_json_empty_segments(self, tmp_path):
        data = {"segments": []}
        jfile = tmp_path / "t.json"
        jfile.write_text(json.dumps(data), encoding="utf-8")
        assert load_transcript_text(jfile) == ""

    def test_json_all_suppressed(self, tmp_path):
        data = {
            "segments": [
                {"keep": False, "global_start_sec": 0.0, "global_end_sec": 1.0,
                 "speaker_final": "Alice", "text": "Hidden"},
            ]
        }
        jfile = tmp_path / "t.json"
        jfile.write_text(json.dumps(data), encoding="utf-8")
        assert load_transcript_text(jfile) == ""


# ---------------------------------------------------------------------------
# get_provider factory
# ---------------------------------------------------------------------------

class TestGetProvider:
    def test_ollama_returns_ollama_provider(self):
        cfg = _make_llm_config(provider="ollama")
        provider = get_provider(cfg)
        assert isinstance(provider, OllamaProvider)

    def test_openai_returns_openai_provider(self):
        cfg = _make_llm_config(provider="openai")
        provider = get_provider(cfg)
        assert isinstance(provider, OpenAIProvider)

    def test_openai_compatible_alias(self):
        cfg = _make_llm_config(provider="openai_compatible")
        provider = get_provider(cfg)
        assert isinstance(provider, OpenAIProvider)

    def test_unknown_provider_raises(self):
        cfg = _make_llm_config(provider="unknown_xyz")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_provider(cfg)

    def test_provider_satisfies_protocol(self):
        cfg = _make_llm_config(provider="ollama")
        provider = get_provider(cfg)
        assert isinstance(provider, AbstractLLMProvider)


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------

class TestOllamaProvider:
    def _provider(self, **kwargs) -> OllamaProvider:
        cfg = _make_llm_config(provider="ollama", **kwargs)
        return OllamaProvider(cfg)

    def test_generate_returns_response_field(self):
        provider = self._provider()
        response_body = json.dumps({"response": "Hello from Ollama"}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = provider.generate("test prompt")

        assert result == "Hello from Ollama"

    def test_generate_url_contains_api_generate(self):
        provider = self._provider(base_url="http://myserver:11434")
        assert "api/generate" in provider._generate_url

    def test_generate_network_error_raises_runtime(self):
        import urllib.error
        provider = self._provider()
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            with pytest.raises(RuntimeError, match="Ollama"):
                provider.generate("prompt")

    def test_stream_yields_tokens(self):
        provider = self._provider()
        lines = [
            json.dumps({"response": "tok1", "done": False}).encode(),
            json.dumps({"response": "tok2", "done": False}).encode(),
            json.dumps({"response": "", "done": True}).encode(),
        ]
        mock_resp = MagicMock()
        mock_resp.__iter__ = lambda s: iter(lines)
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            tokens = list(provider.stream("test prompt"))

        assert tokens == ["tok1", "tok2"]

    def test_stream_skips_empty_tokens(self):
        provider = self._provider()
        lines = [
            json.dumps({"response": "", "done": False}).encode(),
            json.dumps({"response": "real", "done": True}).encode(),
        ]
        mock_resp = MagicMock()
        mock_resp.__iter__ = lambda s: iter(lines)
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            tokens = list(provider.stream("prompt"))

        assert tokens == ["real"]

    def test_stream_network_error_raises_runtime(self):
        import urllib.error
        provider = self._provider()
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            with pytest.raises(RuntimeError, match="Ollama"):
                list(provider.stream("prompt"))

    def test_stream_ignores_malformed_json(self):
        provider = self._provider()
        lines = [
            b"NOT_JSON\n",
            json.dumps({"response": "good", "done": True}).encode(),
        ]
        mock_resp = MagicMock()
        mock_resp.__iter__ = lambda s: iter(lines)
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            tokens = list(provider.stream("prompt"))

        assert tokens == ["good"]


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------

class TestOpenAIProvider:
    def _provider(self, **kwargs) -> OpenAIProvider:
        cfg = _make_llm_config(provider="openai", **kwargs)
        return OpenAIProvider(cfg)

    def _mock_openai_module(self, response_text: str):
        """Return a mock openai module whose client returns response_text."""
        mock_openai = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = response_text
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
        return mock_openai

    def test_generate_returns_content(self):
        provider = self._provider()
        mock_openai = self._mock_openai_module("OpenAI response text")
        with patch.dict(sys.modules, {"openai": mock_openai}):
            result = provider.generate("test prompt")
        assert result == "OpenAI response text"

    def test_generate_uses_configured_model(self):
        provider = self._provider(model="gpt-4o")
        mock_openai = self._mock_openai_module("ok")
        with patch.dict(sys.modules, {"openai": mock_openai}):
            provider.generate("prompt")
        create_call = mock_openai.OpenAI.return_value.chat.completions.create
        assert create_call.call_args.kwargs["model"] == "gpt-4o"

    def test_generate_uses_api_key(self):
        provider = self._provider(api_key="sk-test-123")
        mock_openai = self._mock_openai_module("ok")
        with patch.dict(sys.modules, {"openai": mock_openai}):
            provider.generate("prompt")
        openai_init = mock_openai.OpenAI
        assert openai_init.call_args.kwargs["api_key"] == "sk-test-123"

    def test_generate_uses_env_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-sk-abc")
        provider = self._provider(api_key=None)
        mock_openai = self._mock_openai_module("ok")
        with patch.dict(sys.modules, {"openai": mock_openai}):
            provider.generate("prompt")
        openai_init = mock_openai.OpenAI
        assert openai_init.call_args.kwargs["api_key"] == "env-sk-abc"

    def test_generate_falls_back_to_no_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = self._provider(api_key=None)
        mock_openai = self._mock_openai_module("ok")
        with patch.dict(sys.modules, {"openai": mock_openai}):
            provider.generate("prompt")
        openai_init = mock_openai.OpenAI
        assert openai_init.call_args.kwargs["api_key"] == "sk-no-key-required"

    def test_generate_missing_openai_package_raises(self):
        provider = self._provider()
        with patch.dict(sys.modules, {"openai": None}):
            with pytest.raises((RuntimeError, ImportError)):
                provider.generate("prompt")

    def test_stream_yields_tokens(self):
        provider = self._provider()
        mock_openai = MagicMock()

        def _make_chunk(content):
            c = MagicMock()
            c.choices[0].delta.content = content
            return c

        chunks = [_make_chunk("tok1"), _make_chunk("tok2"), _make_chunk(None)]
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = iter(chunks)

        with patch.dict(sys.modules, {"openai": mock_openai}):
            tokens = list(provider.stream("prompt"))

        assert tokens == ["tok1", "tok2"]


# ---------------------------------------------------------------------------
# generate_report (integration with mocked providers)
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def _cfg(self, **kwargs) -> LLMConfig:
        return _make_llm_config(**kwargs)

    def test_blocking_generate_returns_text(self):
        cfg = self._cfg(stream=False)
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "Full report text"

        with patch("audio2report.llm.base.get_provider", return_value=mock_provider):
            result = generate_report(SAMPLE_TRANSCRIPT, cfg)

        assert result == "Full report text"
        mock_provider.generate.assert_called_once()

    def test_stream_writes_to_stdout(self, capsys):
        cfg = self._cfg(stream=True)
        tokens = ["Report ", "line ", "one"]
        mock_provider = MagicMock()
        mock_provider.stream.return_value = iter(tokens)

        with patch("audio2report.llm.base.get_provider", return_value=mock_provider):
            result = generate_report(SAMPLE_TRANSCRIPT, cfg, stream_to_stdout=True)

        captured = capsys.readouterr()
        assert "Report line one" in captured.out
        assert result == "Report line one"

    def test_stream_false_uses_generate(self):
        """When config.stream=False, provider.generate() is called (not stream)."""
        cfg = self._cfg(stream=False)
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "blocking result"

        with patch("audio2report.llm.base.get_provider", return_value=mock_provider):
            result = generate_report(SAMPLE_TRANSCRIPT, cfg, stream_to_stdout=True)

        mock_provider.stream.assert_not_called()
        mock_provider.generate.assert_called_once()
        assert result == "blocking result"

    def test_stream_to_stdout_false_uses_generate(self):
        """stream_to_stdout=False skips streaming regardless of config.stream."""
        cfg = self._cfg(stream=True)
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "non-streamed"

        with patch("audio2report.llm.base.get_provider", return_value=mock_provider):
            result = generate_report(SAMPLE_TRANSCRIPT, cfg, stream_to_stdout=False)

        mock_provider.stream.assert_not_called()
        assert result == "non-streamed"

    def test_report_with_meta(self):
        cfg = self._cfg(stream=False)
        meta = _make_meta(primes=["Eve", "Frank"])
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "report"

        with patch("audio2report.llm.base.get_provider", return_value=mock_provider):
            generate_report(SAMPLE_TRANSCRIPT, cfg, meta=meta)

        prompt_sent = mock_provider.generate.call_args.args[0]
        assert "Eve" in prompt_sent
        assert "Frank" in prompt_sent

    def test_prompt_passed_to_provider(self):
        cfg = self._cfg(stream=False)
        mock_provider = MagicMock()
        mock_provider.generate.return_value = ""

        with patch("audio2report.llm.base.get_provider", return_value=mock_provider):
            generate_report(SAMPLE_TRANSCRIPT, cfg)

        prompt = mock_provider.generate.call_args.args[0]
        assert SAMPLE_TRANSCRIPT in prompt
