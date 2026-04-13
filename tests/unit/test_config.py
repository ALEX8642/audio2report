"""Unit tests for config loading, validation, and derived helpers."""
from __future__ import annotations

import os

import pytest
import yaml

from audio2report.config import (
    Config,
    DiarizationConfig,
    TranscriptionConfig,
    load_config,
)
from audio2report.utils import auto_compute_type


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

class TestDefaultConfig:
    def test_loads_without_error(self):
        cfg = Config()
        assert cfg is not None

    def test_default_mode(self):
        assert Config().mode == "dual"

    def test_default_model(self):
        assert Config().transcription.model == "large-v3"

    def test_default_diarization_disabled(self):
        assert Config().diarization.enabled is False

    def test_default_deduplication_enabled(self):
        assert Config().deduplication.enabled is True

    def test_default_cache_enabled(self):
        assert Config().cache is True

    def test_default_output_formats(self):
        assert set(Config().output.formats) == {"json", "csv", "txt"}


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

class TestYAMLConfig:
    def test_load_from_yaml(self, tmp_path):
        data = {
            "transcription": {"model": "medium", "compute_type": "int8"},
            "diarization": {"enabled": True},
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(data))
        cfg = load_config(path)
        assert cfg.transcription.model == "medium"
        assert cfg.transcription.compute_type == "int8"
        assert cfg.diarization.enabled is True

    def test_yaml_overrides_defaults_only_for_specified_keys(self, tmp_path):
        # Only override model; everything else stays default
        data = {"transcription": {"model": "tiny"}}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(data))
        cfg = load_config(path)
        assert cfg.transcription.model == "tiny"
        assert cfg.transcription.batch_size == 8  # default untouched

    def test_empty_yaml_returns_defaults(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("")
        cfg = load_config(path)
        assert cfg == Config()

    def test_load_none_returns_defaults(self):
        assert load_config(None) == Config()

    def test_cpu_preset_file(self):
        from pathlib import Path
        preset = Path(__file__).parent.parent.parent / "configs" / "cpu_fast.yaml"
        cfg = load_config(preset)
        assert cfg.transcription.model == "medium"
        assert cfg.transcription.compute_type == "int8"
        assert cfg.diarization.enabled is False

    def test_gpu_preset_file(self):
        from pathlib import Path
        preset = Path(__file__).parent.parent.parent / "configs" / "gpu_full.yaml"
        cfg = load_config(preset)
        assert cfg.transcription.model == "large-v3"
        assert cfg.diarization.enabled is True


# ---------------------------------------------------------------------------
# auto_compute_type
# ---------------------------------------------------------------------------

class TestAutoComputeType:
    def test_float16_on_cpu_becomes_int8(self):
        assert auto_compute_type("float16", "cpu") == "int8"

    def test_float16_on_cuda_unchanged(self):
        assert auto_compute_type("float16", "cuda") == "float16"

    def test_int8_on_cpu_unchanged(self):
        assert auto_compute_type("int8", "cpu") == "int8"

    def test_float32_on_cpu_unchanged(self):
        assert auto_compute_type("float32", "cpu") == "float32"


# ---------------------------------------------------------------------------
# DiarizationConfig.resolved_token
# ---------------------------------------------------------------------------

class TestResolvedToken:
    def test_returns_explicit_token(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        cfg = DiarizationConfig(hf_token="explicit_token_123")
        assert cfg.resolved_token() == "explicit_token_123"

    def test_falls_back_to_env_var(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "env_token_456")
        cfg = DiarizationConfig(hf_token=None)
        assert cfg.resolved_token() == "env_token_456"

    def test_explicit_token_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "env_token")
        cfg = DiarizationConfig(hf_token="explicit_token")
        assert cfg.resolved_token() == "explicit_token"

    def test_returns_none_when_neither_set(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        cfg = DiarizationConfig(hf_token=None)
        assert cfg.resolved_token() is None


# ---------------------------------------------------------------------------
# Pydantic validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_invalid_field_type_raises(self):
        with pytest.raises(Exception):
            TranscriptionConfig(batch_size="not_an_int")

    def test_extra_fields_ignored_gracefully(self, tmp_path):
        # Pydantic v2 default: extra fields ignored
        data = {"transcription": {"model": "medium", "unknown_field": True}}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(data))
        cfg = load_config(path)
        assert cfg.transcription.model == "medium"
