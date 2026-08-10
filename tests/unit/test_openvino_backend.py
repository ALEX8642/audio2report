"""Unit tests for OpenVINOTranscriber, with openvino_genai fully mocked out
(no real OpenVINO install/hardware needed)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from audio2report.config import TranscriptionConfig
from audio2report.transcription.openvino_backend import OpenVINOTranscriber, _to_ov_language_token


def _make_wav(tmp_path, seconds=2.0, samplerate=16000):
    audio = np.zeros(int(seconds * samplerate), dtype=np.float32)
    path = tmp_path / "test.wav"
    sf.write(str(path), audio, samplerate)
    return path


def _chunk(start, end, text):
    return SimpleNamespace(start_ts=start, end_ts=end, text=text)


class TestLanguageToken:
    def test_none_stays_none(self):
        assert _to_ov_language_token(None) is None

    def test_plain_code_wrapped(self):
        assert _to_ov_language_token("en") == "<|en|>"

    def test_already_wrapped_left_alone(self):
        assert _to_ov_language_token("<|en|>") == "<|en|>"


class TestTranscribe:
    def _setup_fake_pipeline(self, monkeypatch, chunks, texts=None, language="en"):
        fake_pipeline = MagicMock()
        fake_pipeline.generate.return_value = SimpleNamespace(
            chunks=chunks, texts=texts or [], language=language
        )
        fake_module = MagicMock()
        fake_module.WhisperPipeline.return_value = fake_pipeline
        monkeypatch.setitem(sys.modules, "openvino_genai", fake_module)

        fake_hub = MagicMock()
        fake_hub.snapshot_download.return_value = "/fake/model/dir"
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
        return fake_pipeline, fake_module

    def test_maps_chunks_to_segments(self, monkeypatch, tmp_path):
        chunks = [_chunk(0.0, 2.0, " Hello there."), _chunk(2.0, 4.0, " General Kenobi.")]
        self._setup_fake_pipeline(monkeypatch, chunks)

        cfg = TranscriptionConfig(backend="openvino", model="fake/repo", device="cpu")
        transcriber = OpenVINOTranscriber(cfg, "cpu")
        wav = _make_wav(tmp_path, seconds=4.0)

        result = transcriber.transcribe(wav)

        assert result["skipped_reason"] is None
        assert result["language"] == "en"
        assert len(result["segments"]) == 2
        assert result["segments"][0] == {"start": 0.0, "end": 2.0, "text": " Hello there."}
        assert result["diarization_segments"] == []

    def test_empty_result_sets_skipped_reason(self, monkeypatch, tmp_path):
        self._setup_fake_pipeline(monkeypatch, chunks=[], texts=[])

        cfg = TranscriptionConfig(backend="openvino", model="fake/repo", device="cpu")
        transcriber = OpenVINOTranscriber(cfg, "cpu")
        wav = _make_wav(tmp_path, seconds=1.0)

        result = transcriber.transcribe(wav)

        assert result["skipped_reason"] == "no_speech_or_too_short"
        assert result["segments"] == []

    def test_no_chunks_falls_back_to_single_segment_from_texts(self, monkeypatch, tmp_path):
        self._setup_fake_pipeline(monkeypatch, chunks=[], texts=["This is the whole clip transcribed."])

        cfg = TranscriptionConfig(backend="openvino", model="fake/repo", device="cpu")
        transcriber = OpenVINOTranscriber(cfg, "cpu")
        wav = _make_wav(tmp_path, seconds=3.0)

        result = transcriber.transcribe(wav)

        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "This is the whole clip transcribed."
        assert result["segments"][0]["start"] == 0.0

    def test_ov_device_uppercased(self, monkeypatch, tmp_path):
        _, fake_module = self._setup_fake_pipeline(
            monkeypatch, chunks=[_chunk(0.0, 1.0, "hi")]
        )
        cfg = TranscriptionConfig(backend="openvino", model="fake/repo", device="npu")
        transcriber = OpenVINOTranscriber(cfg, "npu")
        wav = _make_wav(tmp_path, seconds=1.0)

        transcriber.transcribe(wav)

        fake_module.WhisperPipeline.assert_called_once()
        args, kwargs = fake_module.WhisperPipeline.call_args
        assert args[1] == "NPU"
        # Must be an absolute, per-user cache path — never a bare relative
        # path, which would land wherever the process happens to be run from.
        assert Path(kwargs["CACHE_DIR"]).is_absolute()
        assert Path(kwargs["CACHE_DIR"]).name == "ov_model_cache"

    def test_openvino_genai_not_installed_raises_clear_error(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "openvino_genai", None)

        cfg = TranscriptionConfig(backend="openvino", model="fake/repo", device="cpu")
        transcriber = OpenVINOTranscriber(cfg, "cpu")
        wav = _make_wav(tmp_path, seconds=1.0)

        with pytest.raises(RuntimeError, match="openvino-genai is not installed"):
            transcriber.transcribe(wav)
