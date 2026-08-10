"""Unit tests for the transcriber factory (audio2report/transcription/base.py)."""
from __future__ import annotations

import pytest

from audio2report.config import TranscriptionConfig
from audio2report.transcription.base import get_transcriber, is_effectively_empty
from audio2report.transcription.openvino_backend import OpenVINOTranscriber
from audio2report.transcription.whisperx_backend import WhisperXTranscriber


class TestGetTranscriber:
    def test_whisperx_backend_returns_whisperx_transcriber(self):
        cfg = TranscriptionConfig(backend="whisperx", device=None)
        transcriber = get_transcriber(cfg, "cpu")
        assert isinstance(transcriber, WhisperXTranscriber)

    def test_openvino_backend_returns_openvino_transcriber(self):
        cfg = TranscriptionConfig(backend="openvino", device=None)
        transcriber = get_transcriber(cfg, "gpu")
        assert isinstance(transcriber, OpenVINOTranscriber)

    def test_whisperx_with_intel_gpu_device_raises(self):
        """The factory must reject an incompatible backend/device combo up
        front rather than let it reach WhisperX and fail obscurely."""
        cfg = TranscriptionConfig(backend="whisperx", device=None)
        with pytest.raises(RuntimeError, match="cannot use device"):
            get_transcriber(cfg, "gpu")

    def test_openvino_with_cuda_device_raises(self):
        cfg = TranscriptionConfig(backend="openvino", device=None)
        with pytest.raises(RuntimeError, match="cannot use device"):
            get_transcriber(cfg, "cuda")


class TestIsEffectivelyEmpty:
    def test_no_segments_is_empty(self):
        assert is_effectively_empty({"segments": []}) is True

    def test_missing_segments_key_is_empty(self):
        assert is_effectively_empty({}) is True

    def test_tiny_text_is_empty(self):
        assert is_effectively_empty({"segments": [{"start": 0.0, "end": 0.1, "text": "."}]}) is True

    def test_zero_duration_is_empty(self):
        result = {"segments": [{"start": 1.0, "end": 1.0, "text": "hello there friend"}]}
        assert is_effectively_empty(result) is True

    def test_real_speech_is_not_empty(self):
        result = {"segments": [{"start": 0.0, "end": 3.0, "text": "This is a real sentence."}]}
        assert is_effectively_empty(result) is False
