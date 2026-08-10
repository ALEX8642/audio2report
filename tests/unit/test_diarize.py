"""Unit tests for the shared diarization step (audio2report/transcription/_diarize.py).

pyannote.audio and soundfile are mocked throughout — no real model download or
audio decoding needed.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from audio2report.transcription._diarize import resolve_diarization_device, run_diarization


def _turn(start, end):
    return SimpleNamespace(start=start, end=end)


def _fake_pyannote_modules(diar_turns, wrap_in_output=False):
    """diar_turns: list of (start, end, speaker) tuples."""
    # spec= restricts the mock to only the listed attributes, so accessing
    # anything else (e.g. `.speaker_diarization` on a plain Annotation)
    # raises a real AttributeError instead of MagicMock's default
    # auto-vivification — matching genuine object behavior, which the
    # getattr(..., default) fallback in run_diarization() depends on.
    annotation = MagicMock(spec=["itertracks"])
    annotation.itertracks.return_value = [
        (_turn(s, e), None, spk) for s, e, spk in diar_turns
    ]

    pipeline_instance = MagicMock()
    pipeline_instance.to.return_value = pipeline_instance
    if wrap_in_output:
        pipeline_instance.return_value = SimpleNamespace(speaker_diarization=annotation)
    else:
        pipeline_instance.return_value = annotation

    pipeline_cls = MagicMock()
    pipeline_cls.from_pretrained.return_value = pipeline_instance

    pyannote_audio_mod = MagicMock()
    pyannote_audio_mod.Pipeline = pipeline_cls
    return pyannote_audio_mod, pipeline_instance


def _fake_soundfile(samples=16000, samplerate=16000):
    sf_mod = MagicMock()
    sf_mod.read.return_value = (np.zeros((samples, 1), dtype=np.float32), samplerate)
    return sf_mod


class TestRunDiarization:
    def test_no_token_raises(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "pyannote.audio", _fake_pyannote_modules([])[0])
        monkeypatch.setitem(sys.modules, "soundfile", _fake_soundfile())

        with pytest.raises(RuntimeError, match="no Hugging Face token"):
            run_diarization(tmp_path / "a.wav", {"segments": []}, hf_token=None, device="cpu")

    def test_pyannote_not_installed_raises_clear_error(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "pyannote.audio", None)
        monkeypatch.setitem(sys.modules, "pyannote", None)

        with pytest.raises(RuntimeError, match="pyannote.audio is not installed"):
            run_diarization(tmp_path / "a.wav", {"segments": []}, hf_token="tok", device="cpu")

    def test_passes_preloaded_waveform_not_path(self, monkeypatch, tmp_path):
        """Regression: must not pass a raw path to pipeline() — pyannote's
        own path-based loader depends on torchcodec/FFmpeg shared libs that
        may not be present; soundfile-preloaded waveform avoids that."""
        pyannote_mod, pipeline_instance = _fake_pyannote_modules([(0.0, 1.0, "SPEAKER_00")])
        monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_mod)
        monkeypatch.setitem(sys.modules, "soundfile", _fake_soundfile())

        run_diarization(tmp_path / "a.wav", {"segments": []}, hf_token="tok", device="cpu")

        pipeline_instance.assert_called_once()
        (call_arg,), _ = pipeline_instance.call_args
        assert isinstance(call_arg, dict)
        assert "waveform" in call_arg and "sample_rate" in call_arg

    def test_assigns_speaker_by_max_overlap(self, monkeypatch, tmp_path):
        pyannote_mod, _ = _fake_pyannote_modules([
            (0.0, 5.0, "SPEAKER_00"),
            (5.0, 10.0, "SPEAKER_01"),
        ])
        monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_mod)
        monkeypatch.setitem(sys.modules, "soundfile", _fake_soundfile())

        asr_result = {"segments": [{"start": 4.0, "end": 8.0, "text": "hello"}]}
        result = run_diarization(tmp_path / "a.wav", asr_result, hf_token="tok", device="cpu")

        # Overlaps SPEAKER_00 for 1s (4-5) and SPEAKER_01 for 3s (5-8) -> SPEAKER_01 wins
        assert result["segments"][0]["speaker"] == "SPEAKER_01"
        assert len(result["diarization_segments"]) == 2

    def test_no_overlap_leaves_segment_unlabeled(self, monkeypatch, tmp_path):
        pyannote_mod, _ = _fake_pyannote_modules([(0.0, 1.0, "SPEAKER_00")])
        monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_mod)
        monkeypatch.setitem(sys.modules, "soundfile", _fake_soundfile())

        asr_result = {"segments": [{"start": 50.0, "end": 51.0, "text": "later"}]}
        result = run_diarization(tmp_path / "a.wav", asr_result, hf_token="tok", device="cpu")

        assert "speaker" not in result["segments"][0]

    def test_handles_newer_wrapped_output(self, monkeypatch, tmp_path):
        pyannote_mod, _ = _fake_pyannote_modules(
            [(0.0, 2.0, "SPEAKER_00")], wrap_in_output=True
        )
        monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_mod)
        monkeypatch.setitem(sys.modules, "soundfile", _fake_soundfile())

        asr_result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]}
        result = run_diarization(tmp_path / "a.wav", asr_result, hf_token="tok", device="cpu")

        assert result["segments"][0]["speaker"] == "SPEAKER_00"


def _fake_torch(*, cuda: bool, xpu: bool | None):
    """xpu=None simulates a torch build with no XPU support at all (no
    torch.xpu attribute, e.g. a CPU-only or CUDA-only wheel)."""
    torch_mod = MagicMock()
    torch_mod.cuda.is_available.return_value = cuda
    if xpu is None:
        del torch_mod.xpu
    else:
        torch_mod.xpu.is_available.return_value = xpu
    return torch_mod


class TestResolveDiarizationDevice:
    def test_prefers_cuda_when_available(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True, xpu=True))
        assert resolve_diarization_device() == "cuda"

    def test_falls_back_to_xpu_when_no_cuda(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, xpu=True))
        assert resolve_diarization_device() == "xpu"

    def test_falls_back_to_cpu_when_neither_available(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, xpu=False))
        assert resolve_diarization_device() == "cpu"

    def test_falls_back_to_cpu_when_torch_has_no_xpu_attribute(self, monkeypatch):
        """A torch build without XPU support at all (e.g. CPU-only wheel)
        has no torch.xpu attribute — must not raise, must fall back to cpu."""
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, xpu=None))
        assert resolve_diarization_device() == "cpu"

    def test_falls_back_to_cpu_when_torch_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        assert resolve_diarization_device() == "cpu"
