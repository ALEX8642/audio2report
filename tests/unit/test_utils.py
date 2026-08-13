"""Unit tests for device resolution helpers in audio2report/utils.py."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from audio2report.hardware import DeviceInfo, HardwareReport
from audio2report.utils import pick_device, resolve_backend_and_device, validate_backend_device


def _fake_openvino(available_devices: list[str]):
    ov_mod = MagicMock()
    core = MagicMock()
    core.available_devices = available_devices
    core.get_property.side_effect = lambda dev, prop: f"{dev} device"
    ov_mod.Core.return_value = core
    return ov_mod


# ---------------------------------------------------------------------------
# pick_device — auto mode
# ---------------------------------------------------------------------------

class TestPickDeviceAuto:
    def test_whisperx_auto_ignores_intel_gpu_falls_back_to_cpu(self, monkeypatch):
        """Regression: auto mode must be backend-aware — a whisperx config
        must never be steered toward a device only openvino can use."""
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "openvino", _fake_openvino(["CPU", "GPU"]))

        assert pick_device(None, "whisperx") == "cpu"
        assert pick_device("auto", "whisperx") == "cpu"

    def test_openvino_auto_picks_intel_gpu(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "openvino", _fake_openvino(["CPU", "GPU"]))
        monkeypatch.setitem(sys.modules, "openvino_genai", MagicMock())

        assert pick_device(None, "openvino") == "gpu"

    def test_openvino_auto_never_resolves_to_cuda(self, monkeypatch):
        """Regression: even if CUDA is present, the openvino backend can't
        use it — auto mode must skip straight past it to cpu."""
        torch_mod = MagicMock()
        torch_mod.cuda.is_available.return_value = True
        monkeypatch.setitem(sys.modules, "torch", torch_mod)
        monkeypatch.setitem(sys.modules, "openvino", None)

        assert pick_device(None, "openvino") == "cpu"

    def test_whisperx_auto_picks_cuda_when_available(self, monkeypatch):
        torch_mod = MagicMock()
        torch_mod.cuda.is_available.return_value = True
        monkeypatch.setitem(sys.modules, "torch", torch_mod)
        monkeypatch.setitem(sys.modules, "openvino", None)

        assert pick_device(None, "whisperx") == "cuda"


# ---------------------------------------------------------------------------
# pick_device — explicit device
# ---------------------------------------------------------------------------

class TestPickDeviceExplicit:
    def test_explicit_available_device_returned(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "openvino", _fake_openvino(["CPU", "GPU"]))
        monkeypatch.setitem(sys.modules, "openvino_genai", MagicMock())

        assert pick_device("gpu", "openvino") == "gpu"
        assert pick_device("cpu", "openvino") == "cpu"

    def test_explicit_unavailable_device_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "openvino", None)

        with pytest.raises(RuntimeError, match="not available"):
            pick_device("gpu", "openvino")

    def test_explicit_device_unsupported_by_backend_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "openvino", _fake_openvino(["CPU", "GPU"]))

        with pytest.raises(RuntimeError, match="cannot use device"):
            pick_device("gpu", "whisperx")

        torch_mod = MagicMock()
        torch_mod.cuda.is_available.return_value = True
        monkeypatch.setitem(sys.modules, "torch", torch_mod)
        with pytest.raises(RuntimeError, match="cannot use device"):
            pick_device("cuda", "openvino")

    def test_unknown_device_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown device"):
            pick_device("tpu", "whisperx")

    def test_unknown_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown transcription backend"):
            pick_device("cpu", "not_a_backend")


# ---------------------------------------------------------------------------
# validate_backend_device
# ---------------------------------------------------------------------------

class TestValidateBackendDevice:
    def test_whisperx_cpu_and_cuda_ok(self):
        validate_backend_device("whisperx", "cpu")
        validate_backend_device("whisperx", "cuda")

    def test_whisperx_gpu_npu_rejected(self):
        with pytest.raises(RuntimeError, match="cannot use device"):
            validate_backend_device("whisperx", "gpu")
        with pytest.raises(RuntimeError, match="cannot use device"):
            validate_backend_device("whisperx", "npu")

    def test_openvino_gpu_npu_cpu_ok(self):
        validate_backend_device("openvino", "cpu")
        validate_backend_device("openvino", "gpu")
        validate_backend_device("openvino", "npu")

    def test_openvino_cuda_rejected(self):
        with pytest.raises(RuntimeError, match="cannot use device"):
            validate_backend_device("openvino", "cuda")

    def test_torch_whisper_cpu_and_cuda_ok(self):
        validate_backend_device("torch_whisper", "cpu")
        validate_backend_device("torch_whisper", "cuda")

    def test_torch_whisper_gpu_npu_rejected(self):
        with pytest.raises(RuntimeError, match="cannot use device"):
            validate_backend_device("torch_whisper", "gpu")
        with pytest.raises(RuntimeError, match="cannot use device"):
            validate_backend_device("torch_whisper", "npu")


# ---------------------------------------------------------------------------
# resolve_backend_and_device — joint auto backend+device selection
# ---------------------------------------------------------------------------

def _report(*, cuda=False, gpu=False, npu=False) -> HardwareReport:
    devices = [
        DeviceInfo(id="cpu", available=True),
        DeviceInfo(id="cuda", available=cuda),
        DeviceInfo(id="gpu", available=gpu),
        DeviceInfo(id="npu", available=npu),
    ]
    return HardwareReport(devices=devices, recommended="cpu")


def _patch_capabilities(
    monkeypatch,
    *,
    cuda=False,
    gpu=False,
    npu=False,
    ctranslate2_cuda=False,
    torch_whisper=False,
    whisperx=False,
    openvino=False,
):
    monkeypatch.setattr(
        "audio2report.hardware.detect_hardware",
        lambda: _report(cuda=cuda, gpu=gpu, npu=npu),
    )
    monkeypatch.setattr("audio2report.hardware.ctranslate2_cuda_capable", lambda: ctranslate2_cuda)
    monkeypatch.setattr("audio2report.hardware.torch_whisper_installed", lambda: torch_whisper)
    monkeypatch.setattr("audio2report.hardware.whisperx_installed", lambda: whisperx)
    monkeypatch.setattr("audio2report.hardware.openvino_installed", lambda: openvino)


class TestResolveBackendAndDevice:
    def test_prefers_whisperx_cuda_when_ctranslate2_capable(self, monkeypatch):
        _patch_capabilities(
            monkeypatch, cuda=True, ctranslate2_cuda=True, whisperx=True, torch_whisper=True
        )
        assert resolve_backend_and_device("auto", None) == ("whisperx", "cuda")

    def test_falls_back_to_torch_whisper_cuda_when_ctranslate2_cpu_only(self, monkeypatch):
        """The DGX Spark case: CUDA is present, whisperx is installed, but
        ctranslate2 (its ASR engine) has no CUDA wheel — must not pick
        whisperx+cuda (a guaranteed crash), must land on torch_whisper+cuda
        instead."""
        _patch_capabilities(
            monkeypatch, cuda=True, ctranslate2_cuda=False, whisperx=True, torch_whisper=True
        )
        assert resolve_backend_and_device("auto", None) == ("torch_whisper", "cuda")

    def test_falls_back_to_openvino_gpu_when_no_cuda(self, monkeypatch):
        _patch_capabilities(monkeypatch, gpu=True, whisperx=True, openvino=True)
        assert resolve_backend_and_device("auto", None) == ("openvino", "gpu")

    def test_falls_back_to_openvino_npu_when_no_cuda_or_gpu(self, monkeypatch):
        _patch_capabilities(monkeypatch, npu=True, whisperx=True, openvino=True)
        assert resolve_backend_and_device("auto", None) == ("openvino", "npu")

    def test_cpu_fallback_prefers_whisperx(self, monkeypatch):
        _patch_capabilities(monkeypatch, whisperx=True, torch_whisper=True, openvino=True)
        assert resolve_backend_and_device("auto", None) == ("whisperx", "cpu")

    def test_cpu_fallback_prefers_torch_whisper_when_whisperx_missing(self, monkeypatch):
        _patch_capabilities(monkeypatch, whisperx=False, torch_whisper=True, openvino=True)
        assert resolve_backend_and_device("auto", None) == ("torch_whisper", "cpu")

    def test_cpu_fallback_uses_openvino_when_only_option(self, monkeypatch):
        _patch_capabilities(monkeypatch, whisperx=False, torch_whisper=False, openvino=True)
        assert resolve_backend_and_device("auto", None) == ("openvino", "cpu")

    def test_nothing_installed_raises_clear_error(self, monkeypatch):
        _patch_capabilities(monkeypatch, whisperx=False, torch_whisper=False, openvino=False)
        with pytest.raises(RuntimeError, match="No transcription backend is installed"):
            resolve_backend_and_device("auto", None)

    def test_explicit_backend_never_silently_substituted(self, monkeypatch):
        """Even when torch_whisper+cuda would be the smarter choice, an
        explicit whisperx request is returned exactly as asked — never
        swapped for a different backend. (It can still fail later, deep
        inside whisperx itself, if ctranslate2 truly can't use the
        device — that's a real, if less friendly, failure than a
        silent substitution would be, and out of scope for this
        explicit-request path, which intentionally only delegates to
        pick_device's device-presence check, same as before "auto" got
        smarter.)"""
        _patch_capabilities(
            monkeypatch, cuda=True, ctranslate2_cuda=False, whisperx=True, torch_whisper=True
        )
        assert resolve_backend_and_device("whisperx", "cuda") == ("whisperx", "cuda")

    def test_explicit_backend_with_genuinely_absent_device_still_raises(self, monkeypatch):
        _patch_capabilities(monkeypatch, cuda=False, whisperx=True)
        with pytest.raises(RuntimeError, match="not available"):
            resolve_backend_and_device("whisperx", "cuda")
