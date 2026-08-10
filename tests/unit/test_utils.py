"""Unit tests for device resolution helpers in audio2report/utils.py."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from audio2report.utils import pick_device, validate_backend_device


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
