"""Unit tests for compute-hardware detection (audio2report/hardware.py).

torch/openvino are not hard dependencies, so every scenario is exercised by
injecting fake modules into ``sys.modules`` — setting an entry to ``None``
makes Python raise ``ImportError`` on ``import``, which is how the
"not installed" branches are simulated.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

from audio2report.hardware import detect_hardware


def _fake_torch(cuda_available: bool, name: str = "NVIDIA Fake GPU"):
    torch_mod = MagicMock()
    torch_mod.cuda.is_available.return_value = cuda_available
    torch_mod.cuda.get_device_name.return_value = name
    return torch_mod


def _fake_openvino(available_devices: list[str], names: dict[str, str] | None = None, raise_on_core: bool = False):
    ov_mod = MagicMock()
    if raise_on_core:
        ov_mod.Core.side_effect = RuntimeError("driver not found")
        return ov_mod

    core = MagicMock()
    core.available_devices = available_devices
    names = names or {}
    core.get_property.side_effect = lambda dev, prop: names.get(dev, f"{dev} device")
    ov_mod.Core.return_value = core
    return ov_mod


class TestNothingAvailable:
    def test_cpu_only(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "openvino", None)

        report = detect_hardware()

        assert report.get("cpu").available is True
        assert report.get("cuda").available is False
        assert report.get("gpu").available is False
        assert report.get("npu").available is False
        assert report.recommended == "cpu"
        assert "not installed" in (report.get("gpu").detail or "")

    def test_openvino_without_openvino_genai_is_unavailable(self, monkeypatch):
        """Regression: openvino alone can enumerate devices fine, but
        OpenVINOTranscriber actually needs openvino_genai to run — a
        machine with only the former must be reported as gpu-unavailable,
        not as a false positive that crashes the moment a real
        transcription is attempted."""
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(
            sys.modules, "openvino", _fake_openvino(["CPU", "GPU"]),
        )
        monkeypatch.setitem(sys.modules, "openvino_genai", None)

        report = detect_hardware()

        assert report.get("gpu").available is False
        assert "not installed" in (report.get("gpu").detail or "")


class TestCuda:
    def test_cuda_available_is_recommended(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, "RTX Fake"))
        monkeypatch.setitem(sys.modules, "openvino", None)

        report = detect_hardware()

        assert report.get("cuda").available is True
        assert report.get("cuda").name == "RTX Fake"
        assert report.recommended == "cuda"

    def test_cuda_unavailable(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(False))
        monkeypatch.setitem(sys.modules, "openvino", None)

        report = detect_hardware()

        assert report.get("cuda").available is False
        assert report.recommended == "cpu"


class TestIntelGPU:
    def test_gpu_detected(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(
            sys.modules, "openvino",
            _fake_openvino(["CPU", "GPU"], {"GPU": "Intel Arc Graphics"}),
        )
        monkeypatch.setitem(sys.modules, "openvino_genai", MagicMock())

        report = detect_hardware()

        assert report.get("gpu").available is True
        assert report.get("gpu").name == "Intel Arc Graphics"
        assert report.get("npu").available is False
        assert report.recommended == "gpu"

    def test_multi_gpu_naming(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(
            sys.modules, "openvino",
            _fake_openvino(["CPU", "GPU.0", "GPU.1"]),
        )
        monkeypatch.setitem(sys.modules, "openvino_genai", MagicMock())

        report = detect_hardware()

        gpu = report.get("gpu")
        assert gpu.available is True
        assert "GPU.0" in (gpu.detail or "")
        assert "multiple" in (gpu.detail or "").lower()

    def test_gpu_preferred_over_npu_in_auto_mode(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(
            sys.modules, "openvino",
            _fake_openvino(["CPU", "GPU", "NPU"]),
        )
        monkeypatch.setitem(sys.modules, "openvino_genai", MagicMock())

        report = detect_hardware()

        assert report.get("gpu").available is True
        assert report.get("npu").available is True
        assert report.recommended == "gpu"

    def test_cuda_preferred_over_intel_gpu_in_auto_mode(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(True))
        monkeypatch.setitem(sys.modules, "openvino", _fake_openvino(["CPU", "GPU"]))
        monkeypatch.setitem(sys.modules, "openvino_genai", MagicMock())

        report = detect_hardware()

        assert report.recommended == "cuda"


class TestIntelNPU:
    def test_npu_only(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(
            sys.modules, "openvino",
            _fake_openvino(["CPU", "NPU"], {"NPU": "Intel AI Boost NPU"}),
        )
        monkeypatch.setitem(sys.modules, "openvino_genai", MagicMock())

        report = detect_hardware()

        assert report.get("gpu").available is False
        assert report.get("npu").available is True
        assert report.get("npu").name == "Intel AI Boost NPU"
        assert report.recommended == "npu"


class TestOpenvinoEnumerationFailure:
    def test_core_construction_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "openvino", _fake_openvino([], raise_on_core=True))
        monkeypatch.setitem(sys.modules, "openvino_genai", MagicMock())

        report = detect_hardware()

        assert report.get("gpu").available is False
        assert report.get("npu").available is False
        assert "driver not found" in (report.get("gpu").detail or "")
        assert report.recommended == "cpu"


class TestHardwareReportGet:
    def test_get_unknown_device_returns_none(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "openvino", None)

        report = detect_hardware()

        assert report.get("tpu") is None
