"""Compute-hardware detection: what devices are present and which one 'auto' picks.

Detection is best-effort and defensive — a missing/broken backend (no torch,
no openvino, no drivers) must never prevent detection of the others.  All
heavy imports (torch, openvino) are performed lazily inside
:func:`detect_hardware` so importing this module stays cheap enough for the
Streamlit UI to call it directly on every render.
"""
from __future__ import annotations

from dataclasses import dataclass

# Priority order for "auto"/None device selection.  NPUs are lower-throughput
# than an Arc iGPU for whisper-sized encoder/decoder workloads, so they are
# selectable explicitly but not preferred automatically.
_AUTO_PRIORITY = ["cuda", "gpu", "npu", "cpu"]


@dataclass
class DeviceInfo:
    id: str                 # "cpu" | "cuda" | "gpu" | "npu"
    available: bool
    name: str | None = None      # human-readable device name, if known
    backend: str | None = None   # "cpu" | "torch-cuda" | "openvino"
    detail: str | None = None    # diagnostic note (missing driver, not installed, ...)


@dataclass
class HardwareReport:
    devices: list[DeviceInfo]
    recommended: str

    def get(self, device_id: str) -> DeviceInfo | None:
        for d in self.devices:
            if d.id == device_id:
                return d
        return None

    def best_available(self, allowed: set[str] | None = None) -> str:
        """
        Highest-priority available device, optionally restricted to *allowed*
        ids (e.g. the devices a given transcription backend can actually use).
        Always falls back to ``"cpu"``.
        """
        by_id = {d.id: d for d in self.devices}
        for device_id in _AUTO_PRIORITY:
            if allowed is not None and device_id not in allowed:
                continue
            info = by_id.get(device_id)
            if info is not None and info.available:
                return device_id
        return "cpu"


def _detect_cpu() -> DeviceInfo:
    return DeviceInfo(id="cpu", available=True, name="CPU", backend="cpu")


def _detect_cuda() -> DeviceInfo:
    try:
        import torch

        if torch.cuda.is_available():
            return DeviceInfo(
                id="cuda",
                available=True,
                name=torch.cuda.get_device_name(0),
                backend="torch-cuda",
            )
        return DeviceInfo(id="cuda", available=False, backend="torch-cuda", detail="no CUDA-capable GPU detected")
    except Exception as exc:
        return DeviceInfo(id="cuda", available=False, backend="torch-cuda", detail=f"torch not usable: {exc}")


def _detect_openvino_devices() -> tuple[DeviceInfo, DeviceInfo]:
    """Return (gpu_info, npu_info) via OpenVINO's device enumeration."""
    try:
        import openvino
    except ImportError:
        detail = "openvino not installed (pip install 'audio2report[intel]')"
        return (
            DeviceInfo(id="gpu", available=False, backend="openvino", detail=detail),
            DeviceInfo(id="npu", available=False, backend="openvino", detail=detail),
        )

    try:
        core = openvino.Core()
        available_devices = core.available_devices
    except Exception as exc:
        detail = f"openvino installed but device enumeration failed: {exc}"
        return (
            DeviceInfo(id="gpu", available=False, backend="openvino", detail=detail),
            DeviceInfo(id="npu", available=False, backend="openvino", detail=detail),
        )

    def _full_name(dev: str) -> str | None:
        try:
            return str(core.get_property(dev, "FULL_DEVICE_NAME"))
        except Exception:
            return None

    gpu_devices = sorted(d for d in available_devices if d == "GPU" or d.startswith("GPU."))
    npu_devices = [d for d in available_devices if d == "NPU"]

    if gpu_devices:
        primary = gpu_devices[0]
        detail = None
        if len(gpu_devices) > 1:
            detail = f"multiple Intel GPUs detected ({', '.join(gpu_devices)}); using {primary}"
        gpu_info = DeviceInfo(
            id="gpu", available=True, name=_full_name(primary), backend="openvino", detail=detail
        )
    else:
        gpu_info = DeviceInfo(
            id="gpu", available=False, backend="openvino", detail="no Intel GPU detected by OpenVINO"
        )

    if npu_devices:
        npu_info = DeviceInfo(
            id="npu", available=True, name=_full_name(npu_devices[0]), backend="openvino"
        )
    else:
        npu_info = DeviceInfo(
            id="npu", available=False, backend="openvino", detail="no Intel NPU detected by OpenVINO"
        )

    return gpu_info, npu_info


def detect_hardware() -> HardwareReport:
    """Enumerate CPU / NVIDIA CUDA / Intel GPU / Intel NPU availability."""
    cpu_info = _detect_cpu()
    cuda_info = _detect_cuda()
    gpu_info, npu_info = _detect_openvino_devices()

    devices = [cpu_info, cuda_info, gpu_info, npu_info]
    recommended = HardwareReport(devices=devices, recommended="cpu").best_available()
    return HardwareReport(devices=devices, recommended=recommended)
