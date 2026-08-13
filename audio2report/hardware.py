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
    """Return (gpu_info, npu_info) via OpenVINO's device enumeration.

    Requires both ``openvino`` (device enumeration) and ``openvino_genai``
    (what ``OpenVINOTranscriber`` actually imports to run) — a machine with
    only the former installed can enumerate devices just fine but would
    still crash the moment the openvino backend tried to load a model.
    """
    try:
        import openvino
        import openvino_genai  # noqa: F401
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
        gpu_detail = None
        if len(gpu_devices) > 1:
            gpu_detail = f"multiple Intel GPUs detected ({', '.join(gpu_devices)}); using {primary}"
        gpu_info = DeviceInfo(
            id="gpu", available=True, name=_full_name(primary), backend="openvino", detail=gpu_detail
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


def openvino_installed() -> bool:
    """Whether both packages the openvino backend needs are importable —
    same pairing ``_detect_openvino_devices`` already requires."""
    try:
        import openvino  # noqa: F401
        import openvino_genai  # noqa: F401
        return True
    except ImportError:
        return False


def whisperx_installed() -> bool:
    """Whether the whisperx backend's package is installed at all — a
    separate question from ctranslate2_cuda_capable() below, since a
    machine commonly has whisperx installed with a CPU-only ctranslate2
    build (the default case everywhere except a CUDA GPU host)."""
    try:
        import whisperx  # noqa: F401
        return True
    except ImportError:
        return False


def ctranslate2_cuda_capable() -> bool:
    """Whether the *whisperx* backend can actually use CUDA here.

    Deliberately a different question from "is a CUDA device present"
    (``_detect_cuda`` above, which only checks PyTorch): whisperx's ASR
    engine is ctranslate2, a separately-compiled library whose CUDA
    support is a compile-time property of the installed wheel, not a
    runtime driver question. Confirmed to diverge in practice — PyPI's
    ctranslate2 wheels for aarch64 (e.g. NVIDIA DGX Spark) are CPU-only,
    full stop, even though torch.cuda.is_available() is True there.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False
    except Exception:
        return False

    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def torch_whisper_installed() -> bool:
    """Whether the torch_whisper backend's package (openai-whisper) is
    installed. Actual CUDA usability is governed by plain torch.cuda —
    no separate compiled-engine capability question here, unlike
    ctranslate2 above."""
    try:
        import whisper  # noqa: F401
        return True
    except ImportError:
        return False


def detect_hardware() -> HardwareReport:
    """Enumerate CPU / NVIDIA CUDA / Intel GPU / Intel NPU availability."""
    cpu_info = _detect_cpu()
    cuda_info = _detect_cuda()
    gpu_info, npu_info = _detect_openvino_devices()

    devices = [cpu_info, cuda_info, gpu_info, npu_info]
    recommended = HardwareReport(devices=devices, recommended="cpu").best_available()
    return HardwareReport(devices=devices, recommended=recommended)
