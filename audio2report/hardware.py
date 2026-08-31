"""Compute-hardware detection: what devices are present and which one 'auto' picks.

Detection is best-effort and defensive — a missing/broken backend (no torch,
no openvino, no drivers) must never prevent detection of the others.  All
heavy imports (torch, openvino) are performed lazily inside
:func:`detect_hardware` so importing this module stays cheap enough for the
Streamlit UI to call it directly on every render.
"""
from __future__ import annotations

import os
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


# ---------------------------------------------------------------------------
# Memory detection — backs "auto" Whisper model-size selection
# (TranscriptionConfig.model="auto"; see utils.resolve_model). Detecting
# *device* memory, not just device presence, matters here: pointing a
# low-memory machine at the hardcoded "large-v3" that used to be the
# unconditional default is a real OOM risk, not a hypothetical one.
# ---------------------------------------------------------------------------

def _system_ram_bytes() -> int | None:
    """
    Best-effort total system RAM. Used as the memory budget for every
    device except CUDA: CPU obviously shares system RAM, and on the Intel
    hardware this project targets (Arc iGPU, AI Boost NPU) the GPU/NPU are
    integrated and share system RAM too — there is no separate VRAM pool to
    query, and no reliable cross-platform OpenVINO API to query one even if
    there were on some discrete-GPU machine.
    """
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass

    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
        return int(stat.ullTotalPhys)
    except Exception:
        return None


def _cuda_vram_bytes() -> int | None:
    """Total VRAM on CUDA device 0, or None if CUDA isn't usable."""
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        pass
    return None


# Minimum bytes of available memory required to safely run each model size,
# ordered largest-first — approximate, with headroom for OS/runtime
# overhead beyond the model weights themselves. Deliberately conservative:
# picking a smaller model than strictly necessary just costs some accuracy,
# while picking one too large OOMs or crashes the run outright.
_MODEL_MEMORY_TIERS: list[tuple[int, str]] = [
    (10 * 1024**3, "large-v3"),
    (5 * 1024**3, "medium"),
    (2 * 1024**3, "small"),
    (1 * 1024**3, "base"),
]
_SMALLEST_MODEL = "tiny"


def recommend_model_size(device: str) -> str:
    """
    Recommend a WhisperX-style short model name (``"tiny"`` .. ``"large-v3"``)
    tiered to the memory actually available to *device* — CUDA VRAM for
    ``"cuda"``, system RAM for everything else (see ``_system_ram_bytes``).

    When memory can't be determined at all (detection failed, unsupported
    platform), this deliberately returns a mid-sized, safe default
    (``"small"``) rather than the previous hardcoded ``"large-v3"`` — the
    exact failure mode this function exists to avoid.
    """
    available = _cuda_vram_bytes() if device == "cuda" else _system_ram_bytes()
    if available is None:
        return "small"
    for threshold, size in _MODEL_MEMORY_TIERS:
        if available >= threshold:
            return size
    return _SMALLEST_MODEL
