"""Pure utility functions shared across pipeline stages."""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

AUDIO_EXTS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac",
    ".ogg", ".opus", ".wma", ".mp4", ".mov",
}

TEXT_MIN_LEN_FOR_ANCHOR = 25
TEXT_MIN_LEN_FOR_DEDUPE = 18


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip()).strip("_").lower()


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------

def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = out[-1]
        if s <= pe:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif hasattr(obj, "start") and hasattr(obj, "end"):
        # pyannote Segment
        return {"start": float(obj.start), "end": float(obj.end)}
    elif hasattr(obj, "__dict__"):
        return str(obj)
    return obj


def dump_json(path: Path, obj: Any) -> None:
    safe_obj = make_json_safe(obj)
    with path.open("w", encoding="utf-8") as f:
        json.dump(safe_obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

_VALID_DEVICES = {"cpu", "cuda", "gpu", "npu"}

# Single source of truth for which devices each transcription backend can
# actually use. Both auto-selection and explicit-device validation below
# derive from this — there is no separate blocklist to keep in sync.
_BACKEND_DEVICES = {
    "whisperx": {"cpu", "cuda"},
    "openvino": {"cpu", "gpu", "npu"},
}


def _allowed_devices(backend: str) -> set[str]:
    allowed = _BACKEND_DEVICES.get(backend)
    if allowed is None:
        raise ValueError(
            f"Unknown transcription backend: {backend!r}. "
            f"Valid options: {', '.join(sorted(_BACKEND_DEVICES))}"
        )
    return allowed


def pick_device(device_arg: str | None, backend: str = "whisperx") -> str:
    """
    Resolve the transcription device for *backend*.

    - ``None`` / ``"auto"``: pick the best available device *that this
      backend can use* (CUDA > Intel GPU > Intel NPU > CPU), silently
      degrading to whatever is actually present. Auto-selection is
      backend-aware so it never recommends a device the backend can't use.
    - An explicit device (``"cpu"``/``"cuda"``/``"gpu"``/``"npu"``): returned
      as-is if available *and* supported by *backend*, otherwise raises
      ``RuntimeError`` — an explicit request that can't be honored must fail
      loudly, never fall back.
    """
    from audio2report.hardware import detect_hardware

    allowed = _allowed_devices(backend)

    if device_arg is None or device_arg == "auto":
        return detect_hardware().best_available(allowed)

    if device_arg not in _VALID_DEVICES:
        raise ValueError(
            f"Unknown device '{device_arg}'. Valid options: "
            f"auto, {', '.join(sorted(_VALID_DEVICES))}"
        )

    if device_arg not in allowed:
        raise RuntimeError(
            f"transcription.backend={backend!r} cannot use device={device_arg!r}. "
            f"Supported devices for this backend: {', '.join(sorted(allowed))}."
        )

    report = detect_hardware()
    info = report.get(device_arg)
    if info is not None and info.available:
        return device_arg

    detected = ", ".join(
        f"{d.id}{'' if d.available else '(unavailable)'}" for d in report.devices
    )
    reason = f" ({info.detail})" if info and info.detail else ""
    raise RuntimeError(
        f"Requested device '{device_arg}' is not available{reason}. "
        f"Detected: {detected}. Run 'audio2report hardware' for details."
    )


def validate_backend_device(backend: str, device: str) -> None:
    """
    Defensive re-check that *device* is one *backend* supports — for callers
    (e.g. ``get_transcriber``) that may receive an already-resolved device
    without having gone through ``pick_device`` themselves.
    """
    if device not in _allowed_devices(backend):
        raise RuntimeError(
            f"transcription.backend={backend!r} cannot use device={device!r}. "
            f"Supported devices for this backend: "
            f"{', '.join(sorted(_allowed_devices(backend)))}."
        )


def auto_compute_type(compute_type: str, device: str) -> str:
    """Downgrade float16 → int8 automatically on CPU to avoid silent errors."""
    if device == "cpu" and compute_type == "float16":
        return "int8"
    return compute_type


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------

def format_hms(seconds: float) -> str:
    if seconds < 0:
        return f"-{format_hms(-seconds)}"
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
