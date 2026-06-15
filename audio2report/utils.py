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

def pick_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


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
