"""ffmpeg / ffprobe wrappers for audio normalisation."""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np


def ffprobe_duration_seconds(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}:\n{res.stderr}")
    return float(res.stdout.strip())


def ffmpeg_normalize_to_wav(src: Path, dst: Path) -> None:
    """Convert *src* to mono 16 kHz PCM WAV at *dst*."""
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src}:\n{res.stderr}")


def is_audio_too_short_or_silent(wav_path: Path, min_duration: float = 1.0) -> bool:
    dur = ffprobe_duration_seconds(wav_path)
    return dur < min_duration


def rms_dbfs_for_region(wav_path: str, start_sec: float, end_sec: float) -> float | None:
    """Return RMS dBFS for the given time region, or None on failure."""
    try:
        import soundfile as sf
    except ImportError:
        return None
    try:
        data, sr = sf.read(wav_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        start = max(0, int(start_sec * sr))
        end = min(len(data), int(end_sec * sr))
        if end <= start:
            return None
        x = data[start:end].astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(x))) + 1e-12)
        return 20.0 * math.log10(rms + 1e-12)
    except Exception:
        return None
