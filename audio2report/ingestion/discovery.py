"""Audio file discovery, folder scanning, and timeline construction."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from audio2report._log import get_logger
from audio2report.ingestion.normalize import ffmpeg_normalize_to_wav, ffprobe_duration_seconds
from audio2report.models import AudioFileRecord
from audio2report.utils import AUDIO_EXTS, ensure_dir, safe_slug

logger = get_logger(__name__)


def parse_prime_from_folder(folder_name: str) -> str:
    """
    Extract the prime speaker name from a folder name.

    Examples:
        'TX_MIC - alex prime'   -> 'alex'
        'TX_MIC - Rijad prime'  -> 'Rijad'
    """
    m = re.search(r"-\s*(.*?)\s+prime\s*$", folder_name, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    cleaned = re.sub(r"\btx[_\s-]*mic\b", "", folder_name, flags=re.IGNORECASE).strip(" -_")
    cleaned = re.sub(r"\bprime\b", "", cleaned, flags=re.IGNORECASE).strip(" -_")
    return cleaned if cleaned else folder_name.strip()


def find_prime_folders(root: Path) -> list[Path]:
    """Return all subdirectories whose names contain the word 'prime'."""
    folders = [
        p for p in root.iterdir()
        if p.is_dir() and re.search(r"\bprime\b", p.name, re.IGNORECASE)
    ]
    return sorted(folders, key=lambda x: x.name.lower())


def extract_timestamp(p: Path) -> float | None:
    """
    Extract a wall-clock timestamp from filenames like:
        TX01_MIC001_20260331_181445_orig.wav
    Returns a POSIX timestamp float, or None if the pattern is absent.
    """
    m = re.search(r"_(\d{8})_(\d{6})", p.stem)
    if m:
        dt_str = m.group(1) + m.group(2)
        return datetime.strptime(dt_str, "%Y%m%d%H%M%S").timestamp()
    return None


def file_sort_key(p: Path):
    ts = extract_timestamp(p)
    if ts is not None:
        return (0, ts)
    nums = re.findall(r"\d+", p.stem)
    nums_tuple = tuple(int(n) for n in nums) if nums else (10**12,)
    return (1, nums_tuple, p.stat().st_mtime, p.name.lower())


def list_audio_files(folder: Path) -> list[Path]:
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    files.sort(key=file_sort_key)
    logger.info(f"Sorted files for [bold]{folder.name}[/bold]:")
    for f in files:
        logger.info(f"  {f.name}")
    return files


def build_file_records_for_folder(
    folder: Path,
    normalized_root: Path,
    inter_file_gap_sec: float,
) -> list[AudioFileRecord]:
    """
    Normalize all audio files in *folder* to mono 16 kHz WAV and build
    an ``AudioFileRecord`` list with folder-local timeline timestamps.
    """
    prime = parse_prime_from_folder(folder.name)
    files = list_audio_files(folder)
    if not files:
        return []

    folder_out = normalized_root / safe_slug(folder.name)
    ensure_dir(folder_out)

    first_ts = extract_timestamp(files[0])
    running_t = 0.0
    out: list[AudioFileRecord] = []

    for i, src in enumerate(files):
        dst = folder_out / f"{i:04d}_{safe_slug(src.stem)}.wav"
        ffmpeg_normalize_to_wav(src, dst)
        dur = ffprobe_duration_seconds(dst)

        ts = extract_timestamp(src)
        if ts is not None and first_ts is not None:
            file_start = ts - first_ts
        else:
            file_start = running_t

        out.append(AudioFileRecord(
            path=str(src),
            normalized_wav=str(dst),
            folder_prime=prime,
            folder_name=folder.name,
            order_index=i,
            duration_sec=dur,
            local_file_start_sec=file_start,
            local_file_end_sec=file_start + dur,
        ))
        running_t = file_start + dur + inter_file_gap_sec

    return out
