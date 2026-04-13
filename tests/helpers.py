"""Shared test helpers — not pytest fixtures, just plain factory functions."""
from __future__ import annotations

from typing import Optional

from audio2report.models import SegmentRecord


def make_segment(
    uid: str,
    prime: str,
    folder: str,
    text: str,
    t_start: float,
    t_end: float,
    *,
    file_index: int = 0,
    diar_speaker_raw: Optional[str] = None,
    diar_speaker_role: Optional[str] = None,
    rms_dbfs: float = -18.0,
    avg_logprob: float = -0.2,
    no_speech_prob: float = 0.05,
    speaker_final: Optional[str] = None,
    keep: bool = True,
) -> SegmentRecord:
    """Factory for SegmentRecord with sensible defaults for testing."""
    return SegmentRecord(
        uid=uid,
        source_file=f"/fake/{folder}/audio.wav",
        source_wav=f"/fake/{folder}/norm.wav",
        channel_prime=prime,
        channel_folder=folder,
        file_index=file_index,
        local_file_start_sec=0.0,
        local_file_end_sec=200.0,
        local_seg_start_sec=t_start,
        local_seg_end_sec=t_end,
        root_timeline_start_sec=t_start,
        root_timeline_end_sec=t_end,
        global_start_sec=t_start,
        global_end_sec=t_end,
        text=text,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
        diar_speaker_raw=diar_speaker_raw,
        diar_speaker_role=diar_speaker_role,
        rms_dbfs=rms_dbfs,
        speaker_final=speaker_final,
        speaker_confidence=None,
        duplicate_of=None,
        keep=keep,
    )
