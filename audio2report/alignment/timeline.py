"""Timeline construction and offset application."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from audio2report.ingestion.normalize import rms_dbfs_for_region
from audio2report.models import AudioFileRecord, SegmentRecord
from audio2report.utils import safe_slug


def result_to_segments(
    result: Dict[str, Any],
    file_record: AudioFileRecord,
) -> List[SegmentRecord]:
    """Convert a WhisperX result dict into ``SegmentRecord`` objects."""
    segments: List[SegmentRecord] = []

    for idx, seg in enumerate(result.get("segments", [])):
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()
        diar_raw = seg.get("speaker")

        root_s = file_record.local_file_start_sec + s
        root_e = file_record.local_file_start_sec + e
        rms = rms_dbfs_for_region(file_record.normalized_wav, s, e)

        segments.append(SegmentRecord(
            uid=f"{safe_slug(file_record.folder_name)}__f{file_record.order_index:04d}__s{idx:04d}",
            source_file=file_record.path,
            source_wav=file_record.normalized_wav,
            channel_prime=file_record.folder_prime,
            channel_folder=file_record.folder_name,
            file_index=file_record.order_index,
            local_file_start_sec=file_record.local_file_start_sec,
            local_file_end_sec=file_record.local_file_end_sec,
            local_seg_start_sec=s,
            local_seg_end_sec=e,
            root_timeline_start_sec=root_s,
            root_timeline_end_sec=root_e,
            global_start_sec=root_s,
            global_end_sec=root_e,
            text=text,
            avg_logprob=seg.get("avg_logprob"),
            no_speech_prob=seg.get("no_speech_prob"),
            diar_speaker_raw=diar_raw,
            diar_speaker_role=None,
            rms_dbfs=rms,
            speaker_final=None,
            speaker_confidence=None,
            duplicate_of=None,
            keep=True,
        ))

    return segments


def apply_offset_to_channel(segments: List[SegmentRecord], offset_sec: float) -> None:
    """Shift all global timestamps in *segments* by *offset_sec*."""
    for seg in segments:
        seg.global_start_sec = seg.root_timeline_start_sec + offset_sec
        seg.global_end_sec = seg.root_timeline_end_sec + offset_sec
