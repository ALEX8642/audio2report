"""Output writers: JSON, CSV, TXT, and the cleaned LLM-input text."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from audio2report.config import OutputConfig
from audio2report.models import RunResult, SegmentRecord
from audio2report.utils import dump_json, format_hms

# ---------------------------------------------------------------------------
# Individual writers
# ---------------------------------------------------------------------------

def write_json(path: Path, rows: list[SegmentRecord], extra: dict[str, Any]) -> None:
    payload = {"meta": extra, "segments": [asdict(r) for r in rows]}
    dump_json(path, payload)


def write_csv(path: Path, rows: list[SegmentRecord]) -> None:
    fields = [
        "uid", "speaker_final", "speaker_confidence", "retention_score_value",
        "speaker_score_detail", "speaker_decision_basis",
        "global_start_sec", "global_end_sec",
        "channel_prime", "channel_folder", "file_index", "source_file",
        "diar_speaker_raw", "diar_speaker_role",
        "avg_logprob", "no_speech_prob", "rms_dbfs",
        "duplicate_of", "keep", "flags", "text",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row: dict[str, Any] = {}
            for k in fields:
                if k == "flags":
                    row[k] = ";".join(r.flags)
                elif k == "speaker_score_detail":
                    row[k] = (
                        json.dumps(r.speaker_score_detail, ensure_ascii=False)
                        if r.speaker_score_detail is not None
                        else ""
                    )
                else:
                    row[k] = getattr(r, k)
            w.writerow(row)


def write_txt(path: Path, rows: list[SegmentRecord]) -> None:
    """Write all kept segments with timestamps, speaker, and flag annotations."""
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            if not r.keep:
                continue
            ts = f"[{format_hms(r.global_start_sec)} - {format_hms(r.global_end_sec)}]"
            speaker = r.speaker_final or "UNKNOWN"
            flag_text = f" [{' | '.join(r.flags)}]" if r.flags else ""
            f.write(f"{ts} {speaker}: {r.text}{flag_text}\n")


def write_clean_txt(path: Path, rows: list[SegmentRecord]) -> None:
    """Write post-processed segments — clean, no flag annotations."""
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            speaker = r.speaker_final or "UNKNOWN"
            ts = f"[{format_hms(r.global_start_sec)} - {format_hms(r.global_end_sec)}]"
            f.write(f"{ts} {speaker}: {r.text}\n")


# ---------------------------------------------------------------------------
# Bulk output dispatcher
# ---------------------------------------------------------------------------

def write_all_outputs(result: RunResult, out_root: Path, config: OutputConfig) -> None:
    """Write every configured output format for *result* into *out_root*."""
    from dataclasses import asdict as _asdict

    meta_dict = _asdict(result.meta)
    formats = {f.lower() for f in config.formats}

    if "json" in formats:
        write_json(out_root / "canonical_transcript.json", result.segments, meta_dict)

    if "csv" in formats:
        write_csv(out_root / "canonical_transcript.csv", result.segments)

    if "txt" in formats:
        write_txt(out_root / "canonical_transcript.txt", result.segments)

    # clean LLM input is always written (it is the primary deliverable)
    write_clean_txt(out_root / "cleaned_llm_input.txt", result.cleaned_segments)

    # diagnostics
    from dataclasses import asdict as _asdict2
    dump_json(out_root / "alignment_anchors.json", [_asdict2(a) for a in result.anchors])
    dump_json(out_root / "pair_matches.json", [_asdict2(p) for p in result.pair_matches])
    dump_json(out_root / "run_meta.json", meta_dict)
