"""Shared data models (dataclasses) used as the contract between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AudioFileRecord:
    path: str
    normalized_wav: str
    folder_prime: str
    folder_name: str
    order_index: int
    duration_sec: float
    local_file_start_sec: float
    local_file_end_sec: float


@dataclass
class SegmentRecord:
    uid: str
    source_file: str
    source_wav: str
    channel_prime: str
    channel_folder: str
    file_index: int
    local_file_start_sec: float
    local_file_end_sec: float
    local_seg_start_sec: float
    local_seg_end_sec: float
    root_timeline_start_sec: float
    root_timeline_end_sec: float
    global_start_sec: float
    global_end_sec: float
    text: str
    avg_logprob: float | None
    no_speech_prob: float | None
    diar_speaker_raw: str | None
    diar_speaker_role: str | None
    rms_dbfs: float | None
    speaker_final: str | None
    speaker_confidence: float | None
    duplicate_of: str | None
    keep: bool
    # fields with defaults must follow fields without defaults
    flags: list[str] = field(default_factory=list)
    retention_score_value: float | None = None
    speaker_score_detail: dict[str, float] | None = None
    speaker_decision_basis: str | None = None


@dataclass
class AlignmentAnchor:
    a_uid: str
    b_uid: str
    a_prime: str
    b_prime: str
    sim: float
    delta_sec: float
    a_text: str
    b_text: str


@dataclass
class PairMatch:
    a_uid: str
    b_uid: str
    sim: float
    time_diff_sec: float


@dataclass
class RunMeta:
    root: str
    prime_folders: list[str]
    primes: list[str]
    device: str
    model: str
    language: str | None
    diarize: bool
    estimated_offset_b_minus_a_sec: float
    anchor_count: int
    pair_match_count: int
    total_segments: int
    kept_segments: int
    suppressed_segments: int


@dataclass
class RunResult:
    segments: list[SegmentRecord]          # all segments (kept + suppressed), sorted by time
    cleaned_segments: list[SegmentRecord]  # post-processed, LLM-ready
    anchors: list[AlignmentAnchor]
    pair_matches: list[PairMatch]
    meta: RunMeta
