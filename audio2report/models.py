"""Shared data models (dataclasses) used as the contract between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
    avg_logprob: Optional[float]
    no_speech_prob: Optional[float]
    diar_speaker_raw: Optional[str]
    diar_speaker_role: Optional[str]
    rms_dbfs: Optional[float]
    speaker_final: Optional[str]
    speaker_confidence: Optional[float]
    duplicate_of: Optional[str]
    keep: bool
    # fields with defaults must follow fields without defaults
    flags: List[str] = field(default_factory=list)
    retention_score_value: Optional[float] = None
    speaker_score_detail: Optional[Dict[str, float]] = None
    speaker_decision_basis: Optional[str] = None


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
    prime_folders: List[str]
    primes: List[str]
    device: str
    model: str
    language: Optional[str]
    diarize: bool
    estimated_offset_b_minus_a_sec: float
    anchor_count: int
    pair_match_count: int
    total_segments: int
    kept_segments: int
    suppressed_segments: int


@dataclass
class RunResult:
    segments: List[SegmentRecord]          # all segments (kept + suppressed), sorted by time
    cleaned_segments: List[SegmentRecord]  # post-processed, LLM-ready
    anchors: List[AlignmentAnchor]
    pair_matches: List[PairMatch]
    meta: RunMeta
