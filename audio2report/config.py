"""Pydantic v2 configuration models and YAML loader."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


class AudioConfig(BaseModel):
    inter_file_gap_sec: float = 0.5
    min_duration_sec: float = 1.0


class TranscriptionConfig(BaseModel):
    backend: str = "whisperx"
    model: str = "large-v3"
    language: Optional[str] = None
    compute_type: str = "float16"
    batch_size: int = 8
    device: Optional[str] = None  # None → auto-detect (cuda if available, else cpu)


class DiarizationConfig(BaseModel):
    enabled: bool = False
    hf_token: Optional[str] = None  # None → falls back to HF_TOKEN env var

    def resolved_token(self) -> Optional[str]:
        return self.hf_token or os.environ.get("HF_TOKEN")


class AlignmentConfig(BaseModel):
    anchor_sim_threshold: float = 0.90
    min_anchor_text_len: int = 25


class DeduplicationConfig(BaseModel):
    enabled: bool = True
    sim_threshold: float = 0.86
    time_tolerance_sec: float = 2.5
    min_text_len: int = 18


class OutputConfig(BaseModel):
    formats: List[str] = Field(default_factory=lambda: ["json", "csv", "txt"])
    include_suppressed: bool = True


class LLMConfig(BaseModel):
    enabled: bool = False
    provider: str = "ollama"          # ollama | openai
    model: str = "llama3"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None     # None → reads OPENAI_API_KEY env var
    prompt_template: str = "audit_report"
    max_transcript_chars: int = 50_000  # truncate transcript if longer
    stream: bool = True               # stream response to terminal


class Config(BaseModel):
    mode: str = "dual"
    cache: bool = True  # skip re-transcription when per-file JSON already exists

    audio: AudioConfig = Field(default_factory=AudioConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


def load_config(path: Optional[Path] = None) -> Config:
    """Load config from YAML file, or return defaults if no path given."""
    if path is None:
        return Config()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config.model_validate(data)
