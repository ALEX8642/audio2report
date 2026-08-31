"""Pydantic v2 configuration models and YAML loader."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

_VALID_BACKENDS = {"auto", "whisperx", "openvino", "torch_whisper"}
_VALID_DEVICES = {"auto", "cpu", "cuda", "gpu", "npu"}


class AudioConfig(BaseModel):
    inter_file_gap_sec: float = 0.5
    min_duration_sec: float = 1.0


class TranscriptionConfig(BaseModel):
    # "auto" resolves to whichever installed backend can actually use the
    # best available hardware — see utils.resolve_backend_and_device.
    backend: str = "auto"              # auto | whisperx (CUDA/CPU) | openvino (CPU/Intel GPU/Intel NPU) | torch_whisper (CUDA/CPU)
    model: str = "auto"                # auto | whisperx short model name | openvino HF repo id of an OpenVINO IR model
                                        # "auto" tiers model size to available VRAM/RAM — see utils.resolve_model
    language: str | None = None
    compute_type: str = "float16"      # whisperx-only; ignored by the openvino backend
    batch_size: int = 8                # whisperx-only; ignored by the openvino backend
    device: str | None = None          # None/"auto" → best available; or cpu | cuda | gpu | npu

    @field_validator("backend")
    @classmethod
    def _validate_backend(cls, v: str) -> str:
        if v not in _VALID_BACKENDS:
            raise ValueError(f"Invalid backend {v!r}. Valid options: {', '.join(sorted(_VALID_BACKENDS))}")
        return v

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str | None) -> str | None:
        if v is not None and v not in _VALID_DEVICES:
            raise ValueError(f"Invalid device {v!r}. Valid options: {', '.join(sorted(_VALID_DEVICES))}")
        return v


class DiarizationConfig(BaseModel):
    enabled: bool = True
    hf_token: str | None = None  # None → falls back to HF_TOKEN env var

    def resolved_token(self) -> str | None:
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
    formats: list[str] = Field(default_factory=lambda: ["json", "csv", "txt"])
    include_suppressed: bool = True


class LLMConfig(BaseModel):
    enabled: bool = False
    provider: str = "ollama"          # ollama | openai
    model: str = "llama3"
    base_url: str = "http://localhost:11434"
    api_key: str | None = None     # None → reads OPENAI_API_KEY env var
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


def load_config(path: Path | None = None) -> Config:
    """Load config from YAML file, or return defaults if no path given."""
    if path is None:
        return Config()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config.model_validate(data)
