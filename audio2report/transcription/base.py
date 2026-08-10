"""Abstract transcriber protocol — swap backends without touching the pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from audio2report.config import TranscriptionConfig
from audio2report.utils import validate_backend_device


@runtime_checkable
class AbstractTranscriber(Protocol):
    def transcribe(
        self,
        wav_path: Path,
        *,
        diarize: bool = False,
        hf_token: str | None = None,
    ) -> dict[str, Any]:
        """
        Transcribe *wav_path* and return a WhisperX-style result dict:
        {
            "segments": [...],
            "diarization_segments": [...],
            "language": "en",
            "skipped_reason": None | str,
        }
        """
        ...


def is_effectively_empty(result: dict[str, Any]) -> bool:
    """True if a transcription result has no meaningful speech content."""
    segments = result.get("segments") or []
    if not segments:
        return True
    total_text = " ".join((s.get("text") or "").strip() for s in segments).strip()
    if len(total_text) < 3:
        return True
    total_duration = sum(
        max(0.0, float(s.get("end", 0.0) or 0.0) - float(s.get("start", 0.0) or 0.0))
        for s in segments
    )
    return total_duration < 0.5


def get_transcriber(config: TranscriptionConfig, device: str) -> AbstractTranscriber:
    """Construct the transcriber for *config.backend*, resolved against *device*."""
    # Raises for any backend other than "whisperx"/"openvino", so the if/else
    # below only ever has to distinguish between those two known-valid cases.
    validate_backend_device(config.backend, device)

    if config.backend == "whisperx":
        from audio2report.transcription.whisperx_backend import WhisperXTranscriber
        return WhisperXTranscriber(config, device)
    else:
        from audio2report.transcription.openvino_backend import OpenVINOTranscriber
        return OpenVINOTranscriber(config, device)
