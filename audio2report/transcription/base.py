"""Abstract transcriber protocol — swap backends without touching the pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class AbstractTranscriber(Protocol):
    def transcribe(
        self,
        wav_path: Path,
        *,
        diarize: bool = False,
        hf_token: Optional[str] = None,
    ) -> Dict[str, Any]:
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
