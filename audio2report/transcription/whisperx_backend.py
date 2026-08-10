"""WhisperX transcription backend.

The model is loaded lazily on the first call to ``transcribe()`` and cached
for the lifetime of the object, so a single ``WhisperXTranscriber`` instance
can transcribe many files without reloading weights each time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from audio2report._log import get_logger
from audio2report.config import TranscriptionConfig
from audio2report.transcription._diarize import run_diarization
from audio2report.transcription.base import is_effectively_empty
from audio2report.utils import auto_compute_type

logger = get_logger(__name__)


class WhisperXTranscriber:
    """
    Wraps WhisperX with lazy model loading and cross-file caching.

    Parameters
    ----------
    config:
        Transcription configuration section.
    device:
        Resolved device string (``"cuda"`` or ``"cpu"``).  Must be resolved
        before construction so that ``auto_compute_type`` can adapt.
    """

    def __init__(self, config: TranscriptionConfig, device: str) -> None:
        self._config = config
        self._device = device
        self._compute_type = auto_compute_type(config.compute_type, device)

        self._model: Any = None
        self._align_model: Any = None
        self._align_metadata: Any = None
        self._align_language: str | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import whisperx
        except ImportError as exc:
            raise RuntimeError(
                "whisperx is not installed. Run: pip install 'audio2report[whisperx]'"
            ) from exc

        logger.info(
            f"Loading WhisperX model [bold]{self._config.model}[/bold] "
            f"on {self._device} ({self._compute_type})"
        )
        self._model = whisperx.load_model(
            self._config.model,
            device=self._device,
            compute_type=self._compute_type,
            language=self._config.language,
        )

    def _ensure_align_model(self, language: str) -> None:
        if self._align_model is not None and self._align_language == language:
            return
        try:
            import whisperx
        except ImportError as exc:
            raise RuntimeError("whisperx is not installed.") from exc

        self._align_model, self._align_metadata = whisperx.load_align_model(
            language_code=language,
            device=self._device,
        )
        self._align_language = language

    # ------------------------------------------------------------------
    # Public interface (satisfies AbstractTranscriber)
    # ------------------------------------------------------------------

    def transcribe(
        self,
        wav_path: Path,
        *,
        diarize: bool = False,
        hf_token: str | None = None,
    ) -> dict[str, Any]:
        try:
            import whisperx
        except ImportError as exc:
            raise RuntimeError("whisperx is not installed.") from exc

        self._ensure_model()

        audio = whisperx.load_audio(str(wav_path))
        result: dict[str, Any] = self._model.transcribe(audio, batch_size=self._config.batch_size)

        if is_effectively_empty(result):
            result["diarization_segments"] = []
            result["skipped_reason"] = "no_speech_or_too_short"
            return result

        self._ensure_align_model(result["language"])
        result = whisperx.align(
            result["segments"],
            self._align_model,
            self._align_metadata,
            audio,
            self._device,
            return_char_alignments=False,
        )

        if diarize:
            result = run_diarization(
                wav_path, result, hf_token=hf_token, device=self._device
            )
        else:
            result["diarization_segments"] = []

        result.setdefault("skipped_reason", None)
        return result
