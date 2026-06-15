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
from audio2report.utils import auto_compute_type

logger = get_logger(__name__)


def _is_effectively_empty(result: dict[str, Any]) -> bool:
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

        if _is_effectively_empty(result):
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
            if not hf_token:
                raise RuntimeError(
                    "Diarization requested but no Hugging Face token was provided. "
                    "Pass --hf-token or set the HF_TOKEN environment variable."
                )
            diarize_model = whisperx.diarize.DiarizationPipeline(
                token=hf_token,
                device=self._device,
            )
            diar_segments = diarize_model(str(wav_path))
            result = whisperx.assign_word_speakers(diar_segments, result)
            result["diarization_segments"] = diar_segments.to_dict("records")
        else:
            result["diarization_segments"] = []

        result.setdefault("skipped_reason", None)
        return result
