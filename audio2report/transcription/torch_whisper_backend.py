"""Plain-PyTorch Whisper transcription backend (the ``openai-whisper``
package directly, not WhisperX/ctranslate2).

Exists specifically for hardware where ctranslate2 (WhisperX's ASR engine)
has no CUDA wheel at all — confirmed on NVIDIA DGX Spark (aarch64): PyPI
publishes ctranslate2 for aarch64 as CPU-only, full stop, no CUDA variant
to install. This backend uses torch directly, so it gets real GPU
acceleration anywhere torch itself has it, independent of ctranslate2.

Model naming is a happy coincidence with WhisperX — ``openai-whisper``
uses the same short names (``"large-v3"``, ``"tiny"``, ...), so
``TranscriptionConfig.model`` needs no backend-specific handling here,
unlike the ``openvino`` backend's HF-repo-id convention.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from audio2report._log import get_logger
from audio2report.config import TranscriptionConfig
from audio2report.transcription._diarize import run_diarization
from audio2report.transcription.base import is_effectively_empty

logger = get_logger(__name__)


class TorchWhisperTranscriber:
    """
    Wraps ``openai-whisper`` with lazy model loading and cross-file
    caching, mirroring ``WhisperXTranscriber``/``OpenVINOTranscriber``'s
    lifecycle.

    Parameters
    ----------
    config:
        Transcription configuration section.
    device:
        Resolved device string (``"cuda"`` or ``"cpu"``).
    """

    def __init__(self, config: TranscriptionConfig, device: str) -> None:
        self._config = config
        self._device = device
        self._model: Any = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError(
                "openai-whisper is not installed. Run: pip install 'audio2report[torch_whisper]'"
            ) from exc

        logger.info(
            f"Loading PyTorch Whisper model [bold]{self._config.model}[/bold] on {self._device}"
        )
        self._model = whisper.load_model(self._config.model, device=self._device)

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
        self._ensure_model()

        whisper_result = self._model.transcribe(
            str(wav_path),
            language=self._config.language,
        )

        segments = [
            {
                "start": float(s["start"]),
                "end": float(s["end"]),
                "text": s["text"],
                "avg_logprob": s.get("avg_logprob"),
                "no_speech_prob": s.get("no_speech_prob"),
            }
            for s in whisper_result.get("segments", [])
        ]

        result: dict[str, Any] = {
            "segments": segments,
            "language": whisper_result.get("language") or self._config.language,
        }

        if is_effectively_empty(result):
            result["diarization_segments"] = []
            result["skipped_reason"] = "no_speech_or_too_short"
            return result

        if diarize:
            # Diarization (pyannote/torch) is orthogonal to the ASR
            # engine — same reasoning as the openvino backend: it's
            # resolved to its own device independently.
            result = run_diarization(wav_path, result, hf_token=hf_token, device=self._device)
        else:
            result["diarization_segments"] = []

        result.setdefault("skipped_reason", None)
        return result
