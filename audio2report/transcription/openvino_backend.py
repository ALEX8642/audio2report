"""OpenVINO transcription backend — Intel CPU / Arc iGPU / NPU acceleration.

Uses ``openvino_genai.WhisperPipeline`` against a pre-converted OpenVINO IR
Whisper model.  Unlike :class:`~audio2report.transcription.whisperx_backend.
WhisperXTranscriber`, this backend does not run WhisperX's separate
forced-alignment step — the rest of the pipeline
(:func:`audio2report.alignment.timeline.result_to_segments`) only consumes
segment-level ``start``/``end``/``text``, so word-level alignment is not
needed to satisfy the existing contract.

``TranscriptionConfig.model`` is interpreted as a Hugging Face repo id of a
pre-converted OpenVINO IR model for this backend (e.g.
``"OpenVINO/whisper-large-v3-int4-ov"``), not a WhisperX short model name.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from audio2report._log import get_logger
from audio2report.config import TranscriptionConfig
from audio2report.transcription._diarize import run_diarization
from audio2report.transcription.base import is_effectively_empty

logger = get_logger(__name__)

# Devices for which OpenVINO benefits from an on-disk compiled-model cache
# (compilation is comparatively slow on GPU/NPU; not worthwhile on CPU).
_CACHEABLE_DEVICES = {"GPU", "NPU"}


def _ov_cache_dir() -> Path:
    """
    Per-user compiled-model cache directory — deliberately not a bare
    relative path, which would otherwise land wherever the process happens
    to be run from (e.g. a stray directory in a git checkout).
    """
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    cache_dir = Path(base) / "audio2report" / "ov_model_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _to_ov_language_token(language: str | None) -> str | None:
    if not language:
        return None
    if language.startswith("<|") and language.endswith("|>"):
        return language
    return f"<|{language}|>"


class OpenVINOTranscriber:
    """
    Wraps an OpenVINO ``WhisperPipeline`` with lazy model loading and
    cross-file caching, mirroring ``WhisperXTranscriber``'s lifecycle.

    Parameters
    ----------
    config:
        Transcription configuration section. ``model`` is treated as an HF
        repo id of a pre-converted OpenVINO IR Whisper model.
    device:
        Resolved device string (``"cpu"``, ``"gpu"``, or ``"npu"``).
    """

    def __init__(self, config: TranscriptionConfig, device: str) -> None:
        self._config = config
        self._device = device
        self._ov_device = device.upper()

        self._pipeline: Any = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_model_dir(self) -> str:
        """Download (if needed) and return the local path to the IR model."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is not installed. Run: pip install 'audio2report[intel]'"
            ) from exc

        return snapshot_download(repo_id=self._config.model)

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import openvino_genai
        except ImportError as exc:
            raise RuntimeError(
                "openvino-genai is not installed. Run: pip install 'audio2report[intel]'"
            ) from exc

        model_dir = self._resolve_model_dir()

        ov_config: dict[str, Any] = {}
        if self._ov_device in _CACHEABLE_DEVICES:
            ov_config["CACHE_DIR"] = str(_ov_cache_dir())

        logger.info(
            f"Loading OpenVINO Whisper model [bold]{self._config.model}[/bold] "
            f"on {self._ov_device}"
        )
        self._pipeline = openvino_genai.WhisperPipeline(model_dir, self._ov_device, **ov_config)

    @staticmethod
    def _load_audio(wav_path: Path) -> list[float]:
        import numpy as np
        import soundfile as sf

        audio, _samplerate = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return np.asarray(audio, dtype=np.float32).tolist()

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
        self._ensure_pipeline()

        raw_speech = self._load_audio(wav_path)

        generate_kwargs: dict[str, Any] = {"return_timestamps": True}
        language_token = _to_ov_language_token(self._config.language)
        if language_token:
            generate_kwargs["language"] = language_token

        ov_result = self._pipeline.generate(raw_speech, **generate_kwargs)

        chunks = ov_result.chunks or []
        segments = [
            {"start": float(c.start_ts), "end": float(c.end_ts), "text": c.text}
            for c in chunks
        ]
        if not segments and ov_result.texts:
            # Model didn't return timestamped chunks — fall back to one
            # segment spanning the whole clip so downstream code still works.
            segments = [{
                "start": 0.0,
                "end": len(raw_speech) / 16000.0,
                "text": ov_result.texts[0],
            }]

        result: dict[str, Any] = {
            "segments": segments,
            "language": ov_result.language or self._config.language,
        }

        if is_effectively_empty(result):
            result["diarization_segments"] = []
            result["skipped_reason"] = "no_speech_or_too_short"
            return result

        if diarize:
            # Diarization (pyannote/torch) is orthogonal to the ASR engine and
            # has no Intel GPU/NPU support of its own — always resolved to
            # cuda/cpu independently of the OpenVINO ASR device.
            from audio2report.hardware import detect_hardware

            cuda_info = detect_hardware().get("cuda")
            diar_device = "cuda" if cuda_info and cuda_info.available else "cpu"
            result = run_diarization(wav_path, result, hf_token=hf_token, device=diar_device)
        else:
            result["diarization_segments"] = []

        result.setdefault("skipped_reason", None)
        return result
