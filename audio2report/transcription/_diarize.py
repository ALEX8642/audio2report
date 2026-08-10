"""Shared speaker diarization step, reused by every transcription backend.

Diarization (pyannote.audio) is orthogonal to the ASR engine — it has no
Intel GPU/NPU support of its own, so it always runs on ``cuda`` or ``cpu``
regardless of which device the transcription backend itself is using.

This calls ``pyannote.audio`` directly rather than going through WhisperX's
``whisperx.diarize`` wrapper, so diarization works even when only the
``diarize`` extra (not the full ``whisperx`` extra) is installed — the
``openvino`` backend has no other reason to depend on WhisperX at all.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# Matches the default model used by WhisperX's own DiarizationPipeline, so
# diarization quality is consistent regardless of which ASR backend is used.
_DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


def run_diarization(
    wav_path: Path,
    asr_result: dict[str, Any],
    *,
    hf_token: str | None,
    device: str,
) -> dict[str, Any]:
    """
    Run pyannote diarization on *wav_path* and assign a ``speaker`` label to
    each of *asr_result*'s segments by maximal temporal overlap. Returns the
    updated result dict with a populated ``diarization_segments`` key.
    """
    try:
        import soundfile as sf
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio is not installed (required for diarization). "
            "Run: pip install 'audio2report[diarize]'"
        ) from exc

    if not hf_token:
        raise RuntimeError(
            "Diarization requested but no Hugging Face token was provided. "
            "Pass --hf-token or set the HF_TOKEN environment variable."
        )

    pipeline = Pipeline.from_pretrained(_DEFAULT_DIARIZATION_MODEL, token=hf_token)
    pipeline.to(torch.device(device))

    # Preload the waveform ourselves rather than passing a path — pyannote's
    # own path-based audio loading depends on torchcodec, which needs FFmpeg
    # shared libraries that aren't guaranteed to be present (a system `ffmpeg`
    # binary is not sufficient). soundfile has no such dependency.
    audio, samplerate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T)  # (channel, time), as pyannote expects
    output = pipeline({"waveform": waveform, "sample_rate": samplerate})
    # Newer pyannote pipelines wrap the annotation in an output object
    # (`output.speaker_diarization`); older ones return the Annotation
    # directly — handle both.
    diarization = getattr(output, "speaker_diarization", output)

    diar_records = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": speaker}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    # Assign each ASR segment the speaker with the largest total temporal
    # overlap (mirrors whisperx.diarize.assign_word_speakers's segment-level
    # logic; word-level assignment doesn't apply since these backends don't
    # produce word-level timestamps).
    for seg in asr_result.get("segments", []):
        seg_start = float(seg.get("start", 0.0) or 0.0)
        seg_end = float(seg.get("end", 0.0) or 0.0)
        overlap_by_speaker: dict[str, float] = {}
        for rec in diar_records:
            intersection = min(seg_end, rec["end"]) - max(seg_start, rec["start"])
            if intersection > 0:
                overlap_by_speaker[rec["speaker"]] = (
                    overlap_by_speaker.get(rec["speaker"], 0.0) + intersection
                )
        if overlap_by_speaker:
            seg["speaker"] = max(overlap_by_speaker.items(), key=lambda kv: kv[1])[0]

    asr_result["diarization_segments"] = diar_records
    return asr_result
