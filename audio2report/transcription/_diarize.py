"""Shared speaker diarization step, reused by every transcription backend.

Diarization (pyannote.audio) is orthogonal to the ASR engine, so it's
resolved to its own device independently of whatever device the ASR backend
is using. It supports NVIDIA CUDA, Intel GPU (via PyTorch's native XPU
backend — a different mechanism than OpenVINO, which the ASR backend uses),
and CPU.

Real-hardware timing on an Intel Arc iGPU (Framework 13 Pro Series 3): CPU
runs at a consistent ~2.4x realtime regardless of file length. XPU has a
large fixed warm-up cost per distinct audio length (most likely one-time
kernel JIT compilation per tensor shape) — a net loss on a single short
clip, but on a real 6.6-hour/39-file batch (varied file lengths, so that
warm-up cost recurs more than a single-file test would suggest) it was
still a real ~2.5x wall-clock win over CPU. See
docs/INTEL_ACCELERATION_ROADMAP.md for the full writeup, including why an
earlier same-file-repeated benchmark (~60x steady state) overstated the
real full-batch benefit — trust the full-batch number for planning.

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


def resolve_diarization_device() -> str:
    """
    Best available PyTorch device for diarization: CUDA > Intel GPU (XPU) >
    CPU. Independent of the ASR backend's own device — OpenVINO's device
    concept and PyTorch's are unrelated mechanisms.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    try:
        import torch
        if torch.xpu.is_available():
            return "xpu"
    except (ImportError, AttributeError):
        # AttributeError: torch installed without XPU support (no torch.xpu).
        pass

    return "cpu"


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
