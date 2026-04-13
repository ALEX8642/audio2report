"""SingleMicPipeline — single audio file or single folder to transcript.

No alignment or cross-channel deduplication is performed.  The pipeline is:
    normalise → transcribe → diar roles → speaker attribution → postprocess
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from audio2report._log import get_console, get_logger
from audio2report.alignment.timeline import result_to_segments
from audio2report.config import Config
from audio2report.deduplication.scoring import assign_speaker_for_kept_segment, retention_score
from audio2report.diarization.roles import assign_diar_roles_per_channel
from audio2report.ingestion.discovery import build_file_records_for_folder, parse_prime_from_folder
from audio2report.ingestion.normalize import (
    ffmpeg_normalize_to_wav,
    ffprobe_duration_seconds,
    is_audio_too_short_or_silent,
)
from audio2report.models import AudioFileRecord, RunMeta, RunResult, SegmentRecord
from audio2report.postprocessing.cleanup import postprocess_segments_for_llm
from audio2report.transcription.whisperx_backend import WhisperXTranscriber
from audio2report.utils import dump_json, ensure_dir, load_json, pick_device, safe_slug

logger = get_logger(__name__)


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=get_console(),
        transient=True,
    )


class SingleMicPipeline:
    """
    Transcribes a single audio file or a folder of sequential audio files.

    Parameters
    ----------
    config:
        Fully-resolved :class:`~audio2report.config.Config` object.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self, input_path: Path, out_root: Path, prime_name: Optional[str] = None) -> RunResult:
        """
        Parameters
        ----------
        input_path:
            Either a single audio file or a folder containing audio files.
        out_root:
            Destination directory for all output files.
        prime_name:
            Speaker name to use in attribution labels.  Defaults to the folder
            name (or file stem for single-file inputs).
        """
        cfg = self.config

        normalized_root = out_root / "normalized_audio"
        raw_json_root = out_root / "per_file_whisperx_json"
        ensure_dir(out_root)
        ensure_dir(normalized_root)
        ensure_dir(raw_json_root)

        device = pick_device(cfg.transcription.device)
        logger.info(f"Device: [bold]{device}[/bold]")

        # Build file records
        if input_path.is_dir():
            file_recs = build_file_records_for_folder(
                input_path, normalized_root, cfg.audio.inter_file_gap_sec
            )
            resolved_prime = prime_name or parse_prime_from_folder(input_path.name)
        else:
            # Single file — normalise directly
            resolved_prime = prime_name or input_path.stem
            folder_out = normalized_root / safe_slug(resolved_prime)
            ensure_dir(folder_out)
            dst = folder_out / f"0000_{safe_slug(input_path.stem)}.wav"
            ffmpeg_normalize_to_wav(input_path, dst)
            dur = ffprobe_duration_seconds(dst)
            file_recs = [AudioFileRecord(
                path=str(input_path),
                normalized_wav=str(dst),
                folder_prime=resolved_prime,
                folder_name=resolved_prime,
                order_index=0,
                duration_sec=dur,
                local_file_start_sec=0.0,
                local_file_end_sec=dur,
            )]

        if not file_recs:
            raise RuntimeError(f"No audio files found in {input_path}")

        transcriber = WhisperXTranscriber(cfg.transcription, device)
        hf_token = cfg.diarization.resolved_token()
        all_segments: List[SegmentRecord] = []
        json_dir = raw_json_root / safe_slug(resolved_prime)
        ensure_dir(json_dir)

        with _make_progress() as progress:
            task = progress.add_task("Transcribing", total=len(file_recs))

            for fr in file_recs:
                stem = Path(fr.normalized_wav).stem
                json_path = json_dir / f"{stem}.json"
                progress.update(task, description=f"[cyan]{resolved_prime}[/cyan]  {stem}")

                if cfg.cache and json_path.exists():
                    logger.info(f"Cache hit: {json_path.name}")
                    result = load_json(json_path)
                else:
                    wav_path = Path(fr.normalized_wav)
                    if is_audio_too_short_or_silent(wav_path, cfg.audio.min_duration_sec):
                        logger.info(f"Skipping short/silent file: {wav_path.name}")
                        result = {
                            "segments": [],
                            "diarization_segments": [],
                            "skipped_reason": "short_audio_precheck",
                        }
                        dump_json(json_path, result)
                        progress.advance(task)
                        continue

                    result = transcriber.transcribe(
                        wav_path,
                        diarize=cfg.diarization.enabled,
                        hf_token=hf_token,
                    )
                    dump_json(json_path, result)

                progress.advance(task)

                if result.get("skipped_reason"):
                    logger.info(f"Skipping post-ASR: {stem} ({result['skipped_reason']})")
                    continue

                segs = result_to_segments(result, fr)
                all_segments.extend(segs)

        logger.info(f"Transcription complete: {len(all_segments)} segment(s)")

        # Diarization role normalisation
        assign_diar_roles_per_channel(all_segments)

        # Speaker attribution (no peer evidence in single-mic mode)
        for seg in all_segments:
            spk, conf, flags, score_detail, basis = assign_speaker_for_kept_segment(seg, None)
            seg.speaker_final = spk
            seg.speaker_confidence = conf
            seg.speaker_score_detail = score_detail
            seg.speaker_decision_basis = basis
            seg.flags.extend(flags)
            seg.retention_score_value = retention_score(seg)

        all_segments_sorted = sorted(
            all_segments,
            key=lambda x: (x.global_start_sec, x.global_end_sec, x.uid),
        )
        cleaned_segments = postprocess_segments_for_llm(all_segments_sorted)

        meta = RunMeta(
            root=str(input_path),
            prime_folders=[resolved_prime],
            primes=[resolved_prime],
            device=device,
            model=cfg.transcription.model,
            language=cfg.transcription.language,
            diarize=cfg.diarization.enabled,
            estimated_offset_b_minus_a_sec=0.0,
            anchor_count=0,
            pair_match_count=0,
            total_segments=len(all_segments_sorted),
            kept_segments=len([s for s in all_segments_sorted if s.keep]),
            suppressed_segments=0,
        )

        return RunResult(
            segments=all_segments_sorted,
            cleaned_segments=cleaned_segments,
            anchors=[],
            pair_matches=[],
            meta=meta,
        )
