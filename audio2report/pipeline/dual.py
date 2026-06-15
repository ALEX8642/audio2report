"""DualMicPipeline — the core orchestrator for the dual-mic workflow.

This is the primary value-add of the tool: it aligns two independent
microphone recordings, deduplicates cross-captured speech, and produces a
clean, attributed transcript from recordings that were never hardware-synced.

Execution order
---------------
1. Discover prime folders and validate (exactly 2 required)
2. Build file records and normalise audio (ffmpeg → mono 16 kHz WAV)
3. Transcribe each file — WhisperX model is loaded **once** and reused
   (explicit caching: existing per-file JSON is reused unless cache=False)
4. Assign per-file diarization roles (PRIME_ON_THIS_MIC / OTHER_ON_THIS_MIC)
5. Estimate cross-channel clock offset from transcript anchors
6. Apply offset to align global timelines
7. Detect cross-channel duplicate pairs (O(n log n) bisect sweep)
8. Choose which duplicate copy to keep (retention score)
9. Assign final speaker labels (multi-signal)
10. Sort all segments chronologically
11. Post-process for LLM (ack suppression, fragment merging)
12. Return RunResult (pipelines do not write files themselves)
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from audio2report._log import get_console, get_logger
from audio2report.alignment.anchors import collect_alignment_anchors, robust_median_offset
from audio2report.alignment.timeline import apply_offset_to_channel, result_to_segments
from audio2report.config import Config
from audio2report.deduplication.matching import match_segments_across_channels
from audio2report.deduplication.scoring import (
    assign_speaker_for_kept_segment,
    choose_primary_from_pair,
    retention_score,
)
from audio2report.diarization.roles import assign_diar_roles_per_channel
from audio2report.ingestion.discovery import (
    build_file_records_for_folder,
    find_prime_folders,
    parse_prime_from_folder,
)
from audio2report.ingestion.normalize import is_audio_too_short_or_silent
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


class DualMicPipeline:
    """
    Orchestrates the full dual-mic transcription and deduplication workflow.

    Parameters
    ----------
    config:
        Fully-resolved :class:`~audio2report.config.Config` object.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Dry-run: report what would happen without touching any audio
    # ------------------------------------------------------------------

    def dry_run(self, root: Path, out_root: Path) -> None:
        """Print a summary of what ``run()`` would do, then return."""
        from rich.table import Table

        cfg = self.config
        console = get_console()

        prime_folders = find_prime_folders(root)
        if len(prime_folders) != 2:
            console.print(
                f"[red]Expected 2 prime folders, found {len(prime_folders)}:[/red] "
                f"{[p.name for p in prime_folders]}"
            )
            return

        device = pick_device(cfg.transcription.device)

        table = Table(title="Dry run — files that would be processed", show_lines=True)
        table.add_column("Folder", style="cyan")
        table.add_column("File", style="white")
        table.add_column("Status", style="yellow")

        for folder in prime_folders:
            from audio2report.ingestion.discovery import list_audio_files

            files = list_audio_files(folder)
            out_root / "normalized_audio"
            raw_json_root = out_root / "per_file_whisperx_json"

            for i, src in enumerate(files):
                slug = safe_slug(folder.name)
                from audio2report.utils import safe_slug as _slug
                dst_stem = f"{i:04d}_{_slug(src.stem)}"
                json_path = raw_json_root / slug / f"{dst_stem}.json"
                if cfg.cache and json_path.exists():
                    status = "[green]cached[/green]"
                else:
                    status = "[yellow]will transcribe[/yellow]"
                table.add_row(folder.name, src.name, status)

        console.print(table)
        console.print(
            f"\nDevice: [bold]{device}[/bold]  "
            f"Model: [bold]{cfg.transcription.model}[/bold]  "
            f"Diarize: {cfg.diarization.enabled}  "
            f"Cache: {cfg.cache}"
        )

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self, root: Path, out_root: Path) -> RunResult:
        """
        Run the pipeline on *root* and return a :class:`~audio2report.models.RunResult`.

        The caller is responsible for writing outputs (see
        :func:`~audio2report.output.writers.write_all_outputs`).
        """
        cfg = self.config

        # Setup output directories
        normalized_root = out_root / "normalized_audio"
        raw_json_root = out_root / "per_file_whisperx_json"
        ensure_dir(out_root)
        ensure_dir(normalized_root)
        ensure_dir(raw_json_root)

        # Discover prime folders
        prime_folders = find_prime_folders(root)
        if len(prime_folders) != 2:
            raise ValueError(
                f"Expected exactly 2 folders containing 'prime' under {root}, "
                f"found {len(prime_folders)}: {[p.name for p in prime_folders]}"
            )

        device = pick_device(cfg.transcription.device)
        logger.info(f"Device: [bold]{device}[/bold]")
        logger.info(f"Prime folders: {[p.name for p in prime_folders]}")

        # Build file records and normalise audio
        folder_records: dict[str, list[AudioFileRecord]] = {}
        for folder in prime_folders:
            recs = build_file_records_for_folder(
                folder, normalized_root, cfg.audio.inter_file_gap_sec
            )
            if not recs:
                raise RuntimeError(f"No audio files found in {folder}")
            folder_records[folder.name] = recs
            logger.info(
                f"{folder.name}: {len(recs)} file(s), "
                f"prime='{parse_prime_from_folder(folder.name)}'"
            )

        # Transcribe — model loaded once, reused across all files
        transcriber = WhisperXTranscriber(cfg.transcription, device)
        hf_token = cfg.diarization.resolved_token()

        all_segments: list[SegmentRecord] = []
        total_files = sum(len(recs) for recs in folder_records.values())

        with _make_progress() as progress:
            task = progress.add_task("Transcribing", total=total_files)

            for folder_name, file_recs in folder_records.items():
                folder_json_dir = raw_json_root / safe_slug(folder_name)
                ensure_dir(folder_json_dir)

                for fr in file_recs:
                    stem = Path(fr.normalized_wav).stem
                    json_path = folder_json_dir / f"{stem}.json"
                    progress.update(task, description=f"[cyan]{folder_name}[/cyan]  {stem}")

                    # Cache hit: skip re-transcription
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
                        logger.info(
                            f"Skipping post-ASR: {stem} ({result['skipped_reason']})"
                        )
                        continue

                    segs = result_to_segments(result, fr)
                    all_segments.extend(segs)

        logger.info(
            f"Transcription complete: {len(all_segments)} segment(s) across "
            f"{total_files} file(s)"
        )

        # Diarization role normalisation
        assign_diar_roles_per_channel(all_segments)

        # Split segments by channel
        channels: dict[str, list[SegmentRecord]] = defaultdict(list)
        for seg in all_segments:
            channels[seg.channel_folder].append(seg)

        channel_names = list(channels.keys())
        if len(channel_names) != 2:
            raise RuntimeError(
                f"Expected 2 channels after transcription, got {len(channel_names)}: "
                f"{channel_names}"
            )

        ch_a, ch_b = channel_names[0], channel_names[1]
        a_segments = sorted(channels[ch_a], key=lambda x: x.root_timeline_start_sec)
        b_segments = sorted(channels[ch_b], key=lambda x: x.root_timeline_start_sec)

        # Cross-channel alignment
        logger.info("Estimating cross-channel clock offset from transcript anchors…")
        anchors = collect_alignment_anchors(
            a_segments,
            b_segments,
            min_text_len=cfg.alignment.min_anchor_text_len,
            sim_threshold=cfg.alignment.anchor_sim_threshold,
        )
        offset_b_minus_a = robust_median_offset(anchors)

        if offset_b_minus_a is None:
            logger.warning("No transcript anchors found — falling back to zero offset.")
            offset_b_minus_a = 0.0

        logger.info(
            f"Alignment: [bold]{len(anchors)}[/bold] anchor(s)  "
            f"offset = [bold]{offset_b_minus_a:+.3f} s[/bold] "
            f"({ch_b} relative to {ch_a})"
        )

        apply_offset_to_channel(a_segments, 0.0)
        apply_offset_to_channel(b_segments, -offset_b_minus_a)

        # Cross-channel duplicate detection and suppression (O(n log n))
        if cfg.deduplication.enabled:
            logger.info("Running cross-channel deduplication…")
            pair_matches = match_segments_across_channels(
                a_segments,
                b_segments,
                time_tolerance_sec=cfg.deduplication.time_tolerance_sec,
                sim_threshold=cfg.deduplication.sim_threshold,
                min_text_len=cfg.deduplication.min_text_len,
            )
        else:
            pair_matches = []

        seg_by_uid = {s.uid: s for s in all_segments}
        peer_map: dict[str, str] = {}

        for pm in pair_matches:
            sa = seg_by_uid[pm.a_uid]
            sb = seg_by_uid[pm.b_uid]
            keep, drop, reason, margin = choose_primary_from_pair(sa, sb)

            drop.keep = False
            drop.duplicate_of = keep.uid
            drop.flags.extend([
                "cross_mic_duplicate_suppressed",
                reason,
                f"margin={margin:.3f}",
                (f"drop_retention={drop.retention_score_value:.3f}"
                 if drop.retention_score_value is not None else "drop_retention=na"),
                (f"keep_retention={keep.retention_score_value:.3f}"
                 if keep.retention_score_value is not None else "keep_retention=na"),
            ])
            keep.flags.append("has_cross_mic_match")

            peer_map[sa.uid] = sb.uid
            peer_map[sb.uid] = sa.uid

        kept_count = sum(s.keep for s in all_segments)
        logger.info(
            f"Deduplication: [bold]{len(all_segments)}[/bold] total → "
            f"[bold]{kept_count}[/bold] kept  "
            f"([bold]{len(pair_matches)}[/bold] cross-mic pair(s) suppressed)"
        )

        # Final speaker attribution
        for seg in all_segments:
            if not seg.keep:
                continue
            peer = seg_by_uid.get(peer_map.get(seg.uid, ""))
            spk, conf, flags, score_detail, basis = assign_speaker_for_kept_segment(seg, peer)
            seg.speaker_final = spk
            seg.speaker_confidence = conf
            seg.speaker_score_detail = score_detail
            seg.speaker_decision_basis = basis
            seg.flags.extend(flags)
            if seg.retention_score_value is None:
                seg.retention_score_value = retention_score(seg)

        # Sort chronologically
        all_segments_sorted = sorted(
            all_segments,
            key=lambda x: (x.global_start_sec, x.global_end_sec, x.uid),
        )
        kept = [s for s in all_segments_sorted if s.keep]

        # Post-process for LLM
        cleaned_segments = postprocess_segments_for_llm(all_segments_sorted)

        # Assemble metadata
        meta = RunMeta(
            root=str(root),
            prime_folders=[p.name for p in prime_folders],
            primes=[parse_prime_from_folder(p.name) for p in prime_folders],
            device=device,
            model=cfg.transcription.model,
            language=cfg.transcription.language,
            diarize=cfg.diarization.enabled,
            estimated_offset_b_minus_a_sec=offset_b_minus_a,
            anchor_count=len(anchors),
            pair_match_count=len(pair_matches),
            total_segments=len(all_segments_sorted),
            kept_segments=len(kept),
            suppressed_segments=len(all_segments_sorted) - len(kept),
        )

        return RunResult(
            segments=all_segments_sorted,
            cleaned_segments=cleaned_segments,
            anchors=anchors,
            pair_matches=pair_matches,
            meta=meta,
        )
