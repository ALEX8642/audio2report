"""Integration test for DualMicPipeline.

Strategy
--------
WhisperX and ffmpeg are expensive external dependencies.  This test avoids both:

- ffmpeg / ffprobe are monkeypatched:
    ``ffmpeg_normalize_to_wav`` → copies the source WAV unchanged
    ``ffprobe_duration_seconds`` → returns a fixed 10.0 s duration

- WhisperX transcription is bypassed entirely via the cache mechanism:
  Before the pipeline runs, per-file JSON files (simulating WhisperX output)
  are pre-written to the expected cache location.  With ``cache=True`` the
  pipeline reads those files instead of calling the model.

What is actually exercised end-to-end (no mocking):
    - Folder discovery (find_prime_folders, parse_prime_from_folder)
    - File record construction and timeline building
    - result_to_segments conversion
    - Diarization role assignment
    - Anchor detection and offset estimation
    - Bisect-based cross-channel deduplication
    - Retention + speaker scoring
    - Postprocessing
    - All output writers (JSON, CSV, TXT)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio2report.config import Config
from audio2report.ingestion.discovery import build_file_records_for_folder
from audio2report.output.writers import write_all_outputs
from audio2report.pipeline.dual import DualMicPipeline
from audio2report.utils import dump_json, ensure_dir, safe_slug

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHARED_TEXT = (
    "this utterance was captured by both microphones at the same time today "
    "and is long enough to serve as an alignment anchor for the pipeline"
)
ALICE_UNIQUE_1 = "alice describes the quarterly financial results in great detail here"
ALICE_UNIQUE_2 = "alice summarises the action items from today's discussion clearly"
BOB_UNIQUE_1   = "bob explains the technical architecture of the new system now"
BOB_UNIQUE_2   = "bob outlines the risk mitigation strategy for the project team"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_wav(path: Path, duration_sec: float = 1.0, sr: int = 16000) -> None:
    """Write a minimal valid WAV file (silence at 16 kHz mono)."""
    samples = np.zeros(int(sr * duration_sec), dtype=np.float32)
    sf.write(str(path), samples, sr, subtype="PCM_16")


def _whisperx_result(segments: list) -> dict:
    return {
        "segments": segments,
        "language": "en",
        "diarization_segments": [],
        "skipped_reason": None,
    }


def _seg(start: float, end: float, text: str, speaker: str = "SPEAKER_00") -> dict:
    return {
        "start": start, "end": end, "text": text,
        "avg_logprob": -0.2, "no_speech_prob": 0.03,
        "speaker": speaker,
    }


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def dual_workspace(tmp_path, monkeypatch):
    """
    Build a complete dual-mic workspace with synthetic WAVs and pre-cached
    WhisperX JSON files so no external dependencies are needed.

    Layout:
        tmp_path/
        ├── TX_MIC - alice prime/
        │   ├── file01.wav
        │   └── file02.wav
        ├── TX_MIC - bob prime/
        │   ├── file01.wav
        │   └── file02.wav
        └── output/
            ├── normalized_audio/    (created by pipeline)
            └── per_file_whisperx_json/
                ├── tx_mic_-_alice_prime/
                │   ├── 0000_file01.json   (pre-written)
                │   └── 0001_file02.json   (pre-written)
                └── tx_mic_-_bob_prime/
                    ├── 0000_file01.json   (pre-written)
                    └── 0001_file02.json   (pre-written)
    """
    # ------------------------------------------------------------------ dirs
    alice_dir = tmp_path / "TX_MIC - alice prime"
    bob_dir   = tmp_path / "TX_MIC - bob prime"
    out_dir   = tmp_path / "output"
    alice_dir.mkdir()
    bob_dir.mkdir()

    # ------------------------------------------------------------------ WAVs
    for folder in [alice_dir, bob_dir]:
        for name in ["file01.wav", "file02.wav"]:
            _write_wav(folder / name)

    # ------------------------------------------------------------------ mock ffmpeg / ffprobe
    def _mock_normalize(src: Path, dst: Path) -> None:
        import shutil
        shutil.copy(str(src), str(dst))

    def _mock_duration(_path) -> float:
        return 10.0

    monkeypatch.setattr(
        "audio2report.ingestion.normalize.ffmpeg_normalize_to_wav", _mock_normalize
    )
    monkeypatch.setattr(
        "audio2report.ingestion.normalize.ffprobe_duration_seconds", _mock_duration
    )

    # ------------------------------------------------------------------ build file records
    # Run ingestion to discover the exact normalized WAV paths, then use
    # those paths to determine the correct cache JSON locations.
    cfg = Config()
    normalized_root = out_dir / "normalized_audio"
    raw_json_root   = out_dir / "per_file_whisperx_json"
    ensure_dir(normalized_root)
    ensure_dir(raw_json_root)

    alice_recs = build_file_records_for_folder(alice_dir, normalized_root, 0.5)
    bob_recs   = build_file_records_for_folder(bob_dir,   normalized_root, 0.5)

    # ------------------------------------------------------------------ pre-write JSON cache
    #
    # Transcript design:
    #   Alice file01: unique utterance A1 → then SHARED at t=4
    #   Alice file02: unique utterance A2
    #   Bob   file01: SHARED at t=4.3 (0.3 s offset) → then unique B1
    #   Bob   file02: unique utterance B2
    #
    # Expected pipeline behaviour:
    #   1 anchor found (SHARED), offset ≈ +0.3 s
    #   1 dedup pair suppressed
    #   total kept = 5 (A1, A2, B1, B2, SHARED×1)

    cache_data = {
        alice_dir.name: {
            0: _whisperx_result([
                _seg(1.0, 4.0, ALICE_UNIQUE_1, "SPEAKER_00"),
                _seg(5.0, 9.0, SHARED_TEXT,    "SPEAKER_00"),
            ]),
            1: _whisperx_result([
                _seg(1.0, 4.0, ALICE_UNIQUE_2, "SPEAKER_00"),
            ]),
        },
        bob_dir.name: {
            0: _whisperx_result([
                _seg(5.3, 9.3, SHARED_TEXT, "SPEAKER_00"),
                _seg(10.0, 13.0, BOB_UNIQUE_1, "SPEAKER_00"),
            ]),
            1: _whisperx_result([
                _seg(1.0, 4.0, BOB_UNIQUE_2, "SPEAKER_00"),
            ]),
        },
    }

    for folder_name, recs in [
        (alice_dir.name, alice_recs),
        (bob_dir.name, bob_recs),
    ]:
        json_dir = raw_json_root / safe_slug(folder_name)
        ensure_dir(json_dir)
        for fr in recs:
            stem = Path(fr.normalized_wav).stem
            dump_json(
                json_dir / f"{stem}.json",
                cache_data[folder_name][fr.order_index],
            )

    return {"root": tmp_path, "out": out_dir, "cfg": cfg}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDualMicPipelineIntegration:
    def test_pipeline_runs_without_error(self, dual_workspace):
        cfg  = dual_workspace["cfg"]
        root = dual_workspace["root"]
        out  = dual_workspace["out"]
        pipeline = DualMicPipeline(cfg)
        result = pipeline.run(root, out)
        assert result is not None

    def test_finds_two_channels(self, dual_workspace):
        cfg = dual_workspace["cfg"]
        result = DualMicPipeline(cfg).run(dual_workspace["root"], dual_workspace["out"])
        assert len(result.meta.prime_folders) == 2

    def test_primes_parsed_correctly(self, dual_workspace):
        cfg = dual_workspace["cfg"]
        result = DualMicPipeline(cfg).run(dual_workspace["root"], dual_workspace["out"])
        primes = set(result.meta.primes)
        assert "alice" in primes
        assert "bob" in primes

    def test_alignment_anchor_found(self, dual_workspace):
        cfg = dual_workspace["cfg"]
        result = DualMicPipeline(cfg).run(dual_workspace["root"], dual_workspace["out"])
        assert result.meta.anchor_count >= 1, "Expected at least one alignment anchor"

    def test_offset_estimated_correctly(self, dual_workspace):
        cfg = dual_workspace["cfg"]
        result = DualMicPipeline(cfg).run(dual_workspace["root"], dual_workspace["out"])
        # Bob's file starts 0.3 s after alice's (5.3 vs 5.0)
        # So offset_b_minus_a ≈ 0.3 s
        offset = result.meta.estimated_offset_b_minus_a_sec
        assert abs(offset) < 2.0, f"Offset {offset:.3f} s seems unreasonably large"

    def test_cross_mic_duplicate_suppressed(self, dual_workspace):
        cfg = dual_workspace["cfg"]
        result = DualMicPipeline(cfg).run(dual_workspace["root"], dual_workspace["out"])
        assert result.meta.pair_match_count >= 1, "Expected at least one duplicate pair"
        assert result.meta.suppressed_segments >= 1

    def test_total_plus_suppressed_equals_total(self, dual_workspace):
        cfg = dual_workspace["cfg"]
        result = DualMicPipeline(cfg).run(dual_workspace["root"], dual_workspace["out"])
        m = result.meta
        assert m.kept_segments + m.suppressed_segments == m.total_segments

    def test_all_kept_segments_sorted(self, dual_workspace):
        cfg = dual_workspace["cfg"]
        result = DualMicPipeline(cfg).run(dual_workspace["root"], dual_workspace["out"])
        kept = [s for s in result.segments if s.keep]
        times = [s.global_start_sec for s in kept]
        assert times == sorted(times)

    def test_speaker_labels_assigned(self, dual_workspace):
        cfg = dual_workspace["cfg"]
        result = DualMicPipeline(cfg).run(dual_workspace["root"], dual_workspace["out"])
        kept = [s for s in result.segments if s.keep]
        assert all(s.speaker_final is not None for s in kept)

    def test_cleaned_segments_non_empty(self, dual_workspace):
        cfg = dual_workspace["cfg"]
        result = DualMicPipeline(cfg).run(dual_workspace["root"], dual_workspace["out"])
        assert len(result.cleaned_segments) > 0


class TestDualMicPipelineOutputFiles:
    @pytest.fixture(autouse=True)
    def _run(self, dual_workspace):
        cfg = dual_workspace["cfg"]
        self.result = DualMicPipeline(cfg).run(
            dual_workspace["root"], dual_workspace["out"]
        )
        write_all_outputs(self.result, dual_workspace["out"], cfg.output)
        self.out = dual_workspace["out"]

    def test_canonical_json_written(self):
        assert (self.out / "canonical_transcript.json").exists()

    def test_canonical_csv_written(self):
        assert (self.out / "canonical_transcript.csv").exists()

    def test_canonical_txt_written(self):
        assert (self.out / "canonical_transcript.txt").exists()

    def test_cleaned_llm_input_written(self):
        assert (self.out / "cleaned_llm_input.txt").exists()

    def test_run_meta_written(self):
        assert (self.out / "run_meta.json").exists()

    def test_alignment_anchors_written(self):
        assert (self.out / "alignment_anchors.json").exists()

    def test_pair_matches_written(self):
        assert (self.out / "pair_matches.json").exists()

    def test_json_contains_segments_key(self):
        data = json.loads((self.out / "canonical_transcript.json").read_text())
        assert "segments" in data
        assert "meta" in data

    def test_txt_contains_speaker_labels(self):
        content = (self.out / "canonical_transcript.txt").read_text()
        # Every kept segment line should have a speaker label
        lines = [l for l in content.splitlines() if l.strip()]
        assert len(lines) > 0
        for line in lines:
            assert ":" in line, f"Line missing speaker label: {line!r}"

    def test_csv_has_header_row(self):
        import csv
        with (self.out / "canonical_transcript.csv").open() as f:
            reader = csv.DictReader(f)
            assert "uid" in reader.fieldnames
            assert "text" in reader.fieldnames
            assert "speaker_final" in reader.fieldnames

    def test_run_meta_json_correct_structure(self):
        data = json.loads((self.out / "run_meta.json").read_text())
        assert data["anchor_count"] >= 1
        assert data["pair_match_count"] >= 1
        assert data["total_segments"] > 0
        assert data["kept_segments"] + data["suppressed_segments"] == data["total_segments"]


class TestDualMicPipelineCaching:
    def test_cache_hit_skips_transcription(self, dual_workspace, monkeypatch):
        """Second run with cache=True should not call transcribe()."""
        cfg = dual_workspace["cfg"]
        root, out = dual_workspace["root"], dual_workspace["out"]

        # First run populates cache
        DualMicPipeline(cfg).run(root, out)

        # Patch transcriber.transcribe to detect if it's called
        transcribe_calls = []

        original_transcribe = (
            __import__("audio2report.transcription.whisperx_backend", fromlist=["WhisperXTranscriber"])
            .WhisperXTranscriber.transcribe
        )

        def _tracking_transcribe(self_inner, *args, **kwargs):
            transcribe_calls.append(1)
            return original_transcribe(self_inner, *args, **kwargs)

        monkeypatch.setattr(
            "audio2report.transcription.whisperx_backend.WhisperXTranscriber.transcribe",
            _tracking_transcribe,
        )

        # Second run — all JSONs exist → cache should be hit for all files
        DualMicPipeline(cfg).run(root, out)
        assert len(transcribe_calls) == 0, \
            f"transcribe() called {len(transcribe_calls)} time(s) despite cache=True"


class TestDualMicPipelineErrors:
    def test_wrong_number_of_prime_folders(self, tmp_path, monkeypatch):
        # Only one prime folder → should raise ValueError
        (tmp_path / "TX_MIC - alice prime").mkdir()
        _write_wav(tmp_path / "TX_MIC - alice prime" / "audio.wav")

        monkeypatch.setattr(
            "audio2report.ingestion.normalize.ffmpeg_normalize_to_wav",
            lambda src, dst: None,
        )
        monkeypatch.setattr(
            "audio2report.ingestion.normalize.ffprobe_duration_seconds",
            lambda _: 10.0,
        )

        cfg = Config()
        with pytest.raises(ValueError, match="Expected exactly 2"):
            DualMicPipeline(cfg).run(tmp_path, tmp_path / "out")
