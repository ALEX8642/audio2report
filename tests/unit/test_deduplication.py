"""Unit tests for deduplication.matching and deduplication.scoring."""
from __future__ import annotations

import pytest

from audio2report.deduplication.matching import match_segments_across_channels
from audio2report.deduplication.scoring import (
    choose_primary_from_pair,
    finalize_speaker_from_scores,
    retention_score,
    speaker_score_map,
)
from tests.helpers import make_segment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LONG_TEXT = "this utterance is long enough to meet the minimum text length requirement today"


# ---------------------------------------------------------------------------
# match_segments_across_channels
# ---------------------------------------------------------------------------

class TestMatchSegmentsAcrossChannels:
    def test_finds_duplicate_pair(self, channel_a_segments, channel_b_segments):
        matches = match_segments_across_channels(
            channel_a_segments, channel_b_segments,
            time_tolerance_sec=2.5, sim_threshold=0.86, min_text_len=20,
        )
        assert len(matches) == 1
        assert matches[0].a_uid == "a2"
        assert matches[0].b_uid == "b2"

    def test_rejects_pair_outside_time_window(self):
        a = [make_segment("a1", "alice", "A", LONG_TEXT, 0.0, 4.0)]
        b = [make_segment("b1", "bob",   "B", LONG_TEXT, 100.0, 104.0)]
        matches = match_segments_across_channels(
            a, b, time_tolerance_sec=2.5, sim_threshold=0.86, min_text_len=10,
        )
        assert len(matches) == 0

    def test_rejects_pair_below_similarity_threshold(self):
        a = [make_segment("a1", "alice", "A", "alice discusses engineering topics here at work", 0.0, 4.0)]
        b = [make_segment("b1", "bob",   "B", "bob explains completely different financial matters", 0.2, 4.2)]
        matches = match_segments_across_channels(
            a, b, time_tolerance_sec=2.5, sim_threshold=0.86, min_text_len=10,
        )
        assert len(matches) == 0

    def test_respects_min_text_length(self):
        # Text is below min_text_len → filtered before comparison
        a = [make_segment("a1", "alice", "A", "ok", 0.0, 1.0)]
        b = [make_segment("b1", "bob",   "B", "ok", 0.0, 1.0)]
        matches = match_segments_across_channels(
            a, b, time_tolerance_sec=2.5, sim_threshold=0.50, min_text_len=18,
        )
        assert len(matches) == 0

    def test_no_double_claiming_of_b_segment(self):
        """Two A-segments compete for the same B-segment; only the better match wins."""
        b_seg = make_segment("b1", "bob", "B", LONG_TEXT, 10.0, 14.0)
        a1 = make_segment("a1", "alice", "A", LONG_TEXT, 10.0, 14.0)          # perfect match
        a2 = make_segment("a2", "alice", "A", LONG_TEXT + " extra words here", 10.1, 14.1)  # also close
        matches = match_segments_across_channels(
            [a1, a2], [b_seg],
            time_tolerance_sec=2.5, sim_threshold=0.86, min_text_len=10,
        )
        b_uids_claimed = [m.b_uid for m in matches]
        assert b_uids_claimed.count("b1") <= 1, "B segment claimed more than once"

    def test_empty_channels(self):
        assert match_segments_across_channels([], [], time_tolerance_sec=2.5) == []

    def test_returns_correct_similarity_and_time_diff(self):
        a = [make_segment("a1", "alice", "A", LONG_TEXT, 10.0, 14.0)]
        b = [make_segment("b1", "bob",   "B", LONG_TEXT, 10.4, 14.4)]
        matches = match_segments_across_channels(
            a, b, time_tolerance_sec=2.5, sim_threshold=0.86, min_text_len=10,
        )
        assert len(matches) == 1
        assert matches[0].sim == pytest.approx(1.0, abs=0.01)
        assert matches[0].time_diff_sec == pytest.approx(0.4, abs=0.01)


# ---------------------------------------------------------------------------
# retention_score
# ---------------------------------------------------------------------------

class TestRetentionScore:
    def test_louder_segment_scores_higher(self):
        loud = make_segment("s1", "a", "A", "some text here", 0.0, 3.0, rms_dbfs=-10.0)
        quiet = make_segment("s2", "a", "A", "some text here", 0.0, 3.0, rms_dbfs=-30.0)
        assert retention_score(loud) > retention_score(quiet)

    def test_higher_logprob_scores_higher(self):
        good = make_segment("s1", "a", "A", "text", 0.0, 3.0, avg_logprob=-0.1)
        bad  = make_segment("s2", "a", "A", "text", 0.0, 3.0, avg_logprob=-1.5)
        assert retention_score(good) > retention_score(bad)

    def test_lower_no_speech_prob_scores_higher(self):
        clear  = make_segment("s1", "a", "A", "text", 0.0, 3.0, no_speech_prob=0.01)
        noisy  = make_segment("s2", "a", "A", "text", 0.0, 3.0, no_speech_prob=0.90)
        assert retention_score(clear) > retention_score(noisy)

    def test_none_fields_do_not_crash(self):
        seg = make_segment("s1", "a", "A", "text", 0.0, 3.0)
        seg.rms_dbfs = None
        seg.avg_logprob = None
        seg.no_speech_prob = None
        score = retention_score(seg)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# choose_primary_from_pair
# ---------------------------------------------------------------------------

class TestChoosePrimaryFromPair:
    def test_higher_retention_score_wins(self):
        good = make_segment("g", "alice", "A", "text here now", 0.0, 3.0,
                            rms_dbfs=-10.0, avg_logprob=-0.1)
        poor = make_segment("p", "alice", "A", "text here now", 0.0, 3.0,
                            rms_dbfs=-30.0, avg_logprob=-1.5)
        keep, drop, reason, margin = choose_primary_from_pair(good, poor)
        assert keep.uid == "g"
        assert drop.uid == "p"
        assert reason == "higher_retention_score"
        assert margin >= 0.0

    def test_retention_scores_written_back(self):
        sa = make_segment("a", "alice", "A", "text", 0.0, 3.0)
        sb = make_segment("b", "bob",   "B", "text", 0.0, 3.0)
        choose_primary_from_pair(sa, sb)
        assert sa.retention_score_value is not None
        assert sb.retention_score_value is not None


# ---------------------------------------------------------------------------
# speaker_score_map
# ---------------------------------------------------------------------------

class TestSpeakerScoreMap:
    def test_prime_on_this_mic_promotes_channel_prime(self):
        seg = make_segment("s1", "alice", "A", "text", 0.0, 3.0,
                           diar_speaker_role="PRIME_ON_THIS_MIC")
        scores = speaker_score_map(seg, peer_match=None)
        assert scores["alice"] > scores.get("OTHER_OR_THIRD_SPEAKER", 0.0)

    def test_other_on_this_mic_promotes_other(self):
        seg = make_segment("s1", "alice", "A", "text", 0.0, 3.0,
                           diar_speaker_role="OTHER_ON_THIS_MIC")
        scores = speaker_score_map(seg, peer_match=None)
        assert scores.get("OTHER_OR_THIRD_SPEAKER", 0.0) > 0.0

    def test_louder_on_this_mic_promotes_prime(self):
        seg  = make_segment("s1", "alice", "A", "text", 0.0, 3.0, rms_dbfs=-10.0)
        peer = make_segment("s2", "bob",   "B", "text", 0.0, 3.0, rms_dbfs=-20.0)
        scores = speaker_score_map(seg, peer_match=peer)
        # 10 dB louder on alice's mic → prime alice boosted
        assert scores["alice"] > 0.0

    def test_third_speaker_signal_from_dual_other(self):
        seg  = make_segment("s1", "alice", "A", "text", 0.0, 3.0,
                            diar_speaker_role="OTHER_ON_THIS_MIC")
        peer = make_segment("s2", "bob",   "B", "text", 0.0, 3.0,
                            diar_speaker_role="OTHER_ON_THIS_MIC")
        scores = speaker_score_map(seg, peer_match=peer)
        assert scores.get("THIRD_SPEAKER", 0.0) > 0.0


# ---------------------------------------------------------------------------
# finalize_speaker_from_scores
# ---------------------------------------------------------------------------

class TestFinalizeSpeaker:
    def test_clear_winner(self):
        label, conf, flags, basis = finalize_speaker_from_scores(
            {"alice": 2.0, "THIRD_SPEAKER": 0.0, "OTHER_OR_THIRD_SPEAKER": 0.0}
        )
        assert label == "alice"
        assert conf > 0.5
        assert "speaker_uncertain" not in flags

    def test_small_margin_returns_unknown(self):
        label, conf, flags, _ = finalize_speaker_from_scores(
            {"alice": 1.0, "THIRD_SPEAKER": 0.8, "OTHER_OR_THIRD_SPEAKER": 0.0}
        )
        assert label == "UNKNOWN"
        assert "speaker_uncertain" in flags
        assert conf == pytest.approx(0.3)

    def test_weak_third_speaker_collapses(self):
        label, _, flags, _ = finalize_speaker_from_scores(
            {"alice": 0.0, "THIRD_SPEAKER": 1.0, "OTHER_OR_THIRD_SPEAKER": 0.0}
        )
        assert label == "OTHER_OR_THIRD_SPEAKER"
        assert "third_speaker_evidence_weak" in flags

    def test_strong_third_speaker_preserved(self):
        label, _, flags, _ = finalize_speaker_from_scores(
            {"alice": 0.0, "THIRD_SPEAKER": 1.5, "OTHER_OR_THIRD_SPEAKER": 0.0}
        )
        assert label == "THIRD_SPEAKER"
        assert "third_speaker_evidence_weak" not in flags
