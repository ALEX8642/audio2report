"""Unit tests for alignment.anchors and alignment.timeline."""
from __future__ import annotations

import pytest

from audio2report.alignment.anchors import (
    _bigram_jaccard,
    _word_bigrams,
    collect_alignment_anchors,
    robust_median_offset,
)
from audio2report.alignment.timeline import apply_offset_to_channel
from tests.helpers import make_segment


# ---------------------------------------------------------------------------
# Bigram helpers
# ---------------------------------------------------------------------------

class TestWordBigrams:
    def test_normal_text(self):
        bg = _word_bigrams("the quick brown fox")
        assert ("the", "quick") in bg
        assert ("quick", "brown") in bg
        assert ("brown", "fox") in bg
        assert len(bg) == 3

    def test_single_word_returns_empty(self):
        assert _word_bigrams("hello") == frozenset()

    def test_empty_string_returns_empty(self):
        assert _word_bigrams("") == frozenset()

    def test_normalisation_applied(self):
        # punctuation stripped, lowercased
        assert _word_bigrams("Hello, World!") == _word_bigrams("hello world")


class TestBigramJaccard:
    def test_identical_texts_return_one(self):
        bg = _word_bigrams("the quick brown fox jumps")
        assert _bigram_jaccard(bg, bg) == pytest.approx(1.0)

    def test_completely_different_texts_return_zero(self):
        bg_a = _word_bigrams("alpha beta gamma delta")
        bg_b = _word_bigrams("one two three four")
        assert _bigram_jaccard(bg_a, bg_b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        bg_a = _word_bigrams("the quick brown fox")
        bg_b = _word_bigrams("the quick red fox")
        # shared: ("the","quick"), ("quick","...") differs, ("...","fox") differs
        score = _bigram_jaccard(bg_a, bg_b)
        assert 0.0 < score < 1.0

    def test_both_empty_returns_zero(self):
        assert _bigram_jaccard(frozenset(), frozenset()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# collect_alignment_anchors
# ---------------------------------------------------------------------------

class TestCollectAlignmentAnchors:
    def test_finds_shared_utterance(self, channel_a_segments, channel_b_segments):
        anchors = collect_alignment_anchors(
            channel_a_segments, channel_b_segments,
            min_text_len=20, sim_threshold=0.90,
        )
        assert len(anchors) == 1
        assert anchors[0].a_uid == "a2"
        assert anchors[0].b_uid == "b2"

    def test_computes_correct_delta(self, channel_a_segments, channel_b_segments):
        anchors = collect_alignment_anchors(
            channel_a_segments, channel_b_segments,
            min_text_len=20, sim_threshold=0.90,
        )
        # b_start=20.3, a_start=20.0  →  delta = +0.3
        assert anchors[0].delta_sec == pytest.approx(0.3, abs=1e-6)

    def test_respects_sim_threshold(self, channel_a_segments, channel_b_segments):
        # threshold of 1.0 means only perfect matches
        anchors = collect_alignment_anchors(
            channel_a_segments, channel_b_segments,
            min_text_len=20, sim_threshold=1.0,
        )
        # Texts are identical so this should still match
        assert len(anchors) == 1

    def test_filters_short_texts(self, channel_a_segments, channel_b_segments):
        # min_text_len larger than the shared text → no anchors
        anchors = collect_alignment_anchors(
            channel_a_segments, channel_b_segments,
            min_text_len=9999, sim_threshold=0.90,
        )
        assert len(anchors) == 0

    def test_no_anchor_when_no_shared_utterance(self):
        a = [make_segment("a1", "alice", "A", "alice talks about engineering topics here", 0.0, 3.0)]
        b = [make_segment("b1", "bob",   "B", "bob discusses completely different themes", 0.0, 3.0)]
        anchors = collect_alignment_anchors(a, b, min_text_len=10, sim_threshold=0.90)
        assert len(anchors) == 0

    def test_empty_channels_return_empty(self):
        assert collect_alignment_anchors([], [], min_text_len=10, sim_threshold=0.90) == []

    def test_bigram_prefilter_does_not_suppress_valid_match(self, shared_text):
        # Two identical long texts should pass the pre-filter and return an anchor
        a = [make_segment("a1", "alice", "A", shared_text, 10.0, 14.0)]
        b = [make_segment("b1", "bob",   "B", shared_text, 10.5, 14.5)]
        anchors = collect_alignment_anchors(a, b, min_text_len=10, sim_threshold=0.90)
        assert len(anchors) == 1


# ---------------------------------------------------------------------------
# robust_median_offset
# ---------------------------------------------------------------------------

class TestRobustMedianOffset:
    def _anchor(self, delta: float):
        return type("A", (), {"delta_sec": delta})()

    def test_single_anchor(self):
        anchors = [self._anchor(3.5)]
        assert robust_median_offset(anchors) == pytest.approx(3.5)

    def test_median_of_three(self):
        anchors = [self._anchor(1.0), self._anchor(2.0), self._anchor(3.0)]
        assert robust_median_offset(anchors) == pytest.approx(2.0)

    def test_outlier_pruned(self):
        # 4 consistent deltas around 2.0, 1 extreme outlier
        anchors = [
            self._anchor(2.0), self._anchor(2.1), self._anchor(1.9),
            self._anchor(2.05), self._anchor(100.0),
        ]
        offset = robust_median_offset(anchors)
        assert offset == pytest.approx(2.0, abs=0.2), \
            f"Outlier should be pruned; got {offset}"

    def test_returns_none_on_empty(self):
        assert robust_median_offset([]) is None

    def test_zero_mad_uses_all_deltas(self):
        # All identical → MAD=0 → no filtering → same value returned
        anchors = [self._anchor(5.0)] * 5
        assert robust_median_offset(anchors) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# apply_offset_to_channel
# ---------------------------------------------------------------------------

class TestApplyOffset:
    def test_shifts_global_timestamps(self):
        segs = [
            make_segment("s1", "alice", "A", "text", 10.0, 13.0),
            make_segment("s2", "alice", "A", "text", 20.0, 23.0),
        ]
        apply_offset_to_channel(segs, offset_sec=3.0)
        assert segs[0].global_start_sec == pytest.approx(13.0)
        assert segs[0].global_end_sec == pytest.approx(16.0)
        assert segs[1].global_start_sec == pytest.approx(23.0)

    def test_zero_offset_leaves_timestamps_unchanged(self):
        segs = [make_segment("s1", "alice", "A", "text", 5.0, 8.0)]
        apply_offset_to_channel(segs, offset_sec=0.0)
        assert segs[0].global_start_sec == pytest.approx(5.0)

    def test_negative_offset(self):
        segs = [make_segment("s1", "alice", "A", "text", 10.0, 13.0)]
        apply_offset_to_channel(segs, offset_sec=-2.0)
        assert segs[0].global_start_sec == pytest.approx(8.0)
        assert segs[0].global_end_sec == pytest.approx(11.0)

    def test_does_not_modify_root_timeline_timestamps(self):
        segs = [make_segment("s1", "alice", "A", "text", 10.0, 13.0)]
        apply_offset_to_channel(segs, offset_sec=5.0)
        # root_timeline timestamps must be untouched
        assert segs[0].root_timeline_start_sec == pytest.approx(10.0)
        assert segs[0].root_timeline_end_sec == pytest.approx(13.0)
