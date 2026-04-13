"""Unit tests for postprocessing.cleanup."""
from __future__ import annotations

import pytest

from audio2report.postprocessing.cleanup import (
    dedupe_internal_repetition,
    is_short_ack,
    is_very_short,
    overlap_seconds,
    postprocess_segments_for_llm,
    text_contains,
)
from tests.helpers import make_segment


# ---------------------------------------------------------------------------
# Helper predicates
# ---------------------------------------------------------------------------

class TestIsShortAck:
    @pytest.mark.parametrize("text", ["yeah", "ok", "okay", "right", "sure", "oh", "yep"])
    def test_recognised_acks(self, text):
        assert is_short_ack(text)

    @pytest.mark.parametrize("text", [
        "that is a reasonable point",
        "yes I agree with your assessment",
        "ok let me explain the details",
    ])
    def test_non_acks(self, text):
        assert not is_short_ack(text)


class TestIsVeryShort:
    def test_three_words_or_fewer(self):
        assert is_very_short("yes")
        assert is_very_short("ok sure")
        assert is_very_short("all right then")

    def test_four_words_not_short(self):
        assert not is_very_short("all right let us")


class TestOverlapSeconds:
    def test_full_overlap(self):
        assert overlap_seconds(0.0, 5.0, 0.0, 5.0) == pytest.approx(5.0)

    def test_no_overlap(self):
        assert overlap_seconds(0.0, 2.0, 3.0, 5.0) == pytest.approx(0.0)

    def test_partial_overlap(self):
        assert overlap_seconds(0.0, 3.0, 2.0, 5.0) == pytest.approx(1.0)

    def test_touching_endpoints_no_overlap(self):
        assert overlap_seconds(0.0, 2.0, 2.0, 4.0) == pytest.approx(0.0)


class TestTextContains:
    def test_a_contains_b(self):
        assert text_contains("the quick brown fox", "quick brown")

    def test_b_contains_a(self):
        assert text_contains("quick brown", "the quick brown fox")

    def test_no_containment(self):
        assert not text_contains("hello world", "completely different")


class TestDedupeInternalRepetition:
    def test_repeated_sentence_removed(self):
        text = "Alice said hello. Alice said hello."
        result = dedupe_internal_repetition(text)
        assert result.count("Alice said hello") == 1

    def test_unique_sentences_kept(self):
        # Use clearly distinct sentences (low SequenceMatcher ratio)
        text = "Alice reviewed the quarterly budget figures. Bob outlined the technical risks."
        result = dedupe_internal_repetition(text)
        assert "Alice reviewed" in result
        assert "Bob outlined" in result

    def test_single_sentence_unchanged(self):
        text = "Just one sentence."
        assert dedupe_internal_repetition(text) == text


# ---------------------------------------------------------------------------
# postprocess_segments_for_llm — Pass 1: duplicate suppression
# ---------------------------------------------------------------------------

class TestPostprocessPass1:
    def test_empty_input(self):
        assert postprocess_segments_for_llm([]) == []

    def test_suppressed_segments_excluded(self):
        segs = [
            make_segment("s1", "alice", "A", "kept segment here today", 0.0, 3.0, keep=True),
            make_segment("s2", "alice", "A", "suppressed segment here", 5.0, 8.0, keep=False),
        ]
        result = postprocess_segments_for_llm(segs)
        uids = [r.uid for r in result]
        assert "s1" in uids
        assert "s2" not in uids

    def test_ack_filtered_when_close_to_previous(self):
        segs = [
            make_segment("s1", "alice", "A", "alice explains the financial results here", 0.0, 4.0,
                         speaker_final="alice"),
            make_segment("s2", "bob",   "B", "yeah",   0.5, 1.0, speaker_final="bob"),
        ]
        result = postprocess_segments_for_llm(segs)
        uids = [r.uid for r in result]
        assert "s1" in uids
        assert "s2" not in uids, "Short ack near previous should be suppressed"

    def test_ack_kept_when_far_from_previous(self):
        segs = [
            make_segment("s1", "alice", "A", "alice says something important here today", 0.0, 4.0,
                         speaker_final="alice"),
            make_segment("s2", "bob",   "B", "yeah", 30.0, 31.0, speaker_final="bob"),
        ]
        result = postprocess_segments_for_llm(segs)
        uids = [r.uid for r in result]
        assert "s2" in uids, "Ack far away from previous should be kept"

    def test_overlapping_near_duplicate_longer_wins(self):
        short_seg = make_segment("s1", "alice", "A", "the results show", 0.0, 2.0,
                                 speaker_final="alice")
        long_seg  = make_segment("s2", "alice", "A", "the results show a significant increase today",
                                 0.5, 4.0, speaker_final="alice")
        result = postprocess_segments_for_llm([short_seg, long_seg])
        assert len(result) == 1
        assert result[0].uid == "s2", "Longer segment should be kept"

    def test_twin_capture_prefers_named_over_other(self):
        named = make_segment("s2", "alice", "A",
                             "we reviewed the documents and discussed the findings carefully",
                             1.0, 5.0, speaker_final="alice")
        other = make_segment("s1", "bob",   "B",
                             "we reviewed the documents and discussed the findings carefully",
                             0.8, 4.8, speaker_final="OTHER_OR_THIRD_SPEAKER")
        result = postprocess_segments_for_llm([other, named])
        assert len(result) == 1
        assert result[0].speaker_final == "alice"

    def test_sort_order_maintained(self):
        segs = [
            make_segment("s3", "alice", "A", "third utterance at the end of meeting", 30.0, 33.0,
                         speaker_final="alice"),
            make_segment("s1", "alice", "A", "first utterance at start of the meeting", 0.0, 3.0,
                         speaker_final="alice"),
            make_segment("s2", "bob",   "B", "second utterance in the middle section", 15.0, 18.0,
                         speaker_final="bob"),
        ]
        result = postprocess_segments_for_llm(segs)
        times = [r.global_start_sec for r in result]
        assert times == sorted(times), "Output must be sorted by global start time"


# ---------------------------------------------------------------------------
# postprocess_segments_for_llm — Pass 2: fragment merging
# ---------------------------------------------------------------------------

class TestPostprocessPass2:
    def test_adjacent_same_speaker_merged(self):
        segs = [
            make_segment("s1", "alice", "A", "alice starts talking about the results now",
                         0.0, 3.0, speaker_final="alice"),
            make_segment("s2", "alice", "A", "and continues with more important details here",
                         3.4, 6.0, speaker_final="alice"),
        ]
        result = postprocess_segments_for_llm(segs)
        assert len(result) == 1
        assert "alice starts talking" in result[0].text
        assert "continues with more" in result[0].text

    def test_adjacent_different_speakers_not_merged(self):
        segs = [
            make_segment("s1", "alice", "A", "alice says something important here today",
                         0.0, 3.0, speaker_final="alice"),
            make_segment("s2", "bob",   "B", "bob responds with his answer now here",
                         3.4, 6.0, speaker_final="bob"),
        ]
        result = postprocess_segments_for_llm(segs)
        assert len(result) == 2

    def test_same_speaker_not_merged_when_gap_too_large(self):
        segs = [
            make_segment("s1", "alice", "A", "alice says something important here today",
                         0.0, 3.0, speaker_final="alice"),
            make_segment("s2", "alice", "A", "alice continues speaking much later now",
                         10.0, 13.0, speaker_final="alice"),
        ]
        result = postprocess_segments_for_llm(segs)
        # Gap = 7.0 s >> 0.6 s threshold
        assert len(result) == 2

    def test_merged_end_time_is_last_segment(self):
        segs = [
            make_segment("s1", "alice", "A", "first part of the sentence from alice today",
                         0.0, 3.0, speaker_final="alice"),
            make_segment("s2", "alice", "A", "second part continues the thought further",
                         3.3, 6.5, speaker_final="alice"),
        ]
        result = postprocess_segments_for_llm(segs)
        assert result[0].global_end_sec == pytest.approx(6.5)
