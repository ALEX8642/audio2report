"""Cross-channel alignment: anchor detection and robust offset estimation.

Algorithm
---------
1. Find high-similarity text pairs across channels (anchors).
   These are utterances that both mics captured — i.e., cross-talk.
2. For each anchor, compute the raw time delta: b_root_time - a_root_time.
3. Use a MAD-filtered median to produce a single robust offset estimate.

Performance
-----------
Finding anchors requires comparing every candidate on channel A against every
candidate on channel B — O(n×m) in the worst case — because we don't know the
clock offset yet and can't use time windowing.  We mitigate this with a cheap
bigram-Jaccard pre-filter: SequenceMatcher is only called when the two texts
share enough bigrams to plausibly meet the similarity threshold.  This avoids
SequenceMatcher on the vast majority of non-matching pairs.
"""
from __future__ import annotations

import statistics
from typing import FrozenSet, List, Optional, Set, Tuple

from audio2report.models import AlignmentAnchor, SegmentRecord
from audio2report.utils import normalize_text, text_similarity


# ---------------------------------------------------------------------------
# Bigram pre-filter
# ---------------------------------------------------------------------------

def _word_bigrams(text: str) -> FrozenSet[Tuple[str, str]]:
    words = normalize_text(text).split()
    if len(words) < 2:
        return frozenset()
    return frozenset(zip(words, words[1:]))


def _bigram_jaccard(bg_a: FrozenSet, bg_b: FrozenSet) -> float:
    if not bg_a and not bg_b:
        return 0.0
    intersection = len(bg_a & bg_b)
    union = len(bg_a | bg_b)
    return intersection / union if union else 0.0


# Conservative threshold: if bigram overlap can't possibly produce a
# SequenceMatcher ratio of `sim_threshold`, skip the expensive call.
# Bigram Jaccard ≥ ~0.45 is required for SequenceMatcher ratio ≥ 0.90.
_BIGRAM_PREFILTER_RATIO = 0.45


def collect_alignment_anchors(
    a_segments: List[SegmentRecord],
    b_segments: List[SegmentRecord],
    *,
    min_text_len: int = 25,
    sim_threshold: float = 0.90,
) -> List[AlignmentAnchor]:
    """
    Find high-similarity segment pairs across two channels and compute
    per-pair time deltas.  Delta = b_root_timeline_start − a_root_timeline_start.
    """
    anchors: List[AlignmentAnchor] = []

    a_candidates = [s for s in a_segments if len(normalize_text(s.text)) >= min_text_len]
    b_candidates = [s for s in b_segments if len(normalize_text(s.text)) >= min_text_len]

    # Pre-compute bigrams once per segment
    b_bigrams = [_word_bigrams(sb.text) for sb in b_candidates]

    for sa in a_candidates:
        bg_a = _word_bigrams(sa.text)
        best_sb = None
        best_sim = -1.0

        for sb, bg_b in zip(b_candidates, b_bigrams):
            # Cheap pre-filter: skip SequenceMatcher if bigram overlap is too low
            if _bigram_jaccard(bg_a, bg_b) < _BIGRAM_PREFILTER_RATIO:
                continue
            sim = text_similarity(sa.text, sb.text)
            if sim > best_sim:
                best_sim = sim
                best_sb = sb

        if best_sb is not None and best_sim >= sim_threshold:
            delta = best_sb.root_timeline_start_sec - sa.root_timeline_start_sec
            anchors.append(AlignmentAnchor(
                a_uid=sa.uid,
                b_uid=best_sb.uid,
                a_prime=sa.channel_prime,
                b_prime=best_sb.channel_prime,
                sim=best_sim,
                delta_sec=delta,
                a_text=sa.text,
                b_text=best_sb.text,
            ))

    return anchors


def robust_median_offset(anchors: List[AlignmentAnchor]) -> Optional[float]:
    """
    Return a MAD-filtered median of the per-anchor time deltas.
    Outliers more than max(2.0, 3.5 × MAD) away from the median are pruned.
    Returns None if no anchors are provided.
    """
    if not anchors:
        return None

    deltas = [a.delta_sec for a in anchors]
    median = statistics.median(deltas)

    abs_dev = [abs(x - median) for x in deltas]
    mad = statistics.median(abs_dev) if abs_dev else 0.0
    if mad == 0:
        filtered = deltas
    else:
        threshold = max(2.0, 3.5 * mad)
        filtered = [x for x in deltas if abs(x - median) <= threshold]

    return statistics.median(filtered) if filtered else median
