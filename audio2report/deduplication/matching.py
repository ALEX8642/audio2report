"""Cross-channel duplicate detection.

A pair of segments is considered a duplicate when:
  - their global timestamps are within *time_tolerance_sec* of each other, AND
  - their normalised text similarity exceeds *sim_threshold*.

Only one match per B-side segment is allowed (greedy, similarity-ordered).

Performance
-----------
The original algorithm was O(n×m) — every A segment was compared against every
B segment with only the time filter applied inside the loop.

This version is O(n log n):
  1. Both input lists must be sorted by ``global_start_sec`` (the pipeline
     guarantees this after ``apply_offset_to_channel``).
  2. For each A segment, ``bisect`` locates the contiguous slice of B segments
     whose timestamps fall within ±time_tolerance_sec.  Only that slice is
     compared — typically a handful of segments rather than the entire channel.
"""
from __future__ import annotations

import bisect
from typing import List

from audio2report.models import PairMatch, SegmentRecord
from audio2report.utils import normalize_text, text_similarity


def match_segments_across_channels(
    a_segments: List[SegmentRecord],
    b_segments: List[SegmentRecord],
    *,
    time_tolerance_sec: float = 2.5,
    sim_threshold: float = 0.86,
    min_text_len: int = 18,
) -> List[PairMatch]:
    """
    Return a list of cross-channel duplicate pairs.

    Both lists must be sorted by ``global_start_sec``.
    Segments shorter than *min_text_len* (after normalisation) are ignored to
    avoid spurious matches on short acknowledgements.
    """
    matches: List[PairMatch] = []
    used_b: set = set()

    # Pre-compute sorted B start times for binary search
    b_starts = [sb.global_start_sec for sb in b_segments]
    # Pre-filter B by minimum text length to avoid repeated checks in the inner loop
    b_eligible = [
        len(normalize_text(sb.text)) >= min_text_len
        for sb in b_segments
    ]

    for sa in a_segments:
        if len(normalize_text(sa.text)) < min_text_len:
            continue

        # Binary search: find the window of B segments within ±time_tolerance_sec
        lo = bisect.bisect_left(b_starts, sa.global_start_sec - time_tolerance_sec)
        hi = bisect.bisect_right(b_starts, sa.global_start_sec + time_tolerance_sec)

        best_idx = None
        best_score = -1.0
        best_td = 0.0

        for j in range(lo, hi):
            if j in used_b or not b_eligible[j]:
                continue
            sb = b_segments[j]
            sim = text_similarity(sa.text, sb.text)
            if sim >= sim_threshold and sim > best_score:
                best_score = sim
                best_idx = j
                best_td = abs(sa.global_start_sec - sb.global_start_sec)

        if best_idx is not None:
            used_b.add(best_idx)
            matches.append(PairMatch(
                a_uid=sa.uid,
                b_uid=b_segments[best_idx].uid,
                sim=best_score,
                time_diff_sec=float(best_td),
            ))

    return matches
