"""LLM-prep post-processing pass.

Applies a two-pass cleanup to the kept segments before they are handed to an
LLM or written as the final human-readable transcript:

Pass 1 — duplicate / near-duplicate suppression
    - Short acknowledgements clustered within 1.2 s of the previous segment
    - Very-short utterance pairs with >50 % similarity
    - Cross-speaker near-identical twin captures
    - Overlapping near-duplicates (prefer the longer / better-attributed copy)

Pass 2 — adjacent same-speaker fragment merging
    - Fragments from the same speaker separated by ≤ 0.6 s are merged.
    - Near-identical or contained fragments are collapsed; additive fragments
      are concatenated, with internal repetition removed.
"""
from __future__ import annotations

import re

from audio2report.models import SegmentRecord
from audio2report.utils import normalize_text, text_similarity

# ---------------------------------------------------------------------------
# Short-utterance helpers  (defined before the functions that use them)
# ---------------------------------------------------------------------------

def is_very_short(text: str) -> bool:
    return len(normalize_text(text).split()) <= 3


def is_short_ack(text: str) -> bool:
    t = normalize_text(text)
    return t in {
        "yeah", "yeah?", "yep", "ok", "okay", "right", "sorry",
        "all right", "alright", "i see", "oh okay", "oh", "sure",
    }


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def text_contains(a: str, b: str) -> bool:
    na = normalize_text(a)
    nb = normalize_text(b)
    return na in nb or nb in na


def dedupe_internal_repetition(text: str) -> str:
    """Remove near-duplicate sentences within a single merged segment."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    unique: list[str] = []
    for p in parts:
        p_clean = p.strip()
        if not p_clean:
            continue
        if not any(text_similarity(p_clean, u) > 0.65 for u in unique):
            unique.append(p_clean)
    return " ".join(unique)


# ---------------------------------------------------------------------------
# Main post-processing entry point
# ---------------------------------------------------------------------------

def postprocess_segments_for_llm(segments: list[SegmentRecord]) -> list[SegmentRecord]:
    """
    Return a cleaned, merged list of kept segments suitable for LLM ingestion.

    The input list should already be sorted by global_start_sec.  Only segments
    with ``keep=True`` are included.
    """
    segs = [s for s in segments if s.keep]
    segs = sorted(segs, key=lambda x: x.global_start_sec)

    # ------------------------------------------------------------------
    # Pass 1: per-segment duplicate suppression
    # ------------------------------------------------------------------
    cleaned: list[SegmentRecord] = []

    for seg in segs:
        if not cleaned:
            cleaned.append(seg)
            continue

        prev = cleaned[-1]
        overlap = overlap_seconds(
            prev.global_start_sec, prev.global_end_sec,
            seg.global_start_sec, seg.global_end_sec,
        )
        sim = text_similarity(prev.text, seg.text)
        contains = text_contains(prev.text, seg.text)

        # Case 1: short acknowledgement spam
        if is_short_ack(seg.text) and abs(seg.global_start_sec - prev.global_start_sec) < 1.2:
            continue

        # Case 1b: very short near-duplicate pair
        if (
            abs(seg.global_start_sec - prev.global_start_sec) < 1.2
            and is_very_short(seg.text)
            and is_very_short(prev.text)
            and sim > 0.50
        ):
            continue

        # Case 1c: cross-speaker identical / near-identical twin capture
        if abs(seg.global_start_sec - prev.global_start_sec) < 1.2 and sim > 0.70:
            prev_is_other = (prev.speaker_final or "").startswith("OTHER")
            seg_is_other = (seg.speaker_final or "").startswith("OTHER")
            if prev_is_other and not seg_is_other:
                cleaned[-1] = seg
            elif len(seg.text) > len(prev.text):
                cleaned[-1] = seg
            continue

        # Case 1d: same-speaker repeated short utterance
        if (
            seg.speaker_final == prev.speaker_final
            and abs(seg.global_start_sec - prev.global_start_sec) < 1.5
            and is_very_short(seg.text)
            and is_very_short(prev.text)
        ):
            continue

        # Case 2: overlapping duplicates / near-duplicates
        if overlap >= 0.4 and (sim > 0.78 or contains):
            if len(seg.text) > len(prev.text):
                cleaned[-1] = seg
            continue

        cleaned.append(seg)

    # ------------------------------------------------------------------
    # Pass 2: adjacent same-speaker fragment merging
    # ------------------------------------------------------------------
    merged: list[SegmentRecord] = []

    for seg in cleaned:
        if not merged:
            merged.append(seg)
            continue

        prev = merged[-1]
        gap = seg.global_start_sec - prev.global_end_sec

        if seg.speaker_final == prev.speaker_final and gap <= 0.6:
            sim = text_similarity(prev.text, seg.text)
            contains = text_contains(prev.text, seg.text)

            # Near-duplicate or contained: keep the cleaner / longer version
            if contains or sim > 0.60:
                if len(seg.text) > len(prev.text):
                    prev.text = seg.text
                    prev.global_end_sec = seg.global_end_sec
                else:
                    prev.global_end_sec = max(prev.global_end_sec, seg.global_end_sec)
                prev.text = dedupe_internal_repetition(prev.text)
                continue

            prev_norm = normalize_text(prev.text)
            seg_norm = normalize_text(seg.text)

            # Replace if previous is fully contained in current
            if prev_norm in seg_norm:
                prev.text = seg.text
                prev.global_end_sec = seg.global_end_sec
                prev.text = dedupe_internal_repetition(prev.text)
                continue

            # Extend only if new text adds information
            if seg_norm in prev_norm:
                prev.global_end_sec = max(prev.global_end_sec, seg.global_end_sec)
                prev.text = dedupe_internal_repetition(prev.text)
                continue

            # Concatenate additive content
            prev.text = prev.text.rstrip() + " " + seg.text.lstrip()
            prev.text = dedupe_internal_repetition(prev.text)
            prev.global_end_sec = seg.global_end_sec
            continue

        merged.append(seg)

    return merged
