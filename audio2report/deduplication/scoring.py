"""Retention scoring and speaker attribution.

Two separate scoring concerns are deliberately kept apart:

retention_score
    Answers "which copy of a duplicate is higher quality?" — uses only
    audio/ASR quality signals (RMS, logprob, text length).

speaker_score_map
    Answers "who is speaking in this segment?" — uses diarization roles,
    RMS loudness differential, and cross-mic evidence.
"""
from __future__ import annotations

from audio2report.models import SegmentRecord

# ---------------------------------------------------------------------------
# Retention scoring (which duplicate copy to keep)
# ---------------------------------------------------------------------------

def retention_score(seg: SegmentRecord) -> float:
    x = 0.0
    if seg.rms_dbfs is not None:
        x += seg.rms_dbfs / 20.0
    if seg.avg_logprob is not None:
        x += float(seg.avg_logprob)
    if seg.no_speech_prob is not None:
        x -= float(seg.no_speech_prob) * 0.5
    x += min(len(seg.text), 200) / 200.0
    return x


def choose_primary_from_pair(
    sa: SegmentRecord,
    sb: SegmentRecord,
) -> tuple[SegmentRecord, SegmentRecord, str, float]:
    """
    Choose which of a duplicate pair to keep based solely on quality signals.

    Returns (keep, drop, reason, margin).
    """
    a_score = retention_score(sa)
    b_score = retention_score(sb)
    sa.retention_score_value = a_score
    sb.retention_score_value = b_score

    keep, drop = (sa, sb) if a_score >= b_score else (sb, sa)
    margin = abs(a_score - b_score)
    return keep, drop, "higher_retention_score", margin


# ---------------------------------------------------------------------------
# Speaker attribution
# ---------------------------------------------------------------------------

def speaker_score_map(
    seg: SegmentRecord,
    peer_match: SegmentRecord | None,
) -> dict[str, float]:
    """
    Build a score map for possible speaker labels.

    Signals used:
        - diarization role on this mic
        - RMS loudness differential vs peer mic
        - cross-mic diarization agreement
    """
    scores: dict[str, float] = {
        seg.channel_prime: 0.0,
        "THIRD_SPEAKER": 0.0,
        "OTHER_OR_THIRD_SPEAKER": 0.0,
    }

    if seg.diar_speaker_role == "PRIME_ON_THIS_MIC":
        scores[seg.channel_prime] += 1.5
    if seg.diar_speaker_role == "OTHER_ON_THIS_MIC":
        scores["OTHER_OR_THIRD_SPEAKER"] += 0.5

    if peer_match is not None:
        if seg.rms_dbfs is not None and peer_match.rms_dbfs is not None:
            loud_diff = seg.rms_dbfs - peer_match.rms_dbfs
            if loud_diff >= 2.0:
                scores[seg.channel_prime] += 1.0
            elif loud_diff <= -2.0:
                scores["OTHER_OR_THIRD_SPEAKER"] += 0.5

        if (
            seg.diar_speaker_role == "OTHER_ON_THIS_MIC"
            and peer_match.diar_speaker_role == "OTHER_ON_THIS_MIC"
        ):
            scores["THIRD_SPEAKER"] += 1.5

        if (
            seg.diar_speaker_role == "PRIME_ON_THIS_MIC"
            and peer_match.diar_speaker_role == "PRIME_ON_THIS_MIC"
        ):
            scores[seg.channel_prime] += 0.3
    else:
        # single-channel fallback (no peer evidence)
        if seg.diar_speaker_role is None:
            scores[seg.channel_prime] += 0.6
        elif seg.diar_speaker_role == "PRIME_ON_THIS_MIC":
            scores[seg.channel_prime] += 0.5

    return scores


def finalize_speaker_from_scores(
    score_map: dict[str, float],
) -> tuple[str, float, list[str], str]:
    """Convert a score map into (label, confidence, flags, decision_basis)."""
    ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -999.0
    margin = best_score - second_score

    flags: list[str] = []
    basis = f"best={best_label}:{best_score:.2f}, margin={margin:.2f}, scores={score_map}"

    if margin < 0.5:
        flags.append("speaker_uncertain")
        return "UNKNOWN", 0.3, flags, basis

    # Collapse weak THIRD_SPEAKER decisions
    if best_label == "THIRD_SPEAKER" and best_score < 1.25:
        best_label = "OTHER_OR_THIRD_SPEAKER"
        flags.append("third_speaker_evidence_weak")

    confidence = min(0.95, max(0.20, 0.50 + 0.15 * margin + 0.10 * best_score))
    return best_label, confidence, flags, basis


def assign_speaker_for_kept_segment(
    seg: SegmentRecord,
    peer_match: SegmentRecord | None,
) -> tuple[str, float, list[str], dict[str, float], str]:
    """
    Assign a final speaker label to *seg* using multi-signal scoring.

    Returns (label, confidence, flags, score_detail, decision_basis).
    """
    score_map = speaker_score_map(seg, peer_match)
    label, confidence, flags, basis = finalize_speaker_from_scores(score_map)
    return label, confidence, flags, score_map, basis
