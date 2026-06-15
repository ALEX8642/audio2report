"""Diarization role assignment.

pyannote SPEAKER_XX labels are not stable across files — speaker 00 in file 1
is not necessarily the same person as speaker 00 in file 2.  We normalise this
per-file using a majority-vote heuristic: the speaker who occupies the most
total duration in that file is labelled PRIME_ON_THIS_MIC; all others become
OTHER_ON_THIS_MIC.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from audio2report.models import SegmentRecord


def assign_diar_roles_per_channel(segments: list[SegmentRecord]) -> None:
    """
    Mutate each segment's ``diar_speaker_role`` field in-place.

    Roles assigned:
        - ``"PRIME_ON_THIS_MIC"``  — dominant speaker for that (channel, file) pair
        - ``"OTHER_ON_THIS_MIC"``  — any other diarized speaker
        - ``None``                 — no diarization label present
    """
    by_channel_file: dict[tuple[str, int], list[SegmentRecord]] = defaultdict(list)
    for seg in segments:
        by_channel_file[(seg.channel_folder, seg.file_index)].append(seg)

    for file_segments in by_channel_file.values():
        durations: Counter = Counter()
        for seg in file_segments:
            if seg.diar_speaker_raw:
                durations[seg.diar_speaker_raw] += max(
                    0.0, seg.local_seg_end_sec - seg.local_seg_start_sec
                )

        dominant = durations.most_common(1)[0][0] if durations else None

        for seg in file_segments:
            if not seg.diar_speaker_raw:
                seg.diar_speaker_role = None
            elif seg.diar_speaker_raw == dominant:
                seg.diar_speaker_role = "PRIME_ON_THIS_MIC"
            else:
                seg.diar_speaker_role = "OTHER_ON_THIS_MIC"
