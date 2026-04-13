"""Shared pytest fixtures."""
from __future__ import annotations

import pytest

from tests.helpers import make_segment


# ---------------------------------------------------------------------------
# Reusable segment sets
# ---------------------------------------------------------------------------

SHARED_TEXT = (
    "this is a shared utterance that both microphones captured simultaneously "
    "during the meeting today and the text is long enough to be an anchor"
)


@pytest.fixture
def shared_text() -> str:
    return SHARED_TEXT


@pytest.fixture
def channel_a_segments():
    """Typical channel-A segment list: two unique + one shared utterance."""
    return [
        make_segment("a1", "alice", "A", "alice discusses the quarterly results", 5.0, 8.0),
        make_segment("a2", "alice", "A", SHARED_TEXT, 20.0, 24.0),
        make_segment("a3", "alice", "A", "alice closes the meeting now thank you", 50.0, 53.0),
    ]


@pytest.fixture
def channel_b_segments():
    """Channel-B: one shared utterance (0.3 s offset) + two unique."""
    return [
        make_segment("b1", "bob", "B", "bob introduces himself and his role today", 3.0, 6.0),
        make_segment("b2", "bob", "B", SHARED_TEXT, 20.3, 24.3),
        make_segment("b3", "bob", "B", "bob thanks everyone for attending the session", 55.0, 58.0),
    ]
