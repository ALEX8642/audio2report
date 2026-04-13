"""Report generation engine.

Public surface
--------------
generate_report(transcript_text, config, meta=None)
    Build a prompt and call the configured LLM provider.  Returns the full
    report text.

load_transcript_text(path)
    Load a transcript from either a ``canonical_transcript.json`` or any
    ``*.txt`` file into a plain string ready for the prompt.

build_prompt(transcript_text, config, meta=None)
    Assemble the final prompt string from the template + substitutions.
    Exported for testing and for users who want to inspect the prompt before
    sending it.
"""
from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Iterator, Optional

from audio2report._log import get_logger
from audio2report.config import LLMConfig
from audio2report.models import RunMeta
from audio2report.utils import format_hms

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

_BUILTIN_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _load_template(name_or_path: str) -> str:
    """
    Load a prompt template by name or file path.

    Resolution order:
    1. Treat *name_or_path* as an absolute or relative file path.
    2. Look for ``{name_or_path}.txt`` in the built-in templates directory.
    """
    path = Path(name_or_path)
    if path.is_file():
        return path.read_text(encoding="utf-8")

    builtin = _BUILTIN_TEMPLATES_DIR / f"{name_or_path}.txt"
    if builtin.is_file():
        return builtin.read_text(encoding="utf-8")

    raise ValueError(
        f"Prompt template {name_or_path!r} not found.  "
        f"Built-in templates: {[p.stem for p in _BUILTIN_TEMPLATES_DIR.glob('*.txt')]}"
    )


# ---------------------------------------------------------------------------
# Transcript loading
# ---------------------------------------------------------------------------

def load_transcript_text(path: Path) -> str:
    """
    Load a transcript file into a plain string.

    Supports:
    - ``canonical_transcript.json`` — extracts kept segments formatted as
      ``[HH:MM:SS.mmm] speaker: text``
    - ``cleaned_llm_input.txt`` or any ``.txt`` file — used as-is
    """
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        segments = data.get("segments", [])
        lines = []
        for seg in segments:
            if not seg.get("keep", True):
                continue
            ts = (
                f"[{format_hms(seg.get('global_start_sec', 0.0))} - "
                f"{format_hms(seg.get('global_end_sec', 0.0))}]"
            )
            speaker = seg.get("speaker_final") or "UNKNOWN"
            text = (seg.get("text") or "").strip()
            lines.append(f"{ts} {speaker}: {text}")
        return "\n".join(lines)

    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def _format_duration(meta: RunMeta) -> str:
    total_sec = meta.total_segments * 0  # not directly available
    # Use the last global timestamp from meta if we had it; fall back to a
    # best-effort description based on kept segment count.
    return f"~{meta.kept_segments} transcript segments"


def build_prompt(
    transcript_text: str,
    config: LLMConfig,
    *,
    meta: Optional[RunMeta] = None,
) -> str:
    """
    Assemble the LLM prompt from the configured template.

    Truncates the transcript if it exceeds ``config.max_transcript_chars``
    (keeping the tail — the most recent content — which is usually what
    matters most for an audit summary).
    """
    template = _load_template(config.prompt_template)

    if len(transcript_text) > config.max_transcript_chars:
        original_len = len(transcript_text)
        transcript_text = transcript_text[-config.max_transcript_chars:]
        # Find the first newline so we don't start mid-line
        nl = transcript_text.find("\n")
        if nl > 0:
            transcript_text = transcript_text[nl + 1:]
        transcript_text = (
            f"[... transcript truncated — showing last "
            f"{config.max_transcript_chars:,} of {original_len:,} characters ...]\n\n"
            + transcript_text
        )
        logger.warning(
            f"Transcript truncated to {config.max_transcript_chars:,} chars "
            f"(was {original_len:,}).  Consider using a model with a larger context window."
        )

    primes = ", ".join(meta.primes) if meta else "unknown"
    duration = _format_duration(meta) if meta else "unknown"

    return template.format(
        transcript=transcript_text,
        primes=primes,
        duration=duration,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_report(
    transcript_text: str,
    config: LLMConfig,
    *,
    meta: Optional[RunMeta] = None,
    stream_to_stdout: bool = True,
) -> str:
    """
    Generate an audit report for *transcript_text* using the configured LLM.

    Parameters
    ----------
    transcript_text:
        Plain-text transcript, formatted as ``[timestamp] speaker: text``.
    config:
        LLM configuration section.
    meta:
        Optional run metadata used to populate the prompt header (participants,
        duration).
    stream_to_stdout:
        If True and the provider supports streaming, tokens are written to
        stdout in real-time as they arrive.  The full text is always returned.

    Returns
    -------
    str
        The complete LLM-generated report text.
    """
    from audio2report.llm.base import get_provider

    prompt = build_prompt(transcript_text, config, meta=meta)
    provider = get_provider(config)

    use_stream = stream_to_stdout and config.stream

    if use_stream:
        report_text = ""
        try:
            for token in provider.stream(prompt):
                sys.stdout.write(token)
                sys.stdout.flush()
                report_text += token
        finally:
            sys.stdout.write("\n")
            sys.stdout.flush()
        return report_text
    else:
        return provider.generate(prompt)
