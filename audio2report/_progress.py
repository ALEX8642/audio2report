"""Shared progress reporting for the transcribe+diarize loop, used by both
DualMicPipeline and SingleMicPipeline — the one stage in the pipeline that's
both the dominant cost and already progress-tracked.

Wraps a Rich ``Progress`` bar (unchanged terminal UX, now with an ETA
column) and, if a sink path is given, mirrors each completed item as a
newline-delimited JSON event. That lets a non-interactive consumer (the
Streamlit UI, reading a piped subprocess) track progress without scraping
Rich's ANSI-rendered terminal output — which doesn't render reliably once
``console.is_terminal`` is False, i.e. once stdout is piped rather than a
real terminal.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from audio2report._log import get_console


class ProgressReporter:
    """Context manager wrapping a single "Transcribing" Rich task.

    Usage mirrors the ``Progress`` object it replaces:

        with ProgressReporter(total_files, progress_file) as progress:
            for item in items:
                progress.start_item(description)
                ... do work ...
                progress.advance()
    """

    def __init__(self, total: int, progress_file: Path | None = None) -> None:
        self._rich = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=get_console(),
            transient=True,
        )
        self._total = total
        self._completed = 0
        self._sink_path = progress_file
        self._current_description = ""
        self._started = 0.0
        self._task_id: Any = None

    def __enter__(self) -> ProgressReporter:
        self._rich.__enter__()
        self._task_id = self._rich.add_task("Transcribing", total=self._total)
        self._started = time.monotonic()
        if self._sink_path:
            self._sink_path.parent.mkdir(parents=True, exist_ok=True)
            self._sink_path.write_text("", encoding="utf-8")  # fresh file per run
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._rich.__exit__(*exc_info)

    def start_item(self, description: str) -> None:
        self._current_description = description
        self._rich.update(self._task_id, description=description)

    def advance(self) -> None:
        self._completed += 1
        self._rich.advance(self._task_id)
        if self._sink_path:
            event = {
                "completed": self._completed,
                "total": self._total,
                "description": self._current_description,
                "elapsed_sec": time.monotonic() - self._started,
            }
            with self._sink_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
