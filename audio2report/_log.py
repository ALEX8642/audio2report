"""Centralised logging and console setup.

A single ``Console`` instance is shared across the logger and any Rich
``Progress`` bars so they render on the same output stream without
interleaving artefacts.
"""
from __future__ import annotations

import logging
import warnings

from rich.console import Console
from rich.logging import RichHandler

# One console shared by every module that needs rich output
_console = Console(stderr=False)

# Known-benign third-party warnings that alarm users for no reason — an
# allow-list of specific messages, not a blanket ignore-everything, so a
# genuinely new warning still surfaces. audio2report never uses pyannote's
# path-based audio loader (see transcription/_diarize.py's own comment on
# why), so the torchcodec warning it emits at import time is moot; the
# pooling std() warning is expected floating-point behavior on some
# tensor shapes, not a sign anything is wrong.
warnings.filterwarnings(
    "ignore", message=r"(?s)\s*torchcodec is not installed correctly.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message=r"(?s)\s*std\(\): degrees of freedom is <= 0.*", category=UserWarning
)


def get_console() -> Console:
    return _console


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(
            console=_console,
            rich_tracebacks=True,
            show_path=False,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_log_level(level: int) -> None:
    """Adjust the log level of all audio2report loggers at runtime."""
    root = logging.getLogger("audio2report")
    root.setLevel(level)
    for child in logging.Logger.manager.loggerDict.values():
        if isinstance(child, logging.Logger) and child.name.startswith("audio2report"):
            child.setLevel(level)
