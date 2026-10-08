"""Glue between detector and voting, shared by every entrypoint (CLI now, HTTP server later)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from wastevision.detector import Detection, WasteDetector
from wastevision.settings import Settings
from wastevision.voting import VoteOutcome, tally

if TYPE_CHECKING:
    import numpy as np


def classify_frames(
    detector: WasteDetector, frames: Sequence[np.ndarray], settings: Settings
) -> tuple[VoteOutcome, list[Detection | None]]:
    """Run the detector on each frame, reduce each to one verdict, and vote."""
    picks: list[Detection | None] = []
    for frame in frames:
        picks.append(WasteDetector.best(detector.detect(frame), settings.best_tie_window))
    verdicts = [(p.category, p.confidence) if p else (None, 0.0) for p in picks]
    return tally(verdicts, settings), picks
