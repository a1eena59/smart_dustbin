"""Append-only CSV log of every decision."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from wastevision.settings import CATEGORIES
from wastevision.voting import VoteOutcome

# Vote columns are derived from CATEGORIES so the log can never drift from the index contract.
COLUMNS: tuple[str, ...] = (
    "timestamp",
    "category",
    "decided_by",
    "confidence",
    "frames_used",
    "elapsed_ms",
    *(f"votes_{c}" for c in CATEGORIES),
)


class DecisionLog:
    """Appends one row per decision; writes the header only when the file is new or empty."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not self._path.exists() or self._path.stat().st_size == 0
        # newline="" is required by the csv module, otherwise Windows gets blank rows.
        self._fh = self._path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        if needs_header:
            self._writer.writerow(COLUMNS)
            self._fh.flush()

    @property
    def path(self) -> Path:
        """Location of the log file."""
        return self._path

    def append(self, outcome: VoteOutcome, elapsed_ms: float) -> None:
        """Write one decision row and flush it to disk immediately."""
        self._writer.writerow(
            (
                datetime.now().astimezone().isoformat(timespec="milliseconds"),
                outcome.category,
                outcome.decided_by,
                f"{outcome.confidence:.4f}",
                outcome.frames_used,
                round(elapsed_ms),
                *(outcome.votes[c] for c in CATEGORIES),
            )
        )
        # Flush per row: the bin may lose power at any moment and we want every sort recorded.
        self._fh.flush()

    def close(self) -> None:
        """Close the underlying file."""
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> DecisionLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
