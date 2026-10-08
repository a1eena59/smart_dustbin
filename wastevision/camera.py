"""Burst capture from a webcam that degrades to empty results instead of raising."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cv2

from wastevision.settings import Settings

if TYPE_CHECKING:
    import numpy as np


@dataclass
class BurstStats:
    """Timing of the last burst, so staleness can be checked rather than guessed."""

    warmup_grabs: int = 0
    warmup_ms: float = 0.0
    frame_offsets_ms: list[float] = field(default_factory=list)  # since burst start
    reopened: bool = False


class BurstCamera:
    """Holds one cv2.VideoCapture open and captures short bursts of frames."""

    def __init__(self, settings: Settings, capture_factory: Callable[[int], Any] = cv2.VideoCapture) -> None:
        self._settings = settings
        # Injectable so tests can simulate a driver with a stale queue, no webcam needed.
        self._factory = capture_factory
        self._cap: Any = None
        self.last_stats = BurstStats()
        self._open()

    @property
    def is_alive(self) -> bool:
        """True while the underlying capture device is open."""
        return self._cap is not None and self._cap.isOpened()

    def reopen(self) -> bool:
        """Release and reopen the device. Returns whether it is alive afterwards."""
        self._release()
        return self._open()

    def read_frame(self, flush: bool | None = None) -> np.ndarray | None:
        """Read one frame, or None on any failure.

        With flush (default: settings.flush_between_frames) one queued frame is grabbed
        and dropped first, so a caller that was busy (e.g. running inference) gets a current one.
        """
        if flush is None:
            flush = self._settings.flush_between_frames
        if flush and not self._grab():
            return None
        return self._read()

    def capture_burst(self) -> list[np.ndarray]:
        """Flush stale frames, then capture up to burst_size current frames; [] if the camera is dead."""
        s = self._settings
        start = time.monotonic()
        deadline = start + s.burst_timeout_s
        stats = self.last_stats = BurstStats()
        if not self.is_alive and not self.reopen():
            return []

        frames: list[np.ndarray] = []
        failures = 0
        warm_grabs = 0
        warm_until = start + s.warmup_ms / 1000

        while len(frames) < s.burst_size and time.monotonic() < deadline:
            warming = warm_grabs < s.discard_frames or time.monotonic() < warm_until
            if warming:
                # grab() without retrieve(): the frame is dequeued but never decoded,
                # so draining the driver's backlog costs almost nothing.
                ok = self._grab()
                frame = None
            else:
                if warm_grabs and not stats.warmup_ms:
                    stats.warmup_grabs = warm_grabs
                    stats.warmup_ms = (time.monotonic() - start) * 1000
                # The first read follows the warmup directly, so the queue is already empty
                # and an extra grab would only wait a frame period for nothing.
                flush = s.flush_between_frames and bool(frames)
                frame = self.read_frame(flush=flush)
                ok = frame is not None

            if not ok:
                failures += 1
                if failures >= s.max_consecutive_read_failures:
                    # One recovery attempt only: a second failure means the device is gone
                    # and retrying would just burn the timeout.
                    if stats.reopened or not self.reopen():
                        return []
                    stats.reopened = True
                    failures = 0
                    # A freshly opened sensor is as stale/underexposed as the first one.
                    warm_grabs = 0
                    warm_until = time.monotonic() + s.warmup_ms / 1000
                    stats.warmup_ms = 0.0
                continue

            failures = 0
            if warming:
                warm_grabs += 1
            else:
                frames.append(frame)
                stats.frame_offsets_ms.append((time.monotonic() - start) * 1000)
        return frames

    def close(self) -> None:
        """Release the device. Safe to call more than once."""
        self._release()

    def __enter__(self) -> BurstCamera:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # Driver/backends raise a zoo of errors (cv2.error, OSError, MSMF HRESULTs);
    # a sorting bin must never crash on a camera hiccup, hence the broad excepts below.

    def _grab(self) -> bool:
        if self._cap is None:
            return False
        try:
            return bool(self._cap.grab())
        except Exception:
            return False

    def _read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        try:
            ok, frame = self._cap.read()
        except Exception:
            return None
        return frame if ok and frame is not None else None

    def _open(self) -> bool:
        s = self._settings
        try:
            cap = self._factory(s.camera_index)
            if not cap.isOpened():
                cap.release()
                return False
        except Exception:
            return False
        for prop, value in (
            (cv2.CAP_PROP_FRAME_WIDTH, s.capture_width),
            (cv2.CAP_PROP_FRAME_HEIGHT, s.capture_height),
            (cv2.CAP_PROP_BUFFERSIZE, s.camera_buffer_size),
        ):
            # Each set() separately: drivers that reject one property (BUFFERSIZE is
            # commonly unsupported on Windows/USB) must not prevent the others or the open.
            try:
                cap.set(prop, value)
            except Exception:
                pass
        self._cap = cap
        return True

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
