import math
import time
from collections import deque

import pytest

cv2 = pytest.importorskip("cv2")  # camera.py needs cv2's property ids; skip on a bare test machine

from wastevision.camera import BurstCamera  # noqa: E402
from wastevision.settings import Settings  # noqa: E402

STALE = -math.inf  # production time of frames already queued before the test starts


class FakeCapture:
    """Mimics a webcam driver: frames arrive in real time at `fps` into a bounded queue
    that keeps the newest `depth` frames, and grab() blocks while the queue is empty."""

    def __init__(self, stale: int = 0, fps: float = 60.0, depth: int = 32, opened: bool = True) -> None:
        self.fps = fps
        self.t0 = time.monotonic()
        self.produced = 0
        self.queue: deque[tuple[int, float]] = deque(((-i, STALE) for i in range(stale, 0, -1)), maxlen=depth)
        self.opened = opened
        self.current: tuple[int, float] | None = None
        self.props: dict[int, float] = {}
        self.grabs = 0

    def isOpened(self) -> bool:
        return self.opened

    def set(self, prop: int, value: float) -> bool:
        self.props[prop] = value
        return True

    def release(self) -> None:
        self.opened = False

    def _produce(self) -> None:
        due = int((time.monotonic() - self.t0) * self.fps)
        while self.produced < due:
            self.produced += 1
            self.queue.append((self.produced, self.t0 + self.produced / self.fps))

    def grab(self) -> bool:
        self.grabs += 1
        if not self.opened:
            return False
        self._produce()
        while not self.queue:
            time.sleep(1 / self.fps / 4)
            self._produce()
        self.current = self.queue.popleft()
        return True

    def retrieve(self) -> tuple[bool, tuple[int, float] | None]:
        return self.current is not None, self.current

    def read(self) -> tuple[bool, tuple[int, float] | None]:
        return self.retrieve() if self.grab() else (False, None)


class DeadCapture(FakeCapture):
    """Opens fine, then never delivers a frame (unplugged mid-session)."""

    def grab(self) -> bool:
        self.grabs += 1
        return False

    def read(self) -> tuple[bool, None]:
        return False, None


def make_settings(**overrides) -> Settings:
    base = dict(warmup_ms=100, discard_frames=8, burst_size=7, burst_timeout_s=2.0, flush_between_frames=True)
    return Settings(**{**base, **overrides})


def camera_with(fake: FakeCapture, settings: Settings) -> tuple[BurstCamera, list[int]]:
    opens: list[int] = []

    def factory(index: int) -> FakeCapture:
        opens.append(index)
        return fake

    return BurstCamera(settings, capture_factory=factory), opens


def test_burst_contains_no_frames_from_before_the_item_was_placed():
    # The bug: queue full of pre-placement frames (deeper than discard_frames, as with
    # drivers that ignore BUFFERSIZE) plus a preview loop that let more pile up.
    fake = FakeCapture(stale=20, fps=60, depth=32)
    cam, _ = camera_with(fake, make_settings())
    time.sleep(0.3)
    placed = time.monotonic()

    frames = cam.capture_burst()

    assert len(frames) == 7
    assert all(produced_at >= placed for _, produced_at in frames), frames
    ids = [n for n, _ in frames]
    assert ids == sorted(set(ids)), "frames must be distinct and in order"


def test_warmup_lasts_at_least_warmup_ms_and_discard_frames():
    fake = FakeCapture(stale=50, fps=60, depth=64)
    cam, _ = camera_with(fake, make_settings(warmup_ms=150, discard_frames=8))
    cam.capture_burst()
    stats = cam.last_stats
    assert stats.warmup_ms >= 150
    assert stats.warmup_grabs >= 50  # drained the whole stale queue, not just 8


def test_discard_frames_is_a_minimum_even_with_zero_warmup_time():
    fake = FakeCapture(stale=0, fps=60)
    cam, _ = camera_with(fake, make_settings(warmup_ms=0, discard_frames=5))
    cam.capture_burst()
    assert cam.last_stats.warmup_grabs == 5


def test_flush_between_frames_skips_one_frame_per_read():
    fake = FakeCapture(stale=0, fps=60)
    cam, _ = camera_with(fake, make_settings(flush_between_frames=True))
    ids = [n for n, _ in cam.capture_burst()]
    # Each read after the first is preceded by a grab, so consecutive voting frames are 2 apart.
    assert all(b - a >= 2 for a, b in zip(ids, ids[1:])), ids


def test_no_flush_between_frames_reads_consecutive_frames():
    fake = FakeCapture(stale=0, fps=60)
    cam, _ = camera_with(fake, make_settings(flush_between_frames=False))
    ids = [n for n, _ in cam.capture_burst()]
    assert all(b - a == 1 for a, b in zip(ids, ids[1:])), ids


def test_read_frame_flush_drops_exactly_one_queued_frame():
    fake = FakeCapture(stale=3, fps=0.001)
    cam, _ = camera_with(fake, make_settings())
    assert cam.read_frame(flush=True)[0] == -2  # -3 dropped
    assert cam.read_frame(flush=False)[0] == -1


def test_burst_respects_timeout():
    fake = FakeCapture(stale=0, fps=5)  # slow camera: 7 frames + warmup can't fit
    cam, _ = camera_with(fake, make_settings(burst_timeout_s=0.8, warmup_ms=100, discard_frames=1))
    start = time.monotonic()
    frames = cam.capture_burst()
    # One in-flight grab may finish after the deadline, at most one frame period late.
    assert time.monotonic() - start < 0.8 + 1 / 5 + 0.1
    assert len(frames) < 7


def test_dead_camera_reopens_once_then_returns_empty():
    opens: list[DeadCapture] = []

    def factory(index: int) -> DeadCapture:
        # A real reopen yields a new handle; reusing a released fake would fail the reopen itself.
        opens.append(DeadCapture())
        return opens[-1]

    cam = BurstCamera(make_settings(), capture_factory=factory)
    assert cam.capture_burst() == []
    assert len(opens) == 2  # initial open + exactly one reopen
    assert cam.last_stats.reopened


def test_camera_that_never_opens_returns_empty_without_raising():
    cam, _ = camera_with(FakeCapture(opened=False), make_settings())
    assert not cam.is_alive
    assert cam.capture_burst() == []
    assert cam.read_frame() is None


def test_unsupported_buffersize_does_not_break_open():
    class NoBufferSize(FakeCapture):
        def set(self, prop, value):
            if prop == cv2.CAP_PROP_BUFFERSIZE:
                raise cv2.error("unsupported")
            return super().set(prop, value)

    fake = NoBufferSize()
    cam, _ = camera_with(fake, make_settings())
    assert cam.is_alive
    assert fake.props[cv2.CAP_PROP_FRAME_WIDTH] == 640
    assert fake.props[cv2.CAP_PROP_FRAME_HEIGHT] == 480


def test_settings_reject_warmup_longer_than_timeout():
    with pytest.raises(ValueError, match="warmup_ms"):
        Settings(warmup_ms=3000, burst_timeout_s=3.0)
