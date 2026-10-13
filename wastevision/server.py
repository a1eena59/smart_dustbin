"""HTTP service for the ESP32: one POST /classify -> one category to open.

Run with `python run.py serve`. Single process only: the camera is one physical
device and the burst lock below is per-process.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from wastevision.logging_csv import DecisionLog
from wastevision.pipeline import classify_frames
from wastevision.settings import CATEGORIES, Settings
from wastevision.snapshots import save_burst
from wastevision.voting import VoteOutcome, tally

log = logging.getLogger("wastevision.server")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ClassifyResponse(BaseModel):
    """What the ESP32 receives. `index` is the lid number (CATEGORIES order)."""

    category: str
    index: int
    confidence: float
    decided_by: str
    votes: dict[str, int]
    frames_used: int
    elapsed_ms: int


class BusyResponse(BaseModel):
    """Returned with HTTP 409 when a burst is already in flight."""

    error: str = "busy"
    detail: str = "a classification is already in progress; retry shortly"


def to_response(outcome: VoteOutcome, elapsed_ms: float) -> ClassifyResponse:
    """Convert a VoteOutcome into the wire schema."""
    return ClassifyResponse(
        category=outcome.category,
        index=outcome.index,
        confidence=round(outcome.confidence, 4),
        decided_by=outcome.decided_by,
        votes=dict(outcome.votes),
        frames_used=outcome.frames_used,
        elapsed_ms=round(elapsed_ms),
    )


# ---------------------------------------------------------------------------
# Collaborators (real ones in production, fakes in tests)
# ---------------------------------------------------------------------------


class Camera(Protocol):
    @property
    def is_alive(self) -> bool: ...
    def capture_burst(self) -> list[Any]: ...
    def close(self) -> None: ...


class Detector(Protocol):
    def detect(self, frame: Any) -> list[Any]: ...


class ImageLoader(Protocol):
    def from_path(self, path: str) -> Any | None: ...
    def from_bytes(self, data: bytes) -> Any | None: ...


# (frames, per-frame picks, outcome) -> anything; writes a burst to disk for debugging.
FrameSaver = Callable[[list[Any], list[Any], VoteOutcome], Any]


def _open_camera(settings: Settings) -> Camera:
    from wastevision.camera import BurstCamera

    return BurstCamera(settings)


def _load_detector(settings: Settings) -> Detector:
    import numpy as np

    from wastevision.detector import WasteDetector

    detector = WasteDetector(settings)
    # The first inference initialises torch kernels and takes ~1 s extra; pay it at boot
    # rather than on the ESP32's first request.
    detector.detect(np.zeros((settings.capture_height, settings.capture_width, 3), dtype=np.uint8))
    return detector


class Cv2ImageLoader:
    """Decodes images for /simulate with OpenCV, giving the same BGR arrays the camera does."""

    def from_path(self, path: str) -> Any | None:
        """Read an image file; None if missing or undecodable."""
        import cv2

        return cv2.imread(path)

    def from_bytes(self, data: bytes) -> Any | None:
        """Decode an uploaded image; None if undecodable."""
        import cv2
        import numpy as np

        if not data:
            return None
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Session statistics
# ---------------------------------------------------------------------------


class SessionStats:
    """In-memory totals since server start. Thread-safe."""

    def __init__(self, latency_window: int) -> None:
        self._lock = threading.Lock()
        self.per_category = {c: 0 for c in CATEGORIES}
        self.requests = 0
        self.fallbacks = 0
        self.simulated = 0
        self.rejected_busy = 0
        self.empty_bursts = 0
        self._latencies: deque[float] = deque(maxlen=latency_window)

    def record(self, outcome: VoteOutcome, elapsed_ms: float, *, simulated: bool, empty_burst: bool = False) -> None:
        """Count one completed classification."""
        with self._lock:
            self.requests += 1
            self.per_category[outcome.category] += 1
            self.fallbacks += outcome.decided_by == "fallback"
            self.simulated += simulated
            self.empty_bursts += empty_burst
            self._latencies.append(elapsed_ms)

    def record_busy(self) -> None:
        """Count one request rejected with 409."""
        with self._lock:
            self.rejected_busy += 1

    def snapshot(self) -> dict[str, Any]:
        """Current totals as a JSON-friendly dict."""
        with self._lock:
            latencies = sorted(self._latencies)
            return {
                "requests": self.requests,
                "per_category": dict(self.per_category),
                "fallback_rate": round(self.fallbacks / self.requests, 4) if self.requests else 0.0,
                "latency_mean_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
                "latency_p95_ms": round(percentile(latencies, 95), 1) if latencies else 0.0,
                "simulated": self.simulated,
                "rejected_busy": self.rejected_busy,
                "empty_bursts": self.empty_bursts,
            }


def percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile: always an observed value, no interpolation surprises."""
    rank = max(1, math.ceil(pct / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class WasteService:
    """Owns the camera, detector and log for the lifetime of the server."""

    def __init__(
        self,
        settings: Settings,
        camera: Camera,
        detector: Detector,
        image_loader: ImageLoader,
        decision_log: DecisionLog | None,
        frame_saver: FrameSaver | None = None,
    ) -> None:
        self.settings = settings
        self.camera = camera
        self.detector = detector
        self.image_loader = image_loader
        self.decision_log = decision_log
        self.frame_saver = frame_saver
        self.started = time.monotonic()
        self.stats = SessionStats(settings.stats_latency_window)
        # Non-blocking: overlapping bursts would mix frames of two items and open the wrong lid.
        self.burst_lock = threading.Lock()
        # Blocking: Ultralytics models aren't safe to call from two threads, and /simulate
        # may run while a /classify is inferring.
        self.detector_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._last_burst_empty = False
        self._camera_ok_cache = camera.is_alive

    def try_classify(self) -> ClassifyResponse | None:
        """Capture, vote and log one burst. None means another burst is in flight."""
        if not self.burst_lock.acquire(blocking=False):
            self.stats.record_busy()
            return None
        try:
            start = time.perf_counter()
            frames: list[Any] = []
            picks: list[Any] = []
            try:
                frames = self.camera.capture_burst()
                if not frames:
                    log.warning("camera returned no frames; answering with fallback %r", self.settings.fallback_category)
                with self.detector_lock:
                    outcome, picks = classify_frames(self.detector, frames, self.settings)
            except Exception:
                # The ESP32 must always get a lid to open; never let a bug become an HTTP 500.
                log.exception("classification failed; answering with fallback")
                outcome = tally([], self.settings)
            self._last_burst_empty = not frames
            self._camera_ok_cache = self.camera.is_alive and bool(frames)
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.stats.record(outcome, elapsed_ms, simulated=False, empty_burst=not frames)
            self._append_log(outcome, elapsed_ms)
            self._save_frames(frames, picks, outcome)
            return to_response(outcome, elapsed_ms)
        finally:
            self.burst_lock.release()

    def simulate(self, frame: Any) -> ClassifyResponse:
        """Run the identical vote on one still image repeated burst_size times."""
        start = time.perf_counter()
        with self.detector_lock:
            outcome, _ = classify_frames(self.detector, [frame] * self.settings.burst_size, self.settings)
        elapsed_ms = (time.perf_counter() - start) * 1000
        # Counted in /stats so the whole service can be exercised without hardware,
        # but kept out of the CSV, which records real sorting decisions only.
        self.stats.record(outcome, elapsed_ms, simulated=True)
        return to_response(outcome, elapsed_ms)

    def camera_ok(self) -> bool:
        """Best current knowledge of camera health, without touching it mid-burst."""
        # OpenCV captures aren't thread-safe and a burst may be reopening the device,
        # so only probe when no burst holds the lock; otherwise report the last result.
        if self.burst_lock.acquire(blocking=False):
            try:
                self._camera_ok_cache = self.camera.is_alive and not self._last_burst_empty
            finally:
                self.burst_lock.release()
        return self._camera_ok_cache

    def uptime_s(self) -> float:
        """Seconds since startup."""
        return time.monotonic() - self.started

    def _append_log(self, outcome: VoteOutcome, elapsed_ms: float) -> None:
        if self.decision_log is None:
            return
        try:
            with self._log_lock:
                self.decision_log.append(outcome, elapsed_ms)
        except OSError:
            # A full disk must not stop the bin from sorting.
            log.exception("could not write decision log")

    def _save_frames(self, frames: list[Any], picks: list[Any], outcome: VoteOutcome) -> None:
        if self.frame_saver is None or not frames:
            return
        try:
            # A detector crash leaves no picks; the raw frames are still worth keeping.
            saved = self.frame_saver(frames, picks or [None] * len(frames), outcome)
            log.info("burst frames saved to %s", saved)
        except Exception:
            # Debug output only: never let it cost the ESP32 its answer.
            log.exception("could not save burst frames")

    def close(self) -> None:
        """Release the camera and log file."""
        self.camera.close()
        if self.decision_log is not None:
            self.decision_log.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def create_app(
    settings: Settings | None = None,
    *,
    camera_factory: Callable[[Settings], Camera] = _open_camera,
    detector_factory: Callable[[Settings], Detector] = _load_detector,
    image_loader: ImageLoader | None = None,
    log_decisions: bool = True,
    frame_saver: FrameSaver | None = None,
) -> FastAPI:
    """Build the FastAPI app. Factories are injectable so tests need no camera or weights."""
    settings = settings or Settings.from_env()
    if frame_saver is None and settings.save_frames_dir:
        frame_saver = partial(save_burst, settings.save_frames_dir, settings=settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        mode = "native" if settings.native_waste_classes else "COCO bridge"
        log.info("loading model %s (%s mode)", settings.model_path, mode)
        # A missing or wrong model should stop the server at boot: unlike the camera,
        # there is no useful degraded mode without it.
        detector = detector_factory(settings)
        camera = camera_factory(settings)
        if camera.is_alive:
            log.info("camera %d open at %dx%d", settings.camera_index, settings.capture_width, settings.capture_height)
        else:
            log.warning(
                "camera %d NOT available; /classify will answer %r until it comes back",
                settings.camera_index,
                settings.fallback_category,
            )
        decision_log = DecisionLog(settings.log_path) if log_decisions else None
        service = WasteService(settings, camera, detector, image_loader or Cv2ImageLoader(), decision_log, frame_saver)
        if frame_saver is not None:
            log.info("saving every /classify burst to %s", settings.save_frames_dir or "(custom saver)")
        app.state.service = service
        log.info("ready: model=%s native=%s", settings.model_path, settings.native_waste_classes)
        try:
            yield
        finally:
            service.close()
            log.info("camera released, shutting down")

    app = FastAPI(title="wastevision", lifespan=lifespan)

    def service_of(request: Request) -> WasteService:
        return request.app.state.service

    # Plain `def` endpoints run in FastAPI's threadpool, so a 1-3 s burst never blocks
    # the event loop: /health stays responsive and a second /classify gets its 409 at once.

    @app.post("/classify", response_model=ClassifyResponse, responses={409: {"model": BusyResponse}})
    def classify(request: Request) -> Any:
        response = service_of(request).try_classify()
        if response is None:
            return JSONResponse(status_code=409, content=BusyResponse().model_dump())
        return response

    @app.get("/health")
    def health(request: Request) -> JSONResponse:
        service = service_of(request)
        camera = service.camera_ok()
        body = {
            "status": "ok" if camera else "degraded",
            "model": settings.model_path,
            "camera": camera,
            "uptime_s": round(service.uptime_s(), 1),
        }
        # Non-200 when the camera is down so the ESP32's red-LED check is a status-code test.
        return JSONResponse(status_code=200 if camera else 503, content=body)

    @app.get("/stats")
    def stats(request: Request) -> dict[str, Any]:
        return service_of(request).stats.snapshot()

    @app.post("/simulate", response_model=ClassifyResponse)
    async def simulate(request: Request) -> ClassifyResponse:
        service = service_of(request)
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("file")
            if upload is None or isinstance(upload, str):
                raise HTTPException(422, "multipart body needs a 'file' field")
            data = await upload.read()
            frame = await run_in_threadpool(service.image_loader.from_bytes, data)
            source = getattr(upload, "filename", "upload")
        else:
            try:
                body = await request.json()
            except ValueError:
                raise HTTPException(422, 'send JSON {"image_path": "..."} or multipart with a "file" field') from None
            path = body.get("image_path") if isinstance(body, dict) else None
            if not isinstance(path, str) or not path:
                raise HTTPException(422, 'JSON body needs "image_path"')
            frame = await run_in_threadpool(service.image_loader.from_path, path)
            source = path
        if frame is None:
            raise HTTPException(400, f"could not read an image from {source!r}")
        return await run_in_threadpool(service.simulate, frame)

    return app
