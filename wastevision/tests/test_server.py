import csv
import threading

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # required by fastapi.testclient

from fastapi.testclient import TestClient  # noqa: E402

from wastevision.detector import Detection  # noqa: E402
from wastevision.server import SessionStats, create_app, percentile  # noqa: E402
from wastevision.settings import Settings  # noqa: E402
from wastevision.voting import tally  # noqa: E402

SCHEMA = {"category", "index", "confidence", "decided_by", "votes", "frames_used", "elapsed_ms"}

# Fake frames are (category, confidence) tuples, or None for "nothing in view";
# FakeDetector turns each into one detection, so tests control every frame's verdict.
P, A, O = ("plastic", 0.8), ("paper", 0.8), ("organic", 0.8)


class FakeCamera:
    def __init__(self, frames=(), alive=True, gate: threading.Event | None = None):
        self.frames = list(frames)
        self.alive = alive
        self.gate = gate
        self.entered = threading.Event()
        self.closed = False
        self.bursts = 0

    @property
    def is_alive(self):
        return self.alive

    def capture_burst(self):
        self.bursts += 1
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(5), "test gate never opened"
        # Mirrors BurstCamera: a dead camera yields an empty burst, never an exception.
        return list(self.frames) if self.alive else []

    def close(self):
        self.closed = True


class FakeDetector:
    def __init__(self, fail=False):
        self.fail = fail

    def detect(self, frame):
        if self.fail:
            raise RuntimeError("model exploded")
        if frame is None:
            return []
        category, conf = frame
        return [Detection(category, category, conf, (0.0, 0.0, 10.0, 10.0))]


class FakeLoader:
    def from_path(self, path):
        return ("paper", 0.9) if path.endswith(".jpg") else None

    def from_bytes(self, data):
        return ("paper", 0.9) if data else None


@pytest.fixture
def settings(tmp_path):
    return Settings(log_path=str(tmp_path / "decisions.csv"))


def client_for(settings, camera, detector=None):
    app = create_app(
        settings,
        camera_factory=lambda s: camera,
        detector_factory=lambda s: detector or FakeDetector(),
        image_loader=FakeLoader(),
    )
    return TestClient(app)


# --- /classify -------------------------------------------------------------------


def test_classify_successful_vote(settings):
    camera = FakeCamera([P, P, P, P, P, A, None])
    with client_for(settings, camera) as client:
        r = client.post("/classify")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == SCHEMA
    assert body["category"] == "plastic"
    assert body["index"] == 0
    assert body["decided_by"] == "vote"
    assert body["votes"] == {"plastic": 5, "paper": 1, "organic": 0}
    assert body["frames_used"] == 6
    assert body["confidence"] == pytest.approx(0.8)
    assert isinstance(body["elapsed_ms"], int)


def test_classify_is_logged_to_csv(settings):
    with client_for(settings, FakeCamera([P] * 7)) as client:
        client.post("/classify")
        client.post("/classify")
    with open(settings.log_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["category"] for r in rows] == ["plastic", "plastic"]


def test_classify_fallback_on_split_vote(settings):
    camera = FakeCamera([P, P, P, A, A, A, O])
    with client_for(settings, camera) as client:
        r = client.post("/classify")
    assert r.status_code == 200
    body = r.json()
    assert body["decided_by"] == "fallback"
    assert body["category"] == settings.fallback_category
    assert body["confidence"] == 0.0
    assert body["frames_used"] == 7


def test_dead_camera_still_returns_200_with_fallback(settings):
    with client_for(settings, FakeCamera([P] * 7, alive=False)) as client:
        r = client.post("/classify")
    assert r.status_code == 200
    body = r.json()
    assert body["category"] == settings.fallback_category
    assert body["decided_by"] == "fallback"
    assert body["frames_used"] == 0


def test_detector_crash_still_returns_200_with_fallback(settings):
    with client_for(settings, FakeCamera([P] * 7), FakeDetector(fail=True)) as client:
        r = client.post("/classify")
    assert r.status_code == 200
    assert r.json()["decided_by"] == "fallback"


def test_request_body_is_ignored(settings):
    with client_for(settings, FakeCamera([P] * 7)) as client:
        r = client.post("/classify", content=b"distance=12cm garbage{")
    assert r.status_code == 200


def test_concurrent_classify_gets_409_without_blocking(settings):
    gate = threading.Event()
    camera = FakeCamera([P] * 7, gate=gate)
    with client_for(settings, camera) as client:
        first: dict = {}
        worker = threading.Thread(target=lambda: first.setdefault("r", client.post("/classify")))
        worker.start()
        assert camera.entered.wait(5), "first burst never started"

        second = client.post("/classify")  # gate still closed: first burst is mid-capture
        assert second.status_code == 409
        assert second.json()["error"] == "busy"
        assert camera.bursts == 1, "second request must not start its own burst"

        # /health must not block behind the burst either.
        assert client.get("/health").status_code == 200

        gate.set()
        worker.join(5)
        assert first["r"].status_code == 200
        assert first["r"].json()["category"] == "plastic"
        assert client.get("/stats").json()["rejected_busy"] == 1

        # Lock released: the next request goes through.
        gate.set()
        assert client.post("/classify").status_code == 200


# --- frame saving ----------------------------------------------------------------


def client_with_saver(settings, camera, saver, detector=None):
    app = create_app(
        settings,
        camera_factory=lambda s: camera,
        detector_factory=lambda s: detector or FakeDetector(),
        image_loader=FakeLoader(),
        frame_saver=saver,
    )
    return TestClient(app)


def test_classify_hands_burst_to_frame_saver(settings):
    saved = []
    camera = FakeCamera([P, P, P, P, P, A, None])
    with client_with_saver(settings, camera, lambda *args: saved.append(args)) as client:
        client.post("/classify")
    (frames, picks, outcome), = saved
    assert frames == camera.frames
    assert [p.category if p else None for p in picks] == ["plastic"] * 5 + ["paper", None]
    assert outcome.category == "plastic"


def test_frame_saver_skips_empty_burst_and_keeps_raw_frames_on_detector_crash(settings):
    saved = []
    saver = lambda *args: saved.append(args)  # noqa: E731
    with client_with_saver(settings, FakeCamera(alive=False), saver) as client:
        client.post("/classify")
    assert saved == []
    with client_with_saver(settings, FakeCamera([P] * 7), saver, FakeDetector(fail=True)) as client:
        client.post("/classify")
    (frames, picks, _), = saved
    assert len(frames) == 7 and picks == [None] * 7


def test_frame_saver_failure_still_returns_answer(settings):
    def broken(*args):
        raise OSError("disk full")

    with client_with_saver(settings, FakeCamera([P] * 7), broken) as client:
        r = client.post("/classify")
    assert r.status_code == 200
    assert r.json()["category"] == "plastic"


def test_save_burst_writes_raw_and_annotated_frames(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from wastevision.snapshots import save_burst

    frames = [np.zeros((48, 64, 3), dtype=np.uint8)] * 2
    picks = [Detection("paper", "paper", 0.83, (5.0, 5.0, 30.0, 30.0)), None]
    outcome = tally([("paper", 0.83), (None, 0.0)], Settings(min_valid_votes=1, min_margin=1))
    out = save_burst(tmp_path, frames, picks, outcome, settings=Settings())
    assert out.parent == tmp_path
    assert out.name.endswith("_paper_vote")
    assert sorted(f.name for f in out.iterdir()) == ["01_paper-0.83.jpg", "01_raw.jpg", "02_none.jpg", "02_raw.jpg"]


# --- /health ---------------------------------------------------------------------


def test_health_ok(settings):
    with client_for(settings, FakeCamera([P] * 7)) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["camera"] is True
    assert body["model"] == settings.model_path
    assert body["uptime_s"] >= 0


def test_health_when_camera_down_at_boot(settings):
    with client_for(settings, FakeCamera(alive=False)) as client:
        r = client.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["camera"] is False
    assert body["status"] == "degraded"
    assert set(body) == {"status", "model", "camera", "uptime_s"}


def test_health_reports_camera_that_died_mid_session(settings):
    camera = FakeCamera([P] * 7)
    with client_for(settings, camera) as client:
        assert client.get("/health").status_code == 200
        # Unplugged USB webcams often still report isOpened() but deliver no frames.
        camera.frames = []
        client.post("/classify")
        assert client.get("/health").json()["camera"] is False
        camera.frames = [P] * 7
        client.post("/classify")
        assert client.get("/health").json()["camera"] is True


# --- /simulate -------------------------------------------------------------------


def test_simulate_json_path_matches_classify_schema(settings):
    with client_for(settings, FakeCamera([P] * 7)) as client:
        classify = client.post("/classify").json()
        sim = client.post("/simulate", json={"image_path": "photo.jpg"})
    assert sim.status_code == 200
    assert set(sim.json()) == set(classify) == SCHEMA
    assert sim.json()["category"] == "paper"
    assert sim.json()["frames_used"] == settings.burst_size


def test_simulate_upload_matches_classify_schema(settings):
    with client_for(settings, FakeCamera([P] * 7)) as client:
        r = client.post("/simulate", files={"file": ("x.jpg", b"\xff\xd8fakejpeg", "image/jpeg")})
    assert r.status_code == 200
    assert set(r.json()) == SCHEMA
    assert r.json()["decided_by"] == "vote"


def test_simulate_unreadable_image_is_400(settings):
    with client_for(settings, FakeCamera()) as client:
        assert client.post("/simulate", json={"image_path": "missing.txt"}).status_code == 400


def test_simulate_bad_body_is_422(settings):
    with client_for(settings, FakeCamera()) as client:
        assert client.post("/simulate", json={"wrong": 1}).status_code == 422
        assert client.post("/simulate", content=b"not json").status_code == 422


def test_simulate_does_not_touch_camera_or_csv(settings):
    camera = FakeCamera([P] * 7)
    with client_for(settings, camera) as client:
        client.post("/simulate", json={"image_path": "a.jpg"})
    assert camera.bursts == 0
    with open(settings.log_path, newline="") as fh:
        assert list(csv.DictReader(fh)) == []


# --- /stats ----------------------------------------------------------------------


def test_stats_totals(settings):
    camera = FakeCamera([P] * 7)
    with client_for(settings, camera) as client:
        client.post("/classify")
        client.post("/classify")
        camera.alive = False
        client.post("/classify")
        client.post("/simulate", json={"image_path": "a.jpg"})
        stats = client.get("/stats").json()
    assert stats["requests"] == 4
    assert stats["per_category"] == {"plastic": 2, "paper": 1, "organic": 1}
    assert stats["fallback_rate"] == pytest.approx(0.25)
    assert stats["simulated"] == 1
    assert stats["empty_bursts"] == 1
    assert stats["latency_mean_ms"] >= 0
    assert stats["latency_p95_ms"] >= 0


def test_stats_empty_session(settings):
    with client_for(settings, FakeCamera()) as client:
        stats = client.get("/stats").json()
    assert stats["requests"] == 0
    assert stats["fallback_rate"] == 0.0
    assert stats["latency_p95_ms"] == 0.0


def test_percentile_nearest_rank():
    values = sorted(float(v) for v in range(1, 101))
    assert percentile(values, 95) == 95.0
    assert percentile([7.0], 95) == 7.0


def test_latency_window_is_bounded():
    stats = SessionStats(latency_window=3)
    outcome = tally([], Settings())
    for ms in (1000.0, 1.0, 2.0, 3.0):
        stats.record(outcome, ms, simulated=False)
    assert stats.snapshot()["latency_p95_ms"] == 3.0  # the 1000 ms outlier aged out


# --- lifecycle -------------------------------------------------------------------


def test_camera_released_on_shutdown(settings):
    camera = FakeCamera([P] * 7)
    with client_for(settings, camera):
        assert not camera.closed
    assert camera.closed


def test_model_and_camera_opened_once(settings):
    opened = {"camera": 0, "detector": 0}

    def cam_factory(s):
        opened["camera"] += 1
        return FakeCamera([P] * 7)

    def det_factory(s):
        opened["detector"] += 1
        return FakeDetector()

    app = create_app(settings, camera_factory=cam_factory, detector_factory=det_factory, image_loader=FakeLoader())
    with TestClient(app) as client:
        for _ in range(3):
            client.post("/classify")
    assert opened == {"camera": 1, "detector": 1}
