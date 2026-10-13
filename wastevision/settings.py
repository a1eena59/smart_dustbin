"""Single source of truth for configuration and the category contract."""

import os
from dataclasses import dataclass, fields
from pathlib import Path

# Relative model paths are resolved against this, so run.py works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Stock COCO weights, for comparison runs (`run.py --coco`). Ultralytics downloads them on demand.
COCO_MODEL = "yolov8n.pt"

# Index order is a wire contract: the ESP32 firmware maps lid numbers to these
# indices and the training dataset's class IDs must match. Append only; never reorder.
CATEGORIES: tuple[str, ...] = ("plastic", "paper", "organic")
CATEGORY_INDEX: dict[str, int] = {name: i for i, name in enumerate(CATEGORIES)}

ENV_PREFIX = "WV_"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Settings:
    """Runtime configuration. `Settings()` is pure defaults; `Settings.from_env()` applies WV_* overrides."""

    # Model
    # Custom weights trained on plastic/paper/organic directly (Colab, 2026-10-08).
    model_path: str = "models/waste_v1/best.pt"
    imgsz: int = 640
    conf_floor: float = 0.40
    # True: the model already outputs CATEGORIES names. False: COCO weights via the bridge.
    native_waste_classes: bool = True
    # Two detections whose confidences differ by less than this are treated as a tie,
    # because YOLO confidence differences that small are noise, not signal.
    best_tie_window: float = 0.05

    # Camera
    camera_index: int = 0
    # 640x480 because the model infers at 640 anyway, and many USB webcams only reach
    # full frame rate at 720p+ in MJPG; in YUY2 they drop to ~10 fps, which slows every grab.
    capture_width: int = 640
    capture_height: int = 480
    # Ask the driver for the smallest buffer; many Windows/USB backends ignore it,
    # which is why the warmup flush below exists as well.
    camera_buffer_size: int = 1
    burst_size: int = 7
    # Warmup = grab() (no decode) until BOTH warmup_ms has passed and discard_frames
    # grabs are done. Time-based because the driver queue depth is unknown: draining
    # it by count alone left 6 of 7 frames stale.
    warmup_ms: int = 400
    discard_frames: int = 8
    # grab() once before each voting read() so a frame queued while we were busy is dropped.
    flush_between_frames: bool = True
    burst_timeout_s: float = 3.0
    max_consecutive_read_failures: int = 2

    # Voting
    min_valid_votes: int = 3
    min_margin: int = 2
    fallback_category: str = "organic"

    # Logging
    log_path: str = "logs/decisions.csv"
    # When set, every /classify burst is written here (raw + annotated frames) for
    # debugging misclassifications. Empty = off: each burst is ~1 MB.
    save_frames_dir: str = ""

    # HTTP server (ESP32 bridge)
    # 0.0.0.0, not localhost: the ESP32 is another machine on the LAN.
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    # /stats latency percentiles cover the most recent N requests, so memory stays
    # bounded on a bin that runs for weeks.
    stats_latency_window: int = 1000

    # CLI preview (watch mode only)
    preview_box_thickness: int = 2
    preview_font_scale: float = 0.7
    preview_text_thickness: int = 2
    preview_text_margin_px: int = 8
    preview_status_y_px: int = 30
    preview_retry_wait_ms: int = 100

    def __post_init__(self) -> None:
        if self.fallback_category not in CATEGORY_INDEX:
            raise ValueError(f"fallback_category must be one of {CATEGORIES}, got {self.fallback_category!r}")
        if not 0.0 <= self.conf_floor <= 1.0:
            raise ValueError(f"conf_floor must be in [0, 1], got {self.conf_floor}")
        if self.burst_size < 1 or self.discard_frames < 0:
            raise ValueError("burst_size must be >= 1 and discard_frames >= 0")
        # A margin of 0 would let a tie be "won" by whichever category sorts first.
        if self.min_margin < 1:
            raise ValueError(f"min_margin must be >= 1, got {self.min_margin}")
        if self.min_valid_votes < 1:
            raise ValueError(f"min_valid_votes must be >= 1, got {self.min_valid_votes}")
        # Otherwise every burst silently falls back and the bin looks "stuck on organic".
        if self.burst_size < self.min_valid_votes:
            raise ValueError(
                f"burst_size ({self.burst_size}) < min_valid_votes ({self.min_valid_votes}): no vote could ever succeed"
            )
        if self.burst_timeout_s <= 0:
            raise ValueError(f"burst_timeout_s must be > 0, got {self.burst_timeout_s}")
        if self.warmup_ms < 0:
            raise ValueError(f"warmup_ms must be >= 0, got {self.warmup_ms}")
        # Warmup counts against the burst timeout; if it eats all of it, no frame is ever collected.
        if self.warmup_ms / 1000 >= self.burst_timeout_s:
            raise ValueError(
                f"warmup_ms ({self.warmup_ms}) must be shorter than burst_timeout_s ({self.burst_timeout_s}s)"
            )
        if self.capture_width < 1 or self.capture_height < 1:
            raise ValueError("capture_width and capture_height must be positive")
        if not 1 <= self.server_port <= 65535:
            raise ValueError(f"server_port must be 1-65535, got {self.server_port}")
        if self.stats_latency_window < 1:
            raise ValueError(f"stats_latency_window must be >= 1, got {self.stats_latency_window}")

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "Settings":
        """Build settings from defaults overridden by WV_<FIELD_NAME> environment variables."""
        env = os.environ if environ is None else environ
        overrides: dict[str, object] = {}
        for f in fields(cls):
            raw = env.get(ENV_PREFIX + f.name.upper())
            if raw is None:
                continue
            overrides[f.name] = _parse(raw, type(f.default), f.name)
        return cls(**overrides)


def _parse(raw: str, kind: type, name: str) -> object:
    value = raw.strip()
    # bool("false") is True, so booleans need explicit parsing.
    if kind is bool:
        lowered = value.lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise ValueError(f"{ENV_PREFIX}{name.upper()}: expected a boolean, got {raw!r}")
    try:
        return kind(value)
    except ValueError as exc:
        raise ValueError(f"{ENV_PREFIX}{name.upper()}: expected {kind.__name__}, got {raw!r}") from exc


# BGR colours for preview boxes, indexed like CATEGORIES.
CATEGORY_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "plastic": (255, 160, 0),
    "paper": (200, 200, 200),
    "organic": (0, 200, 0),
}
