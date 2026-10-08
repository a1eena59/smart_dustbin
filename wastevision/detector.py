"""Ultralytics YOLO wrapper that emits waste-category detections."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from wastevision.mapping import check_model_classes, resolve
from wastevision.settings import PROJECT_ROOT, Settings

if TYPE_CHECKING:
    import numpy as np


def resolve_model_path(model_path: str) -> str:
    """Resolve a relative weights path against the working directory, then the project root.

    Bare names that exist in neither place (e.g. "yolov8n.pt") are passed through so
    Ultralytics can download them.
    """
    path = Path(model_path)
    if path.is_absolute() or path.exists():
        return str(path)
    in_project = PROJECT_ROOT / path
    if in_project.exists():
        return str(in_project)
    if len(path.parts) > 1:
        raise FileNotFoundError(f"model weights not found: {model_path} (also looked in {PROJECT_ROOT})")
    return model_path


@dataclass(frozen=True)
class Detection:
    """One object the model found that maps to a waste category."""

    category: str
    raw_class: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

    @property
    def centre(self) -> tuple[float, float]:
        """Box centre (x, y) in pixels."""
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class WasteDetector:
    """Runs YOLO on a frame and keeps only detections that resolve to a waste category."""

    def __init__(self, settings: Settings) -> None:
        # Imported here so tests, `run.py bridge`, and the voting module work without torch.
        from ultralytics import YOLO

        self._settings = settings
        self._model = YOLO(resolve_model_path(settings.model_path))
        self._names: dict[int, str] = self._model.names
        # Fail at startup, not by silently voting "fallback" forever.
        for message in check_model_classes(self._names, settings.native_waste_classes):
            warnings.warn(message, stacklevel=2)

    @property
    def class_names(self) -> dict[int, str]:
        """The loaded model's class-id -> name table."""
        return dict(self._names)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return waste detections in `frame`, highest confidence first."""
        result = self._model.predict(
            frame,
            imgsz=self._settings.imgsz,
            conf=self._settings.conf_floor,
            verbose=False,
        )[0]
        # Classification-only weights produce no boxes; treat that as "nothing found".
        if result.boxes is None:
            return []

        detections: list[Detection] = []
        boxes = result.boxes
        for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
            raw = self._names[int(cls)]
            category = resolve(raw, self._settings.native_waste_classes)
            if category is None:
                continue
            detections.append(Detection(category, raw, float(conf), tuple(xyxy)))
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    @staticmethod
    def best(detections: list[Detection], tie_window: float = Settings.best_tie_window) -> Detection | None:
        """Pick the item on the platform: top confidence, with near-ties broken by largest box."""
        if not detections:
            return None
        top = max(d.confidence for d in detections)
        # The platform item sits closest to the lens, so among near-equal candidates the
        # biggest box is the one we mean; small boxes are usually background clutter.
        contenders = [d for d in detections if top - d.confidence <= tie_window]
        return max(contenders, key=lambda d: d.area)
