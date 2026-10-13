"""Write a classified burst to disk so misclassifications can be inspected afterwards."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from wastevision.settings import CATEGORY_COLORS_BGR, Settings
from wastevision.voting import VoteOutcome

# cv2 is imported inside the functions so importing this module stays cheap for tests.
if TYPE_CHECKING:
    import numpy as np

    from wastevision.detector import Detection


def draw_detection(frame: np.ndarray, det: Detection | None, settings: Settings) -> np.ndarray:
    """Copy of `frame` with `det`'s box and label drawn on it; an unmarked copy if det is None."""
    import cv2

    view = frame.copy()
    if det is not None:
        color = CATEGORY_COLORS_BGR[det.category]
        x1, y1, x2, y2 = (int(v) for v in det.box)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, settings.preview_box_thickness)
        label = f"{det.category} ({det.raw_class} {det.confidence:.2f})"
        y = max(y1 - settings.preview_text_margin_px, settings.preview_status_y_px)
        cv2.putText(
            view, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, settings.preview_font_scale, color, settings.preview_text_thickness
        )
    return view


def save_burst(
    directory: str | Path,
    frames: Sequence[np.ndarray],
    picks: Sequence[Detection | None],
    outcome: VoteOutcome,
    *,
    settings: Settings,
) -> Path:
    """Write one folder per burst, named <time>_<category>_<decided_by>.

    Each frame is saved twice: NN_raw.jpg, untouched so it can be relabelled and added to
    the training set, and NN_<verdict>.jpg with the chosen detection drawn, to see what
    the model actually looked at (often clutter behind the item, not the item itself).
    """
    import cv2

    # Milliseconds in the name: two bursts in the same second must not share a folder.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    out = Path(directory) / f"{stamp}_{outcome.category}_{outcome.decided_by}"
    out.mkdir(parents=True, exist_ok=True)
    for i, (frame, pick) in enumerate(zip(frames, picks), start=1):
        verdict = f"{pick.category}-{pick.confidence:.2f}" if pick else "none"
        cv2.imwrite(str(out / f"{i:02d}_raw.jpg"), frame)
        cv2.imwrite(str(out / f"{i:02d}_{verdict}.jpg"), draw_detection(frame, pick, settings))
    return out
