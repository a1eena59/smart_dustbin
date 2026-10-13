"""Command-line test harness for the wastevision package."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from wastevision.mapping import BLOCKLIST, bridge_by_category
from wastevision.settings import CATEGORIES, CATEGORY_COLORS_BGR, COCO_MODEL, Settings
from wastevision.voting import VoteOutcome

# cv2/ultralytics are imported inside the commands that need them so `bridge`
# (and --help) work on a machine with neither installed.
if TYPE_CHECKING:
    import numpy as np

    from wastevision.detector import Detection, WasteDetector
    from wastevision.logging_csv import DecisionLog

WINDOW = "wastevision"
KEY_SPACE, KEY_ESC = 32, 27


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    # Defaults are None so unset flags fall through to WV_* env vars, then dataclass defaults.
    common.add_argument("--model", dest="model_path", help="YOLO weights path (default models/waste_v1/best.pt)")
    common.add_argument(
        "--coco", action="store_true", help=f"use stock {COCO_MODEL} through the COCO bridge, for comparison"
    )
    common.add_argument("--camera", dest="camera_index", type=int, help="OpenCV camera index")
    common.add_argument("--conf", dest="conf_floor", type=float, help="detection confidence floor, 0-1")

    parser = argparse.ArgumentParser(description="Waste-bin classifier test CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("classify", parents=[common], help="capture one burst, vote, print, log")
    p.add_argument("--no-log", action="store_true", help="don't append to the decision log")
    p.add_argument("--save-frames", metavar="DIR", help="write the burst's voting frames as numbered JPGs")

    p = sub.add_parser("watch", parents=[common], help="live preview; SPACE classifies, Q/ESC quits")
    p.add_argument("--no-log", action="store_true", help="don't append to the decision log")

    p = sub.add_parser("image", parents=[common], help="vote on a still image (no camera needed)")
    p.add_argument("--path", required=True, help="image file to classify")
    p.add_argument("--log", action="store_true", help="also append the result to the decision log")

    sub.add_parser("bridge", parents=[common], help="show which COCO objects map to which bin")

    p = sub.add_parser("serve", parents=[common], help="HTTP service for the ESP32")
    p.add_argument("--host", dest="server_host", help="bind address (default 0.0.0.0 = all interfaces)")
    p.add_argument("--port", dest="server_port", type=int, help="TCP port (default 8000)")
    p.add_argument(
        "--save-frames", dest="save_frames_dir", metavar="DIR", help="save every burst (raw + annotated) under DIR"
    )
    return parser


def load_settings(args: argparse.Namespace) -> Settings:
    overrides = {
        key: getattr(args, key)
        for key in ("model_path", "camera_index", "conf_floor", "server_host", "server_port", "save_frames_dir")
        if getattr(args, key, None) is not None
    }
    if args.coco:
        overrides.setdefault("model_path", COCO_MODEL)
        overrides["native_waste_classes"] = False
    return replace(Settings.from_env(), **overrides)


def print_result(outcome: VoteOutcome, picks: list[Detection | None], elapsed_ms: float, total: int) -> None:
    for i, pick in enumerate(picks, start=1):
        seen = f"{pick.raw_class:<12} {pick.confidence:.2f} -> {pick.category}" if pick else "(nothing)"
        print(f"  frame {i}: {seen}")
    votes = " | ".join(f"{c} {outcome.votes[c]}" for c in CATEGORIES)
    print(f"Result : {outcome.category.upper()} (index {outcome.index})")
    print(f"Method : {outcome.decided_by}" + (f", mean conf {outcome.confidence:.2f}" if outcome.decided_by == "vote" else ""))
    print(f"Votes  : {votes}   ({outcome.frames_used}/{total} frames voted)")
    print(f"Time   : {elapsed_ms:.0f} ms")


def open_log(settings: Settings, enabled: bool) -> DecisionLog | None:
    if not enabled:
        return None
    from wastevision.logging_csv import DecisionLog

    return DecisionLog(settings.log_path)


def run_burst(
    cam, detector: WasteDetector, settings: Settings, log: DecisionLog | None, save_dir: str | None = None
) -> tuple[VoteOutcome, list[Detection | None]]:
    from wastevision.pipeline import classify_frames

    start = time.perf_counter()
    frames = cam.capture_burst()
    capture_ms = (time.perf_counter() - start) * 1000
    stats = cam.last_stats
    offsets = ", ".join(f"{t:.0f}" for t in stats.frame_offsets_ms)
    print(
        f"Capture: {len(frames)} frames in {capture_ms:.0f} ms "
        f"(warmup {stats.warmup_grabs} grabs / {stats.warmup_ms:.0f} ms; frames at {offsets or '-'} ms"
        + ("; camera reopened" if stats.reopened else "")
        + ")"
    )
    if not frames:
        print("WARNING: camera returned no frames; decision is the fallback.", file=sys.stderr)
    if save_dir and frames:
        save_frames(frames, save_dir)
    outcome, picks = classify_frames(detector, frames, settings)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print_result(outcome, picks, elapsed_ms, settings.burst_size)
    if log:
        log.append(outcome, elapsed_ms)
    return outcome, picks


def save_frames(frames: list[np.ndarray], directory: str) -> None:
    import cv2

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    # Timestamp prefix so repeated runs into the same folder never overwrite or mix bursts.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for i, frame in enumerate(frames, start=1):
        cv2.imwrite(str(out / f"{stamp}_{i:02d}.jpg"), frame)
    print(f"Saved  : {len(frames)} frames as {out / stamp}_01..{len(frames):02d}.jpg")


def cmd_classify(args: argparse.Namespace, settings: Settings) -> int:
    from wastevision.camera import BurstCamera
    from wastevision.detector import WasteDetector

    detector = WasteDetector(settings)
    log = open_log(settings, not args.no_log)
    try:
        with BurstCamera(settings) as cam:
            run_burst(cam, detector, settings, log, save_dir=args.save_frames)
    finally:
        if log:
            log.close()
    return 0


def cmd_image(args: argparse.Namespace, settings: Settings) -> int:
    import cv2

    from wastevision.detector import WasteDetector
    from wastevision.pipeline import classify_frames

    frame = cv2.imread(args.path)
    if frame is None:
        print(f"error: cannot read image {args.path!r}", file=sys.stderr)
        return 2
    detector = WasteDetector(settings)
    start = time.perf_counter()
    # Same frame repeated so the result goes through exactly the code path a real burst does.
    outcome, picks = classify_frames(detector, [frame] * settings.burst_size, settings)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print_result(outcome, picks, elapsed_ms, settings.burst_size)
    log = open_log(settings, args.log)
    if log:
        with log:
            log.append(outcome, elapsed_ms)
    return 0


def annotate(frame: np.ndarray, det: Detection | None, status: str, settings: Settings) -> np.ndarray:
    import cv2

    from wastevision.snapshots import draw_detection

    view = draw_detection(frame, det, settings)
    cv2.putText(
        view,
        status,
        (settings.preview_text_margin_px, settings.preview_status_y_px),
        cv2.FONT_HERSHEY_SIMPLEX,
        settings.preview_font_scale,
        CATEGORY_COLORS_BGR["paper"],
        settings.preview_text_thickness,
    )
    return view


def cmd_watch(args: argparse.Namespace, settings: Settings) -> int:
    import cv2

    from wastevision.camera import BurstCamera
    from wastevision.detector import WasteDetector

    detector = WasteDetector(settings)
    log = open_log(settings, not args.no_log)
    status = "SPACE: classify   Q/ESC: quit"
    failures = 0
    try:
        with BurstCamera(settings) as cam:
            if not cam.is_alive:
                print(f"error: camera {settings.camera_index} did not open", file=sys.stderr)
                return 1
            print(status)
            while True:
                frame = cam.read_frame()
                if frame is None:
                    failures += 1
                    if failures >= settings.max_consecutive_read_failures:
                        if not cam.reopen():
                            print("error: camera lost and could not be reopened", file=sys.stderr)
                            return 1
                        failures = 0
                    key = cv2.waitKey(settings.preview_retry_wait_ms) & 0xFF
                else:
                    failures = 0
                    det = WasteDetector.best(detector.detect(frame), settings.best_tie_window)
                    cv2.imshow(WINDOW, annotate(frame, det, status, settings))
                    key = cv2.waitKey(1) & 0xFF
                    # Closing the window with the mouse should end the session, not hang it.
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break

                if key in (ord("q"), KEY_ESC):
                    break
                if key == KEY_SPACE:
                    print("-" * 40)
                    outcome, _ = run_burst(cam, detector, settings, log)
                    status = f"LAST: {outcome.category.upper()} via {outcome.decided_by}   SPACE: again   Q: quit"
    finally:
        cv2.destroyAllWindows()
        if log:
            log.close()
    return 0


def cmd_bridge(args: argparse.Namespace, settings: Settings) -> int:
    if settings.native_waste_classes:
        print(f"NOTE: the active model ({settings.model_path}) outputs {', '.join(CATEGORIES)} directly,")
        print("so the bridge below only applies with --coco.\n")
    print("Hold one of these up to the camera:\n")
    for index, (category, names) in enumerate(bridge_by_category().items()):
        print(f"  {index} {category:<8} <- {', '.join(names)}")
    print(f"\nNever classified (blocklist): {', '.join(sorted(BLOCKLIST))}")
    print("Anything else is ignored. Frames with nothing usable abstain from the vote.")
    return 0


def lan_ip() -> str | None:
    import socket

    # Connecting a UDP socket sends nothing; it just makes the OS pick the outbound
    # interface, whose address is the one the ESP32 should call.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return None


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import logging

    import uvicorn

    from wastevision.server import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ip = lan_ip()
    if settings.server_host == "0.0.0.0" and ip:
        print(f"ESP32 should call: http://{ip}:{settings.server_port}/classify")
    # One process only: the camera is a single device and the burst lock is per-process.
    uvicorn.run(create_app(settings), host=settings.server_host, port=settings.server_port)
    return 0


COMMANDS = {
    "classify": cmd_classify,
    "watch": cmd_watch,
    "image": cmd_image,
    "bridge": cmd_bridge,
    "serve": cmd_serve,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args)
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.command != "bridge":
        mode = "native" if settings.native_waste_classes else "COCO bridge"
        print(f"model: {settings.model_path} ({mode})")
    try:
        return COMMANDS[args.command](args, settings)
    except (ValueError, FileNotFoundError) as exc:
        # Model/class mismatches from WasteDetector: a clear one-liner beats a torch traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
