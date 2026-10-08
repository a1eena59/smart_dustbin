# wastevision

The Python half of an automated waste-sorting bin. A webcam photographs the item on the platform, YOLOv8 detects it, a burst of frames votes, and the result is one of three bins:

| index | category |
|------:|----------|
| 0 | plastic |
| 1 | paper |
| 2 | organic |

**This order is a contract.** The ESP32 firmware and the training dataset both rely on it. It is defined once, in `wastevision/settings.py` (`CATEGORIES`). You can add new categories at the end, but never reorder the existing ones.

## Install

Requires Python 3.11.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux or macOS, run `source .venv/bin/activate` instead of the second line.

## Verify

```powershell
pytest -v                      # no camera, weights or network needed
python run.py bridge           # no model or camera needed
python run.py image --path some_photo.jpg   # no camera needed
```

## Model

By default the bin uses `models/waste_v1/best.pt`, a YOLOv8n fine-tuned on Colab whose classes are exactly `0=plastic, 1=paper, 2=organic`. Because it outputs the categories directly, it runs in **native mode**: the COCO bridge isn't used.

To compare against stock COCO weights through the bridge, add `--coco` to any command. Ultralytics downloads `yolov8n.pt` on first use.

When a model loads, its class names are checked against the mode. If no class could ever produce a category (for example, COCO weights in native mode), the command stops with a clear error instead of quietly falling back on every burst.

## Commands

| command | what it does |
|---|---|
| `python run.py classify` | Captures one burst, votes, prints the result and appends it to the log. |
| `python run.py watch` | Opens a live preview with detection boxes. SPACE classifies; Q or ESC quits. |
| `python run.py image --path x.jpg` | Repeats one still `burst_size` times and runs it through the same vote. Add `--log` to record the result. |
| `python run.py bridge` | Shows which COCO objects count as which bin. |

All four commands accept `--model PATH`, `--coco`, `--camera N` and `--conf 0.xx`. `classify` and `watch` also accept `--no-log`. `classify` also accepts `--save-frames DIR`, which writes the voting frames as `DIR/<timestamp>_01.jpg` and so on.

Precedence, highest first: **CLI flag → `WV_*` environment variable → default.**

## Configuration

Every setting is a field of `Settings` in `wastevision/settings.py`. You can override any of them with `WV_<FIELD_NAME_UPPER>`, for example `WV_BURST_SIZE=9` or `WV_NATIVE_WASTE_CLASSES=true`. Invalid combinations are rejected at startup. For example, `burst_size < min_valid_votes` is rejected because no vote could ever pass.

| field | default | meaning |
|---|---|---|
| model_path | models/waste_v1/best.pt | weights file; a relative path is tried against the working directory first, then the project root |
| imgsz | 640 | inference size |
| conf_floor | 0.40 | below this, a frame abstains |
| camera_index | 0 | OpenCV device index |
| capture_width / capture_height | 640 / 480 | requested capture resolution |
| burst_size | 7 | frames that vote |
| warmup_ms | 400 | minimum time spent flushing the driver queue before collecting |
| discard_frames | 8 | minimum number of frames flushed before collecting |
| flush_between_frames | true | drop one queued frame before each voting read and each preview read |
| burst_timeout_s | 3.0 | hard stop for a burst |
| min_valid_votes | 3 | the minimum number of non-abstaining frames |
| min_margin | 2 | the number of votes the winner must lead the runner-up by |
| fallback_category | organic | used when the vote is not decisive |
| log_path | logs/decisions.csv | append-only decision log |
| native_waste_classes | true | the model already outputs plastic/paper/organic, so the COCO bridge is skipped; set to false only for COCO weights |

## How a decision is made

1. **Capture.** `BurstCamera` first flushes the driver's queue. It calls `grab()` (which discards a frame without decoding it) until both `warmup_ms` have passed and `discard_frames` frames have been dropped. Only then does it collect `burst_size` frames, dropping one queued frame before each read. It stops hard at `burst_timeout_s`. If the camera is dead, the result is an empty burst, never an exception. To check the frames are current, run `python run.py classify --save-frames shots` and look at the saved JPGs.
2. **Per frame.** `WasteDetector.detect` keeps only detections that `mapping.resolve` accepts. `WasteDetector.best` picks one: the highest confidence wins, and among detections within 0.05 of the top, the largest box wins, because the item on the platform is closer than the clutter behind it.
3. **Vote.** `voting.tally` is pure stdlib. A frame abstains if it found nothing or its confidence is under the floor. The winner is accepted only with at least `min_valid_votes` votes and a lead of at least `min_margin` over the runner-up. Otherwise the result is `fallback_category`, with `decided_by="fallback"`.

## Swapping in new weights

Train with classes `0=plastic, 1=paper, 2=organic` (same names, same order). Put the weights in `models/<name>/best.pt` and either change the `model_path` default in `settings.py` or try them first:

```powershell
python run.py watch --model models\waste_v2\best.pt
```

## Layout

```
wastevision/
  settings.py     config + category contract
  mapping.py      COCO -> category bridge and blocklist
  detector.py     YOLO wrapper (ultralytics imported lazily)
  voting.py       pure decision logic
  pipeline.py     detector + voting glue (reuse this in the HTTP server)
  camera.py       burst capture
  logging_csv.py  append-only CSV log
  tests/
models/
  waste_v1/best.pt  default weights (committed to git)
training/           dataset preparation
run.py              CLI
```
