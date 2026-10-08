"""Convert a Roboflow YOLOv8 export into a 3-class (plastic/paper/organic) training set.

    python training/prepare_dataset.py --inspect "Trash Detection.v1i.yolov8.zip"
    python training/prepare_dataset.py --source "Trash Detection.v1i.yolov8.zip" --out data/datasets/v1 --dry-run
    python training/prepare_dataset.py --source "Trash Detection.v1i.yolov8.zip" --out data/datasets/v1

The source may be the export directory or the zip itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath

# Run as a plain script, so the repo root isn't on sys.path; we need it to import the
# one true category order rather than duplicating it here.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wastevision.settings import CATEGORIES, CATEGORY_INDEX  # noqa: E402

# ---------------------------------------------------------------------------
# MAPPING TABLE — edit this.
# Keys are source class names normalised by normalize_name(): lowercase, with
# spaces/underscores turned into hyphens ("Paper Bag" -> "paper-bag").
# Value is a target category, or None to drop the instance.
# Any source class not listed here is UNMAPPED and blocks conversion.
# ---------------------------------------------------------------------------
CLASS_MAP: dict[str, str | None] = {
    # paper
    "paper": "paper",
    "cardboard": "paper",
    # Tetra Pak-style cartons are paper/plastic/aluminium laminate; move to None if
    # your local rules don't accept them as paper.
    "carton": "paper",
    "paper-bag": "paper",
    "newspaper": "paper",
    # plastic
    "plastic": "plastic",
    "plastic-bottle": "plastic",
    "plastic-bag": "plastic",
    "plastic-cup": "plastic",
    # Many datasets label metal crown caps as bottle-cap too; check samples if unsure.
    "bottle-cap": "plastic",
    "styrofoam": "plastic",
    # organic
    "organic": "organic",
    "biodegradable": "organic",
    "food-waste": "organic",
    "compost": "organic",
    "organic-waste": "organic",
    "leaves": "organic",
    "vegetable": "organic",
    "fruit": "organic",
    # no compartment for these
    "metal": None,
    "glass": None,
    "battery": None,
    "e-waste": None,
    "textile": None,
    "trash": None,
    "mixed": None,
}

# Roboflow calls it valid/, Ultralytics convention is val/.
SPLIT_MAP: dict[str, str] = {"train": "train", "valid": "val", "val": "val", "test": "test"}
IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
IMBALANCE_WARN_RATIO = 3.0
COORD_DECIMALS = 6
DEFAULT_OUT = "data/datasets/v1"
# Windows' 260-char MAX_PATH: some Roboflow names are ~200 chars before the folder
# path is even added. 80 leaves room for deep checkouts and Ultralytics' cache files.
MAX_STEM_LEN = 80
STEM_HASH_LEN = 12
# Antivirus and editor file watchers briefly hold handles on freshly written files,
# and Windows refuses to rename a directory while any handle inside it is open.
RENAME_ATTEMPTS = 20
RENAME_BACKOFF_S = 0.5

DROP, UNMAPPED = "DROP", "UNMAPPED"


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def normalize_name(name: str) -> str:
    """Canonical form for CLASS_MAP lookups: lowercase, spaces/underscores -> hyphens."""
    return re.sub(r"[\s_]+", "-", name.strip().lower())


def propose_target(name: str, class_map: Mapping[str, str | None] = CLASS_MAP) -> str:
    """Return a category name, DROP, or UNMAPPED for a source class name."""
    key = normalize_name(name)
    if key not in class_map:
        return UNMAPPED
    target = class_map[key]
    return DROP if target is None else target


def build_index_map(
    source_names: list[str], class_map: Mapping[str, str | None] = CLASS_MAP
) -> tuple[dict[int, int | None], list[str]]:
    """Map source class index -> target index (or None to drop). Also returns unmapped names."""
    index_map: dict[int, int | None] = {}
    unmapped: list[str] = []
    for i, name in enumerate(source_names):
        proposal = propose_target(name, class_map)
        if proposal == UNMAPPED:
            unmapped.append(name)
        else:
            index_map[i] = None if proposal == DROP else CATEGORY_INDEX[proposal]
    return index_map, unmapped


def validate_class_map(class_map: Mapping[str, str | None] = CLASS_MAP) -> list[str]:
    """Return human-readable problems with the mapping table (typos in targets or keys)."""
    problems = []
    for key, target in class_map.items():
        if target is not None and target not in CATEGORY_INDEX:
            problems.append(f"{key!r} -> {target!r}: target must be one of {CATEGORIES} or None")
        if key != normalize_name(key):
            problems.append(f"{key!r}: key is not normalised; use {normalize_name(key)!r}")
    return problems


def safe_stem(stem: str, max_len: int = MAX_STEM_LEN) -> str:
    """Shorten an over-long filename stem deterministically, keeping it unique via a hash suffix."""
    if len(stem) <= max_len:
        return stem
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:STEM_HASH_LEN]
    return f"{stem[: max_len - STEM_HASH_LEN - 1]}_{digest}"


def parse_label_line(line: str) -> tuple[int, tuple[float, float, float, float]] | None:
    """Parse one YOLO line (box or polygon) into (class, (cx, cy, w, h)) clamped to the image.

    Returns None for blank lines; raises ValueError for malformed ones.
    """
    tokens = line.split()
    if not tokens:
        return None
    cls_f = float(tokens[0])
    if not cls_f.is_integer() or cls_f < 0:
        raise ValueError(f"bad class id {tokens[0]!r}")
    coords = [float(t) for t in tokens[1:]]
    if any(math.isnan(c) or math.isinf(c) for c in coords):
        raise ValueError("non-finite coordinate")

    if len(coords) == 4:
        cx, cy, w, h = coords
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        # Segmentation polygon: Ultralytics would convert this to its bounding box
        # internally anyway; doing it here keeps the written labels honest and small.
        xs, ys = coords[0::2], coords[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        raise ValueError(f"expected 4 box values or an even polygon of >=6, got {len(coords)}")

    # Roboflow exports occasionally spill a hair outside [0, 1]; Ultralytics rejects those labels.
    x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        raise ValueError("degenerate box")
    return int(cls_f), ((x1 + x2) / 2, (y1 + y2) / 2, w, h)


@dataclass
class LabelResult:
    """Outcome of rewriting one label file."""

    text: str
    kept: Counter[int] = field(default_factory=Counter)  # by target index
    dropped: int = 0
    malformed: int = 0

    @property
    def n_kept(self) -> int:
        """Number of instances written to the new label file."""
        return sum(self.kept.values())


def rewrite_label(text: str, index_map: Mapping[int, int | None]) -> LabelResult:
    """Rewrite one label file's contents to target indices, dropping rejected instances."""
    out_lines: list[str] = []
    result = LabelResult(text="")
    for line in text.splitlines():
        try:
            parsed = parse_label_line(line)
        except ValueError:
            result.malformed += 1
            continue
        if parsed is None:
            continue
        src_cls, (cx, cy, w, h) = parsed
        if src_cls not in index_map:
            # Class id beyond data.yaml's names list: corrupt label, not a mapping decision.
            result.malformed += 1
            continue
        target = index_map[src_cls]
        if target is None:
            result.dropped += 1
            continue
        result.kept[target] += 1
        d = COORD_DECIMALS
        out_lines.append(f"{target} {cx:.{d}f} {cy:.{d}f} {w:.{d}f} {h:.{d}f}")
    result.text = "\n".join(out_lines) + ("\n" if out_lines else "")
    return result


def count_source_instances(text: str) -> tuple[Counter[int], int, int]:
    """Count instances per source class in a label file. Returns (counts, polygon_lines, malformed)."""
    counts: Counter[int] = Counter()
    polygons = malformed = 0
    for line in text.splitlines():
        try:
            parsed = parse_label_line(line)
        except ValueError:
            malformed += 1
            continue
        if parsed is None:
            continue
        counts[parsed[0]] += 1
        if len(line.split()) > 5:
            polygons += 1
    return counts, polygons, malformed


def _strip_comment(line: str) -> str:
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


def _split_flow(body: str) -> list[str]:
    parts, buf, quote = [], [], None
    for ch in body:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            buf.append(ch)
        elif ch == ",":
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _unquote(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] == "'":
        return v[1:-1].replace("''", "'")
    if len(v) >= 2 and v[0] == v[-1] == '"':
        return v[1:-1].replace('\\"', '"')
    return v


def _names_from_dict(items: Iterable[tuple[str, str]]) -> list[str]:
    by_index = {int(k.strip()): _unquote(v) for k, v in items}
    if sorted(by_index) != list(range(len(by_index))):
        raise ValueError(f"names dict keys must be 0..n-1, got {sorted(by_index)}")
    return [by_index[i] for i in range(len(by_index))]


def parse_names(yaml_text: str) -> list[str]:
    """Extract the `names:` class list from a data.yaml without pyyaml.

    Supports flow lists, block lists, flow dicts and block dicts ({index: name}).
    """
    lines = [_strip_comment(line).rstrip() for line in yaml_text.splitlines()]
    for i, line in enumerate(lines):
        m = re.match(r"^names\s*:\s*(.*)$", line)
        if not m:
            continue
        rest = m.group(1).strip()
        if rest and rest[0] in "[{":
            closer = "]" if rest[0] == "[" else "}"
            j = i
            while closer not in rest:
                j += 1
                if j >= len(lines):
                    raise ValueError("unterminated names: flow collection")
                rest += " " + lines[j].strip()
            body = rest[1 : rest.rindex(closer)]
            items = _split_flow(body)
            if closer == "]":
                return [_unquote(x) for x in items]
            return _names_from_dict(tuple(x.split(":", 1)) for x in items)
        if rest:
            raise ValueError(f"unsupported names: value {rest!r}")
        seq: list[str] = []
        mapping: list[tuple[str, str]] = []
        for sub in lines[i + 1 :]:
            if not sub.strip():
                continue
            if not sub[0].isspace():
                break
            s = sub.strip()
            if s.startswith("-"):
                seq.append(_unquote(s[1:]))
            elif ":" in s:
                k, v = s.split(":", 1)
                mapping.append((k, v))
        if seq and mapping:
            raise ValueError("names: mixes list and dict entries")
        return seq if seq else _names_from_dict(mapping)
    raise ValueError("data.yaml has no names: key")


def parse_roboflow_meta(yaml_text: str) -> dict[str, str]:
    """Pull the flat key/values from the `roboflow:` block, for the manifest."""
    meta: dict[str, str] = {}
    inside = False
    for raw in yaml_text.splitlines():
        line = _strip_comment(raw).rstrip()
        if re.match(r"^roboflow\s*:\s*$", line):
            inside = True
            continue
        if inside:
            if line and not line[0].isspace():
                break
            if ":" in line:
                k, v = line.strip().split(":", 1)
                meta[k.strip()] = _unquote(v)
    return meta


def imbalance(counts: Mapping[str, int]) -> tuple[float, str, str]:
    """Return (max/min ratio, most common class, rarest class). Ratio is inf if any class is 0."""
    common = max(counts, key=lambda c: counts[c])
    rare = min(counts, key=lambda c: counts[c])
    if counts[rare] == 0:
        return math.inf, common, rare
    return counts[common] / counts[rare], common, rare


# ---------------------------------------------------------------------------
# Source access (directory or zip)
# ---------------------------------------------------------------------------


class Source:
    """Read-only view of a Roboflow export rooted at the folder containing data.yaml."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._zip: zipfile.ZipFile | None = None
        if path.is_file() and zipfile.is_zipfile(path):
            self._zip = zipfile.ZipFile(path)
            all_files = [n for n in self._zip.namelist() if not n.endswith("/")]
        elif path.is_dir():
            all_files = [p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()]
        else:
            raise FileNotFoundError(f"{path} is neither a directory nor a zip file")

        yamls = sorted((f for f in all_files if PurePosixPath(f).name == "data.yaml"), key=len)
        if not yamls:
            raise FileNotFoundError(f"no data.yaml in {path}")
        # Zips are often wrapped in one top-level folder; treat data.yaml's folder as the root.
        self._prefix = str(PurePosixPath(yamls[0]).parent)
        self._prefix = "" if self._prefix == "." else self._prefix + "/"
        self.files = [f[len(self._prefix) :] for f in all_files if f.startswith(self._prefix)]
        self.name = path.stem if self._zip else path.name

    def read_bytes(self, rel: str) -> bytes:
        """Read one file relative to the export root."""
        full = self._prefix + rel
        if self._zip:
            return self._zip.read(full)
        return (self.path / full).read_bytes()

    def read_text(self, rel: str) -> str:
        """Read one file as UTF-8 text (BOM tolerated)."""
        return self.read_bytes(rel).decode("utf-8-sig")

    def copy_to(self, rel: str, dest: Path) -> None:
        """Copy one file to dest without loading it all into memory."""
        full = self._prefix + rel
        if self._zip:
            with self._zip.open(full) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.copy2(self.path / full, dest)

    def archive_sha256(self) -> str | None:
        """Hash of the zip itself, so the manifest pins the exact download."""
        if not self._zip:
            return None
        h = hashlib.sha256()
        with self.path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()


@dataclass
class Pair:
    """An image and its (possibly missing) label file, relative to the export root."""

    image: str
    label: str | None


def find_pairs(files: Iterable[str]) -> tuple[dict[str, list[Pair]], dict[str, int]]:
    """Group images by source split and attach labels. Also returns orphan-label counts per split."""
    file_set = set(files)
    pairs: dict[str, list[Pair]] = {}
    orphans: dict[str, int] = {}
    for split in SPLIT_MAP:
        images = sorted(
            f for f in file_set
            if f.startswith(f"{split}/images/") and PurePosixPath(f).suffix.lower() in IMAGE_EXTS
        )
        labels = {f for f in file_set if f.startswith(f"{split}/labels/") and f.endswith(".txt")}
        if not images and not labels:
            continue
        split_pairs = []
        for img in images:
            label = f"{split}/labels/{PurePosixPath(img).stem}.txt"
            split_pairs.append(Pair(img, label if label in labels else None))
        pairs[split] = split_pairs
        orphans[split] = len(labels - {p.label for p in split_pairs})
    return pairs, orphans


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _bar(char: str = "-", width: int = 78) -> str:
    return char * width


def cmd_inspect(source: Source) -> int:
    yaml_text = source.read_text("data.yaml")
    names = parse_names(yaml_text)
    pairs, orphans = find_pairs(source.files)

    per_split: dict[str, Counter[int]] = {}
    polygons = malformed = no_label = 0
    for split, split_pairs in pairs.items():
        counts: Counter[int] = Counter()
        for pair in split_pairs:
            if pair.label is None:
                no_label += 1
                continue
            c, p, m = count_source_instances(source.read_text(pair.label))
            counts += c
            polygons += p
            malformed += m
        per_split[split] = counts

    print(f"Source : {source.path}")
    print(f"Classes: {len(names)}   Splits: " + ", ".join(f"{s}={len(p)} images" for s, p in pairs.items()))
    print(_bar())
    splits = list(per_split)
    header = f"{'idx':>3}  {'source class':<22}" + "".join(f"{s:>8}" for s in splits) + f"{'total':>9}  target"
    print(header)
    print(_bar())
    unmapped = []
    for i, name in enumerate(names):
        row = [per_split[s][i] for s in splits]
        target = propose_target(name)
        if target == UNMAPPED:
            unmapped.append(name)
        print(f"{i:>3}  {name:<22}" + "".join(f"{n:>8}" for n in row) + f"{sum(row):>9}  {target}")
    print(_bar())

    known = set(range(len(names)))
    stray = {k: v for s in per_split.values() for k, v in s.items() if k not in known}
    if stray:
        print(f"!! label files use class ids not in data.yaml: {sorted(stray)}")
    if polygons:
        print(f"Note: {polygons} instances are polygons; conversion writes their bounding boxes.")
    if malformed:
        print(f"Note: {malformed} malformed label lines will be skipped.")
    if no_label:
        print(f"Note: {no_label} images have no label file and will be skipped.")
    if any(orphans.values()):
        print(f"Note: orphan label files (no image): {orphans}")

    problems = validate_class_map()
    for p in problems:
        print(f"!! CLASS_MAP: {p}")

    if unmapped:
        print(f"\n{len(unmapped)} UNMAPPED classes. Paste into CLASS_MAP and choose a target for each:\n")
        for name in unmapped:
            print(f'    "{normalize_name(name)}": None,  # "plastic" | "paper" | "organic" | None')
        print()
    else:
        print("\nAll source classes are mapped. Review the target column, then run with --dry-run.")
    return 0


@dataclass
class SplitStats:
    """Conversion counts for one output split."""

    images_seen: int = 0
    images_kept: int = 0
    images_skipped: int = 0
    instances: Counter[int] = field(default_factory=Counter)
    dropped: int = 0
    malformed: int = 0

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly form with category names instead of indices."""
        return {
            "images": self.images_kept,
            "images_skipped_no_labels": self.images_skipped,
            "instances": {c: self.instances[i] for i, c in enumerate(CATEGORIES)},
            "instances_dropped": self.dropped,
            "lines_malformed": self.malformed,
        }


def print_report(stats: dict[str, SplitStats]) -> None:
    print(_bar("="))
    cols = "".join(f"{c:>10}" for c in CATEGORIES)
    print(f"{'split':<7}{'images':>8}{'skipped':>9}{cols}{'dropped':>9}{'bad':>6}")
    print(_bar())
    total = SplitStats()
    for split, s in stats.items():
        print(
            f"{split:<7}{s.images_kept:>8}{s.images_skipped:>9}"
            + "".join(f"{s.instances[i]:>10}" for i in range(len(CATEGORIES)))
            + f"{s.dropped:>9}{s.malformed:>6}"
        )
        total.images_kept += s.images_kept
        total.images_skipped += s.images_skipped
        total.instances += s.instances
        total.dropped += s.dropped
        total.malformed += s.malformed
    print(_bar())
    print(
        f"{'total':<7}{total.images_kept:>8}{total.images_skipped:>9}"
        + "".join(f"{total.instances[i]:>10}" for i in range(len(CATEGORIES)))
        + f"{total.dropped:>9}{total.malformed:>6}"
    )
    print(_bar("="))

    # Imbalance is judged on train: that's what shapes the model; val/test only measure it.
    basis = "train" if "train" in stats else next(iter(stats))
    counts = {c: stats[basis].instances[i] for i, c in enumerate(CATEGORIES)}
    ratio, common, rare = imbalance(counts)
    if ratio > IMBALANCE_WARN_RATIO:
        shown = "inf" if math.isinf(ratio) else f"{ratio:.1f}x"
        print("!" * 78)
        print(f"!! CLASS IMBALANCE in {basis}: {shown} (threshold {IMBALANCE_WARN_RATIO:.0f}x)")
        print(f"!! '{rare}' is STARVED: {counts[rare]} instances vs '{common}' with {counts[common]}.")
        print("!! The model will under-predict it. Add more data for it or rebalance before training.")
        print("!" * 78)
    for split, s in stats.items():
        missing = [c for i, c in enumerate(CATEGORIES) if s.instances[i] == 0]
        if missing and split != "train":
            print(f"!! {split} has no instances of {missing}; their metrics will be meaningless.")


def write_data_yaml(out: Path, splits: Iterable[str]) -> None:
    lines = [
        "# Generated by training/prepare_dataset.py. Regenerate instead of editing.",
        # No `path:` key on purpose: Ultralytics then resolves splits relative to this
        # file, so the dataset folder can be moved or copied to another machine.
    ]
    lines += [f"{split}: {split}/images" for split in splits]
    lines += [f"nc: {len(CATEGORIES)}", "names:"]
    lines += [f"  {i}: {name}" for i, name in enumerate(CATEGORIES)]
    (out / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_convert(source: Source, out: Path, dry_run: bool, overwrite: bool) -> int:
    problems = validate_class_map()
    if problems:
        for p in problems:
            print(f"error: CLASS_MAP: {p}", file=sys.stderr)
        return 2

    yaml_text = source.read_text("data.yaml")
    names = parse_names(yaml_text)
    index_map, unmapped = build_index_map(names)
    if unmapped:
        # Silently dropping an unknown class could throw away a whole compartment's data.
        print(f"error: unmapped source classes {unmapped}; run --inspect and edit CLASS_MAP.", file=sys.stderr)
        return 2

    if not dry_run and out.exists() and any(out.iterdir()):
        if not overwrite:
            print(f"error: {out} is not empty; pass --overwrite to replace it.", file=sys.stderr)
            return 2
        if not (out / "manifest.json").exists():
            # Only ever delete folders this script produced.
            print(f"error: refusing to overwrite {out}: no manifest.json, so it wasn't made by this script.", file=sys.stderr)
            return 2

    pairs, _ = find_pairs(source.files)
    # Build next to the target and swap in at the end, so a crash never leaves a half dataset.
    staging = out.parent / f".{out.name}.partial"
    if not dry_run:
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

    stats: dict[str, SplitStats] = {}
    labels_hash = hashlib.sha256()
    renamed: dict[str, str] = {}
    for src_split, split_pairs in pairs.items():
        dst_split = SPLIT_MAP[src_split]
        s = stats.setdefault(dst_split, SplitStats())
        if not dry_run:
            (staging / dst_split / "images").mkdir(parents=True, exist_ok=True)
            (staging / dst_split / "labels").mkdir(parents=True, exist_ok=True)
        for pair in split_pairs:
            s.images_seen += 1
            text = source.read_text(pair.label) if pair.label else ""
            if pair.label:
                labels_hash.update(pair.label.encode() + b"\0" + text.encode() + b"\0")
            result = rewrite_label(text, index_map)
            s.dropped += result.dropped
            s.malformed += result.malformed
            if result.n_kept == 0:
                s.images_skipped += 1
                continue
            s.images_kept += 1
            s.instances += result.kept
            src = PurePosixPath(pair.image)
            stem = safe_stem(src.stem)
            if stem != src.stem:
                renamed[f"{dst_split}/{stem}{src.suffix}"] = src.name
            if not dry_run:
                source.copy_to(pair.image, staging / dst_split / "images" / f"{stem}{src.suffix}")
                (staging / dst_split / "labels" / f"{stem}.txt").write_text(result.text, encoding="utf-8")

    print_report(stats)
    if renamed:
        print(f"Note: {len(renamed)} over-long filenames shortened (originals listed in manifest.json).")
    if dry_run:
        print("Dry run: nothing written.")
        return 0

    out_splits = [s for s in ("train", "val", "test") if s in stats and stats[s].images_kept]
    write_data_yaml(staging, out_splits)
    manifest = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tool": "training/prepare_dataset.py",
        "source": {
            "name": source.name,
            "path": str(source.path),
            "archive_sha256": source.archive_sha256(),
            "labels_sha256": labels_hash.hexdigest(),
            "classes": names,
            "roboflow": parse_roboflow_meta(yaml_text),
        },
        "categories": list(CATEGORIES),
        "mapping": {
            name: None if index_map[i] is None else CATEGORIES[index_map[i]] for i, name in enumerate(names)
        },
        "split_map": {k: v for k, v in SPLIT_MAP.items() if k in pairs},
        "label_format": "YOLO boxes; source polygons converted to their bounding boxes",
        "images_without_labels": "skipped",
        "counts": {split: s.as_dict() for split, s in stats.items()},
        "renamed_files": renamed,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not _rename_with_retry(staging, out):
        print(
            f"error: the dataset is complete in {staging} but could not be renamed to {out} "
            "(a file inside is still held open). Close Explorer/editor windows on it and rename it by hand.",
            file=sys.stderr,
        )
        return 1
    print(f"Wrote {out} (data.yaml, manifest.json).")
    return 0


def _rename_with_retry(src: Path, dst: Path) -> bool:
    for attempt in range(RENAME_ATTEMPTS):
        try:
            src.rename(dst)
            return True
        except PermissionError:
            time.sleep(RENAME_BACKOFF_S)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", metavar="SOURCE", type=Path, help="list source classes and proposed targets")
    mode.add_argument("--source", metavar="SOURCE", type=Path, help="convert this export")
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT), help=f"output dir (default {DEFAULT_OUT})")
    parser.add_argument("--dry-run", action="store_true", help="report counts without writing anything")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output made by this script")
    args = parser.parse_args(argv)

    try:
        source = Source(args.inspect or args.source)
    except (FileNotFoundError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.inspect:
        return cmd_inspect(source)
    return cmd_convert(source, args.out, args.dry_run, args.overwrite)


if __name__ == "__main__":
    sys.exit(main())
