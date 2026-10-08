"""Bridge from COCO class names to waste categories."""

from collections.abc import Mapping
from types import MappingProxyType

from wastevision.settings import CATEGORIES

# Stock yolov8n is COCO-trained and knows nothing about waste, so we borrow
# object classes that are usually made of one material.
COCO_BRIDGE: MappingProxyType[str, str] = MappingProxyType(
    {
        "bottle": "plastic",
        "cup": "plastic",
        "wine glass": "plastic",
        "bowl": "plastic",
        "book": "paper",
        "banana": "organic",
        "apple": "organic",
        "orange": "organic",
        "broccoli": "organic",
        "carrot": "organic",
        "sandwich": "organic",
        "pizza": "organic",
        "donut": "organic",
        "cake": "organic",
        "hot dog": "organic",
    }
)

# Things that are always in frame around the platform (the user's hand, the desk,
# a monitor). They must never cast a vote, even if someone later adds them to the bridge.
BLOCKLIST: frozenset[str] = frozenset(
    {
        "person",
        "chair",
        "tv",
        "laptop",
        "keyboard",
        "mouse",
        "cell phone",
        "dining table",
        "couch",
        "bed",
        "remote",
        "clock",
        "potted plant",
    }
)


def resolve(class_name: str, native: bool) -> str | None:
    """Return the waste category for a model class name, or None if it must not vote.

    With native=True (custom-trained weights) the COCO bridge is bypassed and the
    name is accepted only if it already is one of CATEGORIES.
    """
    name = class_name.strip().lower()
    if native:
        return name if name in CATEGORIES else None
    if name in BLOCKLIST:
        return None
    return COCO_BRIDGE.get(name)


MAX_NAMES_IN_ERROR = 6


def check_model_classes(names: Mapping[int, str], native: bool) -> list[str]:
    """Verify a model's class names suit the chosen mode. Returns warnings; raises ValueError if unusable.

    A mismatch here would otherwise be silent: every detection rejected, every burst a fallback.
    """
    reachable = {resolve(n, native) for n in names.values()} - {None}
    if not reachable:
        hint = (
            "these look like COCO weights; set WV_NATIVE_WASTE_CLASSES=false or use --coco"
            if native
            else "these look like custom waste weights; set WV_NATIVE_WASTE_CLASSES=true"
        )
        shown = sorted(names.values())
        sample = ", ".join(shown[:MAX_NAMES_IN_ERROR]) + (", ..." if len(shown) > MAX_NAMES_IN_ERROR else "")
        raise ValueError(f"model's {len(shown)} classes ({sample}) can't produce any of {CATEGORIES}: {hint}")

    warnings = []
    missing = [c for c in CATEGORIES if c not in reachable]
    if missing:
        warnings.append(f"model can never output {missing}; those bins will only open via fallback")
    if native:
        # Our code matches by name, so a different order still works here, but the
        # dataset/firmware contract says IDs must match; flag it before it bites elsewhere.
        by_name = {n.strip().lower(): i for i, n in names.items()}
        wrong = [f"{c}={by_name[c]} (expected {i})" for i, c in enumerate(CATEGORIES) if c in by_name and by_name[c] != i]
        if wrong:
            warnings.append(f"model class IDs differ from CATEGORIES order: {', '.join(wrong)}")
    return warnings


def bridge_by_category() -> dict[str, list[str]]:
    """Group bridged COCO classes under each category, in CATEGORIES order."""
    grouped: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    for coco_name, category in COCO_BRIDGE.items():
        grouped[category].append(coco_name)
    return grouped
