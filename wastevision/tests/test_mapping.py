import pytest

from wastevision.mapping import BLOCKLIST, COCO_BRIDGE, bridge_by_category, check_model_classes, resolve
from wastevision.settings import CATEGORIES


@pytest.mark.parametrize(
    ("coco", "expected"),
    [
        ("bottle", "plastic"),
        ("cup", "plastic"),
        ("wine glass", "plastic"),
        ("bowl", "plastic"),
        ("book", "paper"),
        ("banana", "organic"),
        ("hot dog", "organic"),
        ("broccoli", "organic"),
    ],
)
def test_bridge_hits(coco, expected):
    assert resolve(coco, native=False) == expected


def test_bridge_is_case_and_whitespace_insensitive():
    assert resolve("  Bottle ", native=False) == "plastic"


@pytest.mark.parametrize("name", sorted(BLOCKLIST))
def test_blocklist_rejected(name):
    assert resolve(name, native=False) is None


@pytest.mark.parametrize("name", ["giraffe", "toothbrush", "", "plasticc"])
def test_unknown_classes_rejected(name):
    assert resolve(name, native=False) is None


def test_bridge_and_blocklist_are_disjoint():
    assert not (set(COCO_BRIDGE) & BLOCKLIST)


def test_bridge_targets_are_valid_categories():
    assert set(COCO_BRIDGE.values()) <= set(CATEGORIES)


@pytest.mark.parametrize("category", CATEGORIES)
def test_native_accepts_categories(category):
    assert resolve(category, native=True) == category


def test_native_normalises_case():
    assert resolve("Paper", native=True) == "paper"


@pytest.mark.parametrize("name", ["bottle", "book", "banana", "person", "glass"])
def test_native_bypasses_bridge(name):
    assert resolve(name, native=True) is None


WASTE_V1_NAMES = {0: "plastic", 1: "paper", 2: "organic"}
COCO_NAMES = {0: "person", 39: "bottle", 73: "book", 46: "banana"}


def test_custom_weights_in_native_mode_are_clean():
    assert check_model_classes(WASTE_V1_NAMES, native=True) == []


def test_coco_weights_in_bridge_mode_are_clean():
    assert check_model_classes(COCO_NAMES, native=False) == []


def test_coco_weights_in_native_mode_fail_loudly():
    with pytest.raises(ValueError, match="COCO"):
        check_model_classes(COCO_NAMES, native=True)


def test_custom_weights_in_bridge_mode_fail_loudly():
    with pytest.raises(ValueError, match="NATIVE_WASTE_CLASSES=true"):
        check_model_classes(WASTE_V1_NAMES, native=False)


def test_missing_category_warns():
    (msg,) = check_model_classes({0: "plastic", 1: "paper", 2: "glass"}, native=True)
    assert "organic" in msg


def test_reordered_ids_warn():
    (msg,) = check_model_classes({0: "paper", 1: "plastic", 2: "organic"}, native=True)
    assert "order" in msg


def test_bridge_by_category_follows_category_order():
    grouped = bridge_by_category()
    assert tuple(grouped) == CATEGORIES
    assert "book" in grouped["paper"]
