import math

import pytest

from training.prepare_dataset import (
    CLASS_MAP,
    DROP,
    MAX_STEM_LEN,
    UNMAPPED,
    build_index_map,
    find_pairs,
    imbalance,
    normalize_name,
    parse_label_line,
    parse_names,
    propose_target,
    rewrite_label,
    safe_stem,
    validate_class_map,
)

# Mirrors the real Roboflow export: 0 cardboard, 1 glass, 2 metal, 3 organic, 4 paper, 5 plastic.
SOURCE_NAMES = ["cardboard", "glass", "metal", "organic", "paper", "plastic"]
INDEX_MAP, _ = build_index_map(SOURCE_NAMES)
PLASTIC, PAPER, ORGANIC = 0, 1, 2


# --- rewrite_label: the three required cases ---------------------------------


def test_multi_class_label_file():
    text = (
        "5 0.5 0.5 0.2 0.2\n"           # plastic box  -> 0
        "0 0.1 0.1 0.3 0.1 0.3 0.4\n"   # cardboard polygon -> paper (1), as bbox
        "1 0.7 0.7 0.1 0.1\n"           # glass -> dropped
        "3 0.4 0.6 0.2 0.2\n"           # organic -> 2
        "2 0.2 0.2 0.1 0.1\n"           # metal -> dropped
    )
    r = rewrite_label(text, INDEX_MAP)
    assert r.kept == {PLASTIC: 1, PAPER: 1, ORGANIC: 1}
    assert r.dropped == 2
    assert r.malformed == 0
    lines = r.text.splitlines()
    assert [ln.split()[0] for ln in lines] == ["0", "1", "2"]
    # Polygon (0.1,0.1)-(0.3,0.4) becomes centre (0.2, 0.25), size (0.2, 0.3).
    assert [float(v) for v in lines[1].split()[1:]] == pytest.approx([0.2, 0.25, 0.2, 0.3])
    assert r.text.endswith("\n")


def test_file_where_everything_is_dropped():
    text = "1 0.5 0.5 0.2 0.2\n2 0.3 0.3 0.1 0.1\n"
    r = rewrite_label(text, INDEX_MAP)
    assert r.n_kept == 0
    assert r.dropped == 2
    assert r.text == ""


@pytest.mark.parametrize("text", ["", "\n", "   \n\n"])
def test_empty_file(text):
    r = rewrite_label(text, INDEX_MAP)
    assert r.n_kept == 0
    assert r.dropped == 0
    assert r.malformed == 0
    assert r.text == ""


# --- rewrite_label: robustness ------------------------------------------------


def test_malformed_and_unknown_class_lines_are_counted_not_kept():
    text = "5 0.5 0.5\n9 0.5 0.5 0.2 0.2\nabc 1 2 3 4\n5 0.5 0.5 0.2 0.2\n"
    r = rewrite_label(text, INDEX_MAP)
    assert r.malformed == 3
    assert r.kept == {PLASTIC: 1}


def test_box_spilling_outside_image_is_clamped():
    cls, (cx, cy, w, h) = parse_label_line("0 0.95 0.5 0.2 0.2")
    assert cx + w / 2 == pytest.approx(1.0)
    assert w == pytest.approx(0.15)


def test_degenerate_polygon_is_malformed():
    with pytest.raises(ValueError):
        parse_label_line("0 0.5 0.1 0.5 0.2 0.5 0.3")  # zero width


def test_windows_line_endings():
    r = rewrite_label("5 0.5 0.5 0.2 0.2\r\n3 0.5 0.5 0.2 0.2\r\n", INDEX_MAP)
    assert r.n_kept == 2 and r.malformed == 0


# --- mapping table -------------------------------------------------------------


def test_class_map_is_valid():
    assert validate_class_map() == []


def test_validate_catches_typo():
    assert validate_class_map({"bottle": "plstic"})


@pytest.mark.parametrize(
    ("name", "expected"),
    [("Cardboard", "paper"), ("Paper Bag", "paper"), ("food_waste", "organic"), ("GLASS", DROP), ("widget", UNMAPPED)],
)
def test_propose_target(name, expected):
    assert propose_target(name) == expected


def test_normalize_name():
    assert normalize_name("  Plastic_Bottle ") == "plastic-bottle"
    assert normalize_name("E Waste") == "e-waste"


def test_build_index_map_reports_unmapped():
    index_map, unmapped = build_index_map(["paper", "widget", "glass"])
    assert index_map == {0: PAPER, 2: None}
    assert unmapped == ["widget"]


def test_index_map_for_real_export():
    assert INDEX_MAP == {0: PAPER, 1: None, 2: None, 3: ORGANIC, 4: PAPER, 5: PLASTIC}
    assert all(k in CLASS_MAP for k in map(normalize_name, SOURCE_NAMES))


# --- data.yaml parsing -----------------------------------------------------------


@pytest.mark.parametrize(
    "yaml_text",
    [
        "nc: 3\nnames: ['a', 'b c', \"d\"]\n",
        "names: [a, b c, d]  # comment\n",
        "names:\n  - a\n  - 'b c'\n  - d\nnc: 3\n",
        "names:\n  0: a\n  1: b c\n  2: d\n",
        "names: {0: a, 1: 'b c', 2: d}\n",
        "names:\n  2: d\n  0: a\n  1: b c\n",
        "names: [a,\n  b c,\n  d]\n",
    ],
)
def test_parse_names_forms(yaml_text):
    assert parse_names(yaml_text) == ["a", "b c", "d"]


def test_parse_names_real_roboflow_yaml():
    text = (
        "train: ../train/images\nval: ../valid/images\n\nnc: 6\n"
        "names: ['cardboard', 'glass', 'metal', 'organic', 'paper', 'plastic']\n\n"
        "roboflow:\n  workspace: x\n"
    )
    assert parse_names(text) == SOURCE_NAMES


def test_parse_names_rejects_gappy_dict():
    with pytest.raises(ValueError):
        parse_names("names:\n  0: a\n  2: c\n")


def test_parse_names_missing():
    with pytest.raises(ValueError):
        parse_names("nc: 2\n")


# --- misc ---------------------------------------------------------------------------


def test_find_pairs_maps_images_to_labels():
    files = [
        "train/images/a.jpg", "train/labels/a.txt",
        "train/images/b.PNG",
        "valid/images/c.jpg", "valid/labels/c.txt", "valid/labels/orphan.txt",
        "data.yaml",
    ]
    pairs, orphans = find_pairs(files)
    assert [(p.image, p.label) for p in pairs["train"]] == [
        ("train/images/a.jpg", "train/labels/a.txt"),
        ("train/images/b.PNG", None),
    ]
    assert orphans["valid"] == 1


def test_safe_stem_leaves_short_names_alone():
    assert safe_stem("abc_jpg.rf.123") == "abc_jpg.rf.123"


def test_safe_stem_shortens_long_names_deterministically_and_uniquely():
    a = "food-waste-" + "x" * 190 + "_jpg.rf.aa14"
    b = "food-waste-" + "x" * 190 + "_jpg.rf.bb22"  # same prefix, differs only past the cut
    assert len(safe_stem(a)) == MAX_STEM_LEN
    assert safe_stem(a) == safe_stem(a)
    assert safe_stem(a) != safe_stem(b)


def test_imbalance_names_starved_class():
    ratio, common, rare = imbalance({"plastic": 900, "paper": 100, "organic": 400})
    assert ratio == pytest.approx(9.0)
    assert (common, rare) == ("plastic", "paper")


def test_imbalance_zero_is_infinite():
    assert math.isinf(imbalance({"plastic": 5, "paper": 0, "organic": 5})[0])
