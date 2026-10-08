import math
import subprocess
import sys
from pathlib import Path

import pytest

from wastevision.settings import CATEGORIES, Settings
from wastevision.voting import tally


@pytest.fixture
def settings() -> Settings:
    # Pinned explicitly so the tests don't silently change meaning if defaults are retuned.
    return Settings(conf_floor=0.40, min_valid_votes=3, min_margin=2, burst_size=7, fallback_category="organic")


ROOT = Path(__file__).resolve().parents[2]
P, A, O = "plastic", "paper", "organic"


def test_clear_majority(settings):
    frames = [(P, 0.9), (P, 0.8), (P, 0.85), (P, 0.7), (P, 0.75), (O, 0.6), (None, 0.0)]
    out = tally(frames, settings)
    assert out.decided_by == "vote"
    assert out.category == P
    assert out.index == 0
    assert out.votes == {P: 5, A: 0, O: 1}
    assert out.frames_used == 6


def test_three_three_one_split_falls_back_on_margin(settings):
    frames = [(P, 0.9)] * 3 + [(A, 0.9)] * 3 + [(O, 0.9)]
    out = tally(frames, settings)
    assert out.decided_by == "fallback"
    assert out.category == "organic"
    assert out.index == 2
    assert out.confidence == 0.0
    assert out.votes == {P: 3, A: 3, O: 1}
    assert out.frames_used == 7


def test_all_frames_abstaining(settings):
    # Mix of "nothing detected" and "detected but under the floor".
    frames = [(None, 0.0)] * 4 + [(P, 0.39)] * 3
    out = tally(frames, settings)
    assert out.decided_by == "fallback"
    assert out.frames_used == 0
    assert out.votes == {P: 0, A: 0, O: 0}


def test_exactly_three_valid_votes_unanimous_is_accepted(settings):
    # With exactly 3 votes the only split clearing margin 2 is 3-0 (margin 3).
    frames = [(A, 0.6)] * 3 + [(None, 0.0)] * 4
    out = tally(frames, settings)
    assert out.decided_by == "vote"
    assert out.category == A
    assert out.index == 1
    assert out.frames_used == 3


def test_margin_of_exactly_two_is_accepted(settings):
    # Boundary of min_margin itself: 3 vs 1.
    frames = [(P, 0.8)] * 3 + [(A, 0.8)] + [(None, 0.0)] * 3
    out = tally(frames, settings)
    assert out.decided_by == "vote"
    assert out.category == P


def test_exactly_three_valid_votes_margin_one_falls_back(settings):
    frames = [(P, 0.8), (P, 0.8), (A, 0.8)] + [(None, 0.0)] * 4
    out = tally(frames, settings)
    assert out.decided_by == "fallback"
    assert out.frames_used == 3
    assert out.votes == {P: 2, A: 1, O: 0}


def test_too_few_valid_votes_falls_back_even_if_unanimous(settings):
    frames = [(P, 0.95), (P, 0.95)] + [(None, 0.0)] * 5
    out = tally(frames, settings)
    assert out.decided_by == "fallback"


def test_empty_list(settings):
    out = tally([], settings)
    assert out.decided_by == "fallback"
    assert out.category == "organic"
    assert out.confidence == 0.0
    assert out.frames_used == 0
    assert set(out.votes) == set(CATEGORIES)


def test_confidence_is_mean_of_winning_frames_only(settings):
    frames = [(P, 0.9), (P, 0.8), (P, 0.7), (A, 0.41), (None, 0.99), (O, 0.2)]
    out = tally(frames, settings)
    assert out.decided_by == "vote"
    assert out.confidence == pytest.approx(0.8)


def test_confidence_exactly_at_floor_votes(settings):
    out = tally([(P, 0.40)] * 3, settings)
    assert out.decided_by == "vote"


def test_nan_confidence_abstains(settings):
    out = tally([(P, math.nan)] * 7, settings)
    assert out.frames_used == 0


def test_unknown_category_abstains(settings):
    out = tally([("glass", 0.9)] * 7, settings)
    assert out.frames_used == 0
    assert out.decided_by == "fallback"


def test_voting_imports_without_heavy_dependencies():
    # Fresh interpreter: this process may already have cv2/torch loaded by other tests.
    code = (
        "import sys, wastevision.voting; "
        "loaded = [m for m in ('cv2', 'ultralytics', 'torch', 'numpy') if m in sys.modules]; "
        "sys.exit(','.join(loaded) or 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, f"voting pulled in heavy modules: {proc.stderr or proc.stdout}"
