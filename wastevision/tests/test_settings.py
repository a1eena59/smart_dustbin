import pytest

from wastevision.settings import CATEGORIES, Settings


def test_category_order_is_the_wire_contract():
    assert CATEGORIES == ("plastic", "paper", "organic")


def test_env_overrides_are_typed():
    s = Settings.from_env({"WV_BURST_SIZE": "9", "WV_CONF_FLOOR": "0.5", "WV_NATIVE_WASTE_CLASSES": "true"})
    assert s.burst_size == 9
    assert s.conf_floor == 0.5
    assert s.native_waste_classes is True


def test_env_false_string_is_false():
    assert Settings.from_env({"WV_NATIVE_WASTE_CLASSES": "false"}).native_waste_classes is False


def test_bad_env_value_is_reported():
    with pytest.raises(ValueError, match="WV_BURST_SIZE"):
        Settings.from_env({"WV_BURST_SIZE": "seven"})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fallback_category": "glass"},
        {"min_margin": 0},
        {"burst_size": 2, "min_valid_votes": 3},
        {"conf_floor": 1.5},
    ],
)
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)
