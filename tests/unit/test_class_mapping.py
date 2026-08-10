"""Tests for src/utils/class_mapping.py"""

import pytest

from src.utils.class_mapping import NUSCENES_CLASSES, validate_class


def test_all_canonical_classes_pass():
    """Every canonical class name validates to itself unchanged."""
    for cls in NUSCENES_CLASSES:
        assert validate_class(cls) == cls


def test_alias_resolves_to_canonical():
    """A sample of known raw-model aliases each resolve to their canonical class."""
    assert validate_class("vehicle.car") == "car"
    assert validate_class("human.pedestrian.adult") == "pedestrian"
    assert validate_class("movable_object.trafficcone") == "traffic_cone"
    assert validate_class("vehicle.bus.bendy") == "bus"
    assert validate_class("vehicle.bus.rigid") == "bus"


def test_unknown_class_raises():
    """A name that's neither canonical nor a known alias raises ValueError."""
    with pytest.raises(ValueError, match="Unknown detection class"):
        validate_class("unknown_object")


def test_empty_string_raises():
    """An empty string is treated as an unknown class, not a special case."""
    with pytest.raises(ValueError, match="Unknown detection class"):
        validate_class("")


def test_canonical_class_count():
    """There are exactly the 10 nuScenes detection classes, no more, no fewer."""
    assert len(NUSCENES_CLASSES) == 10
