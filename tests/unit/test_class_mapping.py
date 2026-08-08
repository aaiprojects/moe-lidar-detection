"""Tests for src/utils/class_mapping.py"""

import pytest

from src.utils.class_mapping import NUSCENES_CLASSES, validate_class


def test_all_canonical_classes_pass():
    for cls in NUSCENES_CLASSES:
        assert validate_class(cls) == cls


def test_alias_resolves_to_canonical():
    assert validate_class("vehicle.car") == "car"
    assert validate_class("human.pedestrian.adult") == "pedestrian"
    assert validate_class("movable_object.trafficcone") == "traffic_cone"
    assert validate_class("vehicle.bus.bendy") == "bus"
    assert validate_class("vehicle.bus.rigid") == "bus"


def test_unknown_class_raises():
    with pytest.raises(ValueError, match="Unknown detection class"):
        validate_class("unknown_object")


def test_empty_string_raises():
    with pytest.raises(ValueError, match="Unknown detection class"):
        validate_class("")


def test_canonical_class_count():
    assert len(NUSCENES_CLASSES) == 10
