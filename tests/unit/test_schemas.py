"""Tests for src/io/schemas.py"""

import math

import pytest

from src.io.schemas import DetectionBox, _quat_to_yaw

VALID_BOX_KWARGS = dict(
    sample_token="abc123",
    model_name="test_model",
    translation=[1.0, 2.0, 0.5],
    size=[2.0, 4.0, 1.5],
    rotation=[1.0, 0.0, 0.0, 0.0],  # identity quaternion → yaw = 0
    velocity=[0.0, 0.0],
    detection_name="car",
    detection_score=0.9,
    attribute_name="vehicle.moving",
    frame="global",
)


def make_box(**overrides) -> DetectionBox:
    kwargs = {**VALID_BOX_KWARGS, **overrides}
    return DetectionBox(**kwargs)


def test_valid_box_creates_successfully():
    box = make_box()
    assert box.detection_name == "car"
    assert box.detection_score == 0.9
    assert math.isclose(box.yaw, 0.0, abs_tol=1e-6)


def test_yaw_derived_from_rotation():
    # 90° around z-axis: quaternion is [cos(45°), 0, 0, sin(45°)]
    angle = math.pi / 2
    w = math.cos(angle / 2)
    z = math.sin(angle / 2)
    box = make_box(rotation=[w, 0.0, 0.0, z])
    assert math.isclose(box.yaw, angle, abs_tol=1e-5)


def test_alias_class_normalised():
    box = make_box(detection_name="vehicle.car")
    assert box.detection_name == "car"


def test_unknown_class_raises():
    with pytest.raises(ValueError, match="Unknown detection class"):
        make_box(detection_name="spaceship")


def test_negative_size_raises():
    with pytest.raises(ValueError, match="positive"):
        make_box(size=[-1.0, 4.0, 1.5])


def test_zero_size_raises():
    with pytest.raises(ValueError, match="positive"):
        make_box(size=[0.0, 4.0, 1.5])


def test_non_finite_score_raises():
    with pytest.raises(ValueError, match="finite"):
        make_box(detection_score=float("nan"))

    with pytest.raises(ValueError, match="finite"):
        make_box(detection_score=float("inf"))


def test_score_out_of_range_raises():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        make_box(detection_score=1.5)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        make_box(detection_score=-0.1)


def test_invalid_frame_raises():
    with pytest.raises(ValueError, match="Unknown coordinate frame"):
        make_box(frame="camera")


def test_empty_sample_token_raises():
    with pytest.raises(ValueError, match="sample_token"):
        make_box(sample_token="")


def test_to_nuscenes_dict_keys():
    box = make_box()
    d = box.to_nuscenes_dict()
    expected_keys = {
        "sample_token", "translation", "size", "rotation",
        "velocity", "detection_name", "detection_score", "attribute_name",
    }
    assert set(d.keys()) == expected_keys


def test_quat_to_yaw_identity():
    assert math.isclose(_quat_to_yaw([1.0, 0.0, 0.0, 0.0]), 0.0, abs_tol=1e-9)
