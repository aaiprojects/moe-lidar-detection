"""Tests for src/fusion/bev_iou.py"""

import math

from src.fusion.bev_iou import bev_iou, bev_iou_matrix
from src.io.schemas import DetectionBox

_Q_IDENTITY = [1.0, 0.0, 0.0, 0.0]


def _yaw_quat(yaw: float) -> list[float]:
    """[w, x, y, z] quaternion for a pure z-axis rotation by `yaw` radians."""
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def make_box(x: float, y: float, w: float = 2.0, length: float = 4.0, yaw: float = 0.0) -> DetectionBox:
    """Build a minimal 'car' DetectionBox at (x, y) with the given size/yaw."""
    return DetectionBox(
        sample_token="tok",
        model_name="m",
        translation=[x, y, 0.0],
        size=[w, length, 1.5],
        rotation=_yaw_quat(yaw) if yaw != 0.0 else _Q_IDENTITY,
        velocity=[0.0, 0.0],
        detection_name="car",
        detection_score=0.9,
        attribute_name=None,
        frame="global",
    )


def test_identical_boxes_iou_is_one():
    """A box compared against itself has IoU exactly 1.0."""
    a = make_box(0.0, 0.0)
    assert math.isclose(bev_iou(a, a), 1.0, abs_tol=1e-9)


def test_non_overlapping_iou_is_zero():
    """Two boxes far enough apart to not overlap have IoU 0.0."""
    a = make_box(0.0, 0.0, w=2.0, length=2.0)   # occupies [-1,1] x [-1,1]
    b = make_box(10.0, 10.0, w=2.0, length=2.0)  # far away
    assert bev_iou(a, b) == 0.0


def test_partial_overlap():
    """A hand-computed partial overlap matches bev_iou's output exactly."""
    # Box a: centre (0,0), 2×2 → [-1,1]×[-1,1]
    # Box b: centre (1,0), 2×2 → [0,2]×[-1,1]
    # Overlap: [0,1]×[-1,1] = 1×2 = 2
    # Union: 4 + 4 - 2 = 6
    a = make_box(0.0, 0.0, w=2.0, length=2.0)
    b = make_box(1.0, 0.0, w=2.0, length=2.0)
    expected = 2.0 / 6.0
    assert math.isclose(bev_iou(a, b), expected, abs_tol=1e-9)


def test_iou_is_symmetric():
    """bev_iou(a, b) equals bev_iou(b, a)."""
    a = make_box(0.0, 0.0)
    b = make_box(1.0, 0.5)
    assert math.isclose(bev_iou(a, b), bev_iou(b, a), abs_tol=1e-12)


def test_iou_in_range():
    """IoU is always within the valid [0, 1] range."""
    a = make_box(0.0, 0.0)
    b = make_box(0.5, 0.5)
    val = bev_iou(a, b)
    assert 0.0 <= val <= 1.0


def test_matrix_shape():
    """bev_iou_matrix returns an [len(boxes_a), len(boxes_b)] matrix."""
    boxes_a = [make_box(0.0, 0.0), make_box(5.0, 0.0)]
    boxes_b = [make_box(0.1, 0.0), make_box(10.0, 0.0), make_box(5.1, 0.0)]
    matrix = bev_iou_matrix(boxes_a, boxes_b)
    assert len(matrix) == 2
    assert all(len(row) == 3 for row in matrix)


def test_matrix_diagonal_high_for_near_identical():
    """The vectorized matrix path agrees with bev_iou on near-identical boxes."""
    a = make_box(0.0, 0.0)
    b = make_box(0.0, 0.0)
    matrix = bev_iou_matrix([a], [b])
    assert math.isclose(matrix[0][0], 1.0, abs_tol=1e-9)


def test_yaw_is_ignored_axis_aligned_convention():
    """§40 tried a rotated (yaw-aware) alternative; §41 confirmed its mAP
    regression wasn't recoverable via NMS retuning, so it was reverted.
    bev_iou is axis-aligned by design -- two boxes at the same center with
    different yaw must still show full overlap (yaw has no effect)."""
    a = make_box(0.0, 0.0, w=2.0, length=6.0, yaw=0.0)
    b = make_box(0.0, 0.0, w=2.0, length=6.0, yaw=math.pi / 2)
    assert math.isclose(bev_iou(a, b), 1.0, abs_tol=1e-9)
