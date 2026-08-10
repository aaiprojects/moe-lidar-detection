"""Tests for src/fusion/nms3d.py"""

from src.fusion.nms3d import nms3d
from src.io.schemas import DetectionBox

_Q = [1.0, 0.0, 0.0, 0.0]


def make_box(
    x: float,
    y: float,
    score: float,
    cls: str = "car",
    w: float = 2.0,
    length: float = 4.0,
) -> DetectionBox:
    """Build a DetectionBox at (x, y) with the given score and class."""
    return DetectionBox(
        sample_token="tok",
        model_name="m",
        translation=[x, y, 0.0],
        size=[w, length, 1.5],
        rotation=_Q,
        velocity=[0.0, 0.0],
        detection_name=cls,
        detection_score=score,
        attribute_name=None,
        frame="global",
    )


def test_empty_input_returns_empty():
    """NMS on an empty list returns an empty list."""
    assert nms3d([]) == []


def test_single_box_is_kept():
    """A single box always survives NMS (nothing to suppress it)."""
    boxes = [make_box(0.0, 0.0, 0.9)]
    result = nms3d(boxes)
    assert len(result) == 1


def test_duplicate_box_suppressed():
    # Two identical boxes — lower score should be suppressed
    """Of two identical, fully-overlapping boxes, only the higher-score one survives."""
    high = make_box(0.0, 0.0, 0.9)
    low = make_box(0.0, 0.0, 0.5)
    result = nms3d([low, high], iou_threshold=0.3)
    assert len(result) == 1
    assert result[0].detection_score == 0.9


def test_non_overlapping_boxes_both_kept():
    """Two boxes far apart (no overlap) both survive NMS."""
    a = make_box(0.0, 0.0, 0.9)
    b = make_box(100.0, 100.0, 0.8)
    result = nms3d([a, b], iou_threshold=0.3)
    assert len(result) == 2


def test_different_classes_not_suppressed_in_class_aware_mode():
    """In class-aware mode, co-located boxes of different classes never suppress each other."""
    car = make_box(0.0, 0.0, 0.9, cls="car")
    ped = make_box(0.0, 0.0, 0.8, cls="pedestrian")
    result = nms3d([car, ped], iou_threshold=0.3, class_aware=True)
    assert len(result) == 2


def test_different_classes_suppressed_in_non_class_aware_mode():
    """With class_aware=False, co-located boxes of different classes DO suppress each other."""
    car = make_box(0.0, 0.0, 0.9, cls="car")
    ped = make_box(0.0, 0.0, 0.8, cls="pedestrian")
    result = nms3d([car, ped], iou_threshold=0.3, class_aware=False)
    assert len(result) == 1


def test_output_sorted_by_score_descending():
    """Surviving boxes are always returned sorted by score, highest first."""
    boxes = [
        make_box(0.0, 0.0, 0.5),
        make_box(10.0, 0.0, 0.9),
        make_box(20.0, 0.0, 0.7),
    ]
    result = nms3d(boxes)
    scores = [b.detection_score for b in result]
    assert scores == sorted(scores, reverse=True)


def test_max_boxes_cap():
    """max_boxes hard-caps the number of surviving boxes even when none overlap."""
    boxes = [make_box(float(i) * 10, 0.0, 0.9) for i in range(20)]
    result = nms3d(boxes, max_boxes=5)
    assert len(result) <= 5
