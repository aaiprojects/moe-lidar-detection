"""Tests for src/moe/router_dataset.py"""

import math

import pandas as pd

from src.io.schemas import DetectionBox
from src.moe.features import FEATURE_NAMES
from src.moe.router_dataset import (
    GtBox,
    _gt_iou,
    _match_predictions_to_gt,
    build_dataset,
    make_token_split,
    warm_map_masks,
)

_Q = [1.0, 0.0, 0.0, 0.0]


def make_pred(x, y, score=0.9, cls="car", model="centerpoint", token="tok1"):
    """Build a predicted DetectionBox at (x, y) for the given class/model/token."""
    return DetectionBox(
        sample_token=token,
        model_name=model,
        translation=[x, y, 0.5],
        size=[2.0, 4.0, 1.5],
        rotation=_Q,
        velocity=[0.0, 0.0],
        detection_name=cls,
        detection_score=score,
        attribute_name=None,
        frame="global",
    )


def make_gt(x, y, cls="car"):
    """Build a GtBox at (x, y) for the given class."""
    return GtBox(translation=[x, y, 0.5], size=[2.0, 4.0, 1.5], detection_name=cls)


# --- _gt_iou ---

def test_gt_iou_identical():
    """A prediction and GT box at the same position/size have IoU 1.0."""
    pred = make_pred(0.0, 0.0)
    gt = make_gt(0.0, 0.0)
    assert math.isclose(_gt_iou(pred, gt), 1.0, abs_tol=1e-6)


def test_gt_iou_no_overlap():
    """A prediction and GT box far apart have IoU 0.0."""
    pred = make_pred(0.0, 0.0)
    gt = make_gt(100.0, 100.0)
    assert _gt_iou(pred, gt) == 0.0


# --- _match_predictions_to_gt ---

def test_match_tp():
    """A prediction overlapping a same-class GT box is labeled a true positive."""
    pred = make_pred(0.0, 0.0, score=0.9)
    gt = make_gt(0.0, 0.0)
    labels = _match_predictions_to_gt([pred], [gt])
    assert labels == [1]


def test_match_fp_wrong_class():
    """A prediction overlapping a GT box of a DIFFERENT class is a false positive."""
    pred = make_pred(0.0, 0.0, cls="car")
    gt = make_gt(0.0, 0.0, cls="pedestrian")
    labels = _match_predictions_to_gt([pred], [gt])
    assert labels == [0]


def test_match_fp_too_far():
    """A prediction too far from any GT box (below the IoU threshold) is a false positive."""
    pred = make_pred(0.0, 0.0)
    gt = make_gt(50.0, 0.0)
    labels = _match_predictions_to_gt([pred], [gt])
    assert labels == [0]


def test_match_greedy_one_gt_matched_once():
    """Greedy matching is score-descending: with two predictions on one GT
    box, only the higher-scoring prediction is labeled a true positive."""
    # Two predictions, one GT — only higher-score pred gets TP
    pred_high = make_pred(0.0, 0.0, score=0.9)
    pred_low = make_pred(0.0, 0.0, score=0.5)
    gt = make_gt(0.0, 0.0)
    labels = _match_predictions_to_gt([pred_high, pred_low], [gt])
    assert sum(labels) == 1  # only one TP
    assert labels[0] == 1    # higher score gets the match


def test_match_empty_gt():
    """With no GT boxes at all, every prediction is a false positive."""
    pred = make_pred(0.0, 0.0)
    labels = _match_predictions_to_gt([pred], [])
    assert labels == [0]


def test_match_empty_preds():
    """With no predictions, the label list is empty (nothing to score)."""
    gt = make_gt(0.0, 0.0)
    labels = _match_predictions_to_gt([], [gt])
    assert labels == []


# --- build_dataset ---

def test_build_dataset_returns_dataframe():
    """build_dataset returns a DataFrame with every FEATURE_NAMES column plus
    the label/sample_token metadata columns."""
    preds = {
        "centerpoint": {
            "tok1": [make_pred(0.0, 0.0, token="tok1")],
        }
    }
    gt = {"tok1": [make_gt(0.0, 0.0)]}
    df = build_dataset(preds, gt, ["tok1"])
    assert isinstance(df, pd.DataFrame)
    assert "label" in df.columns
    assert "sample_token" in df.columns
    assert all(f in df.columns for f in FEATURE_NAMES)


def test_build_dataset_tp_label():
    """A row for a prediction that matches GT is labeled 1."""
    preds = {"cp": {"tok1": [make_pred(0.0, 0.0, model="cp", token="tok1")]}}
    gt = {"tok1": [make_gt(0.0, 0.0)]}
    df = build_dataset(preds, gt, ["tok1"])
    assert df["label"].iloc[0] == 1


def test_build_dataset_fp_label():
    """A row for a prediction far from any GT box is labeled 0."""
    preds = {"cp": {"tok1": [make_pred(100.0, 100.0, model="cp", token="tok1")]}}
    gt = {"tok1": [make_gt(0.0, 0.0)]}
    df = build_dataset(preds, gt, ["tok1"])
    assert df["label"].iloc[0] == 0


def test_build_dataset_skips_missing_token():
    """A sample token absent from gt_by_token is skipped entirely (no GT to label against)."""
    preds = {"cp": {"tok_missing": [make_pred(0.0, 0.0, model="cp", token="tok_missing")]}}
    gt = {}  # no GT for this token
    df = build_dataset(preds, gt, ["tok_missing"])
    assert len(df) == 0


# --- make_token_split ---

def test_make_token_split_ratio():
    """The split honors train_ratio and covers every input token exactly once."""
    tokens = [f"tok_{i}" for i in range(100)]
    train, val = make_token_split(tokens, train_ratio=0.7, seed=42)
    assert len(train) == 70
    assert len(val) == 30
    assert set(train) | set(val) == set(tokens)
    assert set(train) & set(val) == set()


def test_make_token_split_deterministic():
    """The same seed always produces the same split."""
    tokens = [f"tok_{i}" for i in range(50)]
    t1, v1 = make_token_split(tokens, seed=42)
    t2, v2 = make_token_split(tokens, seed=42)
    assert t1 == t2
    assert v1 == v2


def test_make_token_split_different_seeds():
    """Different seeds produce different splits."""
    tokens = [f"tok_{i}" for i in range(50)]
    t1, _ = make_token_split(tokens, seed=42)
    t2, _ = make_token_split(tokens, seed=99)
    assert t1 != t2


# --- warm_map_masks (OOM fix, see EXPERIMENT_LOG §46) --------------------

class _FakeMapMask:
    """Records dilation values it's called with, standing in for the real
    (expensive) MapMask.mask() during tests."""

    def __init__(self):
        self.calls: list[float] = []

    def mask(self, dilation: float = 0.0):
        self.calls.append(dilation)
        return dilation


def test_warm_map_masks_calls_each_dilation_once_per_distinct_mask():
    """Each distinct mask object gets warmed at dilation 0.0 plus every level in dilation_levels_m."""
    m1, m2 = _FakeMapMask(), _FakeMapMask()
    mask_by_token = {"t1": m1, "t2": m1, "t3": m2}
    warm_map_masks(mask_by_token, dilation_levels_m=(2.0, 8.0))
    assert m1.calls == [0.0, 2.0, 8.0]
    assert m2.calls == [0.0, 2.0, 8.0]


def test_warm_map_masks_dedupes_by_identity_not_just_count():
    # same mask referenced by every token -- must only be warmed once
    """The same mask object shared by many tokens is warmed only once, not once per token."""
    m = _FakeMapMask()
    mask_by_token = {f"t{i}": m for i in range(50)}
    warm_map_masks(mask_by_token, dilation_levels_m=(2.0, 8.0))
    assert m.calls == [0.0, 2.0, 8.0]


def test_warm_map_masks_empty_input_noop():
    """An empty mask_by_token dict is a safe no-op."""
    warm_map_masks({}, dilation_levels_m=(2.0, 8.0))  # must not raise
