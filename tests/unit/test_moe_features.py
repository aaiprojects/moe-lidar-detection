"""Tests for src/moe/features.py"""

import math

from src.io.schemas import DetectionBox
from src.moe.features import (
    CLASS_AGREEMENT_MISSING,
    DIST_TO_DRIVABLE_AREA_MISSING,
    FEATURE_NAMES,
    extract_features,
    extract_features_for_sample,
)

_Q = [1.0, 0.0, 0.0, 0.0]


def make_box(
    x: float,
    y: float,
    score: float = 0.9,
    cls: str = "car",
    model: str = "centerpoint",
    vx: float = 0.0,
    vy: float = 0.0,
) -> DetectionBox:
    """Build a DetectionBox at (x, y) with the given score/class/model/velocity."""
    return DetectionBox(
        sample_token="tok",
        model_name=model,
        translation=[x, y, 0.5],
        size=[2.0, 4.0, 1.5],
        rotation=_Q,
        velocity=[vx, vy],
        detection_name=cls,
        detection_score=score,
        attribute_name=None,
        frame="global",
    )


def _feats(box, peers=None, n_active=1, n_other=0, max_cls=None):
    """Helper to call extract_features with sensible defaults."""
    if peers is None:
        peers = []
    if max_cls is None:
        max_cls = box.detection_score
    return extract_features(
        box, peers, n_active_experts=n_active, n_other_experts=n_other, max_class_score=max_cls
    )


def test_feature_names_length():
    """FEATURE_NAMES has exactly the 17 production features, no removed/excluded ones."""
    # Original 15 + n_spatial_overlaps + class_agreement + dist_to_drivable_area,
    # minus class_id (excluded: constant within each per-class router's training
    # data, so it carries zero split information -- see features.py comment).
    assert len(FEATURE_NAMES) == 17
    assert "class_agreement" in FEATURE_NAMES
    assert "n_spatial_overlaps" in FEATURE_NAMES
    assert "dist_to_drivable_area" in FEATURE_NAMES
    assert "n_points_inside" not in FEATURE_NAMES
    assert "class_id" not in FEATURE_NAMES


def test_features_no_peers():
    """A box with no peers gets zeroed presence/uncertainty features and the
    class_agreement missing sentinel (not 0.0)."""
    box = make_box(10.0, 5.0, score=0.8, model="centerpoint")
    feats = _feats(box)

    assert feats.expert_id == 0          # centerpoint = 0
    assert feats.class_id == 0           # car = 0
    assert math.isclose(feats.detection_score, 0.8)
    assert math.isclose(feats.dist_from_ego, math.sqrt(10**2 + 5**2))
    assert feats.n_peer_overlaps == 0
    assert feats.n_spatial_overlaps == 0
    assert feats.max_peer_iou == 0.0
    assert feats.mean_peer_score == 0.0
    assert feats.score_variance == 0.0
    assert feats.expert_agreement == 0.0
    # Isolated detection → missing sentinel, NOT 0.0 (disagreement)
    assert math.isclose(feats.class_agreement, CLASS_AGREEMENT_MISSING)


def test_isolated_vs_disagreement_not_conflated():
    """class_agreement=-1 (no peers) must differ from class_agreement=0 (disagreement)."""
    box = make_box(0.0, 0.0, score=0.9, cls="car", model="centerpoint")

    isolated = _feats(box, peers=[], n_active=1, n_other=0)
    assert isolated.n_spatial_overlaps == 0
    assert math.isclose(isolated.class_agreement, CLASS_AGREEMENT_MISSING)

    truck_peer = make_box(0.0, 0.0, score=0.8, cls="truck", model="pointpillars")
    disagree = _feats(box, peers=[truck_peer], n_active=2, n_other=1)
    assert disagree.n_spatial_overlaps == 1
    assert math.isclose(disagree.class_agreement, 0.0)
    assert isolated.class_agreement != disagree.class_agreement


def test_features_with_overlapping_peer():
    """One fully-overlapping, same-class peer drives presence and agreement to 1.0."""
    box = make_box(0.0, 0.0, score=0.9, model="centerpoint")
    peer = make_box(0.0, 0.0, score=0.7, model="pointpillars")
    feats = _feats(box, peers=[peer], n_active=2, n_other=1, max_cls=0.9)

    assert feats.n_peer_overlaps == 1
    assert feats.n_spatial_overlaps == 1
    assert feats.max_peer_iou > 0.9
    assert math.isclose(feats.mean_peer_score, 0.7)
    assert feats.score_variance == 0.0      # only 1 peer score → no variance
    assert math.isclose(feats.expert_agreement, 1.0)  # 1/1 other experts agree
    assert math.isclose(feats.class_agreement, 1.0)   # sole peer is same class


def test_cross_class_peer_still_counts_as_presence():
    """Car + truck at same spot: presence agreement kept; class_agreement = 0."""
    box = make_box(0.0, 0.0, score=0.9, cls="car", model="centerpoint")
    peer = make_box(0.0, 0.0, score=0.8, cls="truck", model="pointpillars")
    feats = _feats(box, peers=[peer], n_active=2, n_other=1, max_cls=0.9)

    # Presence signal preserved (class-agnostic peer features)
    assert feats.n_peer_overlaps == 1
    assert feats.n_spatial_overlaps == 1
    assert feats.max_peer_iou > 0.9
    assert math.isclose(feats.mean_peer_score, 0.8)
    assert math.isclose(feats.expert_agreement, 1.0)
    # Semantic disagreement (peers exist) → 0.0, not the missing sentinel
    assert math.isclose(feats.class_agreement, 0.0)


def test_class_agreement_partial_confusion():
    """One same-class peer + one cross-class peer → class_agreement = 0.5."""
    box = make_box(0.0, 0.0, score=0.9, cls="car", model="centerpoint")
    peer_car = make_box(0.0, 0.0, score=0.8, cls="car", model="pointpillars")
    peer_truck = make_box(0.0, 0.0, score=0.7, cls="truck", model="ssn")
    feats = _feats(box, peers=[peer_car, peer_truck], n_active=3, n_other=2, max_cls=0.9)

    assert feats.n_peer_overlaps == 2
    assert feats.n_spatial_overlaps == 2
    assert math.isclose(feats.expert_agreement, 1.0)
    assert math.isclose(feats.class_agreement, 0.5)


def test_score_variance_two_peers():
    """score_variance matches the hand-computed population variance of two peer scores."""
    box = make_box(0.0, 0.0, score=0.9, model="centerpoint")
    peer1 = make_box(0.0, 0.0, score=0.8, model="pointpillars")
    peer2 = make_box(0.0, 0.0, score=0.4, model="voxelnext")
    feats = _feats(box, peers=[peer1, peer2], n_active=3, n_other=2, max_cls=0.9)

    assert feats.n_peer_overlaps == 2
    assert feats.score_variance > 0.0
    expected_mean = (0.8 + 0.4) / 2
    expected_var = ((0.8 - expected_mean)**2 + (0.4 - expected_mean)**2) / 2
    assert math.isclose(feats.score_variance, expected_var, rel_tol=1e-5)


def test_expert_agreement_partial():
    """expert_agreement is the fraction of OTHER experts with an overlapping box."""
    # Two other experts, only one has an overlapping box
    box = make_box(0.0, 0.0, score=0.9, model="centerpoint")
    peer_close = make_box(0.0, 0.0, score=0.8, model="pointpillars")
    peer_far = make_box(100.0, 100.0, score=0.8, model="voxelnext")
    feats = _feats(box, peers=[peer_close, peer_far], n_active=3, n_other=2, max_cls=0.9)

    assert math.isclose(feats.expert_agreement, 0.5)   # 1 of 2 experts agree


def test_expert_agreement_none():
    """expert_agreement is 0.0 when no other expert has an overlapping box."""
    box = make_box(0.0, 0.0, model="centerpoint")
    peer = make_box(100.0, 100.0, model="pointpillars")
    feats = _feats(box, peers=[peer], n_active=2, n_other=1, max_cls=0.9)
    assert feats.expert_agreement == 0.0


def test_max_class_score_stored():
    """max_class_score is passed through from the caller-supplied value, not recomputed."""
    box = make_box(0.0, 0.0, score=0.6)
    feats = _feats(box, max_cls=0.95)
    assert math.isclose(feats.max_class_score, 0.95)


def test_features_with_non_overlapping_peer():
    """A peer far enough away contributes no overlap and no IoU."""
    box = make_box(0.0, 0.0, model="centerpoint")
    peer = make_box(100.0, 100.0, model="pointpillars")
    feats = _feats(box, peers=[peer], n_active=2, n_other=1)

    assert feats.n_peer_overlaps == 0
    assert feats.max_peer_iou == 0.0


def test_velocity_magnitude():
    """vel_magnitude is the Euclidean norm of (vx, vy) — a 3-4-5 triangle here."""
    box = make_box(0.0, 0.0, vx=3.0, vy=4.0)
    feats = _feats(box)
    assert math.isclose(feats.vel_magnitude, 5.0)


def test_to_list_length_matches_feature_names():
    """to_list() emits exactly one value per entry in FEATURE_NAMES."""
    box = make_box(1.0, 2.0)
    feats = _feats(box)
    assert len(feats.to_list()) == len(FEATURE_NAMES)


def test_extract_features_for_sample_two_experts():
    """Two experts each contributing an overlapping box see each other as peers."""
    boxes_cp = [make_box(0.0, 0.0, model="centerpoint")]
    boxes_pp = [make_box(0.5, 0.0, model="pointpillars")]

    result = extract_features_for_sample({
        "centerpoint": boxes_cp,
        "pointpillars": boxes_pp,
    })

    assert "centerpoint" in result
    assert "pointpillars" in result
    assert len(result["centerpoint"]) == 1
    assert len(result["pointpillars"]) == 1

    cp_feats = result["centerpoint"][0]
    assert cp_feats.n_peer_overlaps >= 1
    assert cp_feats.n_active_experts == 2
    assert math.isclose(cp_feats.expert_agreement, 1.0)


def test_n_active_experts_counted():
    """An expert contributing an empty box list doesn't count as active."""
    result = extract_features_for_sample({
        "centerpoint": [make_box(0.0, 0.0, model="centerpoint")],
        "pointpillars": [make_box(0.0, 0.0, model="pointpillars")],
        "voxelnext": [],   # no predictions → not active
    })
    assert result["centerpoint"][0].n_active_experts == 2


def test_pointpillars_expert_id():
    """pointpillars maps to expert_id 1, per _EXPERT_TO_ID."""
    box = make_box(0.0, 0.0, model="pointpillars")
    feats = _feats(box)
    assert feats.expert_id == 1


def test_centerpoint_pillar_expert_id():
    """centerpoint_pillar maps to expert_id 2, per _EXPERT_TO_ID."""
    box = make_box(0.0, 0.0, model="centerpoint_pillar")
    feats = _feats(box)
    assert feats.expert_id == 2


def test_ssn_expert_id():
    """ssn maps to expert_id 3, per _EXPERT_TO_ID."""
    box = make_box(0.0, 0.0, model="ssn")
    feats = _feats(box)
    assert feats.expert_id == 3


def test_bevfusion_lidar_expert_id():
    """bevfusion_lidar maps to expert_id 6, per _EXPERT_TO_ID."""
    box = make_box(0.0, 0.0, model="bevfusion_lidar")
    feats = _feats(box)
    assert feats.expert_id == 6


def test_unknown_expert_gets_fallback_id():
    """A model name absent from _EXPERT_TO_ID gets a fallback id past the known range."""
    box = make_box(0.0, 0.0, model="some_new_model")
    feats = _feats(box)
    assert feats.expert_id >= 7


# --- dist_to_drivable_area / map-mask tests -----------------------------

class _FakeMask:
    """Minimal stand-in for nuscenes.utils.map_mask.MapMask: on-mask iff
    the point is within `radius` of the origin (plus dilation)."""

    def __init__(self, radius: float = 5.0):
        self.radius = radius

    def is_on_mask(self, xs, ys, dilation: float = 0.0):
        import numpy as np
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        return (xs * xs + ys * ys) <= (self.radius + dilation) ** 2


def test_extract_features_missing_mask_gives_sentinel():
    """With mask=None, every box gets the DIST_TO_DRIVABLE_AREA_MISSING sentinel."""
    box = make_box(0.0, 0.0)
    feats = extract_features_for_sample({"centerpoint": [box]}, mask=None)
    assert feats["centerpoint"][0].dist_to_drivable_area == DIST_TO_DRIVABLE_AREA_MISSING


def test_extract_features_on_drivable_area_is_zero():
    """A box already on the mask gets distance 0.0."""
    box = make_box(0.0, 0.0)  # inside the fake mask's radius=5 disk
    feats = extract_features_for_sample({"centerpoint": [box]}, mask=_FakeMask(radius=5.0))
    assert feats["centerpoint"][0].dist_to_drivable_area == 0.0


def test_extract_features_off_drivable_area_positive_distance():
    """A box off the mask snaps to the smallest dilation level that covers the true gap."""
    box = make_box(10.0, 0.0)  # 10m out, outside radius=5 disk
    feats = extract_features_for_sample({"centerpoint": [box]}, mask=_FakeMask(radius=5.0))
    dist = feats["centerpoint"][0].dist_to_drivable_area
    assert dist > 0.0
    # true gap is 5m; smallest dilation level >= 5 in the candidate set is 8.0
    assert dist == 8.0


def test_extract_features_far_off_road_capped():
    """A box far beyond every dilation level is capped at _DRIVABLE_AREA_OFF_ROAD_CAP_M."""
    box = make_box(1000.0, 0.0)
    feats = extract_features_for_sample({"centerpoint": [box]}, mask=_FakeMask(radius=5.0))
    assert feats["centerpoint"][0].dist_to_drivable_area == 16.0


def test_dist_to_drivable_area_batch_matches_per_box():
    """The batched vectorized distance computation agrees with per-box expectations."""
    boxes = [make_box(0.0, 0.0), make_box(10.0, 0.0), make_box(1000.0, 0.0)]
    feats = extract_features_for_sample({"centerpoint": boxes}, mask=_FakeMask(radius=5.0))
    dists = [f.dist_to_drivable_area for f in feats["centerpoint"]]
    assert dists == [0.0, 8.0, 16.0]
