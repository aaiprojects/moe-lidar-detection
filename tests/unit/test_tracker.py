"""Tests for src/fusion/tracker.py"""

from src.fusion.tracker import TrackInfo, apply_track_rescoring, interpolate_missed_frames, track_scene
from src.io.schemas import DetectionBox

_Q = [1.0, 0.0, 0.0, 0.0]


def make_box(x: float, y: float, score: float, cls: str = "car") -> DetectionBox:
    """Build a DetectionBox at (x, y) with the given score and class."""
    return DetectionBox(
        sample_token="tok",
        model_name="m",
        translation=[x, y, 0.0],
        size=[2.0, 4.0, 1.5],
        rotation=_Q,
        velocity=[0.0, 0.0],
        detection_name=cls,
        detection_score=score,
        attribute_name=None,
        frame="global",
    )


def test_single_frame_scene_all_orphans():
    """With only one frame, every box starts its own track with hit_count=1."""
    frames = [("t0", [make_box(0.0, 0.0, 0.9), make_box(50.0, 0.0, 0.8)])]
    result = track_scene(frames)
    infos = result["t0"]
    assert len(infos) == 2
    assert all(i.hit_count == 1 for i in infos)
    assert infos[0].track_id != infos[1].track_id


def test_stationary_object_tracked_across_frames():
    """A near-stationary object stays under one track_id across 3 frames,
    accumulating hit_count=3."""
    frames = [
        ("t0", [make_box(0.0, 0.0, 0.9)]),
        ("t1", [make_box(0.2, 0.1, 0.85)]),
        ("t2", [make_box(0.1, -0.1, 0.88)]),
    ]
    result = track_scene(frames)
    ids = [result[t][0].track_id for t in ["t0", "t1", "t2"]]
    assert ids[0] == ids[1] == ids[2]
    assert result["t0"][0].hit_count == 3


def test_far_apart_detections_form_separate_tracks():
    """Two detections beyond the per-class distance gate never join the same track."""
    frames = [
        ("t0", [make_box(0.0, 0.0, 0.9)]),
        ("t1", [make_box(100.0, 0.0, 0.9)]),
    ]
    result = track_scene(frames)
    assert result["t0"][0].track_id != result["t1"][0].track_id
    assert result["t0"][0].hit_count == 1
    assert result["t1"][0].hit_count == 1


def test_different_classes_never_share_a_track():
    """Tracking runs per-class internally, so a car and a pedestrian at
    nearly the same spot are never assigned to the same track."""
    frames = [
        ("t0", [make_box(0.0, 0.0, 0.9, cls="car")]),
        ("t1", [make_box(0.1, 0.0, 0.9, cls="pedestrian")]),
    ]
    result = track_scene(frames)
    assert result["t0"][0].track_id != result["t1"][0].track_id
    assert result["t0"][0].hit_count == 1
    assert result["t1"][0].hit_count == 1


def test_missed_frame_still_reconnects_within_max_misses():
    """A track survives one missed frame (default max_misses=1) and reconnects."""
    # object present at t0 and t2, absent at t1 (occlusion) -- should still
    # be one track since max_misses defaults to 1.
    frames = [
        ("t0", [make_box(0.0, 0.0, 0.9)]),
        ("t1", []),
        ("t2", [make_box(0.2, 0.0, 0.9)]),
    ]
    result = track_scene(frames)
    assert result["t0"][0].track_id == result["t2"][0].track_id
    assert result["t0"][0].hit_count == 2


def test_two_close_objects_each_get_own_track():
    """Two distinct objects each keep their own track_id across frames,
    even when both are moving and both are within the association gate."""
    frames = [
        ("t0", [make_box(0.0, 0.0, 0.9), make_box(20.0, 0.0, 0.8)]),
        ("t1", [make_box(0.1, 0.0, 0.9), make_box(20.1, 0.0, 0.8)]),
    ]
    result = track_scene(frames)
    t0_ids = {result["t0"][0].track_id, result["t0"][1].track_id}
    t1_ids = {result["t1"][0].track_id, result["t1"][1].track_id}
    assert len(t0_ids) == 2
    assert t0_ids == t1_ids


def test_rescoring_penalises_orphans_and_boosts_multi_hit():
    """apply_track_rescoring lowers a single-hit orphan's score and raises a
    multi-hit track's score, for every box that track touches."""
    frames = [
        ("t0", [make_box(0.0, 0.0, 0.6), make_box(50.0, 0.0, 0.6)]),
        ("t1", [make_box(0.1, 0.0, 0.6)]),
    ]
    track_info = track_scene(frames)
    rescored = apply_track_rescoring(
        frames, track_info, orphan_penalty=0.5, multi_hit_boost=1.2, multi_hit_min=2
    )
    # t0 box 0 is part of a 2-hit track -> boosted
    assert rescored["t0"][0].detection_score > 0.6
    # t0 box 1 is a 1-hit orphan -> penalised
    assert rescored["t0"][1].detection_score < 0.6
    # t1 box 0 is part of the same 2-hit track -> boosted
    assert rescored["t1"][0].detection_score > 0.6


def test_rescoring_clips_score_to_valid_range():
    """An orphan_penalty > 1.0 that would push the score above 1.0 is clipped to [0, 1]."""
    frames = [("t0", [make_box(0.0, 0.0, 0.95)])]
    track_info = track_scene(frames)
    rescored = apply_track_rescoring(frames, track_info, multi_hit_boost=1.0, orphan_penalty=3.0)
    assert 0.0 <= rescored["t0"][0].detection_score <= 1.0


# --- interpolate_missed_frames --------------------------------------------

def test_interpolates_single_frame_gap():
    """A one-frame gap between two hits of the same track gets a midpoint
    box synthesized at the missed frame, scored at min(neighbor scores) *
    score_discount."""
    frames = [
        ("t0", [make_box(0.0, 0.0, 0.8)]),
        ("t1", []),  # missed
        ("t2", [make_box(2.0, 0.0, 0.6)]),
    ]
    track_info = track_scene(frames)
    result = interpolate_missed_frames(frames, track_info, score_discount=0.5)
    assert len(result["t0"]) == 1
    assert len(result["t2"]) == 1
    assert len(result["t1"]) == 1  # one interpolated box inserted
    interp = result["t1"][0]
    assert interp.translation[0] == 1.0  # midpoint of 0.0 and 2.0
    assert interp.translation[1] == 0.0
    assert interp.detection_score == 0.6 * 0.5  # min(0.8, 0.6) * discount
    assert interp.sample_token == "t1"


def test_no_interpolation_when_no_gap():
    """Consecutive real hits with no gap between them get no interpolated boxes."""
    frames = [
        ("t0", [make_box(0.0, 0.0, 0.8)]),
        ("t1", [make_box(0.2, 0.0, 0.8)]),
    ]
    track_info = track_scene(frames)
    result = interpolate_missed_frames(frames, track_info)
    assert len(result["t0"]) == 1
    assert len(result["t1"]) == 1


def test_no_interpolation_for_orphan_single_frame_track():
    """A track with only one hit has no gap to bridge, so output equals input unchanged."""
    frames = [("t0", [make_box(0.0, 0.0, 0.9)])]
    track_info = track_scene(frames)
    result = interpolate_missed_frames(frames, track_info)
    assert result == {"t0": frames[0][1]}


def test_no_interpolation_for_gap_larger_than_one_frame():
    """A two-frame gap is NOT bridged (only exactly-one-frame gaps qualify)."""
    # track dies after 1 miss (default max_misses=1), so a 2-frame gap
    # produces two SEPARATE tracks, not one bridgeable track -- no
    # interpolation should be inserted for the middle frames.
    frames = [
        ("t0", [make_box(0.0, 0.0, 0.8)]),
        ("t1", []),
        ("t2", []),
        ("t3", [make_box(3.0, 0.0, 0.6)]),
    ]
    track_info = track_scene(frames)
    result = interpolate_missed_frames(frames, track_info)
    assert len(result["t1"]) == 0
    assert len(result["t2"]) == 0


def test_interpolated_box_preserves_class_and_uses_midpoint_size():
    """An interpolated box keeps the track's class and gets the midpoint of
    its two neighbors' sizes (here, identical sizes on both sides)."""
    frames = [
        ("t0", [make_box(0.0, 0.0, 0.8, cls="pedestrian")]),
        ("t1", []),
        ("t2", [make_box(2.0, 0.0, 0.6, cls="pedestrian")]),
    ]
    track_info = track_scene(frames, max_dist_m={"pedestrian": 5.0})
    result = interpolate_missed_frames(frames, track_info)
    interp = result["t1"][0]
    assert interp.detection_name == "pedestrian"
    assert interp.size == [2.0, 4.0, 1.5]


def test_original_boxes_unaffected_by_interpolation():
    """interpolate_missed_frames only appends new boxes; it never mutates
    the scores of the original real detections."""
    frames = [
        ("t0", [make_box(0.0, 0.0, 0.8)]),
        ("t1", []),
        ("t2", [make_box(2.0, 0.0, 0.6)]),
    ]
    track_info = track_scene(frames)
    result = interpolate_missed_frames(frames, track_info)
    assert result["t0"][0].detection_score == 0.8
    assert result["t2"][0].detection_score == 0.6


# --- interpolate_missed_frames: duplicate-avoidance fix ---------------------
# A single real object can get fragmented into two tracks (e.g. barrier
# segments, where a greedy nearest-neighbor tracker can lose the thread
# frame to frame): one track has a real hit at the gap frame, the other
# sees a gap there and would insert a near-duplicate right next to it.
# These bypass track_scene and hand-construct track_info directly, so the
# "two tracks collide at the gap frame" scenario doesn't depend on
# replicating the exact fragmentation mechanics of the greedy tracker.

def test_interpolation_skipped_when_it_would_duplicate_existing_box():
    """An interpolation candidate is dropped when it would overlap a
    same-class box a different track already placed in that frame."""
    frames = [
        ("t0", [make_box(0.0, 0.0, 0.8)]),      # track 1, hit 1
        ("t1", [make_box(1.0, 0.0, 0.5)]),      # track 2 (different object/track), sits
                                                 # exactly where track 1's interpolation
                                                 # candidate (midpoint of t0/t2) would land
        ("t2", [make_box(2.0, 0.0, 0.6)]),      # track 1, hit 2
    ]
    track_info = {
        "t0": [TrackInfo(track_id=1, hit_count=2)],
        "t1": [TrackInfo(track_id=2, hit_count=1)],
        "t2": [TrackInfo(track_id=1, hit_count=2)],
    }
    result = interpolate_missed_frames(frames, track_info, dedup_iou_threshold=0.3)
    # No interpolated box added -- only track 2's original real box remains.
    assert len(result["t1"]) == 1
    assert result["t1"][0].detection_score == 0.5  # untouched original, not the ~0.3 interpolated score


def test_interpolation_still_happens_when_existing_box_is_far_away():
    """A same-class box in the gap frame that's spatially far from the
    interpolation candidate does NOT block the interpolation."""
    frames = [
        ("t0", [make_box(0.0, 0.0, 0.8)]),
        ("t1", [make_box(20.0, 0.0, 0.5)]),  # same class, but nowhere near the gap midpoint
        ("t2", [make_box(2.0, 0.0, 0.6)]),
    ]
    track_info = {
        "t0": [TrackInfo(track_id=1, hit_count=2)],
        "t1": [TrackInfo(track_id=2, hit_count=1)],
        "t2": [TrackInfo(track_id=1, hit_count=2)],
    }
    result = interpolate_missed_frames(frames, track_info, dedup_iou_threshold=0.3)
    assert len(result["t1"]) == 2  # original far-away box + the new interpolated one
    scores = {round(b.detection_score, 3) for b in result["t1"]}
    assert 0.5 in scores  # original
    assert 0.3 in scores  # min(0.8, 0.6) * default discount 0.5


def test_interpolation_still_happens_when_existing_box_is_different_class():
    """A different-class box occupying the same spot in the gap frame does
    NOT block the interpolation (dedup is same-class only)."""
    frames = [
        ("t0", [make_box(0.0, 0.0, 0.8, cls="car")]),
        ("t1", [make_box(1.0, 0.0, 0.5, cls="pedestrian")]),  # same spot, different class
        ("t2", [make_box(2.0, 0.0, 0.6, cls="car")]),
    ]
    track_info = {
        "t0": [TrackInfo(track_id=1, hit_count=2)],
        "t1": [TrackInfo(track_id=2, hit_count=1)],
        "t2": [TrackInfo(track_id=1, hit_count=2)],
    }
    result = interpolate_missed_frames(frames, track_info, dedup_iou_threshold=0.3)
    assert len(result["t1"]) == 2
    assert any(b.detection_name == "car" for b in result["t1"])
    assert any(b.detection_name == "pedestrian" for b in result["t1"])
