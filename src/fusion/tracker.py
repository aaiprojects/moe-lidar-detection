"""Simple offline, per-scene, per-class greedy BEV tracker.

Post-processing only: consumes a scene's final (post-router, post-NMS)
per-token detections in temporal order and associates them into tracks.
Unlike a real-time tracker, this runs offline over a whole scene at once,
so a box's final track evidence (total hit count across the scene) can
draw on both earlier and later frames -- there is no causality constraint
here, only association.

Used to explore whether track-based re-scoring (down-weighting one-off
"orphan" detections with no temporal corroboration, or boosting detections
that are part of a multi-frame track) can improve mAP on top of the
existing MoE + NMS pipeline. See docs/EXPERIMENT_LOG.md for the associated
experiment and its result -- this module by itself makes no claim that
tracking helps; that is an empirical question answered downstream.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace

from src.fusion.bev_iou import bev_iou
from src.io.schemas import DetectionBox
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# Per-class greedy-association gate: max center distance (metres) between a
# track's last-seen position and a new detection for them to be considered
# the same object, at nuScenes' 2Hz keyframe interval (0.5s between frames).
# Rough, unTuned defaults sized to each class's typical speed -- vehicles get
# a generous gate, static/slow classes get a tight one so nearby-but-distinct
# objects (e.g. two parked cars) don't get merged into one track.
DEFAULT_MAX_DIST_M: dict[str, float] = {
    "car": 4.0,
    "truck": 6.0,
    "bus": 6.0,
    "trailer": 6.0,
    "construction_vehicle": 4.0,
    "pedestrian": 1.0,
    "motorcycle": 4.0,
    "bicycle": 2.5,
    "traffic_cone": 1.0,
    "barrier": 1.0,
}
DEFAULT_MAX_MISSES = 1


@dataclass(frozen=True)
class TrackInfo:
    """Per-box track membership, aligned index-for-index with the input
    box list for that token."""

    track_id: int
    hit_count: int  # total frames this track appears in, across the whole scene


def get_scene_ordered_tokens(nusc, token_set: set[str]) -> list[list[str]]:
    """Return each scene's sample tokens in temporal order, filtered to
    ``token_set``, for scenes that have at least one token in the set.

    Same traversal pattern as scripts/make_scene_split.py
    (scene['first_sample_token'] -> sample['next']). Scene-grouped splits
    (docs/EXPERIMENT_LOG.md §23) put an entire scene on one side of the
    train/eval boundary, so filtering a scene's full token sequence down
    to ``token_set`` yields a contiguous run, not a token list with holes.
    """
    scenes: list[list[str]] = []
    for scene in nusc.scene:
        tokens: list[str] = []
        cur = scene["first_sample_token"]
        while cur:
            if cur in token_set:
                tokens.append(cur)
            sample = nusc.get("sample", cur)
            cur = sample["next"]
        if tokens:
            scenes.append(tokens)
    return scenes


class _Track:
    """Mutable, in-progress track state during a single class's pass over a
    scene (internal to track_scene; TrackInfo is the public, frozen
    per-box result)."""

    __slots__ = ("track_id", "last_pos", "last_frame_idx", "misses", "hit_count")

    def __init__(self, track_id: int, pos: tuple[float, float], frame_idx: int) -> None:
        self.track_id = track_id
        self.last_pos = pos
        self.last_frame_idx = frame_idx
        self.misses = 0
        self.hit_count = 1


def _greedy_match(
    tracks: list[_Track],
    dets: list[DetectionBox],
    max_dist: float,
) -> tuple[dict[int, int], set[int]]:
    """Greedy nearest-distance matching within a single class/frame.

    Returns (track_idx -> det_idx, matched_det_indices).
    """
    pairs: list[tuple[float, int, int]] = []
    for ti, t in enumerate(tracks):
        tx, ty = t.last_pos
        for di, d in enumerate(dets):
            dx = d.translation[0] - tx
            dy = d.translation[1] - ty
            dist = math.hypot(dx, dy)
            if dist <= max_dist:
                pairs.append((dist, ti, di))
    pairs.sort(key=lambda p: p[0])

    # Greedily consume the globally closest pairs first: once a track or
    # detection is claimed it's skipped for the rest of the pass, so each
    # track ends up matched to its nearest still-available detection
    # rather than the nearest detection overall.
    matched_tracks: set[int] = set()
    matched_dets: set[int] = set()
    track_to_det: dict[int, int] = {}
    for _, ti, di in pairs:
        if ti in matched_tracks or di in matched_dets:
            continue
        matched_tracks.add(ti)
        matched_dets.add(di)
        track_to_det[ti] = di

    return track_to_det, matched_dets


def track_scene(
    frames: list[tuple[str, list[DetectionBox]]],
    max_dist_m: dict[str, float] | None = None,
    max_misses: int = DEFAULT_MAX_MISSES,
) -> dict[str, list[TrackInfo]]:
    """Greedily track one scene's detections, independently per class.

    Args:
        frames: (sample_token, boxes) in temporal order for one scene.
            ``boxes`` may mix classes; tracking is run per-class internally.
        max_dist_m: Per-class association gate; falls back to
            DEFAULT_MAX_DIST_M for classes not listed, and 4.0m for
            unrecognised classes.
        max_misses: How many consecutive frames a track may go unmatched
            before it is retired (does not affect earlier-assigned hits).

    Returns:
        sample_token -> list[TrackInfo], aligned index-for-index with the
        ``boxes`` list passed in for that token. hit_count is the track's
        FINAL total hit count across the whole scene (this is an offline,
        non-causal pass -- see module docstring).
    """
    gates = dict(DEFAULT_MAX_DIST_M)
    if max_dist_m:
        gates.update(max_dist_m)

    classes: set[str] = set()
    for _, boxes in frames:
        classes.update(b.detection_name for b in boxes)

    # box_track[token][index] = _Track object (mutated in place; hit_count
    # read out at the end once every frame has been processed).
    box_track: dict[str, list[_Track | None]] = {
        token: [None] * len(boxes) for token, boxes in frames
    }

    next_track_id = 0
    for cls in classes:
        max_dist = gates.get(cls, 4.0)
        active: list[_Track] = []

        for frame_idx, (token, boxes) in enumerate(frames):
            cls_indices = [i for i, b in enumerate(boxes) if b.detection_name == cls]
            cls_dets = [boxes[i] for i in cls_indices]

            track_to_det, matched_dets = _greedy_match(active, cls_dets, max_dist)

            # Update every currently active track: matched tracks advance
            # to the new detection's position and reset their miss streak;
            # unmatched tracks accrue a miss and are dropped once they
            # exceed max_misses (retired tracks simply aren't carried into
            # `still_active`, so later frames can no longer match them).
            still_active: list[_Track] = []
            for ti, t in enumerate(active):
                if ti in track_to_det:
                    di = track_to_det[ti]
                    box_idx = cls_indices[di]
                    d = cls_dets[di]
                    t.last_pos = (d.translation[0], d.translation[1])
                    t.last_frame_idx = frame_idx
                    t.misses = 0
                    t.hit_count += 1
                    box_track[token][box_idx] = t
                    still_active.append(t)
                else:
                    t.misses += 1
                    if t.misses <= max_misses:
                        still_active.append(t)
            active = still_active

            # Any detection this frame that didn't match an existing track
            # starts a brand-new one (hit_count=1 until/unless a later
            # frame extends it).
            for di, d in enumerate(cls_dets):
                if di in matched_dets:
                    continue
                box_idx = cls_indices[di]
                new_track = _Track(
                    next_track_id, (d.translation[0], d.translation[1]), frame_idx
                )
                next_track_id += 1
                box_track[token][box_idx] = new_track
                active.append(new_track)

    result: dict[str, list[TrackInfo]] = {}
    for token, boxes in frames:
        infos = []
        for t in box_track[token]:
            assert t is not None
            infos.append(TrackInfo(track_id=t.track_id, hit_count=t.hit_count))
        result[token] = infos
    return result


def apply_track_rescoring(
    frames: list[tuple[str, list[DetectionBox]]],
    track_info: dict[str, list[TrackInfo]],
    orphan_penalty: float = 1.0,
    multi_hit_boost: float = 1.0,
    multi_hit_min: int = 2,
) -> dict[str, list[DetectionBox]]:
    """Multiply each box's detection_score by a track-evidence factor.

    Args:
        orphan_penalty: Multiplier applied to boxes whose track has
            hit_count == 1 (no temporal corroboration at all). <1.0
            down-weights them.
        multi_hit_boost: Multiplier applied to boxes whose track has
            hit_count >= multi_hit_min. >1.0 boosts them.
        multi_hit_min: Hit-count threshold for multi_hit_boost to apply.

    Returns:
        sample_token -> list[DetectionBox] with adjusted detection_score
        (scores are clipped to [0, 1]; all other fields unchanged).
    """
    out: dict[str, list[DetectionBox]] = {}
    for token, boxes in frames:
        infos = track_info[token]
        new_boxes = []
        for box, info in zip(boxes, infos):
            factor = 1.0
            if info.hit_count == 1:
                factor = orphan_penalty
            elif info.hit_count >= multi_hit_min:
                factor = multi_hit_boost
            new_score = min(1.0, max(0.0, box.detection_score * factor))
            new_boxes.append(replace(box, detection_score=new_score))
        out[token] = new_boxes
    return out


def _interpolate_box(
    b1: DetectionBox, b2: DetectionBox, token: str, score_discount: float
) -> DetectionBox:
    """Synthesize a box at the midpoint frame between two hits of the same
    track. Position/size linearly interpolated; rotation/velocity taken
    from the earlier hit (nuScenes TP matching is center-distance-based,
    not IoU, so position -- not orientation -- determines whether this
    recovers a match); score is the weaker of the two neighbors, discounted
    further since it is unobserved evidence, not a real detection."""
    mid_translation = [(x + y) / 2.0 for x, y in zip(b1.translation, b2.translation)]
    mid_size = [(x + y) / 2.0 for x, y in zip(b1.size, b2.size)]
    mid_velocity = [(x + y) / 2.0 for x, y in zip(b1.velocity, b2.velocity)]
    new_score = min(1.0, max(0.0, min(b1.detection_score, b2.detection_score) * score_discount))
    return replace(
        b1,
        sample_token=token,
        translation=mid_translation,
        size=mid_size,
        velocity=mid_velocity,
        detection_score=new_score,
        model_name=b1.model_name + "_track_interp",
    )


def interpolate_missed_frames(
    frames: list[tuple[str, list[DetectionBox]]],
    track_info: dict[str, list[TrackInfo]],
    score_discount: float = 0.5,
    dedup_iou_threshold: float = 0.3,
) -> dict[str, list[DetectionBox]]:
    """Recover likely false negatives: for every track with a hit at frame
    i and frame i+2 but no detection at frame i+1 (the exactly-one-frame
    gap that track_scene's default max_misses=1 tolerates when
    reconnecting), insert one synthesized box at frame i+1.

    Unlike apply_track_rescoring (which only reweights existing boxes),
    this adds new candidate boxes -- it changes recall directly, not just
    ranking. A wrong interpolation is a new false positive, not just a
    misranked box, so this is a higher-risk lever than rescoring; see
    docs/EXPERIMENT_LOG.md for its validated result (if any).

    This is a *per-track* gap check: it only looks at whether the track
    being interpolated has a hit at frame i+1, not whether some OTHER
    track already covers that same spot. A single real object can get
    fragmented into two tracks (most likely for long, thin, static classes
    like barrier, where a greedy nearest-neighbor tracker can lose the
    thread frame to frame) -- one track has a real hit at frame i+1, the
    other sees a gap there and would insert a near-duplicate right next to
    it. dedup_iou_threshold guards against exactly this: an interpolation
    candidate is dropped if it would overlap (same class, BEV IoU >= this
    threshold) an already-present box in the target frame -- the same
    "same object" criterion NMS itself uses (see configs/moe_final.yaml's
    ensemble.iou_threshold, which callers should pass here for consistency).

    Returns:
        sample_token -> list[DetectionBox]: original boxes unchanged, plus
        any interpolated insertions appended (skipping ones that would
        duplicate an already-present box).
    """
    track_hits: dict[int, list[tuple[int, DetectionBox]]] = defaultdict(list)
    for frame_idx, (token, boxes) in enumerate(frames):
        infos = track_info[token]
        for box, info in zip(boxes, infos):
            track_hits[info.track_id].append((frame_idx, box))

    out: dict[str, list[DetectionBox]] = {token: list(boxes) for token, boxes in frames}

    n_skipped_dupes = 0
    for hits in track_hits.values():
        if len(hits) < 2:
            continue
        hits.sort(key=lambda h: h[0])
        for (i1, b1), (i2, b2) in zip(hits, hits[1:]):
            if i2 - i1 == 2:
                gap_token = frames[i1 + 1][0]
                candidate = _interpolate_box(b1, b2, gap_token, score_discount)
                if _duplicates_existing_box(candidate, out[gap_token], dedup_iou_threshold):
                    n_skipped_dupes += 1
                    continue
                out[gap_token].append(candidate)

    if n_skipped_dupes:
        log.info(
            "interpolate_missed_frames: skipped %d candidate(s) that would have "
            "duplicated an already-present box (BEV IoU >= %.2f)",
            n_skipped_dupes, dedup_iou_threshold,
        )

    return out


def _duplicates_existing_box(
    candidate: DetectionBox, existing_boxes: list[DetectionBox], iou_threshold: float
) -> bool:
    """True if candidate has BEV IoU >= iou_threshold with any same-class
    box already in existing_boxes (real detections or earlier-inserted
    interpolations in this same frame)."""
    for box in existing_boxes:
        if box.detection_name != candidate.detection_name:
            continue
        if bev_iou(candidate, box) >= iou_threshold:
            return True
    return False
