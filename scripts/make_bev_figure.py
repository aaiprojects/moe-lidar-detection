"""Render the qualitative BEV comparison figure for the report.

One held-out keyframe seen by each frozen expert and by the fused system, laid
out 2x3 so the whole grid stays legible at the report's 3.25 in column width.
Panel titles carry the quantitative content (ground-truth objects recovered),
so the figure still reads when the individual boxes are small.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.bev_viz import (  # noqa: E402
    CLASS_COLOURS,
    quat_to_yaw,
    read_pcd_bin,
)

# Defaults to data/nuscenes inside the repo; override with the NUSCENES_ROOT
# environment variable if the dataset lives elsewhere on your machine.
NUSCENES_ROOT = Path(os.environ.get("NUSCENES_ROOT", REPO / "data" / "nuscenes"))
OUT = REPO / "docs" / "figures" / "fig13_bev_comparison.png"

EXPERTS = {
    "centerpoint": "CenterPoint-Voxel",
    "centerpoint_pillar": "CenterPoint-Pillar",
    "pointpillars": "PointPillars",
    "ssn": "SSN",
    "bevfusion_lidar": "BEVFusion-LiDAR",
}
PANEL_ORDER = [
    "CenterPoint-Voxel", "CenterPoint-Pillar", "PointPillars",
    "SSN", "BEVFusion-LiDAR", "Fused MoE",
]
SCORE_MIN = 0.35
MATCH_DIST = 2.0
TARGET_SCENE = "scene-0097"

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 400, "savefig.bbox": "tight"})


def to_ego(box: dict, ego_t: np.ndarray, ego_yaw: float) -> tuple[float, float, float]:
    """Global box centre and heading -> ego-frame centre and heading."""
    c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
    g = np.array(box["translation"][:2]) - ego_t[:2]
    return (
        float(c * g[0] - s * g[1]),
        float(s * g[0] + c * g[1]),
        quat_to_yaw(box["rotation"]) - ego_yaw,
    )


def matched_set(preds: list[dict], gts: list[dict]) -> set[int]:
    """Indices of ground-truth objects recovered, greedy one-to-one by score."""
    used: set[int] = set()
    for p in sorted(preds, key=lambda b: -b["score"]):
        best, best_d = None, MATCH_DIST
        for j, g in enumerate(gts):
            if j in used or g["name"] != p["name"]:
                continue
            d = np.hypot(p["x"] - g["x"], p["y"] - g["y"])
            if d < best_d:
                best, best_d = j, d
        if best is not None:
            used.add(best)
    return used


def recovered(preds: list[dict], gts: list[dict]) -> int:
    """Count of ground-truth objects recovered by these predictions."""
    return len(matched_set(preds, gts))


def draw(ax, points, preds, gts, title):
    """Draw one panel: LiDAR points, dashed GT boxes, solid prediction boxes."""
    ax.scatter(points[:, 0], points[:, 1], s=0.12, c="#8e979f", alpha=0.75,
               linewidths=0, rasterized=True)
    for g in gts:
        _rect(ax, g, "#000000", 0.75, (0, (1.8, 1.2)))
    for p in preds:
        _rect(ax, p, CLASS_COLOURS.get(p["name"], "#333333"), 1.15, "solid")
    ax.set_title(title, fontsize=8.2, pad=2.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("0.55")


def _rect(ax, b, colour, lw, ls):
    """Draw one box dict (ego-frame x/y/yaw/w/l) as a rotated rectangle outline."""
    w, l = b["w"], b["l"]
    corners = np.array([[l / 2, w / 2], [-l / 2, w / 2], [-l / 2, -w / 2], [l / 2, -w / 2]])
    c, s = np.cos(b["yaw"]), np.sin(b["yaw"])
    corners = (np.array([[c, -s], [s, c]]) @ corners.T).T + np.array([b["x"], b["y"]])
    ax.add_patch(mpatches.Polygon(corners, closed=True, edgecolor=colour,
                                  facecolor="none", linewidth=lw, linestyle=ls))


def main() -> None:
    """Pick the densest held-out keyframe in TARGET_SCENE and render the
    six-panel per-expert-vs-fused BEV comparison figure."""
    from nuscenes import NuScenes

    split = json.loads((REPO / "training_data" / "token_split_3way.json").read_text())
    test_tokens = set(split["test_tokens"])
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(NUSCENES_ROOT), verbose=False)

    scene = next((s for s in nusc.scene if s["name"] == TARGET_SCENE), None)
    if scene is None:
        raise SystemExit(f"{TARGET_SCENE} not found")

    tokens, tok = [], scene["first_sample_token"]
    while tok:
        if tok in test_tokens:
            tokens.append(tok)
        tok = nusc.get("sample", tok)["next"]
    if not tokens:
        raise SystemExit(f"{TARGET_SCENE} is not in the held-out test split")
    print(f"{TARGET_SCENE}: {len(tokens)} held-out keyframes")

    # Prediction sources, loaded once and indexed by sample token.
    sources: dict[str, dict[str, list]] = {}
    for key, label in EXPERTS.items():
        raw = json.loads((REPO / "predictions" / key / "predictions.json").read_text())
        res = raw.get("results", raw)
        sources[label] = res
        print(f"  loaded {label}")
    moe = json.loads((REPO / "outputs" / "submission.json").read_text())["results"]
    sources["Fused MoE"] = moe
    print("  loaded Fused MoE")

    # Pick the keyframe with the most ground truth, for a dense, readable scene.
    best_tok = max(tokens, key=lambda t: len(nusc.get("sample", t)["anns"]))
    sample = nusc.get("sample", best_tok)
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    ego_t, ego_yaw = np.array(ego["translation"]), quat_to_yaw(ego["rotation"])

    gts = []
    for ann_tok in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_tok)
        name = _nusc_to_det(ann["category_name"])
        if name is None or ann["num_lidar_pts"] == 0:
            continue
        x, y, yaw = to_ego(ann, ego_t, ego_yaw)
        gts.append({"name": name, "x": x, "y": y, "w": ann["size"][0],
                    "l": ann["size"][1], "yaw": yaw})
    print(f"keyframe {best_tok}: {len(gts)} ground-truth objects")

    # LiDAR points, sensor frame -> ego frame.
    cal = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pts = read_pcd_bin(NUSCENES_ROOT / sd["filename"])[:, :3]
    cy = quat_to_yaw(cal["rotation"])
    c, s = np.cos(cy), np.sin(cy)
    pts = np.column_stack([
        c * pts[:, 0] - s * pts[:, 1] + cal["translation"][0],
        s * pts[:, 0] + c * pts[:, 1] + cal["translation"][1],
        pts[:, 2] + cal["translation"][2],
    ])

    panels = {}
    for label in PANEL_ORDER:
        boxes = []
        for b in sources[label].get(best_tok, []):
            if b["detection_score"] < SCORE_MIN:
                continue
            x, y, yaw = to_ego(b, ego_t, ego_yaw)
            boxes.append({"name": b["detection_name"], "x": x, "y": y,
                          "w": b["size"][0], "l": b["size"][1], "yaw": yaw,
                          "score": b["detection_score"]})
        panels[label] = boxes

    # Keep the full annotated extent. A tighter crop looks better but removes
    # the far-field objects that separate the systems: inside 27 m three of the
    # six panels saturate at 22 of 22 and the comparison disappears.
    # Frame on the annotated objects, not on predictions: a single far-field
    # false positive would otherwise stretch the view and shrink everything.
    xs = [g["x"] for g in gts]
    ys = [g["y"] for g in gts]
    pad = 7.0
    xlim = (max(min(xs) - pad, -55), min(max(xs) + pad, 55))
    ylim = (max(min(ys) - pad, -55), min(max(ys) + pad, 55))
    pts = pts[(pts[:, 0] > xlim[0]) & (pts[:, 0] < xlim[1])
              & (pts[:, 1] > ylim[0]) & (pts[:, 1] < ylim[1])]
    print(f"view: x {xlim[0]:.0f}..{xlim[1]:.0f} m, y {ylim[0]:.0f}..{ylim[1]:.0f} m")

    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.6))
    for ax, label in zip(axes.ravel(), PANEL_ORDER):
        hits = recovered(panels[label], gts)
        draw(ax, pts, panels[label], gts, f"{label}\n{hits} of {len(gts)} found")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        print(f"  {label:<20} {hits:>3} of {len(gts)}  ({len(panels[label])} boxes)")

    # How much of this frame's signal is genuinely complementary, and how much
    # of that the fused system keeps.
    per_expert = {lbl: matched_set(panels[lbl], gts) for lbl in PANEL_ORDER[:5]}
    counts = {j: sum(j in s for s in per_expert.values()) for j in range(len(gts))}
    unique = {j for j, n in counts.items() if n == 1}
    missed_by_all = {j for j, n in counts.items() if n == 0}
    fused = matched_set(panels["Fused MoE"], gts)
    print(f"\n  found by exactly one expert : {len(unique)}"
          f"  (fused keeps {len(unique & fused)})")
    print(f"  found by no expert          : {len(missed_by_all)}")
    print(f"  found by all five experts   : {sum(1 for n in counts.values() if n == 5)}")

    handles = [
        mpatches.Patch(edgecolor="#111111", facecolor="none", linestyle="--",
                       linewidth=0.8, label="ground truth"),
        mpatches.Patch(edgecolor=CLASS_COLOURS["car"], facecolor="none",
                       linewidth=1.0, label="prediction (colour = class)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.012))
    fig.tight_layout(rect=(0, 0.022, 1, 1), h_pad=0.9, w_pad=0.6)
    fig.savefig(OUT)
    print(f"\nwrote {OUT}")


def _nusc_to_det(category: str) -> str | None:
    """Map a raw nuScenes annotation category to a detection class name,
    or None if this category has no detection-eval counterpart."""
    mapping = {
        "vehicle.car": "car", "vehicle.truck": "truck", "vehicle.bus.rigid": "bus",
        "vehicle.bus.bendy": "bus", "vehicle.trailer": "trailer",
        "vehicle.construction": "construction_vehicle",
        "human.pedestrian.adult": "pedestrian", "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "vehicle.motorcycle": "motorcycle", "vehicle.bicycle": "bicycle",
        "movable_object.trafficcone": "traffic_cone",
        "movable_object.barrier": "barrier",
    }
    return mapping.get(category)


if __name__ == "__main__":
    main()
