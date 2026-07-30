"""Static bird's-eye-view (BEV) visualization of predictions vs. ground truth.

Renders one keyframe at a time as a matplotlib figure (point cloud + box
overlays) and returns/saves it -- no interactive window, so this works in a
notebook or a headless script. Adapted from the original project's
interactive players (scripts/play_nuscenes.py / play_nuscenes_viz.py in the
research repo), keeping only the non-interactive drawing primitives.

Requires the raw nuScenes LiDAR sweeps (data/nuscenes/samples/LIDAR_TOP/) --
NOT part of the Drive-hosted CSVs/predictions/weights, since that's ~306GB.
This is an optional, advanced-path utility for anyone with the full dataset;
see docs/expert_regeneration.md.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

CLASS_COLOURS: dict[str, str] = {
    "car": "#2196F3",
    "truck": "#FF9800",
    "bus": "#9C27B0",
    "trailer": "#795548",
    "construction_vehicle": "#F44336",
    "pedestrian": "#4CAF50",
    "motorcycle": "#FFEB3B",
    "bicycle": "#00BCD4",
    "traffic_cone": "#E91E63",
    "barrier": "#607D8B",
}


def read_pcd_bin(path: Path) -> np.ndarray:
    """Read a nuScenes .pcd.bin file -> (N, 5) array [x, y, z, intensity, ring]."""
    pts = np.fromfile(path, dtype=np.float32)
    if pts.size % 5 != 0:
        pts = pts[: (pts.size // 4) * 4].reshape(-1, 4)
        pts = np.hstack([pts, np.zeros((pts.shape[0], 1), dtype=np.float32)])
    else:
        pts = pts.reshape(-1, 5)
    return pts


def quat_to_yaw(q: list[float]) -> float:
    """Extract yaw from a [qw, qx, qy, qz] quaternion."""
    qw, qx, qy, qz = q
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2)))


def global_box_to_ego(box: dict, ego_pose: dict) -> dict:
    """Convert a box dict (translation/rotation in global frame) to ego frame."""
    ego_t = np.array(ego_pose["translation"])
    ego_yaw = quat_to_yaw(ego_pose["rotation"])

    c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
    r_inv = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    g_t = np.array(box["translation"])
    local_t = r_inv @ (g_t - ego_t)

    local_yaw = quat_to_yaw(box["rotation"]) - ego_yaw

    new_box = dict(box)
    new_box["translation"] = local_t.tolist()
    new_box["rotation"] = [float(np.cos(local_yaw / 2)), 0.0, 0.0, float(np.sin(local_yaw / 2))]
    return new_box


def bev_color_by_height(z: np.ndarray, z_min: float = -3.0, z_max: float = 5.0) -> np.ndarray:
    z_norm = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)
    return plt.cm.plasma(z_norm)


def draw_bev_box(ax: plt.Axes, box: dict, style: str = "solid", label_score: bool = True) -> None:
    """Draw one box as a BEV rectangle. style='solid' for predictions
    (filled outline, class-coloured, score label), 'dashed' for ground
    truth (white dashed outline, no score)."""
    name = box.get("detection_name", "car")
    colour = "white" if style == "dashed" else CLASS_COLOURS.get(name, "#ffffff")
    tx, ty, _ = box["translation"]
    w, l, _ = box["size"]
    heading = quat_to_yaw(box.get("rotation", [1, 0, 0, 0]))

    corners = np.array([[l / 2, w / 2], [-l / 2, w / 2], [-l / 2, -w / 2], [l / 2, -w / 2]])
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    corners = (rot @ corners.T).T + np.array([tx, ty])

    poly = mpatches.Polygon(
        corners, closed=True, edgecolor=colour, facecolor="none",
        linewidth=1.4 if style == "solid" else 1.0,
        linestyle="solid" if style == "solid" else (0, (4, 2)),
        alpha=0.9,
    )
    ax.add_patch(poly)

    front = rot @ np.array([l / 2, 0]) + np.array([tx, ty])
    ax.plot([tx, front[0]], [ty, front[1]], color=colour, linewidth=0.8, alpha=0.7)

    if label_score and style == "solid":
        score = box.get("detection_score", 1.0)
        ax.text(tx, ty, f"{name[:3]} {score:.2f}", fontsize=5, color=colour, ha="center", va="center")


def _draw_panel(
    ax: plt.Axes,
    points: np.ndarray,
    pred_boxes: list[dict],
    gt_boxes: list[dict],
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    point_size: float = 0.5,
) -> None:
    ax.set_facecolor("#0b0b0b")
    colours = bev_color_by_height(points[:, 2])
    ax.scatter(points[:, 0], points[:, 1], s=point_size, c=colours, alpha=0.6)

    for box in gt_boxes:
        draw_bev_box(ax, box, style="dashed", label_score=False)
    for box in pred_boxes:
        draw_bev_box(ax, box, style="solid", label_score=True)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_title(title, color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("white")


def _legend_handles() -> list[mpatches.Patch]:
    handles = [mpatches.Patch(edgecolor=c, facecolor="none", label=n) for n, c in CLASS_COLOURS.items()]
    handles.append(mpatches.Patch(edgecolor="white", facecolor="none", label="ground truth", linestyle="--"))
    return handles


def render_bev_frame(
    points: np.ndarray,
    pred_boxes: list[dict],
    gt_boxes: list[dict],
    title: str = "",
    xlim: tuple[float, float] = (-50, 50),
    ylim: tuple[float, float] = (-50, 50),
    figsize: tuple[float, float] = (9, 9),
) -> plt.Figure:
    """Render one BEV frame: point cloud + predicted boxes (solid, class-
    coloured) + ground truth boxes (white dashed) for visual comparison.
    All boxes must already be in ego-vehicle frame (see global_box_to_ego).
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0b0b0b")
    _draw_panel(ax, points, pred_boxes, gt_boxes, title, xlim, ylim)
    ax.legend(
        handles=_legend_handles(), loc="upper right", fontsize=6,
        facecolor="#1a1a1a", labelcolor="white", ncol=2,
    )
    fig.tight_layout()
    return fig


def render_expert_comparison_grid(
    points: np.ndarray,
    pred_boxes_by_source: dict[str, list[dict]],
    gt_boxes: list[dict],
    suptitle: str = "",
    xlim: tuple[float, float] = (-50, 50),
    ylim: tuple[float, float] = (-50, 50),
    n_cols: int = 3,
    panel_size: float = 4.5,
) -> plt.Figure:
    """Grid of BEV panels, one per prediction source (e.g. each of the 5
    frozen experts plus the final MoE ensemble), all sharing the same point
    cloud and ground truth so they're directly comparable side by side.

    Args:
        pred_boxes_by_source: source name -> predicted boxes (ego frame),
            in the order panels should appear. Put the ensemble last so it
            reads as "the answer" after seeing each expert alone.
    """
    n = len(pred_boxes_by_source)
    n_cols = min(n_cols, n)
    n_rows = -(-n // n_cols)  # ceil

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(panel_size * n_cols, panel_size * n_rows), squeeze=False,
    )
    fig.patch.set_facecolor("#0b0b0b")

    n_gt = len(gt_boxes)
    for i, (source, boxes) in enumerate(pred_boxes_by_source.items()):
        ax = axes[i // n_cols][i % n_cols]
        _draw_panel(
            ax, points, boxes, gt_boxes,
            title=f"{source}  ({len(boxes)} pred / {n_gt} GT)",
            xlim=xlim, ylim=ylim, point_size=0.35,
        )

    # Hide unused axes if the grid isn't fully filled.
    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.legend(
        handles=_legend_handles(), loc="lower center", fontsize=8,
        facecolor="#1a1a1a", labelcolor="white", ncol=6, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(suptitle, color="white", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig
