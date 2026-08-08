#!/usr/bin/env python3
"""Draw the system architecture figure used in the report (Figure 9).

The previous version of this diagram lived only as a bitmap pasted into the
.docx, which meant it silently went stale: it still showed 15 features, a
single GBT gating model, and 10 saved models, none of which describe the
current pipeline. Generating it from code keeps it in step with
configs/moe_final.yaml, and puts the layout under version control.

The figure is reproduced in a two-column report at roughly 3.3 in wide, so it
is drawn tall and narrow rather than wide: the offline and online phases are
stacked instead of placed side by side, which keeps the type legible after the
figure is scaled down to fit the column.

    python scripts/make_architecture_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "figures" / "fig9_architecture.png"

EXPERT = "#3f8f5b"
POOL = "#d1732a"
FEATURE = "#2c6fad"
MODEL = "#9c3535"
NEUTRAL = "#6b7280"

EXPERTS = ["CenterPoint\nVoxel", "CenterPoint\nPillar", "Point\nPillars", "SSN", "BEVFusion\nLiDAR"]

# The figure is placed across both columns at 6.9 in. Type is scaled up so it
# stays legible after that reduction.
FONT_SCALE = 1.15


def box(ax, cx, cy, w, h, text, color, fontsize=7.5, text_color="white"):
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            linewidth=0, facecolor=color, zorder=2,
        )
    )
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize * FONT_SCALE,
            color=text_color, zorder=3, linespacing=1.25)
    return cy - h / 2, cy + h / 2


def arrow(ax, x0, y0, x1, y1, color="#44464a"):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
            linewidth=0.8, color=color, shrinkA=0, shrinkB=0, zorder=1,
        )
    )


def expert_row(ax, y, h=0.062):
    """Five experts side by side, returning the bottom edge."""
    n = len(EXPERTS)
    w = 0.175
    gap = (1.0 - 0.04 - n * w) / (n - 1)
    x = 0.02 + w / 2
    for label in EXPERTS:
        box(ax, x, y, w, h, label, EXPERT, fontsize=5.6)
        x += w + gap
    return y - h / 2


def fan_in(ax, y_from, y_to):
    """Arrows from the five experts down to a single pooling box."""
    n = len(EXPERTS)
    w = 0.175
    gap = (1.0 - 0.04 - n * w) / (n - 1)
    xs = [0.02 + w / 2 + i * (w + gap) for i in range(n)]
    mid = (y_from + y_to) / 2
    for x in xs:
        ax.plot([x, x], [y_from, mid], color="#44464a", linewidth=0.8, zorder=1)
    ax.plot([min(xs), max(xs)], [mid, mid], color="#44464a", linewidth=0.8, zorder=1)
    arrow(ax, 0.5, mid, 0.5, y_to)


def panel(ax, title, facecolor, edgecolor):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.005, 0.005), 0.99, 0.99,
            boxstyle="round,pad=0.002,rounding_size=0.015",
            linewidth=0.9, edgecolor=edgecolor, facecolor=facecolor, zorder=0,
        )
    )
    ax.text(0.5, 0.965, title, ha="center", va="center",
            fontsize=8.5 * FONT_SCALE, color=edgecolor, fontweight="bold")


def draw_fitting(ax):
    panel(ax, "ROUTER FITTING  —  once, offline", "#eef3fa", "#1f5fa8")

    bottom = expert_row(ax, 0.875)
    fan_in(ax, bottom, 0.795)

    b, _ = box(ax, 0.5, 0.762, 0.60, 0.052, "Pool all predicted boxes", POOL)
    arrow(ax, 0.5, b, 0.5, 0.688)

    b, _ = box(ax, 0.5, 0.655, 0.86, 0.052,
               "Extract 17 features per box", FEATURE)
    arrow(ax, 0.5, b, 0.5, 0.578)

    b, _ = box(ax, 0.5, 0.535, 0.90, 0.072,
               "Label against ground truth\nBEV IoU $\\geq$ 0.5 (6 classes)  ·  centre dist. $\\leq$ 2 m (4 classes)",
               FEATURE, fontsize=6.6)
    arrow(ax, 0.5, b, 0.5, 0.442)

    b, _ = box(ax, 0.5, 0.395, 0.90, 0.084,
               "Train two scorers per class\nXGBoost  +  FT-Transformer",
               MODEL, fontsize=7.5)
    arrow(ax, 0.5, b, 0.5, 0.288)

    b, _ = box(ax, 0.5, 0.245, 0.90, 0.072,
               "20 saved scorers\n(10 classes $\\times$ 2 families)", MODEL, fontsize=7.5)
    arrow(ax, 0.5, b, 0.5, 0.152)

    box(ax, 0.5, 0.105, 0.90, 0.076,
        "Calibration scenes fix $\\lambda$, $\\tau$\nand per-class thresholds",
        NEUTRAL, fontsize=7)

    ax.text(0.5, 0.030, "ground truth: nuScenes annotations   ·   85 fitting + 20 calibration scenes",
            ha="center", va="center", fontsize=5.8 * FONT_SCALE, color="#4b5563", style="italic")


def draw_labeling(ax):
    panel(ax, "LABELING  —  per scene, offline", "#eefaf1", "#1d7a45")

    b, _ = box(ax, 0.5, 0.905, 0.46, 0.050, "Unlabeled LiDAR scene", NEUTRAL)
    arrow(ax, 0.5, b, 0.5, 0.845)

    bottom = expert_row(ax, 0.782)
    fan_in(ax, bottom, 0.706)

    b, _ = box(ax, 0.5, 0.673, 0.60, 0.050, "Pool all predicted boxes", POOL)
    arrow(ax, 0.5, b, 0.5, 0.603)

    b, _ = box(ax, 0.5, 0.570, 0.86, 0.050, "Extract 17 features per box", FEATURE)
    arrow(ax, 0.5, b, 0.5, 0.494)

    b, _ = box(ax, 0.5, 0.450, 0.90, 0.078,
               "Route by the box's own class label\nto that class's XGBoost + FT-Transformer",
               MODEL, fontsize=6.8)
    arrow(ax, 0.5, b, 0.5, 0.358)

    b, _ = box(ax, 0.5, 0.312, 0.90, 0.080,
               "Blend  $\\lambda\\cdot p_{XGB} + (1-\\lambda)\\cdot p_{FT}$\n"
               "then soft-temperature gate with the expert's own score",
               FEATURE, fontsize=6.4)
    arrow(ax, 0.5, b, 0.5, 0.224)

    b, _ = box(ax, 0.5, 0.180, 0.90, 0.074,
               "Per-class score threshold\n+ class-aware BEV NMS", POOL, fontsize=7)
    arrow(ax, 0.5, b, 0.5, 0.104)

    box(ax, 0.5, 0.062, 0.90, 0.070,
        "Temporal refinement (uses past and future frames)\n"
        "$\\rightarrow$  draft boxes for human review", "#1d7a45", fontsize=6.4)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.9))
    draw_fitting(axes[0])
    draw_labeling(axes[1])
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01, wspace=0.04)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
