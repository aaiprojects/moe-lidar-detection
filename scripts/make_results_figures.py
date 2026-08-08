"""Build the Assignment 6.1 results figures and print the tables that back them.

Combines the frozen per-expert test-split metrics computed in the research repo
with the fused-system metrics reproduced here (outputs/test_metrics/moe_fused.json).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "docs" / "figures"

# Per-expert test-split metrics. Produced by scripts/eval_test_split.py; override
# the location with the EXPERT_METRICS environment variable if you keep the file
# outside the repo.
EXPERT_METRICS = Path(
    os.environ.get(
        "EXPERT_METRICS",
        REPO / "outputs" / "test_metrics" / "experts_comparison.json",
    )
)
MOE_METRICS = REPO / "outputs" / "test_metrics" / "moe_fused.json"

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

CLASSES = [
    "car", "truck", "bus", "trailer", "construction_vehicle",
    "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier",
]
PRETTY = {
    "construction_vehicle": "constr. vehicle",
    "traffic_cone": "traffic cone",
}
EXPERT_ORDER = [
    "BEVFusion-LiDAR", "CenterPoint-Voxel", "CenterPoint-Pillar", "SSN", "PointPillars",
]

# Optimization trajectory: the accepted configuration lineage on the held-out
# partition, from EXPERIMENT_LOG sections 36 through 56.
# Ground-truth recovery on the held-out partition at a common confidence of
# 0.35, one-to-one greedy match within 2.0 m, GT filtered to nuScenes' per-class
# evaluation ranges. Computed in the research repo over all 1,804 keyframes.
TOTAL_GT = 56_670
RECOVERY = [
    ("MoE ensemble", 35_669, 0.69),
    ("CenterPoint-Voxel", 30_591, 0.76),
    ("CenterPoint-Pillar", 29_741, 0.70),
    ("BEVFusion-LiDAR", 24_597, 0.95),
    ("PointPillars", 21_900, 0.80),
    ("SSN", 19_475, 0.85),
]

TRAJECTORY = [
    ("Blended router (baseline)", 0.5865, 0.6507, "§36"),
    ("+ orphan penalty", 0.5903, 0.6580, "§42"),
    ("+ map feature", 0.5955, 0.6603, "§45"),
    ("+ track interpolation", 0.6014, 0.6624, "§47"),
    ("+ XGBoost for GBT", 0.6125, 0.6696, "§50"),
    ("+ tracking exclusions", 0.6138, 0.6686, "§51"),
    ("+ bicycle calibration", 0.6152, 0.6694, "§53"),
    ("- class_id feature", 0.6180, 0.6761, "§56"),
]
BEVFUSION_MAP = 0.6009


def load() -> tuple[dict, dict]:
    for label, path in (("expert", EXPERT_METRICS), ("fused-system", MOE_METRICS)):
        if not path.exists():
            raise SystemExit(
                f"Missing {label} metrics: {path}\n"
                "Run scripts/eval_test_split.py first to produce them."
            )
    experts = json.loads(EXPERT_METRICS.read_text())
    moe = json.loads(MOE_METRICS.read_text())
    return experts, moe


def print_tables(experts: dict, moe: dict) -> None:
    cols = ["mAP", "NDS", "mATE", "mASE", "mAOE", "mAVE", "mAAE"]
    print("\n=== TABLE 5.3  system-level metrics, 45 held-out test scenes ===")
    print(f"{'System':<22}" + "".join(f"{c:>9}" for c in cols))
    row = f"{'Fused MoE':<22}"
    for c in cols:
        row += f"{moe['summary'][c]:>9.4f}"
    print(row)
    for name in EXPERT_ORDER:
        row = f"{name:<22}"
        for c in cols:
            row += f"{experts[name][c]:>9.4f}"
        print(row)

    print("\n=== per-class AP: fused vs. best single expert ===")
    print(f"{'class':<18}{'MoE':>8}{'best expert':>14}{'name':>20}{'delta':>9}")
    deltas = []
    for cls in CLASSES:
        moe_ap = moe["per_class_ap"][cls]
        best_name, best_ap = max(
            ((n, experts[n][f"AP_{cls}"]) for n in EXPERT_ORDER), key=lambda t: t[1]
        )
        deltas.append(moe_ap - best_ap)
        print(
            f"{cls:<18}{moe_ap:>8.4f}{best_ap:>14.4f}{best_name:>20}"
            f"{moe_ap - best_ap:>+9.4f}"
        )
    wins = sum(d > 0 for d in deltas)
    print(f"\nfused beats the best single expert on {wins}/10 classes")


def fig_per_class(experts: dict, moe: dict) -> None:
    """Dot plot: each class shows the five experts in grey and the fused system."""
    order = sorted(CLASSES, key=lambda c: moe["per_class_ap"][c])
    fig, ax = plt.subplots(figsize=(4.6, 4.4))

    for i, cls in enumerate(order):
        aps = [experts[n][f"AP_{cls}"] for n in EXPERT_ORDER]
        ax.plot(aps, [i] * len(aps), "o", color="0.62", markersize=6,
                markeredgecolor="0.35", markeredgewidth=0.6, zorder=2,
                label="Single expert" if i == 0 else None)
        moe_ap = moe["per_class_ap"][cls]
        best = max(aps)
        ax.plot([best, moe_ap], [i, i], "-", color="#c0392b" if moe_ap > best else "#7f8c8d",
                linewidth=1.4, alpha=0.55, zorder=1)
        ax.plot(moe_ap, i, "D", color="#c0392b", markersize=7.5, zorder=3,
                label="Fused MoE" if i == 0 else None)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([PRETTY.get(c, c) for c in order])
    ax.set_xlabel("Average precision (held-out test scenes)")
    ax.set_xlim(-0.02, 1.0)
    ax.set_title("Per-class AP: fused system vs. five frozen experts")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", alpha=0.15)
    fig.savefig(FIG_DIR / "fig10_per_class_ap.png")
    plt.close(fig)
    print(f"\nwrote {FIG_DIR / 'fig10_per_class_ap.png'}")


def fig_trajectory() -> None:
    """Step plot of the accepted optimization sequence against the best expert."""
    labels = [t[0] for t in TRAJECTORY]
    maps = [t[1] for t in TRAJECTORY]
    x = np.arange(len(TRAJECTORY))

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.axhline(BEVFUSION_MAP, color="#c0392b", linestyle="--", linewidth=1.3,
               label=f"Best single expert ({BEVFUSION_MAP:.4f})")
    ax.plot(x, maps, "-o", color="#1f4e79", markersize=6, linewidth=1.8,
            label="Fused system")

    for xi, m in zip(x, maps):
        ax.annotate(f"{m:.4f}", (xi, m), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, rotation=38, ha="right")
    ax.set_ylabel("mAP (held-out test scenes)")
    ax.set_ylim(0.578, 0.628)
    ax.set_title("Optimization trajectory of accepted changes")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(FIG_DIR / "fig11_optimization_trajectory.png")
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'fig11_optimization_trajectory.png'}")


def fig_recovery() -> None:
    """Ground-truth coverage per source, sized for a single report column."""
    names = [r[0] for r in RECOVERY][::-1]
    recall = [r[1] / TOTAL_GT for r in RECOVERY][::-1]
    prec = [r[2] for r in RECOVERY][::-1]
    colors = ["#1f7a34" if n == "MoE ensemble" else "#7f95a6" for n in names]

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.barh(names, recall, color=colors, edgecolor="0.25", linewidth=0.6)
    for i, (r, p) in enumerate(zip(recall, prec)):
        ax.text(r + 0.012, i, f"{r * 100:.1f}%  (P = {p:.2f})",
                va="center", fontsize=9)

    ax.set_xlim(0, 0.86)
    ax.set_xlabel("Fraction of ground-truth objects recovered")
    ax.set_title("Ground-truth coverage, 45 held-out scenes")
    ax.grid(axis="y", alpha=0)
    fig.savefig(FIG_DIR / "fig12_recovery_by_source.png")
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'fig12_recovery_by_source.png'}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    experts, moe = load()
    print_tables(experts, moe)
    fig_per_class(experts, moe)
    fig_trajectory()
    fig_recovery()


if __name__ == "__main__":
    main()
