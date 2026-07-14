"""
Execute all data cleaning and EDA steps from data_cleaning_eda.ipynb
and save figures to docs/figures/.

Run from the repo root:
    python notebooks/run_eda.py
"""
import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = pathlib.Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "data" / "router_data" / "train.csv"
EVAL_CSV  = ROOT / "data" / "router_data" / "eval.csv"
FIG_DIR   = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading CSVs …")
df_train = pd.read_csv(TRAIN_CSV)
df_eval  = pd.read_csv(EVAL_CSV)
df_train["split"] = "train"
df_eval["split"]  = "eval"
df_raw = pd.concat([df_train, df_eval], ignore_index=True)
print(f"  Raw rows: {len(df_raw):,}")

# ===========================================================================
# PART 2 — DATA CLEANING
# ===========================================================================

# Step 1: Missing values
missing = df_raw.isnull().sum()
if missing.any():
    print("Missing values found:", missing[missing > 0].to_dict())
else:
    print("Step 1 — Missing values: none found.")

# Step 2: Infinite values
numeric_cols = df_raw.select_dtypes(include=np.number).columns
inf_mask = np.isinf(df_raw[numeric_cols]).any(axis=1)
print(f"Step 2 — Rows with infinite values: {inf_mask.sum():,}")
df = df_raw[~inf_mask].copy()

# Step 3: Out-of-range detection scores
bad_score_mask = ~df["detection_score"].between(0.0, 1.0)
print(f"Step 3 — Rows with detection_score outside [0,1]: {bad_score_mask.sum():,}")
df = df[~bad_score_mask].copy()

# Step 4: Non-positive box dimensions
dim_mask = (df["box_width"] <= 0) | (df["box_length"] <= 0) | (df["box_height"] <= 0)
print(f"Step 4 — Rows with non-positive box dimensions: {dim_mask.sum():,}")
df = df[~dim_mask].copy()

# Step 5: Label validation
VALID_CLASS_IDS = set(range(10))
VALID_MODELS    = {"centerpoint", "centerpoint_pillar", "pointpillars", "ssn", "bevfusion_lidar"}
bad_class = ~df["class_id"].isin(VALID_CLASS_IDS)
bad_model = ~df["model_name"].isin(VALID_MODELS)
print(f"Step 5 — Invalid class_id rows: {bad_class.sum():,}")
print(f"Step 5 — Invalid model_name rows: {bad_model.sum():,}")
df = df[~bad_class & ~bad_model].copy()

# Step 6: BEVFusion size-convention verification
bev_cars   = df[(df["model_name"] == "bevfusion_lidar") & (df["class_id"] == 0)]
other_cars = df[(df["model_name"] != "bevfusion_lidar") & (df["class_id"] == 0)]
print(f"Step 6 — BEVFusion cars: median width={bev_cars['box_width'].median():.2f} m, "
      f"length={bev_cars['box_length'].median():.2f} m")
print(f"         Others  cars: median width={other_cars['box_width'].median():.2f} m, "
      f"length={other_cars['box_length'].median():.2f} m")

# Step 7: Clip extreme distance values
p99 = df["dist_from_ego"].quantile(0.99)
cap = 3 * p99
n_clipped = (df["dist_from_ego"] > cap).sum()
print(f"Step 7 — Clipping dist_from_ego at {cap:,.1f} m ({n_clipped:,} rows affected)")
df["dist_from_ego"] = df["dist_from_ego"].clip(upper=cap)

n_removed = len(df_raw) - len(df)
print(f"\nCleaning summary: {len(df_raw):,} → {len(df):,} rows "
      f"({n_removed:,} removed, {100*n_removed/len(df_raw):.3f}%)")

# Split back
df["class_name"] = df["class_id"].astype(int).map(dict(enumerate(CLASS_NAMES)))
df_tr = df[df["split"] == "train"].copy()
df_ev = df[df["split"] == "eval"].copy()
print(f"  Train: {len(df_tr):,}  |  Eval: {len(df_ev):,}")

# ===========================================================================
# PART 3 — EDA
# ===========================================================================

FEATURE_COLS = [
    "detection_score", "dist_from_ego", "box_width", "box_length", "box_height",
    "vel_magnitude", "n_peer_overlaps", "max_peer_iou", "mean_peer_score",
    "score_variance", "expert_agreement", "max_class_score", "n_active_experts",
    "class_id", "expert_id", "label",
]

# ── Figure 1: class distribution + TP rate ──────────────────────────────────
print("\nGenerating figures …")
class_counts = df_tr["class_name"].value_counts().reindex(CLASS_NAMES)
pos_rate     = df_tr.groupby("class_name")["label"].mean().reindex(CLASS_NAMES)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].bar(CLASS_NAMES, class_counts.values, color="steelblue", edgecolor="white")
axes[0].set_xticklabels(CLASS_NAMES, rotation=38, ha="right", fontsize=8)
axes[0].set_title("Predicted Boxes per Class (train split)")
axes[0].set_ylabel("Number of predicted boxes")
axes[0].yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{int(x/1e3)}K"))

axes[1].bar(CLASS_NAMES, pos_rate.values * 100, color="seagreen", edgecolor="white")
axes[1].set_xticklabels(CLASS_NAMES, rotation=38, ha="right", fontsize=8)
axes[1].set_title("True-Positive Rate per Class (% of predicted boxes)")
axes[1].set_ylabel("True-positive rate (%)")
axes[1].axhline(df_tr["label"].mean() * 100, color="crimson", linestyle="--",
                label=f"Overall mean = {df_tr['label'].mean()*100:.1f}%")
axes[1].legend()
plt.tight_layout()
fig.savefig(FIG_DIR / "fig1_class_distribution.png", bbox_inches="tight")
plt.close()
print("  Saved fig1_class_distribution.png")

# ── Figure 2: per-expert stats ───────────────────────────────────────────────
expert_stats = df_tr.groupby("model_name").agg(
    n_boxes=("label", "count"), tp_rate=("label", "mean")).reset_index()
expert_stats = expert_stats.sort_values("tp_rate", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
axes[0].barh(expert_stats["model_name"], expert_stats["tp_rate"] * 100, color=colors, edgecolor="white")
axes[0].set_xlabel("True-positive rate (%)")
axes[0].set_title("Precision (TP rate) per Expert")
for i, v in enumerate(expert_stats["tp_rate"]):
    axes[0].text(v * 100 + 0.3, i, f"{v*100:.1f}%", va="center", fontsize=9)
axes[1].barh(expert_stats["model_name"], expert_stats["n_boxes"] / 1e6, color=colors, edgecolor="white")
axes[1].set_xlabel("Predicted boxes (millions)")
axes[1].set_title("Volume of Predictions per Expert")
for i, v in enumerate(expert_stats["n_boxes"]):
    axes[1].text(v / 1e6 + 0.01, i, f"{v/1e6:.2f}M", va="center", fontsize=9)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig2_expert_stats.png", bbox_inches="tight")
plt.close()
print("  Saved fig2_expert_stats.png")

# ── Figure 3: detection score distributions ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
for label, color, name in [(0, "tomato", "False positive"), (1, "steelblue", "True positive")]:
    subset = df_tr[df_tr["label"] == label]["detection_score"]
    ax.hist(subset, bins=60, alpha=0.65, color=color,
            label=f"{name} (n={len(subset):,})", density=True, range=(0, 1))
ax.set_xlabel("Detection score (expert confidence)")
ax.set_ylabel("Density")
ax.set_title("Fig 3. Detection Score Distribution: True Positives vs False Positives")
ax.legend()
plt.tight_layout()
fig.savefig(FIG_DIR / "fig3_score_distribution.png", bbox_inches="tight")
plt.close()
print("  Saved fig3_score_distribution.png")

# ── Figure 4: key feature distributions ──────────────────────────────────────
FEAT_LABELS = {
    "detection_score"  : "Expert confidence score",
    "max_peer_iou"     : "Max overlap with peer (IoU)",
    "n_peer_overlaps"  : "Number of peer overlaps",
    "expert_agreement" : "Fraction of experts agreeing",
    "dist_from_ego"    : "Distance from ego (m)",
}
fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
for ax, (col, lbl) in zip(axes, FEAT_LABELS.items()):
    for label, color, name in [(0, "tomato", "FP"), (1, "steelblue", "TP")]:
        vals = df_tr[df_tr["label"] == label][col].clip(
            df_tr[col].quantile(0.01), df_tr[col].quantile(0.99))
        ax.hist(vals, bins=40, alpha=0.6, color=color, label=name, density=True)
    ax.set_title(lbl, fontsize=8)
    ax.legend(fontsize=7)
fig.suptitle("Fig 4. Feature Distributions: True Positives vs False Positives", fontsize=11)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig4_feature_distributions.png", bbox_inches="tight")
plt.close()
print("  Saved fig4_feature_distributions.png")

# ── Figure 5: correlation heatmap ────────────────────────────────────────────
corr = df_tr[FEATURE_COLS].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            annot_kws={"size": 7}, ax=ax)
ax.set_title("Fig 5. Pearson Correlation Matrix of Router Features", fontsize=12)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig5_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("  Saved fig5_correlation_heatmap.png")

label_corr = corr["label"].drop("label").sort_values(key=abs, ascending=False)
print("\nFeature correlation with label (|r| descending):")
print(label_corr.round(3).to_string())

# ── Figure 6: TP rate heatmap by expert × class ──────────────────────────────
pivot = df_tr.pivot_table(index="model_name", columns="class_name",
                           values="label", aggfunc="mean")[CLASS_NAMES]
fig, ax = plt.subplots(figsize=(13, 4))
sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="YlGn",
            linewidths=0.5, ax=ax, vmin=0, vmax=60,
            cbar_kws={"label": "True-positive rate (%)"})
ax.set_title("Fig 6. True-Positive Rate (%) by Expert x Object Class", fontsize=11)
ax.set_xlabel("Object class")
ax.set_ylabel("Expert model")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
fig.savefig(FIG_DIR / "fig6_tp_rate_heatmap.png", bbox_inches="tight")
plt.close()
print("  Saved fig6_tp_rate_heatmap.png")

# ── Figure 7: box geometry scatter ───────────────────────────────────────────
sample = df_tr.sample(n=min(60_000, len(df_tr)), random_state=42)
fig, ax = plt.subplots(figsize=(9, 6))
palette = sns.color_palette("tab10", n_colors=10)
for i, cls in enumerate(CLASS_NAMES):
    sub = sample[sample["class_name"] == cls]
    ax.scatter(sub["box_width"], sub["box_length"], s=4, alpha=0.3, color=palette[i], label=cls)
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.set_xlabel("Box width (m)")
ax.set_ylabel("Box length (m)")
ax.set_title("Fig 7. Predicted Box Width vs Length by Class\n(60k sample, clipped at 10m x 20m)")
ax.legend(fontsize=7, ncol=2, markerscale=3)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig7_box_geometry.png", bbox_inches="tight")
plt.close()
print("  Saved fig7_box_geometry.png")

# ── Figure 8: expert agreement vs TP rate ────────────────────────────────────
bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
bin_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
df_tr = df_tr.copy()
df_tr["agree_bin"] = pd.cut(df_tr["expert_agreement"], bins=bins, labels=bin_labels, right=False)
tp_by_agree = df_tr.groupby("agree_bin", observed=True).agg(
    tp_rate=("label", "mean"), n=("label", "count")).reset_index()

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(tp_by_agree["agree_bin"], tp_by_agree["tp_rate"] * 100,
              color="mediumpurple", edgecolor="white")
for bar, n in zip(bars, tp_by_agree["n"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"n={n/1e3:.0f}K", ha="center", va="bottom", fontsize=8)
ax.set_xlabel("Fraction of experts agreeing on detection")
ax.set_ylabel("True-positive rate (%)")
ax.set_title("Fig 8. Cross-Expert Agreement vs True-Positive Rate\n"
             "Boxes seen by more experts are far more likely to be real objects")
plt.tight_layout()
fig.savefig(FIG_DIR / "fig8_agreement_vs_tp.png", bbox_inches="tight")
plt.close()
print("  Saved fig8_agreement_vs_tp.png")

# ── Print agreement table ─────────────────────────────────────────────────────
print()
print(tp_by_agree.to_string(index=False))

print("\nAll figures saved to:", FIG_DIR)
