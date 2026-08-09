# Training Data

The router training CSVs are too large to store in this repository (GitHub's file size limit is 100 MB).

> **Note:** the CSVs behind the Drive link below predate the current
> feature set (they were built before `dist_to_drivable_area` and
> `class_agreement` were added) — that's the version
> `notebook/data_cleaning_eda.ipynb` was written against. The current
> pipeline (`src/moe/features.py`, `configs/moe_final.yaml`) needs the
> full 18-feature schema described below. **The Drive link needs to be
> updated to point at the new files** — replace it once uploaded.

## Download

Download the dataset from Google Drive:

(https://drive.google.com/drive/folders/1uSW8jr36o29bpaMzKMNZJnsiyn_MT_CZ?usp=drive_link)

After downloading, place the files in `router_data/`:

```
training_data/
└── router_data/
    ├── train.csv
    └── eval.csv
```

## Files

| File | Description | Size |
|------|-------------|------|
| `router_data/train.csv` | Router training split (85 scenes, 3,411 keyframes) | ~447 MB |
| `router_data/eval.csv` | Router calibration split (20 scenes, 804 keyframes) | ~96 MB |
| `token_split_3way.json` | Scene-grouped 3-way split used to build the CSVs above, plus the held-out `test_tokens` (45 scenes, 1,804 keyframes) used for the project's official reported mAP/NDS. Small enough to commit directly — no Drive download needed. |

## Schema

Each row is one candidate box (from one expert, on one keyframe), labeled
true/false positive. 18 model features (`src.moe.features.FEATURE_NAMES`)
plus 3 metadata columns:

```
expert_id, class_id, detection_score, dist_from_ego, box_width, box_length,
box_height, vel_magnitude, n_peer_overlaps, max_peer_iou, mean_peer_score,
score_variance, expert_agreement, n_spatial_overlaps, class_agreement,
max_class_score, n_active_experts, dist_to_drivable_area,   # 18 features
label, sample_token, model_name                             # metadata
```

Six classes (car, truck, bus, trailer, barrier, traffic_cone) are labeled
via bird's-eye-view IoU (≥0.5) against ground truth; four classes
(motorcycle, bicycle, pedestrian, construction_vehicle) are labeled via
center distance (≤2.0 m) instead — see `configs/moe_final.yaml` →
`matching` for which classes use which criterion. These two CSVs already
have the correct label applied per row; no further merging is needed.
