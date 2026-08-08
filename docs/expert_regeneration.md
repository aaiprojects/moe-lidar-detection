# Regenerating Expert Predictions From Scratch (Advanced, Optional)

This is documentation only — there is no code for this path in this
repository. Almost nobody on the team needs it: the router training and
evaluation notebook (`notebook/model_training_evaluation.ipynb`) works
entirely from pre-computed artifacts hosted on Google Drive (training
CSVs, expert prediction JSONs, trained router weights — see
[`training_data/README.md`](../training_data/README.md),
[`predictions/README.md`](../predictions/README.md), and
[`model_weights/README.md`](../model_weights/README.md)).

Follow this only if you specifically want to reproduce the five experts'
raw predictions yourselves — e.g. to verify them independently, or to swap
in a different checkpoint or expert. It requires a large dataset download
and a separate, heavyweight framework install that are both out of scope
for this repo's dependencies.

## What you need

| Item | Size | Notes |
|---|---|---|
| nuScenes `trainval` LiDAR sensor data | ~255 GB (`samples/LIDAR_TOP` + `sweeps/LIDAR_TOP`) | All five experts are LiDAR-only — camera/radar blobs (~50 GB) are not used and can be skipped if your download tooling supports partial downloads |
| nuScenes `trainval` metadata + maps | ~2.5 GB | JSON tables + rasterized map masks (base maps, not the vector expansion pack) |
| mmdetection3d v1.4.0 | — | Editable/dev install; see below |
| Five pretrained checkpoints | ~47 MB total | Hosted on Google Drive — see below |

The 255 GB LiDAR figure is for the full dataset; nuScenes is licensed and
cannot be redistributed, so each teammate must download it themselves
directly from Motional.

## 1. Get the nuScenes dataset

1. Register at **nuscenes.org** (free, requires accepting the dataset
   license — redistribution is not permitted, which is why this isn't
   hosted on our Drive).
2. Download the **Full dataset (v1.0) — Trainval** split. At minimum you
   need the metadata, map expansion (base), and the LiDAR blob archives
   (`v1.0-trainvalXX_blobs_lidar.tgz`, 10 parts). Camera/radar blobs are
   not required for this project's five LiDAR-only experts.
3. Extract everything into a single directory so it has this layout:

   ```
   data/nuscenes/
   ├── maps/
   ├── samples/LIDAR_TOP/
   ├── sweeps/LIDAR_TOP/
   └── v1.0-trainval/            # JSON tables (scene.json, sample.json, ...)
   ```
4. Generate the mmdetection3d `.pkl` info files (`nuscenes_infos_val.pkl`
   etc.) using mmdetection3d's own `create_data.py` tool — see its docs
   for the nuScenes dataset prep instructions. This project's inference
   scripts expect `data/nuscenes/nuscenes_infos_val.pkl` to exist.

## 2. Install mmdetection3d v1.4.0

Four of the five experts (CenterPoint-Voxel, CenterPoint-Pillar,
PointPillars, SSN) run through mmdetection3d's standard `tools/test.py`
entry point. Follow mmdetection3d's own installation guide for v1.4.0
(an editable/dev clone is recommended, since BEVFusion needs a patched
module registered against it — see step 4). This is a version-sensitive,
fairly heavyweight install (PyTorch + MMCV + MMDetection + MMDetection3D,
each pinned to compatible versions) — expect it to take some time and to
need its own dedicated virtual environment, separate from this repo's
`requirements.txt`.

## 3. Get the checkpoints

Five pretrained checkpoints (~47 MB total, small enough that hosting is
straightforward):

**\<ADD_GOOGLE_DRIVE_LINK_HERE\>**

Place them at:

```
data/checkpoints/
├── centerpoint/checkpoint.pth
├── centerpoint_pillar/checkpoint.pth
├── pointpillars/checkpoint.pth
├── ssn/checkpoint.pth
└── bevfusion_lidar/checkpoint.pth
```

These are the original public pretrained weights for each architecture —
none of them are fine-tuned in this project.

## 4. Run inference — CenterPoint, CenterPoint-Pillar, PointPillars, SSN

These four use mmdetection3d's stock configs and `tools/test.py` directly.
Substitute `$MMDET3D` (your mmdetection3d clone), `$DATA` (your
`data/nuscenes`), and `$CKPT` (your `data/checkpoints`) below, and run each
sequentially (they're GPU-memory-hungry enough to conflict if run
concurrently on a single GPU):

```bash
# CenterPoint (voxel)
python3 $MMDET3D/tools/test.py \
    $MMDET3D/configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
    $CKPT/centerpoint/checkpoint.pth \
    --cfg-options \
        test_dataloader.dataset.data_root=$DATA/ \
        test_dataloader.dataset.ann_file=$DATA/nuscenes_infos_val.pkl \
        test_evaluator.jsonfile_prefix=predictions/centerpoint/predictions

# CenterPoint (pillar)
python3 $MMDET3D/tools/test.py \
    $MMDET3D/configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
    $CKPT/centerpoint_pillar/checkpoint.pth \
    --cfg-options \
        test_dataloader.dataset.data_root=$DATA/ \
        test_dataloader.dataset.ann_file=$DATA/nuscenes_infos_val.pkl \
        test_evaluator.jsonfile_prefix=predictions/centerpoint_pillar/predictions

# PointPillars
python3 $MMDET3D/tools/test.py \
    $MMDET3D/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py \
    $CKPT/pointpillars/checkpoint.pth \
    --cfg-options \
        test_dataloader.dataset.data_root=$DATA/ \
        test_dataloader.dataset.ann_file=$DATA/nuscenes_infos_val.pkl \
        test_evaluator.jsonfile_prefix=predictions/pointpillars/predictions

# SSN
python3 $MMDET3D/tools/test.py \
    $MMDET3D/configs/ssn/ssn_hv_secfpn_sbn-all_16xb2-2x_nus-3d.py \
    $CKPT/ssn/checkpoint.pth \
    --cfg-options \
        test_dataloader.dataset.data_root=$DATA/ \
        test_dataloader.dataset.ann_file=$DATA/nuscenes_infos_val.pkl \
        test_evaluator.jsonfile_prefix=predictions/ssn/predictions
```

Each run writes a nuScenes submission JSON alongside the path given in
`jsonfile_prefix`; rename/move it to `predictions/<expert>/predictions.json`
to match the layout the rest of this repo expects.

## 5. Run inference — BEVFusion-LiDAR

BEVFusion isn't a stock mmdetection3d config — it uses a patched module
(originally at `bevfusion_lidar/` in the main research repo, not part of
this trimmed package) registered against mmdetection3d at import time, plus
its own inference script. If you need to regenerate this expert
specifically, get that patched module and script from the main research
repo and run:

```bash
python3 run_bevfusion_inference.py \
    --checkpoint $CKPT/bevfusion_lidar/checkpoint.pth \
    --output predictions/bevfusion_lidar/predictions.json \
    --data-root $DATA \
    --score-thr 0.1 \
    --device cuda:0 \
    --split val
```

## 6. Verify

Each expert's `predictions.json` should be a nuScenes submission-format
file (top-level `results` and `meta` keys). You can sanity-check one with:

```python
import json
d = json.load(open("predictions/centerpoint/predictions.json"))
print(list(d.keys()))                 # ['meta', 'results']
print(len(d["results"]))              # should be 6019 (nuScenes val keyframes)
```

Once all five are in place, `notebook/model_training_evaluation.ipynb`
Step 5 can run the full pipeline end to end (in addition to still needing
the nuScenes metadata from step 1 for scene ordering and map masks).
