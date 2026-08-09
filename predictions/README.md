# Expert Predictions

Raw predictions from the five frozen LiDAR detectors, in nuScenes
submission JSON format, on the nuScenes val split (6,019 keyframes). These
are the inputs to the MoE fusion pipeline (`src/moe/pipeline.py`) — no
expert is fine-tuned in this project, so these files are generated once
and reused for every router experiment.

Too large for GitHub (~1.6 GB total; GitHub's file size limit is 100 MB
per file, and even individually most of these exceed it).

## Download

Download from Google Drive:

**([https://drive.google.com/drive/folders/10OW2u6xL-iqs06J0pgwi-I0xhcNj3pLG?usp=sharing](https://drive.google.com/drive/u/1/folders/10OW2u6xL-iqs06J0pgwi-I0xhcNj3pLG))**

After downloading, place each expert's file at:

```
predictions/
├── centerpoint/predictions.json
├── centerpoint_pillar/predictions.json
├── pointpillars/predictions.json
├── ssn/predictions.json
└── bevfusion_lidar/predictions.json
```

## Files

| File | Expert | Approx. size |
|---|---|---|
| `centerpoint/predictions.json` | CenterPoint (0.075m voxel, SECOND backbone) | 220 MB |
| `centerpoint_pillar/predictions.json` | CenterPoint (0.2m pillar, SECOND backbone) | 320 MB |
| `pointpillars/predictions.json` | PointPillars | 310 MB |
| `ssn/predictions.json` | SSN (Shape Signature Network) | 390 MB |
| `bevfusion_lidar/predictions.json` | BEVFusion (LiDAR-only) | 65 MB |

## Only needed for the optional full-pipeline notebook section

`notebook/model_training_evaluation.ipynb` Steps 1–4 (training and
evaluating the router models) do **not** need these files — they use the
pre-featurized `training_data/router_data/{train,eval}.csv` instead. These
JSONs are only required for Step 5, which runs the complete pipeline
(fusion + NMS + temporal refinement) end to end and reproduces the
project's official nuScenes mAP/NDS.

## Regenerating from scratch

If you'd rather regenerate these from the original pretrained checkpoints
instead of downloading them, see
[`docs/expert_regeneration.md`](../docs/expert_regeneration.md). That path
additionally requires a full mmdetection3d install and the nuScenes
sensor dataset (large — see that doc for exact sizes) and is not part of
the code in this repository.
