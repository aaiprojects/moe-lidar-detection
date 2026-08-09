# Trained Router Weights

Pre-trained per-class router models — the output of
`notebook/model_training_evaluation.ipynb` Steps 2–3 — for teammates who
want to run inference or evaluation without retraining from
`training_data/router_data/`.

Hosted on Google Drive rather than committed directly, for consistency
with how the training CSVs and expert predictions are distributed in this
repo (keeps the git history small and avoids GitHub's 100 MB file limit,
even though these particular files are individually small).

## Download

Download from Google Drive:

**\<https://drive.google.com/drive/folders/1jxE-sz2fj0nd_DuxEGkAwJyUVLRUadgY?usp=sharing>**

After downloading, place the files at:

```
model_weights/
├── router_xgboost/
│   ├── router_car.pkl
│   ├── router_truck.pkl
│   ├── ... (one per class, 10 total)
│   └── router_bicycle.pkl        # sigmoid-calibrated (CalibratedClassifierCV)
└── router_nn/
    ├── router_car.pkl
    ├── ... (one per class, 10 total)
    └── router_per_class_meta.json
```

## Files

| Directory | Contents | Approx. size |
|---|---|---|
| `router_xgboost/` | 10 pickled `XGBClassifier` (or `CalibratedClassifierCV`-wrapped, for bicycle) models | ~8 MB |
| `router_nn/` | 10 pickled `NNRouterWrapper` objects (FT-Transformer weights + StandardScaler + IsotonicRegression calibrator) | ~17 MB |

## Loading

```python
from src.moe.xgboost_router import load_xgboost_per_class_routers
import joblib

xgb_routers = load_xgboost_per_class_routers("model_weights/router_xgboost")
nn_routers = {
    cls: joblib.load(f"model_weights/router_nn/router_{cls}.pkl")
    for cls in xgb_routers
}
```

These are exactly the objects `src.moe.pipeline.build_ensemble_routers`
expects — see `notebook/model_training_evaluation.ipynb` Step 5.
