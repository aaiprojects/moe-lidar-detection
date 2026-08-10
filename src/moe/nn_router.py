"""Neural-network router models for the MoE LiDAR detection pipeline.

Two architectures are provided, both compatible with the existing inference
pipeline via the NNRouterWrapper (implements predict_proba like sklearn):

  MLPRouter
    A 3-layer feed-forward network with BatchNorm and Dropout.
    Fast, simple, effective baseline neural network.

  FTTransformerRouter
    Feature Tokenizer + Transformer (Gorishniy et al., 2021).
    Each of the 15 numeric features is projected to a d_token-dimensional
    embedding.  A learnable [CLS] token is prepended.  Self-attention across
    all feature tokens lets the model capture interactions between, e.g.,
    detection_score and expert_agreement without explicit feature crossing.
    The [CLS] output is passed through a linear head to produce the
    true-positive probability.

Both models are trained with:
  - BCEWithLogitsLoss with pos_weight to handle the ~14% positive rate
  - Adam optimiser with ReduceLROnPlateau scheduling
  - Per-class training mirrors the GBT approach (one model per nuScenes class)
  - StandardScaler normalisation of inputs (tree models don't need this; NNs do)

The NNRouterWrapper wraps a trained model + scaler and exposes
predict_proba(X) so it can be dropped in wherever the GBT routers are used.

Reference:
  Gorishniy et al. (2021). Revisiting Deep Learning Models for Tabular Data.
  NeurIPS 2021.  https://arxiv.org/abs/2106.11959
"""

from __future__ import annotations

import copy
import gc
import json
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# Order matches the class_id encoding used when the training dataset was
# built (src.moe.features._CLASS_TO_ID / nuscenes_infos_val.pkl metainfo
# order) -- NOT alphabetical or nuScenes' benchmark-table order. This list
# previously used the wrong order, which silently trained each per-class
# router on the WRONG class's rows (e.g. the "bicycle" router was actually
# fit on pedestrian's data) for 6 of the 10 classes -- car, truck, bus, and
# motorcycle were unaffected only because their positions happened to match
# between the two orderings. See notebook/model_training_evaluation.ipynb
# Step 5 for how this was caught (per-class row counts didn't match the
# named class) and src/moe/xgboost_router.py's _CLASS_TO_ID for the
# already-correct reference mapping.
NUSCENES_CLASSES: list[str] = [
    "car", "truck", "trailer", "bus", "construction_vehicle",
    "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier",
]

_CLS_TO_ID: dict[str, int] = {c: i for i, c in enumerate(NUSCENES_CLASSES)}


def get_device() -> torch.device:
    """Resolve the active torch device at call time (not import time)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve an explicit device preference ('cpu', 'cuda', or None -> auto)."""
    if device is None:
        return get_device()
    return torch.device(device)


# Backwards-compatible alias; prefer get_device() in new code.
DEVICE = get_device()

DEFAULT_INFERENCE_BATCH_SIZE = 8192


# ══════════════════════════════════════════════════════════════════════════════
# MLP Router
# ══════════════════════════════════════════════════════════════════════════════

class MLPRouter(nn.Module):
    """Three-layer MLP with BatchNorm and Dropout for binary box scoring.

    Architecture:
        Input (n_features)
        → Linear → BatchNorm → ReLU → Dropout(0.3)
        → Linear → BatchNorm → ReLU → Dropout(0.2)
        → Linear → BatchNorm → ReLU → Dropout(0.1)
        → Linear(64 → 1)   [logit; sigmoid applied outside in loss / predict]

    Args:
        n_features: Number of input features (15 for this project).
        hidden_dims: Sizes of the three hidden layers.
        dropout: Dropout probabilities for each layer.
    """

    def __init__(
        self,
        n_features: int = 15,
        hidden_dims: tuple[int, int, int] = (256, 128, 64),
        dropout: tuple[float, float, float] = (0.3, 0.2, 0.1),
    ) -> None:
        super().__init__()
        dims = [n_features, *hidden_dims]
        layers: list[nn.Module] = []
        for i in range(len(hidden_dims)):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout[i]),
            ]
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (shape [B, 1])."""
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# FT-Transformer Router
# ══════════════════════════════════════════════════════════════════════════════

class FeatureTokenizer(nn.Module):
    """Project each scalar feature to a d_token-dimensional embedding.

    Each feature x_i is mapped as:
        token_i = W_i * x_i + b_i       (linear projection, shape [d_token])

    A learnable [CLS] token is prepended, giving a sequence of length
    n_features + 1 for the transformer encoder.
    """

    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        # One weight vector and bias per feature
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls    = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.kaiming_uniform_(self.weight, a=0.01)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, n_features]
        Returns:
            tokens: [B, n_features+1, d_token]
        """
        # [B, n_features, d_token]
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        # Prepend CLS token: [B, 1, d_token]
        cls = self.cls.expand(x.size(0), -1, -1)
        return torch.cat([cls, tokens], dim=1)


class FTTransformerRouter(nn.Module):
    """Feature Tokenizer + Transformer for tabular binary classification.

    Architecture:
        FeatureTokenizer  →  n_layers × TransformerEncoderLayer
        →  CLS token output  →  LayerNorm  →  Linear(d_token, 1)

    Self-attention across the n_features+1 tokens lets the model capture
    interactions such as "high detection_score + high expert_agreement →
    very likely true positive" without requiring manual feature crosses.

    Args:
        n_features: Number of input features (15).
        d_token: Embedding dimension for each feature token.
        n_heads: Number of attention heads (must divide d_token).
        n_layers: Number of transformer encoder layers.
        dropout: Dropout applied inside attention and feed-forward layers.
        d_ffn_factor: Feed-forward hidden dim = d_token * d_ffn_factor.
    """

    def __init__(
        self,
        n_features: int = 15,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        d_ffn_factor: int = 4,
    ) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * d_ffn_factor,
            dropout=dropout,
            batch_first=True,   # [B, S, d_token] convention
            norm_first=True,    # pre-LN for stable training (Gorishniy et al.)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm    = nn.LayerNorm(d_token)
        self.head    = nn.Linear(d_token, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, n_features]
        Returns:
            logits: [B, 1]
        """
        tokens = self.tokenizer(x)           # [B, n_features+1, d_token]
        out    = self.encoder(tokens)        # [B, n_features+1, d_token]
        cls    = self.norm(out[:, 0, :])     # [B, d_token]  — CLS position
        return self.head(cls)                # [B, 1]


# ══════════════════════════════════════════════════════════════════════════════
# sklearn-compatible wrapper (drop-in replacement for GBT routers)
# ══════════════════════════════════════════════════════════════════════════════

class NNRouterWrapper:
    """Wraps a trained PyTorch router + StandardScaler + optional calibrator.

    Exposes predict_proba(X) and predict(X) so it is a drop-in replacement
    for the sklearn CalibratedClassifierCV objects used by the GBT routers
    in infer_router.py and per_class_router.py.

    The optional calibrator is an IsotonicRegression fitted on a held-out
    calibration set to correct the raw sigmoid probabilities so they are
    well-ranked for nuScenes AP evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        scaler: StandardScaler,
        calibrator=None,
        inference_device: str | torch.device = "cpu",
        predict_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    ) -> None:
        # Keep routers on CPU after training — per-frame inference batches are
        # tiny and GPU residency of 10 class models adds up on unified-memory
        # systems (e.g. GB10) where GPU pressure can reboot the machine.
        self.inference_device = resolve_device(inference_device)
        self.predict_batch_size = predict_batch_size
        self.model      = model.to(self.inference_device).eval()
        self.scaler     = scaler
        self.calibrator = calibrator   # IsotonicRegression or None

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability array with shape [N, 2] (FP prob, TP prob).

        Compatible with sklearn's predict_proba convention so that
        ``predict_proba(X)[:, 1]`` gives the true-positive probability.
        """
        device = self.inference_device
        self.model.to(device)
        X_scaled = self.scaler.transform(X).astype(np.float32)
        n = len(X_scaled)
        p_pos = np.empty(n, dtype=np.float64)

        for start in range(0, n, self.predict_batch_size):
            end = min(start + self.predict_batch_size, n)
            tensor = torch.from_numpy(X_scaled[start:end]).to(device)
            logits = self.model(tensor).squeeze(-1)
            p_pos[start:end] = torch.sigmoid(logits).cpu().numpy()

        if self.calibrator is not None:
            p_pos = self.calibrator.predict(p_pos)
        return np.column_stack([1.0 - p_pos, p_pos])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return hard 0/1 labels by thresholding predict_proba's TP column."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def __setstate__(self, state: dict) -> None:
        """Restore pickles written before inference_device/predict_batch_size existed."""
        state.setdefault("inference_device", torch.device("cpu"))
        state.setdefault("predict_batch_size", DEFAULT_INFERENCE_BATCH_SIZE)
        state["inference_device"] = resolve_device(state["inference_device"])
        self.__dict__.update(state)
        self.model = self.model.to(self.inference_device).eval()


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_nn_per_class_routers(
    train_df,
    val_df=None,
    model_type: Literal["mlp", "ft_transformer"] = "ft_transformer",
    epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    min_positives: int = 20,
    seed: int = 42,
    feature_names: list[str] | None = None,
    patience: int = 5,
    use_gpu: bool = True,
    inference_device: str = "cpu",
    predict_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
) -> dict[str, NNRouterWrapper | None]:
    """Train one neural-network router per nuScenes detection class.

    Overfitting control: the same held-out 10% calibration slice used for
    isotonic calibration (disjoint from the externally-supplied ``val_df``)
    also drives early stopping -- validation loss is tracked every epoch,
    the best-validation-loss epoch's weights are checkpointed, and training
    stops after ``patience`` epochs without improvement. The checkpointed
    (not final-epoch) weights are what get calibrated and returned. This
    was found necessary for low-count classes (e.g. construction_vehicle),
    which otherwise memorize the training partition well before epoch 30
    (see notebook/model_training_evaluation.ipynb Step 5).

    Args:
        train_df: Training DataFrame from build_dataset().
        val_df: Optional held-out validation DataFrame for reporting
            metrics. Never used for early stopping -- purely diagnostic.
        model_type: "mlp" or "ft_transformer".
        epochs: Maximum training epochs per class.
        batch_size: Mini-batch size (large batches work well with 2M rows).
        lr: Initial Adam learning rate.
        weight_decay: L2 regularisation weight.
        min_positives: Skip classes with too few positive training labels.
        seed: Random seed for reproducibility.
        feature_names: Feature columns to use (defaults to FEATURE_NAMES).
        patience: Epochs of no validation-loss improvement before stopping.
        use_gpu: Train on CUDA when available; set False to reduce memory pressure
            on unified-memory GPUs (GB10 / DGX Spark).
        inference_device: Device stored models use at predict time (default CPU).
        predict_batch_size: Chunk size for predict_proba to cap GPU spikes.

    Returns:
        Dict mapping class_name → NNRouterWrapper (or None if skipped).
    """
    from src.moe.features import FEATURE_NAMES as DEFAULT_FEATURES
    if feature_names is None:
        feature_names = DEFAULT_FEATURES

    torch.manual_seed(seed)
    np.random.seed(seed)

    routers: dict[str, NNRouterWrapper | None] = {}
    n_features = len(feature_names)
    train_device = resolve_device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

    log.info(
        "Training %s router per class on %s | features=%d | epochs=%d | batch=%d",
        model_type.upper(), train_device, n_features, epochs, batch_size,
    )

    for cls in NUSCENES_CLASSES:
        cls_id   = _CLS_TO_ID[cls]
        sub_train = train_df[train_df["class_id"] == cls_id].copy()

        n_pos   = int(sub_train["label"].sum())
        n_total = len(sub_train)

        if n_pos < min_positives:
            log.warning(
                "Class %-22s: %d positives / %d total — skipping", cls, n_pos, n_total
            )
            routers[cls] = None
            continue

        log.info(
            "Training %-22s: %d rows, %.1f%% positive",
            cls, n_total, 100.0 * n_pos / n_total,
        )

        # Hold out 10% of training data for isotonic calibration.
        # This avoids using the eval split for calibration (no data leakage).
        rng = np.random.default_rng(seed)
        cal_mask = rng.random(len(sub_train)) < 0.10
        sub_cal   = sub_train[cal_mask]
        sub_fit   = sub_train[~cal_mask]

        X_train = sub_fit[feature_names].values.astype(np.float32)
        y_train = sub_fit["label"].values.astype(np.float32)

        X_cal = sub_cal[feature_names].values.astype(np.float32)
        y_cal = sub_cal["label"].values.astype(np.float32)

        # ── Normalise features (essential for NNs; ignored by trees) ──────────
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_cal   = scaler.transform(X_cal).astype(np.float32)

        # ── pos_weight to counteract class imbalance ──────────────────────────
        n_neg      = n_total - n_pos
        device = train_device
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

        # ── Build model ───────────────────────────────────────────────────────
        if model_type == "mlp":
            model: nn.Module = MLPRouter(n_features=n_features)
        else:
            model = FTTransformerRouter(n_features=n_features)
        model = model.to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=3
        )

        # ── DataLoader ────────────────────────────────────────────────────────
        X_t = torch.from_numpy(X_train)
        y_t = torch.from_numpy(y_train).unsqueeze(1)
        dataset    = torch.utils.data.TensorDataset(X_t, y_t)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        # Validation loss (on the held-out calibration slice) drives both the
        # LR scheduler and early stopping -- training loss alone can't detect
        # overfitting since it keeps improving even as generalization worsens.
        X_cal_t = torch.from_numpy(X_cal).to(device)
        y_cal_t = torch.from_numpy(y_cal).unsqueeze(1).to(device)

        best_val_loss = float("inf")
        best_state: dict | None = None
        best_epoch = 0
        epochs_without_improvement = 0

        # ── Training epochs ───────────────────────────────────────────────────
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for xb, yb in dataloader:
                xb, yb = xb.to(device), yb.to(device)
                optimiser.zero_grad()
                logits = model(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()
                epoch_loss += loss.item() * len(xb)

            avg_train_loss = epoch_loss / len(sub_fit)

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_cal_t), y_cal_t).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch % 5 == 0 or epoch == epochs:
                log.info("  epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  lr=%.2e",
                         epoch, epochs, avg_train_loss, val_loss,
                         optimiser.param_groups[0]["lr"])

            if epochs_without_improvement >= patience:
                log.info(
                    "  Early stopping at epoch %d (best val_loss=%.4f at epoch %d)",
                    epoch, best_val_loss, best_epoch,
                )
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        # ── Isotonic calibration on the held-out 10% calibration set ─────────
        # The NN's sigmoid outputs with pos_weight may be poorly calibrated
        # for score-ranked evaluation (nuScenes AP sorts boxes by score).
        # Isotonic regression maps raw probabilities to calibrated ones that
        # better reflect the true positive rate at each score level.
        from sklearn.isotonic import IsotonicRegression
        with torch.no_grad():
            cal_logits = model(X_cal_t).squeeze(-1)
            cal_probs  = torch.sigmoid(cal_logits).cpu().numpy()
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(cal_probs, y_cal)
        log.info(
            "  Isotonic calibration fitted on %d samples (%.1f%% positive)",
            len(y_cal), 100.0 * y_cal.mean(),
        )

        wrapper = NNRouterWrapper(
            model.cpu(),
            scaler,
            calibrator,
            inference_device=inference_device,
            predict_batch_size=predict_batch_size,
        )

        if train_device.type == "cuda":
            del X_cal_t, y_cal_t, model
            gc.collect()
            torch.cuda.empty_cache()

        # ── Validation metrics ────────────────────────────────────────────────
        if val_df is not None:
            sub_val = val_df[val_df["class_id"] == cls_id]
            if len(sub_val) > 0 and sub_val["label"].sum() > 0:
                X_val = sub_val[feature_names].values.astype(np.float32)
                y_val = sub_val["label"].values.astype(int)
                proba = wrapper.predict_proba(X_val)[:, 1]
                auc   = roc_auc_score(y_val, proba)
                ap    = average_precision_score(y_val, proba)
                log.info("  └─ val AUC=%.4f  AP=%.4f", auc, ap)

        routers[cls] = wrapper

    return routers


# ══════════════════════════════════════════════════════════════════════════════
# Save / Load
# ══════════════════════════════════════════════════════════════════════════════

def save_nn_per_class_routers(
    routers: dict[str, NNRouterWrapper | None],
    output_dir: Path,
    meta: dict | None = None,
) -> None:
    """Save each per-class NN router to output_dir.

    Saves PyTorch state_dict + scaler as a single .pkl per class, using the
    same filename convention as the GBT routers so the inference script
    can load either type transparently.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for cls, wrapper in routers.items():
        if wrapper is None:
            continue
        path = output_dir / f"router_{cls}.pkl"
        with path.open("wb") as f:
            pickle.dump(wrapper, f)
        saved.append(cls)

    meta_out = {
        "router_type": "nn",
        "feature_names": None,  # populated by caller
        "classes_with_router": saved,
        "classes_fallback_to_global": [c for c in routers if routers[c] is None],
        **(meta or {}),
    }
    (output_dir / "router_per_class_meta.json").write_text(
        json.dumps(meta_out, indent=2)
    )
    log.info("Saved NN per-class routers for %d classes → %s", len(saved), output_dir)


def load_nn_per_class_routers(
    input_dir: Path,
    inference_device: str = "cpu",
) -> dict[str, NNRouterWrapper | None]:
    """Load per-class NN routers saved by save_nn_per_class_routers.

    Mirrors load_xgboost_per_class_routers so Section 7 of the notebook can run
    in a fresh kernel without re-training. Classes with no saved file map to
    None, matching the training function's contract.

    Weights written before routers were moved to CPU at save time contain CUDA
    tensors and cannot be unpickled on a CPU-only host; retrain to regenerate
    them if that happens.
    """
    input_dir = Path(input_dir)
    routers: dict[str, NNRouterWrapper | None] = {}

    for cls in NUSCENES_CLASSES:
        path = input_dir / f"router_{cls}.pkl"
        if not path.exists():
            routers[cls] = None
            continue
        with path.open("rb") as f:
            wrapper = pickle.load(f)
        wrapper.inference_device = resolve_device(inference_device)
        wrapper.model = wrapper.model.to(wrapper.inference_device).eval()
        routers[cls] = wrapper

    n = sum(r is not None for r in routers.values())
    log.info("Loaded NN per-class routers for %d classes ← %s", n, input_dir)
    return routers
