#!/usr/bin/env python
"""
fit_encoding.py – HRF-convolved ridge-regression encoding model.

* Canonical Glover HRF per event
* GroupKFold (2-fold, split by run)
* RidgeCV across log-alphas 1e-2…1e4
* Batched voxel fitting to keep RAM < 8 GB
Outputs  float16 pred / true  →  results/sub-<ID>_<layer>_{pred,true}.npy
"""
from __future__ import annotations
import argparse, time, gc, sys
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold
from nilearn.glm.first_level import compute_regressor
from bids import BIDSLayout
import psutil

from config import RAW, DERIV, FEAT_DIR, RESULT_DIR, LOGGER, RAM_LIMIT

# ───────── CLI ─────────
cli = argparse.ArgumentParser()
cli.add_argument("--layer", required=True, help="e.g. resnet50_layer4.npy")
cli.add_argument("--subj",  nargs="*", default=None)
args = cli.parse_args()
LAYER = args.layer.replace(".npy", "")

# ───────── hyper-params ─────────
VOX_BATCH = 64
ALPHAS    = np.logspace(-2, 4, 24)     # 0.01 → 10 000


# ───────── helpers ─────────
def mem_gb() -> str:
    return f"{psutil.Process().memory_info().rss/2**30:5.2f} GB"


def cleaned_npy(run_path: str, subj: str) -> Path:
    return (DERIV / f"sub-{subj}" /
            Path(run_path).name.replace("_bold.nii.gz", "_bold.npy")
                            .replace("_bold.nii",    "_bold.npy"))

def load_events(ev_path: Path) -> pd.DataFrame:
    return (pd.read_csv(ev_path, sep="\t")
              .dropna(subset=["feat_row"])[["onset", "feat_row"]])

# ───────── helpers ─────────
from nilearn.glm.first_level import compute_regressor
import numpy as np

def build_design(n_tr: int,
                 events: pd.DataFrame,
                 features: np.memmap,
                 tr_sec: float) -> np.ndarray:
    """
    HRF-convolved design matrix (n_tr × p).

    For every stimulus onset we:
      • build a single-trial HRF regressor (Glover)
      • multiply that column-vector by the feature row (p dims)
      • add it into the big design matrix
    This keeps memory tiny and is still fast (<0.2 s/run on an M2 Max).
    """
    p  = features.shape[1]
    X  = np.zeros((n_tr, p), dtype=np.float32)
    ft = np.arange(n_tr) * tr_sec     # frame times in seconds

    for onset, feat_row in events.itertuples(index=False):
        # 1-element arrays → what compute_regressor expects
        hrf_col, _ = compute_regressor(
            exp_condition=(np.array([onset], dtype=float),
                           np.array([0.0],   dtype=float),   # duration 0 = impulse
                           np.array([1.0],   dtype=float)),  # amplitude = 1
            hrf_model="glover",
            frame_times=ft,
            con_id="stim")
        # hrf_col is shape (n_tr, 1).Ravel to 1-D
        X += hrf_col.ravel()[:, None] * features[int(feat_row)][None, :]

    return X



# ───────── load features ─────────
X_mmap = np.load(FEAT_DIR / f"{LAYER}.npy", mmap_mode="r")
LOGGER.info("Feature matrix %s  shape=%s", LAYER, X_mmap.shape)

layout   = BIDSLayout(RAW, validate=False)
subjects = (args.subj or
            [p.name.split("-")[1] for p in DERIV.glob("sub-*")])

for subj in subjects:
    LOGGER.info("=== %s | %s ===", subj, LAYER)
    tic = time.time()

    runs = sorted(layout.get(subject=subj, task="5000scenes",
                             suffix="bold", extension=[".nii", ".nii.gz"]),
                  key=lambda r: r.path)

    X_parts, Y_parts, groups = [], [], []
    for i, run in enumerate(runs, 1):
        tr = float(run.get_metadata().get("RepetitionTime", 1.0))
        ev_p = Path(run.path.replace("_bold.nii.gz", "_events_with_feat_row.tsv")
                             .replace("_bold.nii",    "_events_with_feat_row.tsv"))
        if not ev_p.exists():
            LOGGER.warning("  run%03d  → events missing, skip", i); continue

        Y_run = np.load(cleaned_npy(run.path, subj))  # (TR, vox)
        n_tr  = Y_run.shape[0]

        X_run = build_design(n_tr, load_events(ev_p), X_mmap, tr)
        keep  = X_run.any(1)
        if keep.sum() == 0:
            LOGGER.info("  run%03d  → no stimulus TRs", i); continue

        X_parts.append(X_run[keep].astype(np.float32))
        Y_parts.append(Y_run[keep].astype(np.float32))
        groups.extend([i] * keep.sum())
        gc.collect()

    if not X_parts:
        LOGGER.warning("  no usable data – abort subject"); continue

    # match voxel count
    min_vox = min(y.shape[1] for y in Y_parts)
    Y_parts = [y[:, :min_vox] for y in Y_parts]

    X_all = np.vstack(X_parts)
    Y_all = np.vstack(Y_parts)
    groups = np.asarray(groups, dtype=int)

    if X_all.nbytes > RAM_LIMIT:
        LOGGER.error("Design matrix %.2f GB > limit – abort",
                     X_all.nbytes/2**30); sys.exit(1)

    # z-score
    X_all = (X_all - X_all.mean(0)) / (X_all.std(0) + 1e-6)
    Y_all = (Y_all - Y_all.mean(0)) / (Y_all.std(0) + 1e-6)

    tr_idx, te_idx = next(GroupKFold(2).split(X_all, groups=groups))

    # alpha CV on ≤ 500 vox
    vox_sample = np.random.choice(Y_all.shape[1],
                                  min(500, Y_all.shape[1]),
                                  replace=False)
    α = float(RidgeCV(ALPHAS, scoring="r2", cv=3,
                      fit_intercept=False).fit(
                      X_all[tr_idx], Y_all[tr_idx][:, vox_sample]).alpha_)
    LOGGER.info("  α (CV) = %.3g", α)

    ridge = Ridge(alpha=α, fit_intercept=True)

    Y_pred_parts = []
    for j, s in enumerate(range(0, Y_all.shape[1], VOX_BATCH), 1):
        e = min(s + VOX_BATCH, Y_all.shape[1])
        ridge.fit(X_all[tr_idx], Y_all[tr_idx, s:e])
        Y_pred_parts.append(ridge.predict(X_all[te_idx]))
        if j % 20 == 0:
            LOGGER.info("    batch %3d vox %d/%d  mem %s",
                        j, e, Y_all.shape[1], mem_gb())
    Y_pred = np.hstack(Y_pred_parts)

    RESULT_DIR.mkdir(exist_ok=True)
    np.save(RESULT_DIR / f"sub-{subj}_{LAYER}_pred.npy",
            Y_pred.astype(np.float16), allow_pickle=False)
    np.save(RESULT_DIR / f"sub-{subj}_{LAYER}_true.npy",
            Y_all[te_idx].astype(np.float16), allow_pickle=False)
    LOGGER.info("  done (%.1f min)  final mem %s",
                (time.time()-tic)/60, mem_gb())
