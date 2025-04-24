#!/usr/bin/env python
"""
fit_encoding.py
Voxel-wise ridge regression for ONE DNN layer, truly laptop-safe.

Usage examples
--------------
python scripts/fit_encoding.py --layer resnet50_layer1 --subj CSI1
python scripts/fit_encoding.py --layer resnet50_layer3              # all subjects
"""
from __future__ import annotations
import argparse, gc
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from bids import BIDSLayout

from config import RAW, DERIV, FEAT_DIR, MODEL_DIR, RESULT_DIR, LOGGER

# ────────── helpers ──────────
def cleaned_npy(run_nifti_path: str, subj: str) -> Path:
    return (DERIV / f"sub-{subj}" /
            Path(run_nifti_path).name
                .replace("_bold.nii.gz", "_bold.npy")
                .replace("_bold.nii",  "_bold.npy"))

def trial_rows(ev_path: Path) -> np.ndarray:
    ev = pd.read_csv(ev_path, sep="\t").dropna(subset=["feat_row"])
    return ev["feat_row"].astype(int).to_numpy()

# ────────── CLI ──────────
ap = argparse.ArgumentParser()
ap.add_argument("--layer", required=True)
ap.add_argument("--subj", nargs="*", default=None,
                help="IDs without 'sub-'; omit to run all")
args = ap.parse_args()
LAYER = args.layer

# ────────── load DNN features (on-disk) ──────────
X_mmap = np.load(FEAT_DIR / f"{LAYER}.npy", mmap_mode="r")   # (n_img, p)
LOGGER.info("Loaded %s (mmap, shape=%s)", LAYER, X_mmap.shape)

EVENT_SUFFIX = "_events_with_feat_row.tsv"
layout = BIDSLayout(RAW, validate=False)
subjects = args.subj or [p.name.split("-")[1] for p in DERIV.glob("sub-*")]

for subj in subjects:
    LOGGER.info("=== %s : %s ===", subj, LAYER)

    runs = layout.get(subject=subj, task="5000scenes",
                      suffix="bold", extension=[".nii", ".nii.gz"])
    if not runs:
        LOGGER.warning("  no raw runs – skipping")
        continue

    # ---------- assemble design matrices ----------
    X_rows, Y_parts = [], []
    for r in sorted(runs, key=lambda r: r.path):
        ev_p = Path(r.path.replace("_bold.nii.gz", EVENT_SUFFIX)
                              .replace("_bold.nii",  EVENT_SUFFIX))
        idx = trial_rows(ev_p)              # rows for *stimuli* in this run
        if idx.size == 0:
            continue

        run_Y = np.load(cleaned_npy(r.path, subj))     # (TR, vox)
        Y_parts.append(run_Y[idx])                     # keep only stimulus TRs
        X_rows.append(idx)

    if not X_rows:
        LOGGER.warning("  no usable trials – skipping")
        continue

    #  match voxel count across runs
    min_vox = min(a.shape[1] for a in Y_parts)
    Y_parts = [a[:, :min_vox] for a in Y_parts]

    # ── slice once, copy to float32 ──
    X_all = np.asarray(X_mmap[np.concatenate(X_rows)], dtype=np.float32)
    Y_all = np.concatenate(Y_parts, axis=0).astype(np.float32)

    # free the per-run arrays immediately
    del Y_parts; gc.collect()

    kfold = KFold(n_splits=2, shuffle=True, random_state=42)
    tr_idx, te_idx = next(kfold.split(X_all))

    # ---------- cross-validate α on 500 random voxels ----------
    vox_sample = np.random.choice(Y_all.shape[1], 500, replace=False)
    ridge_cv = RidgeCV(alphas=np.logspace(1, 5, 5),
                       scoring="r2", cv=3, fit_intercept=False)
    ridge_cv.fit(X_all[tr_idx], Y_all[tr_idx][:, vox_sample])
    best_alpha = float(ridge_cv.alpha_)
    LOGGER.info("    best α (500-voxel CV) = %.0f", best_alpha)

    base_ridge = Ridge(alpha=best_alpha, fit_intercept=False)

    # ---------- batched fit & predict ----------
    VOX_BATCH = 100
    Y_pred_parts = []
    te_X = X_all[te_idx]

    for start in range(0, Y_all.shape[1], VOX_BATCH):
        stop = min(start + VOX_BATCH, Y_all.shape[1])
        base_ridge.fit(X_all[tr_idx], Y_all[tr_idx, start:stop])
        Y_pred_parts.append(base_ridge.predict(te_X))

        if (start // VOX_BATCH) % 20 == 0:
            LOGGER.info("    voxels %d-%d / %d", start, stop, Y_all.shape[1])

    Y_pred = np.concatenate(Y_pred_parts, axis=1)      # (n_te, vox)

    # ---------- save ----------
    MODEL_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

    # coefficients only (float32) → ~1.3 GB for layer4, ≪ that for others
    np.save(RESULT_DIR / f"sub-{subj}_{LAYER}_coef.npy",
            base_ridge.coef_.astype(np.float32))

    # predictions + ground truth for evaluation
    np.save(RESULT_DIR / f"sub-{subj}_{LAYER}_pred.npy", Y_pred)
    np.save(RESULT_DIR / f"sub-{subj}_{LAYER}_true.npy", Y_all[te_idx])
    LOGGER.info("    saved coef, pred, and truth arrays")
