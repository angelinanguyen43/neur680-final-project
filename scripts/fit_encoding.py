#!/usr/bin/env python
"""
fit_encoding.py
FIR (lag-search) voxel-wise ridge with group-wise CV – laptop safe.

• Uses a 7-lag FIR basis (0…6 TR) so the model learns its optimal delay.
• GroupKFold ensures train/test are different runs.
• Saves float16 predictions to keep disk/RAM small.
"""
from __future__ import annotations
import argparse, gc
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold
from bids import BIDSLayout

from config import RAW, DERIV, FEAT_DIR, RESULT_DIR, LOGGER

# ───────── parameters ──────────
LAGS       = np.arange(0, 7)                # 0,1,2,3,4,5,6 TR  (~0–6 s)
NUM_LAGS   = len(LAGS)
VOX_BATCH  = 50                             # lower → safer RAM
ALPHAS     = np.logspace(1, 5, 5)

# ───────── helpers ──────────
def cleaned_npy(run_path: str, subj: str) -> Path:
    return (DERIV / f"sub-{subj}" /
            Path(run_path).name.replace("_bold.nii.gz", "_bold.npy")
                              .replace("_bold.nii",  "_bold.npy"))

def load_events(ev_path: Path) -> pd.DataFrame:
    return (pd.read_csv(ev_path, sep="\t")
              .dropna(subset=["feat_row"])[["onset", "feat_row"]])

def build_design(n_tr: int,
                 events: pd.DataFrame,
                 features: np.memmap,
                 tr_sec: float) -> np.ndarray:
    """
    Build (n_tr, p0*NUM_LAGS) design where each lag is a feature block.
    """
    p0 = features.shape[1]
    X  = np.zeros((n_tr, p0 * NUM_LAGS), dtype=np.float32)

    on_tr = (events["onset"].to_numpy() / tr_sec).round().astype(int)
    rows  = events["feat_row"].to_numpy().astype(int)

    for t0, r in zip(on_tr, rows):
        for j, lag in enumerate(LAGS):
            t = t0 + lag
            if t < n_tr:
                X[t, j*p0:(j+1)*p0] += features[r]
    return X

# ───────── CLI ──────────
ap = argparse.ArgumentParser()
ap.add_argument("--layer", required=True)
ap.add_argument("--subj",  nargs="*", default=None)
args   = ap.parse_args()
LAYER  = args.layer

# ───────── feature matrix ────────
X_mmap = np.load(FEAT_DIR / f"{LAYER}.npy", mmap_mode="r")
p0     = X_mmap.shape[1]
LOGGER.info("Loaded %s features (%d dims)", LAYER, p0)

# ───────── subjects ─────────────
layout   = BIDSLayout(RAW, validate=False)
subjects = args.subj or [p.name.split("-")[1] for p in DERIV.glob("sub-*")]

for subj in subjects:
    LOGGER.info("=== %s | %s ===", subj, LAYER)

    runs = sorted(layout.get(subject=subj, task="5000scenes",
                             suffix="bold", extension=[".nii", ".nii.gz"]),
                  key=lambda r: r.path)
    if not runs:
        LOGGER.warning("  no runs – skipping")
        continue

    X_parts, Y_parts, groups = [], [], []
    for idx, r in enumerate(runs):
        tr = float(r.get_metadata().get("RepetitionTime", 1.0))
        ev_tsv = Path(str(r.path).replace("_bold", "_events_with_feat_row")
                                   .replace(".nii.gz", ".tsv")
                                   .replace(".nii", ".tsv"))
        if not ev_tsv.exists():
            continue

        ev_df = load_events(ev_tsv)
        Y_run = np.load(cleaned_npy(r.path, subj))          # (TR, vox)
        X_run = build_design(Y_run.shape[0], ev_df, X_mmap, tr)

        keep = X_run.any(1)
        if not keep.sum():
            continue
        X_parts.append(X_run[keep])
        Y_parts.append(Y_run[keep])
        groups.extend([idx] * keep.sum())

    if not X_parts:
        LOGGER.warning("  no usable trials – skipping")
        continue

    # equalise voxel count
    min_vox = min(y.shape[1] for y in Y_parts)
    Y_parts = [y[:, :min_vox] for y in Y_parts]

    X_all = np.vstack(X_parts).astype(np.float32)
    Y_all = np.vstack(Y_parts).astype(np.float32)
    groups = np.array(groups, int)
    del X_parts, Y_parts; gc.collect()

    # z-score
    X_all = (X_all - X_all.mean(0)) / (X_all.std(0) + 1e-6)
    Y_all = (Y_all - Y_all.mean(0)) / (Y_all.std(0) + 1e-6)

    tr_idx, te_idx = next(GroupKFold(2).split(X_all, groups=groups))

    vox_sample = np.random.choice(Y_all.shape[1], min(500, Y_all.shape[1]), replace=False)
    α = float(RidgeCV(ALPHAS, scoring="r2", cv=3, fit_intercept=False)
              .fit(X_all[tr_idx], Y_all[tr_idx][:, vox_sample]).alpha_)
    LOGGER.info("  best α = %.0f", α)

    ridge = Ridge(alpha=α, fit_intercept=True)

    # batched fit
    Y_pred = np.empty_like(Y_all[te_idx], dtype=np.float16)
    for s in range(0, Y_all.shape[1], VOX_BATCH):
        e = min(s + VOX_BATCH, Y_all.shape[1])
        ridge.fit(X_all[tr_idx], Y_all[tr_idx, s:e])
        Y_pred[:, s:e] = ridge.predict(X_all[te_idx]).astype(np.float16)

    # save
    RESULT_DIR.mkdir(exist_ok=True)
    np.save(RESULT_DIR / f"sub-{subj}_{LAYER}_pred.npy", Y_pred)
    np.save(RESULT_DIR / f"sub-{subj}_{LAYER}_true.npy",
            Y_all[te_idx].astype(np.float16))
    LOGGER.info("  saved predictions (%s TR × %s vox) [float16]",
                Y_pred.shape[0], Y_pred.shape[1])
