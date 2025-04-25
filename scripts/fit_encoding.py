#!/usr/bin/env python
"""
fit_encoding.py
Voxel-wise ridge regression with an 8-TR FIR design and group-wise CV.
"""
from __future__ import annotations
import argparse, gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold
from bids import BIDSLayout

from config import RAW, DERIV, FEAT_DIR, RESULT_DIR, LOGGER

# ────────── parameters ──────────
HRF_WEIGHTS = np.ones(8, dtype=np.float32)  # FIR window: 8 TRs = 0–7 s
HRF_LAG_TR  = 2                             # start 2 TR (≈2 s) after onset
VOX_BATCH   = 100                           # keep RAM in check
ALPHAS      = np.logspace(1, 5, 5)

# ────────── helpers ──────────
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
    """Return (n_tr, p) FIR design matrix."""
    p  = features.shape[1]
    X  = np.zeros((n_tr, p), dtype=np.float32)

    onsets_tr = (events["onset"].to_numpy() / tr_sec).round().astype(int)
    rows      = events["feat_row"].to_numpy().astype(int)

    for t0, row in zip(onsets_tr + HRF_LAG_TR, rows):
        for k, w in enumerate(HRF_WEIGHTS):
            t = t0 + k
            if t < n_tr:
                X[t] += w * features[row]
    return X

# ────────── CLI ──────────
ap = argparse.ArgumentParser()
ap.add_argument("--layer", required=True)
ap.add_argument("--subj",  nargs="*", default=None)
args    = ap.parse_args()
LAYER   = args.layer

# ────────── feature matrix ──────────
X_mmap = np.load(FEAT_DIR / f"{LAYER}.npy", mmap_mode="r")
LOGGER.info("Loaded features shape = %s", X_mmap.shape)

# ────────── iterate subjects ────────
layout   = BIDSLayout(RAW, validate=False)
subjects = args.subj or [p.name.split("-")[1] for p in DERIV.glob("sub-*")]

for subj in subjects:
    LOGGER.info("=== Subject %s | %s ===", subj, LAYER)

    runs = sorted(layout.get(subject=subj, task="5000scenes",
                             suffix="bold", extension=[".nii", ".nii.gz"]),
                  key=lambda r: r.path)
    if not runs:
        LOGGER.warning("  no runs – skipping")
        continue

    X_parts, Y_parts, group_ids = [], [], []
    for run_idx, r in enumerate(runs):
        tr_sec = float(r.get_metadata().get("RepetitionTime", 1.0))
        ev_p   = Path(r.path.replace("_bold.nii.gz", "_events_with_feat_row.tsv")
                              .replace("_bold.nii",  "_events_with_feat_row.tsv"))
        if not ev_p.exists():
            continue

        ev_df = load_events(ev_p)
        Y_run = np.load(cleaned_npy(r.path, subj))          # (TR, vox)
        X_run = build_design(Y_run.shape[0], ev_df, X_mmap, tr_sec)

        keep = X_run.any(1)
        if not keep.sum():
            continue
        X_parts.append(X_run[keep])
        Y_parts.append(Y_run[keep])
        group_ids.extend([run_idx] * keep.sum())

    if not X_parts:
        LOGGER.warning("  no usable trials – skipping")
        continue

    # equalise voxel count across runs
    min_vox   = min(y.shape[1] for y in Y_parts)
    Y_parts   = [y[:, :min_vox] for y in Y_parts]

    X_all = np.vstack(X_parts).astype(np.float32)
    Y_all = np.vstack(Y_parts).astype(np.float32)
    groups = np.array(group_ids, dtype=int)
    del X_parts, Y_parts; gc.collect()

    # z-score
    X_all = (X_all - X_all.mean(0, keepdims=True)) / (X_all.std(0, keepdims=True) + 1e-6)
    Y_all = (Y_all - Y_all.mean(0, keepdims=True)) / (Y_all.std(0, keepdims=True) + 1e-6)

    # group-wise split (train/test on different runs)
    tr_idx, te_idx = next(GroupKFold(n_splits=2).split(X_all, groups=groups))

    # hyper-parameter CV
    vox_sample = np.random.choice(Y_all.shape[1], min(500, Y_all.shape[1]), replace=False)
    α = float(RidgeCV(alphas=ALPHAS, scoring="r2", cv=3,
                      fit_intercept=False).fit(
                      X_all[tr_idx], Y_all[tr_idx][:, vox_sample]).alpha_)
    LOGGER.info("  best α = %.0f", α)

    base_ridge = Ridge(alpha=α, fit_intercept=True)

    # batched fit & predict
    Y_pred_parts = []
    for s in range(0, Y_all.shape[1], VOX_BATCH):
        e = min(s + VOX_BATCH, Y_all.shape[1])
        base_ridge.fit(X_all[tr_idx], Y_all[tr_idx, s:e])
        Y_pred_parts.append(base_ridge.predict(X_all[te_idx]))
    Y_pred = np.hstack(Y_pred_parts)

    # save (float16)
    RESULT_DIR.mkdir(exist_ok=True)
    np.save(RESULT_DIR / f"sub-{subj}_{LAYER}_pred.npy",  Y_pred.astype(np.float16))
    np.save(RESULT_DIR / f"sub-{subj}_{LAYER}_true.npy",  Y_all[te_idx].astype(np.float16))
    LOGGER.info("  saved predictions (%s TR × %s vox)  [float16]",
                Y_pred.shape[0], Y_pred.shape[1])
