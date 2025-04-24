#!/usr/bin/env python
"""
evaluate.py
Compute voxel-wise Pearson r and R² for held-out trials.
"""
from __future__ import annotations
import argparse, numpy as np, nibabel as nib
from pathlib import Path
from config import DERIV, RESULT_DIR, LOGGER

p = argparse.ArgumentParser()
p.add_argument("--layer", required=True,
               help="Layer name used in fit_encoding, e.g. resnet50_layer1")
p.add_argument("--subj",  nargs="*", default=None,
               help="Subject IDs; omit to evaluate all")
args = p.parse_args()
LAYER: str = args.layer
EPS = 1e-10

subjects = args.subj or [p.name.split("-")[1] for p in DERIV.glob("sub-*")]

for subj in subjects:
    subj_prefix = f"sub-{subj}"
    LOGGER.info(subj_prefix)

    try:
        mask_img = nib.load(next((DERIV / subj_prefix).glob("*_brainmask.nii.gz")))
    except StopIteration:
        LOGGER.warning("  mask not found – skipping")
        continue

    Y_pred = np.load(RESULT_DIR / f"{subj_prefix}_{LAYER}_pred.npy").astype(np.float64)
    Y_true = np.load(RESULT_DIR / f"{subj_prefix}_{LAYER}_true.npy").astype(np.float64)
    LOGGER.info("  computing metrics  (%d trials, %d vox)",
            Y_true.shape[0], Y_true.shape[1])
    cov = (Y_true * Y_pred).mean(0) - Y_true.mean(0) * Y_pred.mean(0)
    r  = cov / (Y_true.std(0) * Y_pred.std(0) + EPS)
    r2 = 1 - ((Y_true - Y_pred)**2).sum(0) / (((Y_true - Y_true.mean(0))**2).sum(0) + EPS)

    for arr, name in [(r, "r"), (r2, "r2")]:
        vol = np.zeros(mask_img.shape, dtype=np.float32)
        vol[mask_img.get_fdata() > 0] = arr
        out = RESULT_DIR / f"{subj_prefix}_{LAYER}_{name}.nii.gz"
        nib.save(nib.Nifti1Image(vol, mask_img.affine), out)
        LOGGER.info("  wrote %s", out.name)
