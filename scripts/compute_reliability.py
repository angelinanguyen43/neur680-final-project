#!/usr/bin/env python
"""
compute_reliability.py – Split-half voxel reliability (odd vs even runs).

Outputs full-resolution reliability volume  (float32 .npy).
"""
from __future__ import annotations
import argparse, gc
from pathlib import Path
import numpy as np, nibabel as nib
from nilearn.masking import apply_mask
from bids import BIDSLayout
from config import RAW, DERIV, RESULT_DIR, LOGGER

cli = argparse.ArgumentParser()
cli.add_argument("--subj", required=True, help="CSI1 (without 'sub-')")
args = cli.parse_args(); subj = args.subj

subj_dir = DERIV / f"sub-{subj}"
mask_img = nib.load(next(subj_dir.glob("*_brainmask.nii.gz")))
mask     = mask_img.get_fdata() > 0

runs = sorted(subj_dir.glob("*_bold.npy"))
if len(runs) < 2:
    raise RuntimeError("Need ≥2 runs for split-half reliability.")

odd_arr, even_arr = [], []
for i, p in enumerate(runs):
    data = np.load(p)
    keep = ~np.all(data == 0, axis=1)
    if not keep.any():
        continue
    (odd_arr if i % 2 else even_arr).append(data[keep])
    gc.collect()

if not odd_arr or not even_arr:
    raise RuntimeError("Odd or even split ended up empty.")

odd  = np.vstack(odd_arr)
even = np.vstack(even_arr)
min_len = min(len(odd), len(even))
odd, even = odd[:min_len], even[:min_len]

mu_o, mu_e = odd.mean(0), even.mean(0)
cov = ((odd - mu_o) * (even - mu_e)).mean(0)
r   = cov / (odd.std(0)*even.std(0) + 1e-12)

rel_vol = np.zeros(mask.shape, np.float32)
rel_vol[mask] = r
out = RESULT_DIR / f"sub-{subj}_reliability.npy"
np.save(out, rel_vol)
LOGGER.info("Reliability map → %s  (min %.3f median %.3f max %.3f)",
            out, r.min(), np.median(r), r.max())
