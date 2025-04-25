#!/usr/bin/env python
"""
evaluate.py – Compute voxel-wise r and R² in small chunks (128 vox).

Writes NIfTI maps (r & r²) and optionally deletes heavy .npy files.
"""
from __future__ import annotations
import argparse, gc
from pathlib import Path
import numpy as np, nibabel as nib
from tqdm import tqdm
from config import DERIV, RESULT_DIR, LOGGER

EPS        = 1e-7
VOX_BATCH  = 128

cli = argparse.ArgumentParser()
cli.add_argument("--layer", required=True)
cli.add_argument("--subj",  nargs="*", default=None)
cli.add_argument("--keep",  action="store_true")
args  = cli.parse_args()
LAYER = args.layer.replace(".npy", "")

subjects = (args.subj or
            [p.name.split("-")[1] for p in DERIV.glob("sub-*")])

for subj in subjects:
    prefix = f"sub-{subj}"
    LOGGER.info(prefix)

    try:
        mask_img = nib.load(next((DERIV / prefix).glob("*_brainmask.nii.gz")))
    except StopIteration:
        LOGGER.warning("  mask not found – skipping"); continue

    pred_f = RESULT_DIR / f"{prefix}_{LAYER}_pred.npy"
    true_f = RESULT_DIR / f"{prefix}_{LAYER}_true.npy"
    if not (pred_f.exists() and true_f.exists()):
        LOGGER.warning("  missing pred/true – skipping"); continue

    Y_pred = np.load(pred_f).astype(np.float32, copy=False)
    Y_true = np.load(true_f).astype(np.float32, copy=False)
    n_tr, n_vox = Y_true.shape
    LOGGER.info("  trials=%d   vox=%d", n_tr, n_vox)

    r_out  = np.empty(n_vox, dtype=np.float32)
    r2_out = np.empty(n_vox, dtype=np.float32)

    for s in tqdm(range(0, n_vox, VOX_BATCH), desc="voxels"):
        e   = min(s + VOX_BATCH, n_vox)
        yt  = Y_true[:, s:e]
        yp  = Y_pred[:, s:e]

        cov = (yt * yp).mean(0) - yt.mean(0) * yp.mean(0)
        r_out[s:e]  = cov / (yt.std(0) * yp.std(0) + EPS)

        mse   = ((yt - yp) ** 2).sum(0)
        var_t = ((yt - yt.mean(0)) ** 2).sum(0).clip(min=EPS)
        r2_out[s:e] = 1.0 - mse / var_t
        gc.collect()

    for arr, tag in [(r_out, "r"), (r2_out, "r2")]:
        vol = np.zeros(mask_img.shape, dtype=np.float32)
        vol[mask_img.get_fdata() > 0] = arr
        out = RESULT_DIR / f"{prefix}_{LAYER}_{tag}.nii.gz"
        nib.save(nib.Nifti1Image(vol, mask_img.affine, mask_img.header), out)
        LOGGER.info("  wrote %s", out.name)

    if not args.keep:
        pred_f.unlink(missing_ok=True)
        true_f.unlink(missing_ok=True)
        LOGGER.info("  deleted temporary .npy files")
