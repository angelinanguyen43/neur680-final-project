#!/usr/bin/env python
"""
preprocess_bids.py – FAST single-subject preprocessing for BOLD5000

* One brain mask per subject
* Minimal clean: detrend only (no high-pass, no standardise here)
* Output: one .npy per run  (shape = TR × vox, float32)
"""
from __future__ import annotations
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np, nibabel as nib
from nilearn.masking import apply_mask, compute_epi_mask
from nilearn.signal import clean
from bids import BIDSLayout
from tqdm import tqdm
from datalad import api as dl     # type: ignore

from config import RAW, DERIV, LOGGER, N_CORES, RAM_LIMIT

# ───────── parameters ─────────
PLACEHOLDER_MAX = 1_000_000      # ≤1 MB ⇒ DataLad stub
SUBJECTS_TO_RUN = {"CSI1"}       # empty set → all subjects


def _proc_one(run_path: str,
              mask_img: nib.Nifti1Image,
              out_path: Path,
              tr: float) -> tuple[int, int]:
    """Load BOLD, apply mask, detrend, save .npy; return (TR, vox)."""
    if Path(run_path).stat().st_size < PLACEHOLDER_MAX:
        dl.get(run_path)        # ensure data downloaded

    data_2d = apply_mask(run_path, mask_img)          # TR × vox
    cleaned = clean(data_2d,
                    detrend=True,
                    standardize=False,    # we z-score later
                    high_pass=None,
                    low_pass=None,
                    t_r=tr,
                    confounds=None).astype(np.float32)
    np.save(out_path.with_suffix(".npy"), cleaned, allow_pickle=False)
    return cleaned.shape


def main() -> None:
    layout = BIDSLayout(RAW, validate=False)

    subjects = sorted(SUBJECTS_TO_RUN or
                      {p.entities["subject"] for p in layout.get_suffix("bold")})
    for subj in subjects:
        LOGGER.info("Subject %s", subj)
        subj_dir = DERIV / f"sub-{subj}"
        subj_dir.mkdir(exist_ok=True, parents=True)

        runs = layout.get(subject=subj, task="5000scenes",
                          suffix="bold", extension=[".nii", ".nii.gz"])
        if not runs:
            LOGGER.warning("  no 5000-scenes runs – skipping"); continue

        # ── brain mask ──
        first_bold = runs[0].path
        if Path(first_bold).stat().st_size < PLACEHOLDER_MAX:
            dl.get(first_bold)
        mask_path = subj_dir / f"sub-{subj}_brainmask.nii.gz"
        mask_img = (nib.load(mask_path) if mask_path.exists()
                    else compute_epi_mask(first_bold))
        nib.save(mask_img, mask_path)

        # ── parallel clean ──
        jobs = [(r.path, float(r.get_metadata().get("RepetitionTime", 1.0)))
                for r in runs]
        LOGGER.info("  %d runs → %d workers", len(jobs), N_CORES)

        def _job(rp: str, tr: float):
            out = subj_dir / Path(rp).name.replace(".nii.gz", "").replace(".nii", "")
            return _proc_one(rp, mask_img, out, tr)

        shapes = Parallel(n_jobs=N_CORES, backend="loky", max_nbytes=None)(
            delayed(_job)(rp, tr) for rp, tr in tqdm(jobs, desc="runs")
        )

        vox = shapes[0][1]
        trs = sum(s[0] for s in shapes)
        LOGGER.info("  finished: %s vox × %s TRs", f"{vox:,}", f"{trs:,}")


if __name__ == "__main__":
    main()
