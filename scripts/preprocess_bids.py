#!/usr/bin/env python
"""
preprocess_bids.py – FAST single-subject preprocessing for BOLD5000

Creates *one* brain mask per subject so every run ends up with the same
(voxel) columns.  Output: one .npy per run in data/sub-<ID>/.
"""
from pathlib import Path
import numpy as np
import nibabel as nib
from nilearn.masking import apply_mask, compute_epi_mask
from nilearn.signal import clean
from bids import BIDSLayout
from joblib import Parallel, delayed
from tqdm import tqdm
from datalad import api as dl     # type: ignore

from config import RAW, DERIV, LOGGER

# ───────── parameters ─────────
N_JOBS          = 6
HIGH_PASS       = 0.01
DTYPE           = np.float32
PLACEHOLDER_MAX = 1_000_000       # ≤1 MB ⇒ DataLad stub
SUBJECTS_TO_RUN = {"CSI1"}        # edit or set empty for all subjects

# ───────── helper ─────────
def _proc_one(run_path: str,
              mask_img: nib.Nifti1Image,
              out_path: Path,
              tr: float) -> tuple[int, int]:
    """Load a BOLD run, apply subject mask, temporal clean, save .npy."""
    if Path(run_path).stat().st_size < PLACEHOLDER_MAX:
        dl.get(run_path) # pylint: disable=no-member

    data_2d = apply_mask(run_path, mask_img)          # TR × vox
    cleaned = clean(data_2d,
                    detrend=True,
                    standardize="zscore_sample",
                    high_pass=HIGH_PASS,
                    low_pass=None,
                    t_r=tr,
                    confounds=None).astype(DTYPE)

    np.save(out_path.with_suffix(".npy"), cleaned, allow_pickle=False)
    return cleaned.shape            # (n_TR, n_vox)

# ───────── main ─────────
def main() -> None:
    layout = BIDSLayout(RAW, validate=False)

    for subj in sorted(SUBJECTS_TO_RUN):
        LOGGER.info("Subject %s", subj)
        subj_dir = DERIV / f"sub-{subj}"
        subj_dir.mkdir(parents=True, exist_ok=True)

        runs = layout.get(subject=subj, task="5000scenes",
                          suffix="bold", extension=[".nii", ".nii.gz"])
        if not runs:
            LOGGER.warning("  no 5000-scenes runs – skipping")
            continue

        # ── one mask per subject ─────────────────────────────────────────
        first_bold = runs[0].path
        if Path(first_bold).stat().st_size < PLACEHOLDER_MAX:
            dl.get(first_bold) # pylint: disable=no-member

        mask_path = subj_dir / f"sub-{subj}_brainmask.nii.gz"
        if mask_path.exists():
            mask_img = nib.load(mask_path)
        else:
            mask_img = compute_epi_mask(first_bold)
            nib.save(mask_img, mask_path)
            LOGGER.info("  saved mask → %s", mask_path)

        # ── job list (run_path, TR) ─────────────────────────────────────
        jobs = [(r.path, float(r.get_metadata().get("RepetitionTime", 1.0)))
                for r in runs]
        LOGGER.info("  processing %d runs with %d job(s)…", len(jobs), N_JOBS)

        def _job(run_path: str, tr: float):
            out = subj_dir / Path(run_path).name.replace(".nii.gz", "").replace(".nii", "")
            return _proc_one(run_path, mask_img, out, tr)

        # ── execute in parallel ─────────────────────────────────────────
        shapes = Parallel(n_jobs=N_JOBS, backend="loky", max_nbytes=None)(
            delayed(_job)(rp, tr) for rp, tr in tqdm(jobs, desc="runs")
        )

        n_vox     = shapes[0][1]
        total_trs = sum(s[0] for s in shapes)
        LOGGER.info("  finished: %s vox × %s TRs", f"{n_vox:,}", f"{total_trs:,}")

if __name__ == "__main__":
    main()
