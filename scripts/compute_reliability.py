# scripts/compute_reliability.py
from pathlib import Path
import numpy as np, nibabel as nib
from nilearn import masking
from config import DERIV, RESULT_DIR

subj   = "CSI1"
mask   = nib.load(next((DERIV / f"sub-{subj}").glob("*_brainmask.nii.gz")))
runs   = sorted((DERIV / f"sub-{subj}").glob("*run-*_bold.npy"))

split1, split2 = np.array_split(runs, 2)
def mean_img(run_list):
    arr = np.vstack([np.load(p) for p in run_list]).mean(0)
    vol = np.zeros(mask.shape, np.float32); vol[mask.get_fdata() > 0] = arr
    return vol

r = np.corrcoef(mean_img(split1)[mask.get_fdata()>0],
                mean_img(split2)[mask.get_fdata()>0])[0,1]
np.save(RESULT_DIR / f"sub-{subj}_reliability_mask.npy",
        (r > 0.1).astype("uint8"))
