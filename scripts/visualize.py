#!/usr/bin/env python
"""
visualize.py
Example: project R² map to fsaverage surface and scatter true vs. predicted
for the best voxel.
"""
from nilearn import datasets, plotting, surface
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from config import RESULT_DIR, DERIV

SUBJ  = "sub-CSI1"
LAYER = "resnet50_layer3"

r2_img = nib.load(RESULT_DIR / f"{SUBJ}_{LAYER}_r2.nii.gz")
mask   = nib.load(DERIV / SUBJ / f"{SUBJ}_brainmask.nii.gz")

fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
tex = surface.vol_to_surf(r2_img, fsavg.pial_left)

plotting.view_surf(fsavg.infl_left, tex, hemi='left',
                   cmap='viridis', threshold=0.0).open_in_browser()

best = np.nanargmax(tex)
y_true = np.load(RESULT_DIR / f"{SUBJ}_{LAYER}_true.npy")[:, best]
y_pred = np.load(RESULT_DIR / f"{SUBJ}_{LAYER}_pred.npy")[:, best]

plt.figure()
plt.scatter(y_true, y_pred, alpha=.5)
plt.xlabel("True response")
plt.ylabel("Predicted")
plt.title(f"{SUBJ}  |  {LAYER}  |  voxel {best}")
plt.show()
