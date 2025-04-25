#!/usr/bin/env python
"""
visualize_results.py  –  best-layer view (trimmed, ceiling-norm)

• Glass-brain: voxel-wise *max* R² across layers, threshold = 95th pct.
• ROI bars   : (top-25 % mean R²) ÷ (ROI noise ceiling)  → 0-to-1 scale.
               Colour = layer that wins the ROI.
"""
from __future__ import annotations
import argparse, logging, gc
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from nilearn import image, masking, plotting, datasets, surface
from nilearn.image import resample_to_img

# ───────────────────────── paths & logger ──────────────────────────────
try:
    from config import RESULT_DIR, DERIV, LOGGER
except ModuleNotFoundError:
    RESULT_DIR = Path(__file__).resolve().parents[1] / "results"
    DERIV      = Path(__file__).resolve().parents[1] / "data"
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    LOGGER = logging.getLogger("visualize")

# ──────────────────────────── CLI ───────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--subj", required=True, help="e.g. CSI1 (without 'sub-')")
p.add_argument("--layers", nargs="*")
p.add_argument("--metric", default="r2", choices=["r", "r2"])
p.add_argument("--surface", action="store_true")
p.add_argument("--top-n", type=int, default=15)
args = p.parse_args()
subj_prefix = f"sub-{args.subj}"

# ───────────────── layer list ───────────────────────────────────────────
if args.layers:
    layers: List[str] = args.layers
else:
    pattern = RESULT_DIR.glob(f"{subj_prefix}_*_*_{args.metric}.nii.gz")
    layers = sorted({"_".join(Path(p).stem.split("_")[1:-1]) for p in pattern})
if not layers:
    raise RuntimeError("No layers found – did you run evaluate.py?")
LOGGER.info("Layers: %s", ", ".join(layers))

# ─────────── atlas & resampling to subject space ───────────────────────
atlas  = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
a_img  = image.load_img(atlas.maps)
ref    = image.load_img(RESULT_DIR / f"{subj_prefix}_{layers[-1]}_{args.metric}.nii.gz")
a_img  = resample_to_img(a_img, ref, interpolation="nearest")
a_dat  = a_img.get_fdata().astype("int16")
labels = atlas.labels
roi_ids= np.arange(1, len(labels))

# ───────────── stack maps & voxel-wise max ─────────────────────────────
stack = np.stack(
    [image.load_img(RESULT_DIR / f"{subj_prefix}_{ly}_{args.metric}.nii.gz")
        .get_fdata(dtype="float32") for ly in layers],
    axis=-1
)                                           # (X,Y,Z,L)
best_img = image.new_img_like(ref, stack.max(-1))

# ───────────── optional reliability (noise ceiling) ────────────────────
rel_path = RESULT_DIR / f"{subj_prefix}_reliability_mask.npy"
rel_map  = (np.load(rel_path) if rel_path.exists() else
            np.ones_like(stack[...,0], dtype="float32"))
EPS = 1e-6

# ───────────── ROI table (top-25 % trimmed, ceiling-norm) ──────────────
records = []
for roi in roi_ids:
    mask = (a_dat == roi) & (rel_map > 0)        # keep reliable voxels only
    if mask.sum() < 100:                         # ignore tiny ROIs
        continue
    roi_vals = stack[mask]                       # (vox, L)
    rel_vals = rel_map[mask]
    # trimmed mean: top 25 %
    cut = int(0.75 * roi_vals.shape[0])
    roi_vals = np.sort(roi_vals, axis=0)[cut:]

    mean_by_layer = roi_vals.mean(0)
    best_idx = int(mean_by_layer.argmax())
    ceiling = rel_vals.mean()
    records.append(dict(
        ROI       = labels[roi],
        BestLayer = layers[best_idx],
        FracVar   = float(mean_by_layer[best_idx] / (ceiling + EPS))
    ))
    gc.collect()

if not records:
    LOGGER.warning("No ROI passed the reliability / size filter; "
                    "falling back to untrimmed mean R².")
    for roi in roi_ids:
        roi_mask = (a_dat == roi)
        if not roi_mask.any():
            continue
        vals = stack[roi_mask].mean(0)
        best = int(vals.argmax())
        records.append(dict(ROI=labels[roi],
                            BestLayer=layers[best],
                            FracVar=float(vals[best])))

df = (pd.DataFrame(records)
        .sort_values("FracVar", ascending=False)
        .head(args.top_n))

# ───────────── colour palette ──────────────────────────────────────────
cmap = cm.get_cmap("tab10")
layer_colors = {l: cmap(i % 10) for i, l in enumerate(layers)}

# ───────────── figure layout ───────────────────────────────────────────
n_rows = 2 if args.surface else 1
fig = plt.figure(figsize=(11, 3.5 * n_rows + 0.5 * len(df)))
gs  = fig.add_gridspec(n_rows, 2, height_ratios=[3] + ([3] if args.surface else []))

# (A) glass-brain – auto threshold (95th pct of non-zero voxels)
nonzero = best_img.get_fdata()[best_img.get_fdata()>0]
thr = np.percentile(nonzero, 95) if nonzero.size else 0.05
ax_brain = fig.add_subplot(gs[0, 0])
plotting.plot_glass_brain(best_img, threshold=thr,
                          display_mode="lyrz", colorbar=True, axes=ax_brain,
                          title=f"Voxel-wise best {args.metric.upper()} (thr={thr:.3f})")

# optional surface
if args.surface:
    fsavg   = datasets.fetch_surf_fsaverage()
    txt     = surface.vol_to_surf(best_img, fsavg.pial_left)
    ax_surf = fig.add_subplot(gs[0, 1], projection="3d")
    plotting.plot_surf_stat_map(fsavg.infl_left, txt, hemi="left",
                                threshold=thr, bg_map=fsavg.sulc_left,
                                axes=ax_surf, colorbar=False)
else:
    fig.add_subplot(gs[0,1]).axis("off")

# (B) bar chart
ax_bar = fig.add_subplot(gs[-1, :])
bars   = ax_bar.bar(np.arange(len(df)), df["FracVar"],
                    color=[layer_colors[l] for l in df["BestLayer"]])
ax_bar.set_xticks(np.arange(len(df)))
ax_bar.set_xticklabels(df["ROI"], rotation=90)
ax_bar.set_ylabel("Explained / ceiling")
ax_bar.margins(x=0.01)
# legend
handles = [plt.Rectangle((0,0),1,1,color=layer_colors[l]) for l in layers]
ax_bar.legend(handles, layers, title="Winning layer",
              bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()

# ───────────── save ────────────────────────────────────────────────────
out_dir = RESULT_DIR / "figures"; out_dir.mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(out_dir / f"{subj_prefix}_encoding_summary.{ext}", dpi=300)
LOGGER.info("Saved → %s", out_dir / f"{subj_prefix}_encoding_summary.png")
