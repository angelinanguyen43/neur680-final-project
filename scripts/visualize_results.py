#!/usr/bin/env python
"""
visualize_results.py – Best-layer glass-brain + ROI bars.

Threshold = 95th pct of non-zero voxels.
"""
from __future__ import annotations
import argparse, gc
from pathlib import Path
from typing import List
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from nilearn import image, masking, plotting, datasets, surface
from nilearn.image import resample_to_img
from config import RESULT_DIR, DERIV, FIG_DIR, LOGGER

# ───────── CLI ─────────
cli = argparse.ArgumentParser()
cli.add_argument("--subj", required=True)
cli.add_argument("--layers", nargs="*")
cli.add_argument("--metric", default="r2", choices=["r", "r2"])
cli.add_argument("--surface", action="store_true")
cli.add_argument("--top-n", type=int, default=15)
args = cli.parse_args()
prefix = f"sub-{args.subj}"

# ───────── assemble layer list ─────────
if args.layers:
    layers: List[str] = args.layers
else:
    pattern = RESULT_DIR.glob(f"{prefix}_*_*_{args.metric}.nii.gz")
    layers = sorted({"_".join(Path(p).stem.split("_")[1:-1]) for p in pattern})
if not layers:
    raise RuntimeError("No layers found; run evaluate.py first.")
LOGGER.info("Layers: %s", ", ".join(layers))

# ───────── atlas resampling ─────────
atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
a_img = image.load_img(atlas.maps)
ref   = image.load_img(RESULT_DIR / f"{prefix}_{layers[-1]}_{args.metric}.nii.gz")
a_img = resample_to_img(a_img, ref, interpolation="nearest")
a_dat = a_img.get_fdata().astype("int16")
labels = atlas.labels
roi_ids = np.arange(1, len(labels))

# ───────── stack maps & voxel-wise max ─────────
stack = np.stack(
    [image.load_img(RESULT_DIR / f"{prefix}_{ly}_{args.metric}.nii.gz")
         .get_fdata(dtype="float32") for ly in layers],
    axis=-1
)
best_img = image.new_img_like(ref, stack.max(-1))

# ───────── optional noise-ceiling ─────────
rel_path = RESULT_DIR / f"{prefix}_reliability.npy"
rel_map  = (np.load(rel_path) if rel_path.exists()
            else np.ones_like(stack[...,0], dtype="float32"))
EPS = 1e-6

# ───────── ROI summary table ─────────
records = []
for roi in roi_ids:
    mask = (a_dat == roi) & (rel_map > 0)
    if mask.sum() < 100:
        continue
    roi_vals = stack[mask]
    rel_vals = rel_map[mask]
    cut = int(0.75 * roi_vals.shape[0])          # top 25 %
    roi_vals = np.sort(roi_vals, axis=0)[cut:]

    mean_layer = roi_vals.mean(0)
    best_idx   = int(mean_layer.argmax())
    records.append(dict(
        ROI       = labels[roi],
        BestLayer = layers[best_idx],
        FracVar   = float(mean_layer[best_idx] / (rel_vals.mean() + EPS))
    ))
    gc.collect()

df = (pd.DataFrame(records)
        .sort_values("FracVar", ascending=False)
        .head(args.top_n))

# ───────── colour palette ─────────
cmap = cm.get_cmap("tab10")
layer_colors = {l: cmap(i % 10) for i, l in enumerate(layers)}

# ───────── figure layout ─────────
n_rows = 2 if args.surface else 1
fig = plt.figure(figsize=(11, 3.5 * n_rows + 0.5 * len(df)))
gs  = fig.add_gridspec(n_rows, 2, height_ratios=[3] + ([3] if args.surface else []))

# (A) glass-brain
nonzero = best_img.get_fdata()[best_img.get_fdata() > 0]
thr = np.percentile(nonzero, 95) if nonzero.size else 0.05
ax_brain = fig.add_subplot(gs[0, 0])
plotting.plot_glass_brain(best_img, threshold=thr,
                          display_mode="lyrz", colorbar=True, axes=ax_brain,
                          title=f"Voxel-wise best {args.metric.upper()} (thr {thr:.3f})")

# surface view
if args.surface:
    fsavg = datasets.fetch_surf_fsaverage()
    txt   = surface.vol_to_surf(best_img, fsavg.pial_left)
    ax_surf = fig.add_subplot(gs[0, 1], projection="3d")
    plotting.plot_surf_stat_map(fsavg.infl_left, txt, hemi="left",
                                threshold=thr, bg_map=fsavg.sulc_left,
                                axes=ax_surf, colorbar=False)
else:
    fig.add_subplot(gs[0,1]).axis("off")

# (B) bar chart
ax_bar = fig.add_subplot(gs[-1, :])
bars = ax_bar.bar(np.arange(len(df)), df["FracVar"],
                  color=[layer_colors[l] for l in df["BestLayer"]])
ax_bar.set_xticks(np.arange(len(df)))
ax_bar.set_xticklabels(df["ROI"], rotation=90)
ax_bar.set_ylabel("Explained / ceiling")
ax_bar.margins(x=0.01)
handles = [plt.Rectangle((0,0),1,1,color=layer_colors[l]) for l in layers]
ax_bar.legend(handles, layers, title="Winning layer",
              bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()

for ext in ("png", "pdf"):
    fig.savefig(FIG_DIR / f"{prefix}_encoding_summary.{ext}", dpi=300)
LOGGER.info("Saved → %s", FIG_DIR / f"{prefix}_encoding_summary.png")
