#!/usr/bin/env python
"""
visualize_results.py   –  ‘best-layer’ view
Shows (1) a glass-brain of the voxel-wise *max* R² across layers and
(2) a bar chart of the best-explained ROIs colour-coded by the winning
layer.

Usage
-----
python scripts/visualize_results.py --subj CSI1 --metric r2 --surface
# optional:
#   --top-n 20         show 20 best ROIs (default 15)
#   --layers layer1 layer4   restrict to a subset of layers
"""
from __future__ import annotations
import argparse, logging, gc
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nilearn import image, masking, plotting, datasets, surface
from nilearn.image import resample_to_img
from matplotlib import cm




# ── paths & logger ──────────────────────────────────────────────────────
try:
    from config import RESULT_DIR, DERIV, LOGGER
except ModuleNotFoundError:
    RESULT_DIR = Path(__file__).resolve().parents[1] / "results"
    DERIV      = Path(__file__).resolve().parents[1] / "data"
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    LOGGER = logging.getLogger("visualize")

# ── CLI ─────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--subj", required=True, help="e.g. CSI1 (without 'sub-')")
p.add_argument("--layers", nargs="*",
               help="Layers to include; default = all layers found")
p.add_argument("--metric", default="r2", choices=["r", "r2"],
               help="Which map to summarise (default=r2)")
p.add_argument("--threshold", type=float, default=0.05,
               help="Display threshold for glass-brain (default=0.05)")
p.add_argument("--surface", action="store_true",
               help="Also draw inflated cortical surface")
p.add_argument("--top-n", type=int, default=15,
               help="Show this many best ROIs in the bar chart (default=15)")
args = p.parse_args()
subj_prefix = f"sub-{args.subj}"

# ── layer list ──────────────────────────────────────────────────────────
if args.layers:
    layers: List[str] = args.layers
else:
    pattern = RESULT_DIR.glob(f"{subj_prefix}_*_*_{args.metric}.nii.gz")
    layers = sorted({
        "_".join(Path(p).stem.split("_")[1:-1])   # strip subj & metric
        for p in pattern
    })
    LOGGER.info("Auto-detected layers: %s", ", ".join(layers))
if not layers:
    raise RuntimeError("No layers found – did you run evaluate.py?")

# ── atlas & resampling ──────────────────────────────────────────────────
atlas          = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
atlas_img_raw  = image.load_img(atlas.maps)
metric_path    = RESULT_DIR / f"{subj_prefix}_{layers[-1]}_{args.metric}.nii.gz"
metric_img_ref = image.load_img(metric_path)
atlas_img      = resample_to_img(atlas_img_raw, metric_img_ref,
                                 interpolation="nearest")
atlas_data     = atlas_img.get_fdata().astype("int16")
atlas_labels   = atlas.labels
roi_ids        = np.arange(1, len(atlas_labels))

# ── load maps & compute voxel-wise max ──────────────────────────────────
maps, stack = {}, []
for layer in layers:
    img = image.load_img(RESULT_DIR / f"{subj_prefix}_{layer}_{args.metric}.nii.gz")
    maps[layer] = img
    stack.append(img.get_fdata(dtype="float32"))
stack = np.stack(stack, axis=-1)                     # (X, Y, Z, L)
max_data = stack.max(-1)
best_idx = stack.argmax(-1)                          # which layer won per voxel
best_img = image.new_img_like(metric_img_ref, max_data.astype("float32"))

# ── ROI table: best layer per region ────────────────────────────────────
records = []
for roi in roi_ids:
    roi_mask = (atlas_data == roi)
    if roi_mask.sum() == 0:
        continue
    reliab   = np.load(RESULT_DIR / f"{subj_prefix}_reliability_mask.npy")
    roi_vals = stack[roi_mask & (reliab.astype(bool))]
    if roi_vals.size == 0:
        continue
    mean_by_layer = roi_vals.mean(0)
    best_layer_idx = int(mean_by_layer.argmax())
    records.append({
        "ROI":        atlas_labels[roi],
        "BestLayer":  layers[best_layer_idx],
        "MeanR2":     float(mean_by_layer[best_layer_idx])
    })
    gc.collect()

df = (pd.DataFrame(records)
        .sort_values("MeanR2", ascending=False)
        .head(args.top_n))

# colour palette: one colour per layer
cmap = cm.get_cmap("tab10")
layer_colors = {l: cmap(i % 10) for i, l in enumerate(layers)}

# ── figure layout ───────────────────────────────────────────────────────
n_rows = 2 if args.surface else 1
fig = plt.figure(figsize=(11, 3.5 * n_rows + 0.5 * len(df)))
gs  = fig.add_gridspec(n_rows, 2,
                       height_ratios=[3] + ([3] if args.surface else []))

# ── (A) Glass-brain of voxel-wise max ───────────────────────────────────
ax_brain = fig.add_subplot(gs[0, 0])
plotting.plot_glass_brain(best_img, threshold=args.threshold,
                          display_mode="lyrz", colorbar=True, axes=ax_brain,
                          title=f"Voxel-wise best {args.metric.upper()} (max over layers)")

# optionally surface plot
if args.surface:
    fsavg   = datasets.fetch_surf_fsaverage()
    texture = surface.vol_to_surf(best_img, fsavg.pial_left)
    ax_surf = fig.add_subplot(gs[0, 1], projection="3d")
    plotting.plot_surf_stat_map(fsavg.infl_left, texture, hemi="left",
                                threshold=args.threshold,
                                bg_map=fsavg.sulc_left,
                                axes=ax_surf, colorbar=False)
else:
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")

# ── (B) ROI bar chart of best layer ─────────────────────────────────────
ax_bar = fig.add_subplot(gs[-1, :])
bars = ax_bar.bar(np.arange(len(df)), df["MeanR2"],
                  color=[layer_colors[l] for l in df["BestLayer"]],
                  alpha=0.9)
ax_bar.set_xticks(np.arange(len(df)))
ax_bar.set_xticklabels(df["ROI"], rotation=90)
ax_bar.set_ylabel(f"Mean {args.metric.upper()}  (best layer)")
ax_bar.margins(x=0.01)
# legend for layers
handles = [plt.Rectangle((0,0),1,1,color=layer_colors[l]) for l in layers]
ax_bar.legend(handles, layers, title="Winning layer",
              bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()

# ── save ────────────────────────────────────────────────────────────────
out_dir = RESULT_DIR / "figures"; out_dir.mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(out_dir / f"{subj_prefix}_encoding_summary.{ext}", dpi=300)
LOGGER.info("Saved figure → %s", out_dir / f"{subj_prefix}_encoding_summary.png")
