#!/usr/bin/env python
"""
visualize_results.py
Create glass-brain + ROI bar-plot summary for one subject.
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

# ── project paths & logger ──────────────────────────────────────────────
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
p.add_argument("--metric", default="r", choices=["r", "r2"],
               help="Which map to plot (default=r)")
p.add_argument("--threshold", type=float, default=0.10,
               help="|value| threshold for overlay (default=0.10)")
p.add_argument("--surface", action="store_true",
               help="Also draw inflated cortical surface")
args = p.parse_args()
subj_prefix = f"sub-{args.subj}"

# ── layer list ──────────────────────────────────────────────────────────
if args.layers:
    layers: List[str] = args.layers
else:
    layers = sorted({p.name.split("_")[1]
                     for p in RESULT_DIR.glob(f"{subj_prefix}_*_*_{args.metric}.nii.gz")})
    LOGGER.info("Auto-detected layers: %s", ", ".join(layers))
if not layers:
    raise RuntimeError("No layers found – did you run evaluate.py?")

# ── Harvard–Oxford atlas ────────────────────────────────────────────────
atlas         = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
atlas_img_raw = image.load_img(atlas.maps)

# --- NEW: resample atlas into native space of the metric image -------------
metric_path   = RESULT_DIR / f"{subj_prefix}_{layers[-1]}_{args.metric}.nii.gz"
metric_img    = image.load_img(metric_path)
atlas_img     = resample_to_img(atlas_img_raw, metric_img, interpolation="nearest")
# ----------------------------------------------------------------------------

atlas_data    = atlas_img.get_fdata().astype("int16")
atlas_labels  = atlas.labels
roi_ids       = np.arange(1, len(atlas_labels))         # 0 = background

# ── figure layout ───────────────────────────────────────────────────────
n_rows = 2 if args.surface else 1
fig = plt.figure(figsize=(11, 3.5 * n_rows + 0.6 * len(layers)))
gs  = fig.add_gridspec(n_rows, 2, height_ratios=[3] + ([3] if args.surface else []))

# ── brain overlay (deepest layer) ───────────────────────────────────────
metric_path = RESULT_DIR / f"{subj_prefix}_{layers[-1]}_{args.metric}.nii.gz"
metric_img  = image.load_img(metric_path)

ax_brain = fig.add_subplot(gs[0, 0])
plotting.plot_glass_brain(metric_img, threshold=args.threshold,
                          display_mode="lyrz", colorbar=True, axes=ax_brain,
                          title=f"{args.metric.upper()} map  ({layers[-1]})")

# ── optional inflated surface ───────────────────────────────────────────
if args.surface:
    fsavg   = datasets.fetch_surf_fsaverage()
    texture = surface.vol_to_surf(metric_img, fsavg.pial_left)
    ax_surf = fig.add_subplot(gs[0, 1], projection="3d")
    plotting.plot_surf_stat_map(fsavg.infl_left, texture, hemi="left",
                                threshold=args.threshold, bg_map=fsavg.sulc_left,
                                axes=ax_surf, colorbar=False)
else:
    ax_dummy = fig.add_subplot(gs[0, 1])
    ax_dummy.axis("off")

# ── ROI × layer dataframe ───────────────────────────────────────────────
records = []
for layer in layers:
    r2_img = image.load_img(RESULT_DIR / f"{subj_prefix}_{layer}_r2.nii.gz")
    for roi in roi_ids:
        # build binary mask for this ROI
        mask_img = image.new_img_like(atlas_img, (atlas_data == roi).astype("uint8"))
        vals     = masking.apply_mask(r2_img, mask_img)
        if vals.size:
            records.append({"Layer": layer,
                            "ROI":   atlas_labels[roi],
                            "Mean_R2": float(np.nanmean(vals))})
    gc.collect()

df = pd.DataFrame(records)

# ── bar plot ────────────────────────────────────────────────────────────
ax_bar = fig.add_subplot(gs[-1, :])
bar_w  = 0.8 / len(layers)
for i, layer in enumerate(layers):
    sub_df = df[df["Layer"] == layer]
    ax_bar.bar(np.arange(len(sub_df)) + i*bar_w, sub_df["Mean_R2"],
               width=bar_w, label=layer, alpha=0.9)

ax_bar.set_xticks(np.arange(len(sub_df)) + bar_w*(len(layers)-1)/2)
ax_bar.set_xticklabels(sub_df["ROI"], rotation=90)
ax_bar.set_ylabel("Mean R²")
ax_bar.margins(x=0.01)
ax_bar.legend(title="Layer", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()

# ── save ────────────────────────────────────────────────────────────────
out_dir = RESULT_DIR / "figures"; out_dir.mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(out_dir / f"{subj_prefix}_encoding_summary.{ext}", dpi=300)
LOGGER.info("Saved figure → %s", out_dir / f"{subj_prefix}_encoding_summary.png")
