#!/usr/bin/env python
"""
Global paths & helpers for the BOLD5000-DNN encoding project.
Only edit ROOT if you ever move this folder.
"""
from pathlib import Path
import logging

# ───────── paths ──────────
ROOT = Path(__file__).resolve().parents[1]          # FINAL_PROJ/

RAW        = ROOT / "raw"  / "ds001499"            # BIDS dataset (DataLad)
STIM_BASE  = ROOT / "raw"  / "stimuli"
SCENE_ROOT = (STIM_BASE /
              "BOLD5000_Stimuli" /
              "Scene_Stimuli"   /
              "Original_Images")   # parent of COCO, ImageNet, Scene

DERIV      = ROOT / "data"                         # pre-proc NIfTI
FEAT_DIR   = ROOT / "features"                     # DNN features
MODEL_DIR  = ROOT / "models"                       # fitted weights
RESULT_DIR = ROOT / "results"                      # predictions & metrics

for p in (DERIV, FEAT_DIR, MODEL_DIR, RESULT_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ───────── logging ─────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt = "%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# ───────── helpers ─────────
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def get_all_scene_files():
    """Return **sorted** list of scene-stimulus image paths that *exist*."""
    if not SCENE_ROOT.exists():
        raise FileNotFoundError(f"Scene root {SCENE_ROOT} not found")
    return sorted(
        p for p in SCENE_ROOT.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )
