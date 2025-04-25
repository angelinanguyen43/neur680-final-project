#!/usr/bin/env python
"""
config.py – Global paths, logging, and helpers
Adjusted for FINAL_PROJ/ 2025-04-25
"""
from __future__ import annotations
from pathlib import Path
import logging, multiprocessing as mp

# ────────── root folder (one level *above* scripts/) ──────────
ROOT = Path(__file__).resolve().parents[1]     # FINAL_PROJ/

RAW        = ROOT / "raw"   / "ds001499"
STIM_BASE  = ROOT / "raw"   / "stimuli"
SCENE_ROOT = (STIM_BASE / "BOLD5000_Stimuli"
                         / "Scene_Stimuli"
                         / "Original_Images")

DERIV      = ROOT / "data"
FEAT_DIR   = ROOT / "features"
RESULT_DIR = ROOT / "results"
FIG_DIR    = RESULT_DIR / "figures"
MODEL_DIR  = ROOT / "models"          # (optional; not used by v2 pipeline)

for p in (DERIV, FEAT_DIR, RESULT_DIR, FIG_DIR):
    p.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("bold-dnn")

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
N_CORES    = max(1, mp.cpu_count() - 2)
RAM_LIMIT  = 8 * 2**30    # 8 GiB


def get_scene_files():
    """Sorted list of stimulus image paths that actually exist."""
    if not SCENE_ROOT.exists():
        raise FileNotFoundError(f"{SCENE_ROOT} not found")
    return sorted(
        p for p in SCENE_ROOT.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )
