#!/usr/bin/env python
"""
extract_features.py – ResNet-50 global-avg-pooled features for every image.

Produces four files:
    features/resnet50_layer{1-4}.npy   (float32, n_img × C)
and   features/stimulus_index.csv
"""
from __future__ import annotations
from pathlib import Path
from typing import List
from tqdm import tqdm
from PIL import Image
import numpy as np, pandas as pd, torch
from torchvision import models

from config import FEAT_DIR, LOGGER, get_scene_files, SCENE_ROOT

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS  = ["layer1", "layer2", "layer3", "layer4"]


def main() -> None:
    scene_files: List[Path] = get_scene_files()
    LOGGER.info("Found %d stimulus images", len(scene_files))

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    resnet  = models.resnet50(weights=weights).to(DEVICE).eval()

    activations = {ln: [] for ln in LAYERS}

    def _capture(name: str):
        def hook(_, __, out):
            pooled = out.mean(dim=(2, 3))         # global-avg-pool
            activations[name].append(pooled.cpu().to(torch.float32))
        return hook

    for ln in LAYERS:
        getattr(resnet, ln).register_forward_hook(_capture(ln))

    preprocess = weights.transforms()

    with torch.inference_mode():
        for p in tqdm(scene_files, desc="ResNet-50"):
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)
            resnet(img)           # hooks fill `activations`

    FEAT_DIR.mkdir(exist_ok=True)
    for ln in LAYERS:
        arr = torch.cat(activations[ln], dim=0).numpy()
        np.save(FEAT_DIR / f"resnet50_{ln}.npy", arr, allow_pickle=False)
        LOGGER.info("%s: saved %s", ln, arr.shape)

    pd.DataFrame({
        "stimulus_id": [p.relative_to(SCENE_ROOT).as_posix() for p in scene_files],
        "filepath"   : [str(p) for p in scene_files],
    }).to_csv(FEAT_DIR / "stimulus_index.csv", index=False)
    LOGGER.info("stimulus_index.csv written")


if __name__ == "__main__":
    main()
