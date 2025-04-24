#!/usr/bin/env python
"""
extract_features.py
Global-average-pooled ResNet-50 features for every stimulus JPEG/PNG.
Outputs one .npy per layer (float32) + stimulus_index.csv.
"""
from pathlib import Path
from typing import List

from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import models
from tqdm import tqdm

from config import FEAT_DIR, LOGGER, get_all_scene_files

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS  = ["layer1", "layer2", "layer3", "layer4"]     # conv2_x – conv5_x

def main() -> None:
    scene_files: List[Path] = get_all_scene_files()
    LOGGER.info("Found %d stimulus images", len(scene_files))

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    resnet  = models.resnet50(weights=weights).to(DEVICE).eval()

    # ─── register forward hooks ───
    activations = {ln: [] for ln in LAYERS}

    def _capture(name: str):
        """
        Hook that global-avg-pools H×W so each output → (B, C),
        keeping only the *channel-wise* activation pattern.
        """
        def hook(_, __, out):
            pooled = out.mean(dim=(2, 3))        # (B, C, H, W) → (B, C)
            activations[name].append(pooled.cpu().to(torch.float32))
        return hook

    for ln in LAYERS:
        getattr(resnet, ln).register_forward_hook(_capture(ln))

    preprocess = weights.transforms()

    with torch.inference_mode():
        for img_path in tqdm(scene_files, desc="ResNet-50"):
            img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
            resnet(img)          # hooks fill `activations`

    # ─── save ───
    FEAT_DIR.mkdir(exist_ok=True)
    for ln, acts in activations.items():
        arr = torch.cat(acts, dim=0).numpy()          # (n_img, channels)
        np.save(FEAT_DIR / f"resnet50_{ln}.npy", arr, allow_pickle=False)
        LOGGER.info("%s: saved %s", ln, arr.shape)

    pd.DataFrame({
        "stimulus_id": [p.name for p in scene_files],
        "filepath"   : [str(p)  for p in scene_files],
    }).to_csv(FEAT_DIR / "stimulus_index.csv", index=False)

if __name__ == "__main__":
    main()
