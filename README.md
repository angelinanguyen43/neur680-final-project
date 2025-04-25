## Overview

We set out to test **how well a modern computer-vision network can
predict single-voxel fMRI responses to thousands of real-world
pictures**. Using the BOLD5000 data set (≈5 000 images, 16 scans per
subject) we built a full **encoding pipeline**:

1. **Clean** every BOLD run into a common voxel space.
2. **Extract** hierarchical features from a pre-trained ResNet-50.
3. **Align** each stimulus TR with its feature row—including a
   voxel-specific hemodynamic lag learned from the data.
4. **Fit** voxel-wise ridge regressions with train/test runs kept
   separate.
5. **Evaluate & visualise** variance explained, masking out voxels whose
   own responses are not reliable.

After a series of debugging steps the deepest ResNet block explained
**~0.08 raw R² (≈30 % of the explainable variance)** in fusiform gyrus
and lateral occipital cortex—values fully consistent with the published
literature but obtained on a laptop-friendly code base that first-year
students can read end-to-end.

---

## Motivation (Why)

Visual neuroscience proposes a progression from simple edge filters in
V1 to category-selective units in ventral temporal cortex. Deep
convolutional networks, trained only to classify images, spontaneously
develop a similar hierarchy. Demonstrating that a CNN can linearly
predict fMRI activity at multiple cortical levels strengthens the link
between artificial and biological vision and provides a quantitative
benchmark for future models.

---

## Data and Pre-processing (How)

- **Dataset** BOLD5000 subject CSI1 (3 T, TR = 1 s, 5 000 distinct
  images).
- **Pre-processing** Slice-timing, high-pass (0.01 Hz), z-score per run;
  one EPI mask per subject; output is a NumPy matrix (_TR × voxel_) for
  each run.
- The entire cleaning script (`preprocess_bids.py`) runs in < 10 minutes
  and keeps everything in native space to avoid interpolation blur.

---

## Feature Extraction (What)

- **Network** ResNet-50 (ImageNet weights).
- **Layers tapped** `layer1`–`layer4` (after conv2_x … conv5_x).
- **Pooling** Global average over spatial dimensions → one vector per
  image per layer (256–2 048 D).
- **Storage** Float32 `.npy`; 4 916 images × 4 layers ≈ 180 MB total.

These features are frozen—they never see fMRI data—so the encoding model
tests their neuroscientific validity.

---

## Stimulus–Feature Alignment

Early runs produced near-zero R² because the same numeric ID may appear
in COCO, ImageNet, and Scene folders. We remapped every event file on
**basename only** (`airplanecabin1.jpg`, `COCO_train2014_…`) guaranteeing
a collision-free one-to-one link from TR to feature row. This single fix
raised global correlation from ≈0.001 to ≈0.02.

---

## Design Matrix and Temporal Modeling

A fixed HRF is fragile. We replaced it with a **finite-impulse-response
basis** covering lags 0…6 s (seven TRs). Each image’s feature vector is
inserted at all seven lags, letting the ridge learn its own lag and
shape. This boosted correlation a further order of magnitude.

---

## Encoding Model Details

- **Ridge regression** with α from 3-fold cross-validation on 500 random
  voxels.
- **GroupKFold** ensures training and test data come from _different
  runs_, eliminating temporal leakage.
- Mini-batch fitting (50 voxels) and float16 predictions keep RAM under
  4 GB.

---

## Reliability Mask & Ceiling

Split-half reliability (odd vs. even runs) defines a noise ceiling per
voxel. We keep voxels with ceiling > 0.05 and express every ROI bar as
R² / ceiling, so 1.0 means “we explained everything that was explainable”.

---

## Results (Subject CSI1, Held-out Runs)

- The glass-brain overlay of **voxel-wise maximum R² across layers**
  reveals a clear V1 → IT gradient: early visual cortex peaks for
  `layer1`, high-level ventral stream peaks for `layer4`.
- In fusiform gyrus, `layer4` explains **0.082 R² = 29 % of the ceiling**;
  lateral occipital cortex reaches 24 %.
- Occipital pole peaks for `layer1`, consistent with edge-like features
  in that block.

These magnitudes replicate those in published CNN-fMRI papers, obtained
here with an undergraduate-scale code base.

---

## Why Performance Was Initially Poor

1. **File-name collision** between stimulus folders mis-aligned half the
   trials.
2. **HRF mis-timing** (fixed lag of 4 s) sampled the down-slope of the
   BOLD peak.
3. **Row-shuffled K-fold CV** leaked temporal information.
4. **ROI means** across unreliable voxels diluted signal.

Each was fixed in turn, yielding the final numbers.

---

## Open Issues and Future Work

- Swap in CLIP or ViT backbones; prior work suggests ×2 improvement in
  high-level cortex.
- Concatenate multiple layers into a single design; layer synergy often
  captures another few percent of variance.
- Extend to the remaining subjects and test cross-subject transfer.
- Try kernel ridge or a shallow MLP for potential non-linear gains,
  especially in V1/V2.

---

### Conclusion

With correct stimulus alignment, an FIR design that lets the model learn
its own hemodynamic delay, and a reliability-aware evaluation metric,
**pre-trained ResNet-50 features explain roughly one-third of the
explainable fMRI variance in high-level human visual cortex**. The
layer-wise topography recapitulates the known cortical hierarchy,
reinforcing deep networks as a compelling computational model of visual
processing.
