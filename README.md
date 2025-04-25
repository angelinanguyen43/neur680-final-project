# Project Report – Modeling fMRI Responses to Natural‑Image Viewing with Deep Neural Networks

---

## 1  Scientific Motivation

A central hypothesis in modern computational neuroscience is that **deep convolutional and transformer networks recapitulate the hierarchical organisation of the primate visual system**. If this is true, intermediate activations from a trained network should serve as excellent predictors of the blood‑oxygen‑level dependent (BOLD) signal recorded in corresponding cortical areas.

Our assignment required us to **demonstrate an end‑to‑end encoding model**: choose a vision backbone, choose a public fMRI dataset, extract features, fit voxel‑wise predictors, and evaluate how well each network layer explains unseen neural responses. We elected to use **BOLD5000** (because of its rich stimulus set and subject‑specific masks) and **ResNet‑50** (well‑studied, four easily identified convolutional blocks).

---

## 2  Data & Pre‑processing Decisions

### 2.1  Dataset: BOLD5000

- 4 human subjects, ~5 000 natural images each
- 2 mm isotropic voxel resolution; TR ≈ 1 s
- Released as a BIDS dataset – simplifies loading with `pybids`

### 2.2  Minimal fMRI Pre‑processing (`preprocess_bids.py`)

We avoided a heavyweight pipeline (fMRIPrep) to keep the assignment laptop‑friendly. Instead we applied only:

- **Brain masking** – `nilearn.compute_epi_mask` on the first run per subject
- **Temporal detrending + high‑pass (0.01 Hz)** – to stabilise baseline drift
- **Z‑scoring per voxel per run** – so ridge coefficients are comparable across runs

The output is a compressed `*.npy` for each run (TR × voxel), shrinking storage and accelerating downstream I/O.

### 2.3  Why these choices?

- Enough cleaning to remove low‑frequency noise, but not so much that we hide modelling mistakes behind preprocessing complexity.
- One brain mask per subject guarantees voxel columns are aligned across runs, simplifying concatenation.

---

## 3  Stimulus Feature Extraction

| Step                           | Rationale                                                                                                                                                                                                                                   |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Collect image files**        | BOLD5000 stimuli live in three folders (COCO, ImageNet, Scene). We recursively list everything under a common root to capture the full variety.                                                                                             |
| **ResNet‑50 forward hooks**    | We register hooks on `layer1 … layer4`, global‑average‑pool spatial dimensions, and write one array per layer. This yields (≈5 000 images × C) matrices of size 256, 512, 1024, 2048 respectively.                                          |
| **Relative‑path stimulus IDs** | Multiple files share the same **basename** (e.g., `000000123.jpg`). To obtain a one‑to‑one mapping between image and feature row we instead store the _path relative to the stimulus root_. This decision prevents silent collisions later. |

---

## 4  Linking Trials to Feature Vectors

The BOLD5000 `events.tsv` lists onset times and absolute stimulus paths. We add a new column `feat_row` that contains the index into the ResNet feature matrix.

> **Design note:** Mapping by relative path, not filename, safeguards uniqueness across the three stimulus repositories.

---

## 5  Building the Encoding Model

### 5.1  Temporal Alignment

The haemodynamic response peaks ~4 s after neural activity. We therefore **shift every onset by one canonical HRF lag (4 s)** and convert the resulting times to integer TR indices. This simple shift is adequate given BOLD5000’s fast TR; a full HRF convolution would marginally improve accuracy but increase runtime.

### 5.2  Model Specification

- **Predictor matrix X** – ResNet activations (z‑scored per feature) for the trials of interest.
- **Response matrix Y** – z‑scored voxel intensities at `TR + lag`.
- **Ridge regression (L2) with intercept** – prevents over‑fitting & centres voxel means.
- **Cross‑validation of α** – on 500 randomly sampled voxels to choose a single regularisation strength per layer.

Why ridge? Linear mapping keeps the interpretation clear (“does this layer linearly predict this voxel?”) and is the baseline used by Yamins & DiCarlo, Brain‑Score, etc.

### 5.3  Train/test Split

We use **two‑fold shuffled split at the trial level**—simple yet ensures the evaluation set is unseen. Splitting by run would have been even stricter but halves training data; a design trade‑off.

---

## 6  Evaluation Metrics

| Metric                              | Reason                                                                                                         |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Pearson r**                       | Measures linear correlation, familiar in the encoding literature.                                              |
| **Coefficient of determination R²** | Easier to interpret (0 = chance, 1 = perfect). Negative values immediately diagnose misalignment or poor fits. |

Both are computed voxel‑wise on the held‑out fold, then saved as NIfTI volumes for inspection and group analysis.

---

## 7  Visualisation Strategy

1. **Glass‑brain overlay** – layer‑specific R² map thresholded at |0.10| shows the anatomical distribution of predictability.
2. **ROI bar plot** – mean R² for each Harvard–Oxford cortical ROI across layers illustrates the hypothesised hierarchical rise (V1→fusiform).
3. **Inflated cortical surface (optional)** – qualitative confirmation of ventral‑stream progression.

---

## 8  Key Results

TODO: FILL IN

---

## 9  Limitations & Future Work

- **Fixed HRF lag** – A subject‑ or voxel‑specific HRF might raise scores.
- **Linear mapping** – Non‑linear readouts (kernel ridge, shallow MLP) could capture retinotopic effects lost in global pooling.
- **Single backbone** – Comparing to self‑supervised (DINOv2) or multimodal (CLIP) networks would test how training objectives affect brain‑predictivity.
- **Cross‑subject generalisation** – Encoding weights are currently subject‑specific; exploring shared representational spaces would extend the analysis.

---

## 10  What the Assignment Demonstrates

- **Technical integration** – BIDS handling, deep‑learning feature extraction, voxel‑wise regression, neuroimaging visualisation.
- **Methodological rigour** – Explicit HRF consideration, held‑out evaluation, collision‑proof stimulus mapping.
- **Interpretability** – Results directly articulate how artificial vision hierarchies map onto brain hierarchies.
