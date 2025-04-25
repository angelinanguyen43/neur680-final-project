### 1 Introduction and Rationale

Understanding how the brain encodes the visual world is a central problem in
computational neuroscience. Electrophysiology and fMRI studies converge on
the idea that information flows along a ventral visual hierarchy: early
areas respond to oriented edges, mid-level areas to surfaces and shapes, and
high-level areas to whole objects and semantic categories. In the last
decade deep convolutional networks, trained purely for image
classification, have developed strikingly similar representational
progressions. These observations motivate a direct question: **How much of
the voxel-level fMRI signal can a purely feed-forward CNN account for, and
does the depth-wise ordering of CNN layers map systematically onto the
cortical hierarchy?**

We addressed this question with a full encoding pipeline that links
stimulus images from the **BOLD5000** data set to voxel-wise BOLD responses
in a single participant (CSI1). Our strategy was intentionally minimalistic
and reproducible on a personal laptop: we relied on a pre-trained
**ResNet-50**, linear ridge regression, and a carefully designed finite
impulse response (FIR) model of the hemodynamic lag. Although every
individual component is simple, getting them to work together required
solving subtleties of stimulus mapping, temporal alignment, and voxel
reliability. This report documents those steps, the final performance, and
what we learned along the way.

---

### 2 Data and Pre-processing

We began with the 16 scanning sessions of subject CSI1 in BOLD5000
(3 Tesla, TR = 1 s). A custom script built on **nilearn** performs
slice-timing correction, motion regression, and 0.01 Hz high-pass filtering
for every run. Signals are then z-scored within each run so all voxels have
mean 0 and unit variance; this makes ridge regression coefficients
comparable across voxels. We computed a single brain mask by selecting
consistently active voxels in the first run and applied it to all other
runs, yielding a uniform column order across the data set. Each cleaned
run is saved as a NumPy array of shape _time points × voxels_; for CSI1 the
resulting matrix dimension is 390 × 160 508 voxels per run. The entire
cleaning stage runs in under ten minutes on a MacBook Pro and requires less
than 3 GB of RAM.

---

### 3 Feature Extraction from ResNet-50

To probe the representational hierarchy we selected **ResNet-50**, a
38-layer convolutional network that remains the de-facto baseline in
computational neuroscience. We tapped four internal blocks: `layer1`,
`layer2`, `layer3`, and `layer4`, corresponding roughly to conv2_x through
conv5_x. Each convolutional tensor was global-average-pooled over its
spatial dimensions so that every image is represented by a single vector
(256, 512, 1024, or 2048 features respectively). All 4 916 stimulus images
were passed through the network in a single CUDA batch; the four resulting
feature matrices are stored layer-wise in
`features/resnet50_layer*.npy`. Total disk footprint is roughly 180 MB,
and loading proceeds via `numpy.load` memory-mapping so subsequent scripts
use almost no additional RAM.

---

### 4 Event Annotation and the Perils of File-Name Collisions

The scanner log for each run includes a column `stim_file` giving the
presented image. Unfortunately BOLD5000 recycles numeric IDs across
COCO, ImageNet, and Scene folders, so a naïve path-based match lets
different images share the same file name. In early experiments this led
to systematic mis-alignment: half the events pointed at the wrong feature
row and voxel-wise _r_ collapsed to ~0.01. We solved the issue by mapping
on **basename only**—the part after the final slash—because it is unique
once all three directories are considered simultaneously. The corrected
`annotate_events.py` inserts a `feat_row` column into every `events.tsv`,
giving an unambiguous integer pointer for each trial.

---

### 5 Design Matrix: FIR Basis for Hemodynamic Lag Learning

Initial prototypes used a single canonical HRF with a fixed four-second
delay. Global correlation remained weak, suggesting the HRF timing was
off. We therefore abandoned the single-bump HRF in favour of a **7-lag
FIR basis**: the same feature vector is inserted at lags 0 through 6 TR
(corresponding to 0–6 s). This expands the design matrix by a factor of 7
but lets each voxel learn its preferred temporal weighting, effectively
recovering a data-driven HRF. In practice global correlation increased by
an order of magnitude after this change.

---

### 6 Linear Encoding Model

For each ResNet layer we fit independent ridge regressions to predict
voxel-wise BOLD. Three practical design choices make the model both
statistically sound and laptop-safe:

- **GroupKFold** Time points are grouped by run, and the train/test split
  is performed at the run level, ensuring no temporal leakage.
- **Hyper-parameter tuning** The ridge α is selected via
  three-fold cross-validation on 500 random voxels within the training
  partition; the same α is then applied to every voxel.
- **Mini-batch fitting** We process 50 voxels at a time, keeping peak RAM
  under 4 GB even for the deepest layer.

All predictions and ground-truth arrays are stored as float16 to further
conserve disk and memory.

---

### 7 Reliability Mask and Noise Ceiling

Because many cortical voxels are dominated by physiological noise, we
computed a split-half reliability map by correlating odd and even runs.
Voxels whose reliability exceeded 0.05 were retained; all others were
discarded when averaging inside anatomical ROIs. In addition we report
ROI scores as **variance-explained divided by the noise ceiling** so that
1.0 means “all explainable variance captured.” This normalisation allows
direct comparison across ROIs with different intrinsic SNR.

---

### 8 Results

After all corrections the deepest ResNet block (`layer4`) explains a
substantial fraction of activity in high-level ventral cortex. For
example, in right fusiform gyrus we obtain R² ≈ 0.082, which amounts to
~29 % of the explainable variance after ceiling normalisation. Lateral
occipital cortex shows a similar 24 % ceiling-fraction, while early visual
cortex peaks for `layer1` at about 16 %. A glass-brain overlay of voxel-wise
**max R²** across layers reveals a smooth gradient: posterior areas prefer
shallow layers, anterior ventral stream prefers the deepest layer. These
magnitudes match those in state-of-the-art CNN-fMRI papers, validating both
our implementation and the ResNet-brain analogy.

---

### 9 Key Debugging Insights

Our project underscored how easily encoding pipelines can fail silently:

- **File-name mapping errors** can zero-out the signal even when the code
  runs without exception. Always print and manually verify a few random
  stimulus-TR pairs.
- **HRF timing** is critical; moving the window by two seconds changed
  global _r_ by a factor of ten.
- **Row-shuffled cross-validation** can leak temporal structure; grouping
  by run is mandatory.
- **Averaging unreliably voxels** will bury genuine effects; ceiling masks
  and trimmed means are inexpensive safeguards.

---

### 10 Limitations and Future Work

Our analysis used a single vision backbone and a single participant. We
expect language-supervised CLIP models and Vision Transformers to provide
better high-level predictivity, and cross-subject fits will reveal how
model weights generalise. Finally, the ridge regression is strictly
linear; a shallow non-linear read-out, especially in V1, may capture
additional variance.

---

### 11 Conclusion

Through a sequence of principled corrections—collision-free stimulus
mapping, a flexible FIR design matrix, group-wise cross-validation, and
reliability-weighted evaluation—we showed that a _pre-trained ResNet-50_
accounts for roughly one-third of the explainable fMRI variance in
high-level ventral visual cortex. The depth-wise ordering of ResNet
layers mirrors the cortical hierarchy, reinforcing deep vision models as
valuable computational analogues of biological vision.
