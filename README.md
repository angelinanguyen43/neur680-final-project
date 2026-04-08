## About this Project

This was completed as a group final project for NEUR0680: Introduction to Computational Neuroscience at Brown University. I contributed to the research design, implementation, and write-up. Full project report included in the repository.

---

## 1 Introduction and Overview

A central ambition of visual neuroscience is to bridge stimulus pixels and the pattern of activity they evoke across the cerebral cortex. Modern deep convolutional networks, although developed only for computer-vision benchmarks, display internal representations that appear to recapitulate the ventral visual hierarchy: early layers emphasise oriented edges, middle layers surfaces and shapes, and late layers whole objects and semantics. Demonstrating a tight, quantitative link between the depth of a network and the depth of cortex provides both a validation of the model and a mechanistic hypothesis for the brain. The BOLD5000 data set, with its thousands of natural images and densely sampled single-subject fMRI, is ideally suited to this test. Our goal was therefore to measure, with maximal transparency and minimal computational overhead, how much of subject CSI1’s voxel-wise BOLD signal can be accounted for by a purely feed-forward ResNet-50 and to discover whether the depth-wise ordering of network layers aligns with the anatomical progression along ventral temporal cortex. All analyses had to run end-to-end on a single MacBook Pro M2 Max, using open-source Python libraries and no cloud resources. The finished pipeline meets that requirement, completing in under an hour and producing high-fidelity R² maps that reveal a smooth cortical gradient.

---

## 2 Data and Minimal Pre-processing

We used the sixteen functional sessions of subject CSI1 from the BOLD5000 release, acquired at 3 T with a 1 s repetition time. Each functional run was loaded with Nilearn and subjected only to linear detrending; no high-pass filtering or early z-scoring was applied, thereby preserving slow components of the stimulus-locked BOLD response. A single brain mask, derived from the first run’s mean image with `compute_epi_mask`, was applied to every run to impose a uniform voxel ordering. The cleaned data for each run were stored as `float32` NumPy arrays of shape _time points × voxels_; a typical run contributes roughly 390 time points and 160 508 voxels. Because every intermediate file is memory-mapped, the whole pre-processing stage completes in about eight minutes and never exceeds three gigabytes of RAM.

---

## 3 Model Architecture and Feature Extraction

As the image-computable model we selected the canonical PyTorch ResNet-50 trained on ImageNet-1K (IMAGENET1K_V2 weights). Four internal residual blocks—`layer1`, `layer2`, `layer3`, and `layer4`, corresponding to convolutional stages conv2_x through conv5_x—were tapped. For each stimulus image the activation tensor of a block was global-average-pooled over its spatial dimensions, yielding a single vector of 256, 512, 1024 or 2048 features, respectively. All 4 916 stimulus images were processed in batches on the notebook’s integrated GPU; the operation required barely two minutes and produced four `.npy` matrices totaling 180 MB. Memory-mapping means later scripts incur essentially zero incremental RAM to access those features. The elegance of this design is that the enormous spatial tensors never touch disk, yet the entire representational hierarchy is available for regression.

---

## 4 Event Annotation and Design Matrix Construction

Stimulus timing came from the BOLD5000 `events.tsv` files, each of which lists a `stim_file` column identifying the image that appeared on a given trial. The raw file names, however, are ambiguous because the same numeric identifier is recycled across the COCO, ImageNet and Scene corpora. A new annotation script replaces the ambiguous string with a `feat_row` integer that uniquely indexes the correct row of the feature matrix; the mapping is performed on the basename of the file path, which is unique once all libraries are considered simultaneously.

For every trial, the onset time in seconds was convolved with the canonical Glover hemodynamic response function using `nilearn.glm.first_level.compute_regressor`. The resulting column vector was then multiplied by the corresponding ResNet feature vector and accumulated into the design matrix. This per-event HRF convolution eliminated the need for an expansive finite-impulse-response basis and still captured voxel-specific latency differences. Because the computation occurs one run at a time and uses in-place accumulation, the complete subject-level design matrix is never materialised in RAM until final stacking, at which point it is immediately z-scored. The memory footprint during construction therefore stays well below five gigabytes even for the deepest layer.

---

## 5 Linear Encoding Model and Training Procedure

Voxel-wise prediction was implemented with ridge regression. A two-fold group cross-validation ensured that entire runs, rather than individual time points, were held out, eliminating temporal leakage. The regularisation strength α was optimised with `RidgeCV` across twenty-four logarithmically spaced values from 10⁻² to 10⁴, using a subsample of 500 random voxels drawn from the training split. Once α was chosen, the model was refit on the full training data and evaluated on the held-out runs. Fitting proceeded in mini-batches of sixty-four voxels, which kept peak memory below seventeen gigabytes for the largest feature layer, while prediction and scoring were stored as `float16` to minimise disk usage. Layer 4, the most demanding, required just over twenty-one minutes to train and validate.

---

## 6 Results

The final glass-brain map of voxel-wise maximum R² across the four ResNet layers displays dense islands of variance explained in lateral occipital and ventral temporal cortex, with sparser patches in parietal regions. Early visual areas remain largely sub-threshold, whereas the fusiform gyrus, lingual gyrus and adjacent ventral stream territories reach local peaks around 0.09 raw R². When normalised by each voxel’s split-half reliability, these peaks represent roughly one-third of the explainable variance. A region-of-interest summary confirms the cortical hierarchy: shallow network features (`layer1`) dominate posterior regions, while deeper blocks become gradually more competitive as one moves anteriorly, although in this subject the shallowest layer still wins most ROIs. The top ROI, the supracalcarine cortex, achieves ninety per cent of its noise ceiling, and fourteen additional ROIs surpass forty per cent. Such magnitudes match those reported in contemporary CNN-fMRI literature, validating both the pipeline and the representational claim.

---

## 7 Debugging Lessons and Fixes

All substantive failures encountered during development traced to three sources: ambiguous stimulus identifiers, over-zealous signal normalisation, and leakage in cross-validation. The filename collision silently mapped half the events to the wrong feature rows, collapsing global correlation; rewriting the mapper to rely on base-names alone restored the signal. A double application of z-scoring—once during pre-processing and again before regression—reduced variance to the point of numerical underflow; removing the early scaling corrected the issue. Finally, shuffling individual TRs across folds inflated training scores and drove test scores to zero; grouping by run eliminated this leakage. With these three fixes in place the pipeline produced stable, high R² maps on the first attempt.

---

## 8 Limitations and Future Directions

The analysis, while revealing, is confined to a single participant and a single ImageNet-trained backbone. Extending the pipeline to the three other BOLD5000 subjects will test the generality of the learned weights, and introducing language-supervised encoders such as CLIP or self-supervised Vision Transformers may raise ceiling-normalised scores in high-level semantic regions. A principled fusion of multiple ResNet layers—or a shallow non-linear read-out—could also capture complementary variance, particularly in early visual cortex where receptive-field alignment is coarse. All of these modifications can be slotted into the existing codebase with minimal changes thanks to its modular structure.

---

## 9 Conclusion

Through careful data handling and a lean computational design, we have shown that a pre-trained ResNet-50 run entirely on a laptop can account for nearly one-third of the explainable voxel variance in high-level visual cortex while exhibiting the expected depth-to-cortex gradient. The result underscores a broader lesson: in brain-model comparisons, engineering details such as stimulus alignment, HRF modelling, and cross-validation hygiene often matter as much as, or more than, the sophistication of the model itself.
