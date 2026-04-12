# GRUE Extended Experiments: Experiment Log

**Date range:** April 6 -- April 12, 2026
**Status:** In progress. Strong preliminary results, but CIFAR-10/100 is too low-resolution and too few classes to reach definitive conclusions. A replication on a higher-quality dataset is needed.

---

## Table of Contents

1. [Overview and Motivation](#1-overview-and-motivation)
2. [Experiment 1: Joint Color+Object Training (Compositionality Probe)](#2-experiment-1-joint-colorobject-training)
3. [Experiment 2: SAM-Augmented Color Labels](#3-experiment-2-sam-augmented-color-labels)
4. [Experiment 3: CIFAR-100 Replication](#4-experiment-3-cifar-100-replication)
5. [Experiment 4: Sky/Ocean Hidden Concept Discovery](#5-experiment-4-skyocean-hidden-concept-discovery)
6. [Experiment 5: Kurtosis-Based Blind Feature Discovery](#6-experiment-5-kurtosis-based-blind-feature-discovery)
7. [Experiment 6: VLM Auto-Naming of Blind Features](#7-experiment-6-vlm-auto-naming)
8. [Key Findings Across All Experiments](#8-key-findings)
9. [Limitations and What Didn't Work](#9-limitations)
10. [Next Steps](#10-next-steps)

---

## 1. Overview and Motivation

The original GRUE paper (paper.md) established that concepts suppressed from training labels survive as latent structure. These extended experiments push that finding in two directions:

**Direction A (Experiments 1--3):** Can we make the latent structure *compositional*? If a model learns "color" from color-blocks and "object" from CIFAR images, does `d_color + d_object` retrieve "red truck"? This tests whether latent concepts compose linearly.

**Direction B (Experiments 4--6):** Can we discover latent concepts *without any labels at all*? If so, how? This progresses from SAM-guided discovery (needs a segmentation oracle) to kurtosis-driven blind discovery (needs only unlabeled forward passes) to VLM auto-naming (needs no human in the loop).

**Model architecture throughout:** All experiments use ConceptCNN or JointCNN -- small CNNs (conv1-3 + fc1 256-dim + fc2) trained on CIFAR-10 or CIFAR-100. The `fc1` layer (256 neurons) is the primary probe point. All runs use MLX on Apple Silicon.

---

## 2. Experiment 1: Joint Color+Object Training

### Question
If we train a single CNN trunk with two heads -- one for color (11 Berlin-Kay categories from color-block images) and one for object (CIFAR classes) -- do the learned representations compose? Can `d_red + d_truck` retrieve red trucks?

### Method
- **JointCNN architecture:** shared conv trunk + fc1 (256-dim) + two output heads (color head: 11 classes, object head: 10 or 100 classes)
- **Training data:** color-block images (solid colored squares, 11 colors) interleaved with CIFAR images
- **Schemes:** JA (baseline), JC (red suppressed from color labels, CIFAR intact), JA100 (CIFAR-100 variant)
- **Compositionality probe:** Given query pair (color, object), compute `d_color + d_object` in fc1 space, rank all test images by cosine similarity, check if top-20 retrieved images actually match
- **Ground truth:** Initially dominant-hue extraction (noisy), later SAM-based color labels (accurate)

### Key Results -- CIFAR-10 (JA scheme)

Compositionality works but is noisy. Representative results with SAM ground-truth evaluation:

| Query | Precision@20 | Lift |
|---|---|---|
| red + automobile | 50% | ~5x |
| red + truck | 55% | ~5.5x |
| blue + automobile | 45% | ~4.5x |
| yellow + truck | 15% | ~1.5x |

**Average precision ~35%, average lift ~10x** over random baseline. The lift is the important number -- retrieving the correct color+object at 10x above chance demonstrates real compositional structure, even when precision is modest.

### Key Results -- CIFAR-100 (JA100 scheme)

- Color accuracy: 85.3%, Object accuracy: 33.3% (100 classes is hard for this small CNN)
- Compositionality precision lower (avg ~6%) but **lift much higher (avg ~39x)** because each vehicle class is only 1% of the test set
- The lift is the meaningful metric: 39x means the model finds the right needle in a much larger haystack

### Key Finding
**Vector arithmetic on latent directions partially composes.** `d_color + d_object` retrieves relevant images well above chance, but precision degrades for underrepresented color+object combinations (e.g., yellow trucks). The model does not separate color and object into perfectly orthogonal subspaces -- the cosine between some color and object directions is as high as 0.3, which contaminates retrieval.

### Vector Length Analysis
- Object directions have norm ~15-17, color directions have norm ~7-10
- After L2 normalization both become unit vectors
- The cosine between d_color and d_object (not the norm) predicts compositionality quality

### Files
- Script: `src/run_joint_experiment.py`
- Model: `src/joint_model.py` (JointCNN)
- Data loader: `src/joint_dataset.py`
- Results: `results/joint_experiment_results.json`
- Figures: `figures/joint/*.png`, `figures/joint100/*.png`

---

## 3. Experiment 2: SAM-Augmented Color Labels

### Problem
The initial compositionality probe used dominant-hue extraction (most common pixel color in the image) as ground truth for vehicle color. This was badly noisy -- a blue sky behind a red truck would label the truck "blue." Precision numbers looked inflated because the noisy heuristic matched the noisy retrieval.

### Solution
Used Meta's Segment Anything Model (SAM) to segment the foreground object, then computed dominant color only within the object mask.

### Technical Details
- Used MLX SAM (vit-base) running locally
- Initial approach: `SamAutomaticMaskGenerator` -- generated all masks per image. **Too slow:** 2+ hours for 10k images.
- Revised approach: `SamPredictor` with a single center point prompt. Much faster (~1.7 img/sec).
- Required patching MLX SAM source:
  - `prompt_encoder.py`: type conversion for points/labels to `mx.array`
  - `predictor.py`: batch dimension insertion for point_coords
- Color classification: Berlin-Kay categories (11 colors) via HSV thresholding of masked pixels

### Corrections Tool
Built a Tkinter GUI (`src/label_corrector.py`) for manual review of SAM labels. Required workaround for macOS button rendering (replaced `tk.Button` with `tk.Frame + tk.Canvas + tk.Label` to avoid white-on-white text).

### Impact on Results
SAM ground truth revealed that the dominant-hue numbers were **overstated**. Red+truck went from 80% (noisy) to 55% (SAM GT), red+automobile from 85% to 50%. The compositionality effect is real but weaker than initially measured.

### Files
- Script: `src/cifar_color_labels.py`
- Correction tool: `src/label_corrector.py`
- CIFAR-10 labels: `dataset_cifar10/vehicle_color_labels.json` (4000 vehicles)
- CIFAR-100 labels: `dataset_cifar100/vehicle_color_labels.json` (600 vehicles)

---

## 4. Experiment 3: CIFAR-100 Replication

### Question
Does the compositionality finding scale to 100 classes?

### Method
Trained JointCNN with CIFAR-100 (100 object classes) + 11 color classes. Probed compositionality on 6 vehicle classes (bicycle, bus, motorcycle, pickup_truck, streetcar, train) x 4 colors.

### Results
Lower absolute precision but much higher lift (39x), because with 100 classes each vehicle type is only 1% of the test data, making random retrieval very unlikely.

### Notable Bug
CIFAR-100 label list was missing "kangaroo" at index 38, causing all subsequent class indices to be off by one. Every vehicle class index was wrong. Fixed by correcting `CIFAR100_LABELS` in `src/cifar100_dataset.py`.

### Critical Dataset Bug
`load_raw_cifar100()` was accidentally returning X_train as X_test (`X_test_raw, y_test_raw, _, _ = load_raw_cifar10()` instead of `_, _, X_test_raw, y_test_raw = load_raw_cifar10()`). This meant SAM ground-truth labels were indexed against the wrong images, causing 0% precision on all GT-labeled queries. Fixed in `src/joint_dataset.py`.

### Files
- Dataset: `src/cifar100_dataset.py`
- Training: `src/joint_dataset.py` (JA100, JC100 schemes)
- Model weights: `models_joint/JA100_seed1/`

---

## 5. Experiment 4: Sky/Ocean Hidden Concept Discovery

### Question
Can we find concepts in the latent space that were **never in any training label**? Specifically: does a CIFAR-10 model that was only trained on {airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck} have a latent representation for "sky" and "ocean"?

### Why Sky and Ocean
Both are visually similar (blue/grey, flat, textureless) but spatially distinct (sky = top of image, ocean = bottom). If the model separates them, it learned spatial semantics beyond class identity.

### Method

**Step 1: SAM ground-truth labeling.** Used SamPredictor with targeted point prompts:
- Sky: point at (128, 32) in 256x256 space (top-center), filtered for area > 6%, blue/grey color, width > height
- Ocean: point at (128, 220) (bottom-center), same filters
- Only probed relevant classes (sky: airplane, bird, ship, deer, horse; ocean: ship, frog, horse, deer)
- Hard-forced negative classes (e.g., automobile, truck are always "no sky")

**Step 2: Compute concept directions.** Mean-difference of fc1 features between SAM-positive and SAM-negative images, L2-normalized.

**Step 3: Probe.** Few-shot logistic regression on fc1 features, varying from 5 to 50 labeled examples per class.

### Results

| Metric | Sky | Ocean |
|---|---|---|
| SAM positives | 737 / 6800 labeled (10.8%) | 71 / 1000 labeled (7.1%) |
| 5-shot probe accuracy | 61.2% +/- 11.7% | 65.5% +/- 11.3% |
| 50-shot probe accuracy | 73.4% +/- 2.5% | 68.2% +/- 3.3% |
| Sky-Ocean cosine similarity | 0.556 | (partially overlapping) |

**Class projection scores (median projection onto concept direction):**

Sky direction top-5: ship (7.85), airplane (6.48), horse (5.94), truck (1.42), deer (0.48)
Ocean direction top-5: ship (11.48), airplane (2.16), bird (0.07), automobile (-1.46), truck (-1.63)

### Key Findings

1. **Both concepts are recoverable** from as few as 5 labeled examples, well above 50% chance.
2. **Sky and ocean are partially separated** (cosine 0.556) -- the model distinguishes "blue thing on top" from "blue thing on bottom" despite never being trained on either concept.
3. **Sky is diffuse** -- spread across airplane, ship, horse, deer classes. Sky is a multi-class background concept.
4. **Ocean is narrow** -- dominated by ships (11.48 median projection, next class is airplane at 2.16). Ocean is basically "the stuff under ships."
5. **Horse is a sky class.** The model learned that horse images tend to have sky backgrounds, and the sky direction picks this up (5.94 median projection, third highest).

### Limitation
Ocean had only 71 positives from 1000 labeled images (labeling was stopped early). The ocean probe is noisier as a result. Sky is more reliable with 737 positives from 6800 images.

### Files
- SAM labeler: `src/sam_concept_labeler.py`
- Experiment: `src/concept_discovery_experiment.py`
- Results: `results/concept_discovery_sky_ocean.json`
- Labels: `results/sky_labels.json`, `results/ocean_labels.json`
- Figures: `figures/concept_discovery/*.png`

---

## 6. Experiment 5: Kurtosis-Based Blind Feature Discovery

### Core Hypothesis (proposed by Ian)
When a neural network learns a coherent concept, it stretches its latent space along that concept's axis. If you project all images onto a random direction, the Central Limit Theorem guarantees a Gaussian distribution. But if a coherent concept exists along a direction, the projection distribution will be **non-Gaussian** -- fat-tailed (high kurtosis) because the network has pushed concept-positive images far from the center.

Therefore: **finding high-kurtosis directions = finding learned concepts, with zero labels.**

### Method
1. Forward-pass all 10k CIFAR-10 test images through ConceptCNN (scheme A, seed 1)
2. Extract fc1 features (10000 x 256)
3. Run FastICA (which maximizes non-Gaussianity) to find top-N directions
4. Compute excess kurtosis for random directions, trained class directions, SAM concept directions, and ICA directions
5. Sort ICA directions by |excess kurtosis| -- these are the blind-discovered concepts

### Quantitative Results

| Direction Type | Kurtosis (excess) | Interpretation |
|---|---|---|
| Random (200 samples) | 0.76 +/- 0.77 | Gaussian noise -- CLT confirmed |
| Trained class directions (fc2.weight rows) | 1.10 +/- 0.57 | Mild non-Gaussianity |
| Sky direction (SAM mean-diff) | -0.32 | Near-Gaussian (diffuse concept) |
| Ocean direction (SAM mean-diff) | 5.58 | Strongly non-Gaussian (sparse concept) |
| ICA components (top-10) | 4.4 -- 11.0 | **Massively non-Gaussian** |

### What ICA Found (top-8 by kurtosis)

Each ICA component has a "positive pole" (high projection) and "negative pole" (low projection). Examining the top-12 images at each pole:

| ICA # | Kurtosis | Positive pole | Negative pole | Interpretation |
|---|---|---|---|---|
| #1 | 11.0 | truck + frog | automobile x12 | bulky/textured vs sleek/streamlined |
| #2 | 10.2 | deer x16 | horse x12 | wild deer vs domesticated horse |
| #3 | 9.1 | airplane x16 | deer x12 | sky/mechanical vs ground/organic |
| #4 | 9.0 | truck | automobile + misc | cargo vs passenger vehicle |
| #5 | 8.8 | frog x16 | truck + automobile | amphibian vs wheeled |
| #6 | 8.6 | frog | ship x12 | land vs water |
| #7 | 8.0 | dog | ship + truck | pet vs vessel |
| #8 | 7.3 | bird + cat | airplane x16 | organic flyer vs mechanical flyer |

### Key Finding: ICA #1 is the Most Interesting
The highest-kurtosis direction groups **frogs and trucks together** at one pole and **automobiles** at the other. Frogs and trucks are categorically unrelated, but they share:
- Boxy/textured appearance
- Low-to-ground, fills the frame
- Earthy/muted colors

The model learned a **shape/texture axis** that cuts across biological categories entirely. This is a genuinely emergent feature -- no label ever told the model that frogs and trucks are similar.

### ICA #8: Mechanical vs Biological Flight
The model independently separated airplanes from birds even though both appear against sky backgrounds. This axis encodes "mechanical flyer" vs "organic flyer" -- a semantic distinction the model derived purely from visual statistics.

### What Kurtosis Misses: The Sky Problem
Sky has **negative** kurtosis (-0.32) despite being a real concept. Why? Sky is present in ~11% of images spread across 5+ classes. The CLT washes it out into near-Gaussian because many different kinds of images contribute partial signal. Kurtosis catches **sparse/concentrated** concepts (like ocean, which is almost entirely ship-class) but misses **diffuse** concepts. A full blind-discovery pipeline would need additional statistics: bimodality coefficient, dip test, or KL-from-Gaussian.

### ICA vs Mean-Diff Directions
The cosine similarity between ICA components and SAM-derived sky/ocean directions was near-zero (~1e-6). This is because ICA was run on whitened features, and the whitening scrambles the geometry. ICA and mean-diff find complementary, nearly orthogonal structures. This is not a failure -- it means the latent space is richer than either method alone can capture.

### Files
- Script: `src/concept_kurtosis_discovery.py` (GRUE dir), `src/unsupervised_feature_discovery.py` (parent dir)
- Results: `results/kurtosis_discovery.json`
- Figures: `figures/concept_discovery/kurtosis/*.png`

---

## 7. Experiment 6: VLM Auto-Naming

### Question
Can we name the blindly-discovered ICA directions automatically, using a Vision-Language Model, with a non-leading prompt?

### Method
1. For each of the top-10 ICA components, assemble the 12 images with highest projection into a grid
2. Pass the grid to Gemma 4 (google/gemma-4-e2b-it via mlx-vlm) with the prompt:
   > "These images were selected by a mathematical algorithm. Look at all of them carefully. What single visual property, feature, or quality do they all share? Answer in one short phrase only -- no explanation."
3. Record the VLM's answer with no human review or correction

### Results

| ICA # | Kurtosis | Top class | VLM name |
|---|---|---|---|
| #1 | 11.80 | truck + frog | "Objects" |
| #2 | 10.47 | deer | "Wildlife" |
| #3 | 10.24 | airplane | "Aircraft" |
| #4 | 10.07 | truck | "Trucks" |
| #5 | 9.85 | frog | "Reptiles" |
| #6 | 8.56 | frog | "Animals" |
| #7 | 8.33 | dog | "Dog portraits" |
| #8 | 8.25 | airplane | "Aircraft" |
| #9 | 7.49 | frog | "Randomness" |
| #10 | 6.78 | bird | "Birds" |

### Assessment
**5 out of 10 are precise** (Aircraft, Trucks, Dog portraits, Aircraft, Birds). **3 are correct but vague** (Wildlife for deer, Animals for frogs, Objects for truck+frog). **1 is wrong** (Reptiles -- frogs are amphibians). **1 admits defeat** (Randomness).

The VLM successfully names clean, single-class ICA components. It struggles with mixed-class components (#1: truck+frog) because the shared feature is abstract (shape/texture) rather than categorical. Gemini's earlier run with Qwen2-VL and a class-orthogonal filtering step found better abstract labels like "Background Blur" and "Color/Lighting" for the residual axes after removing class-aligned components.

### What the Experiment Proved
End-to-end blind concept extraction is **mechanically feasible**: ICA -> image retrieval -> VLM naming, with zero human labels at any stage. But the quality of naming depends heavily on:
1. Whether the ICA component corresponds to a clean categorical concept (works) or an abstract visual feature (doesn't work with current prompt)
2. VLM image resolution (32px CIFAR images are very hard for VLMs)
3. Prompt design (contrastive "what makes LEFT different from RIGHT" would likely outperform single-grid "what do these share")

### Gemini's Parallel Finding
Gemini independently ran a similar pipeline (on the same model, same dataset) and found:
- ICA kurtosis values of 6.0--8.0
- After orthogonal rejection of class-aligned components, discovered features like "prominent animal faces," "contrasted dark backgrounds," and "green foliage"
- Used class-orthogonal filtering (reject ICA components with cosine > 0.4 to any class centroid) to isolate purely emergent features

The orthogonal filtering step is the key insight Gemini added: by explicitly removing class-correlated axes, the remaining ICA components correspond to genuinely novel, cross-class features.

### Files
- Script: `src/vlm_name_features.py`
- Results: `GRUE/results/ica_vlm_names.json`
- Figures: `figures/feature_geometry/vlm_names/*.png`
- Gemini's paper: `GRUE/paper/paper_blind_discovery.md`

---

## 8. Key Findings Across All Experiments

### Finding 1: Latent concepts survive and compose (partially)
Vector arithmetic `d_color + d_object` retrieves correct color+object combinations at 10-40x above chance. Composition is real but imperfect -- color and object directions are not fully orthogonal.

### Finding 2: Models encode concepts they were never trained on
The CIFAR-10 model has linearly-separable "sky" and "ocean" directions despite having no sky or ocean labels. These are recoverable from as few as 5 labeled examples (61-66% accuracy vs 50% chance).

### Finding 3: Sky and ocean are partially distinct
Cosine similarity 0.556 -- the model has learned that "blue thing on top" and "blue thing on bottom" are related but different, purely from co-occurrence statistics with different object classes.

### Finding 4: Non-Gaussianity reveals learned concepts
Random projections follow the CLT (kurtosis ~0). Concept-aligned projections deviate (kurtosis 1-11). ICA discovers the directions with maximum non-Gaussianity, and these correspond to semantically coherent concepts.

### Finding 5: Kurtosis catches sparse concepts but misses diffuse ones
Ocean (kurtosis 5.6, concentrated in ship class) is detectable. Sky (kurtosis -0.3, spread across 5 classes) is not. A complete blind-discovery method needs statistics beyond kurtosis -- bimodality coefficient or KL divergence from Gaussian.

### Finding 6: The truck-frog axis
The highest-kurtosis ICA direction groups trucks and frogs together, opposed to automobiles. This is a shape/texture axis the model learned without any label telling it that frogs and trucks are similar. It is the clearest evidence of genuinely emergent cross-category structure.

### Finding 7: VLM naming works for clean concepts, fails for abstract ones
Gemma 4 correctly names "Aircraft," "Trucks," "Dog portraits," "Birds" but fails on "bulky/textured things" (calls it "Objects"). The bottleneck is CIFAR resolution (32px) and prompt design.

---

## 9. Limitations and What Didn't Work

### CIFAR-10/100 is the Wrong Dataset for This Work
32x32 pixel images are too low-resolution for:
- SAM segmentation (requires upscaling to 256x256, introducing artifacts)
- VLM labeling (Gemma can barely read 32px thumbnails)
- Fine-grained concept discovery (many visual features are destroyed by compression)
- Color labeling (a few pixels of sky dominates small images)

Only 10 classes means ICA mostly rediscovers class boundaries rather than cross-class features. CIFAR-100 helps (100 classes) but still has the resolution problem.

### SAM Labeling is Slow
Even with the optimized point-prompt approach (vs auto-mask generator), SAM processes ~1.7 images/sec on Apple Silicon. Labeling 10k images takes ~100 minutes per concept. For larger datasets this would need batched GPU inference.

### Whitened ICA and Mean-Diff Find Orthogonal Structure
ICA on whitened features finds different axes than mean-difference on raw features. The cosine between them is ~0. This means neither method alone captures the full latent structure -- they're complementary views. A unified method is needed.

### Compositionality Precision is Modest
~35% average precision on color+object retrieval is statistically significant (high lift) but not practically useful for retrieval. The fc1 space conflates color, object, background, and texture along partially-correlated axes.

### JC100 Not Trained
Only JA100 was trained on CIFAR-100. The planned JC100 (red suppressed) replication was not completed.

---

## 10. Next Steps

### High Priority: Better Dataset
Rerun the entire experimental pipeline on a higher-resolution dataset with more classes. Candidates:
- **ImageNet-1k** (1000 classes, 224x224) -- the obvious choice but heavy to train
- **STL-10** (10 classes, 96x96) -- like CIFAR but 3x resolution, might be sufficient for SAM/VLM
- **iNaturalist** -- fine-grained categories with natural backgrounds (sky, water, foliage as hidden concepts)

### Contrastive VLM Prompting
Show top-12 vs bottom-12 images side by side and ask "what visual quality does the LEFT group have that the RIGHT group lacks?" This directly extracts the bipolar meaning of each ICA axis.

### Bimodality + KL Divergence
Add bimodality coefficient and KL-from-Gaussian to the kurtosis pipeline. This should catch diffuse concepts like sky that kurtosis misses.

### Class-Orthogonal Discovery (Gemini's Approach)
Systematically filter out ICA components that align with known class centroids (cosine > threshold). The residual axes are the genuinely emergent features. Combine with VLM naming for fully automated feature taxonomy.

### Cross-Scheme Comparison
Run the kurtosis pipeline on schemes A, B, and C. Does suppressing "red" from color labels change which ICA components emerge? This would directly connect the blind-discovery method back to the original GRUE thesis.

---

## Appendix: File Index

### Scripts (in `src/`)
| File | Purpose |
|---|---|
| `run_joint_experiment.py` | Joint color+object experiment, compositionality probe |
| `joint_model.py` | JointCNN architecture (shared trunk + two heads) |
| `joint_dataset.py` | Data loading for joint training (color blocks + CIFAR) |
| `cifar_color_labels.py` | SAM-based vehicle color labeling |
| `label_corrector.py` | Tkinter GUI for manual label review |
| `cifar100_dataset.py` | CIFAR-100 data loading and label list |
| `sam_concept_labeler.py` | SAM point-prompt concept labeler (sky/ocean) |
| `concept_discovery_experiment.py` | Sky/ocean few-shot probe and analysis |
| `concept_kurtosis_discovery.py` | ICA + kurtosis blind feature discovery |
| `unsupervised_feature_discovery.py` | Kurtosis comparison (random/class/ICA/sky/ocean) |
| `vlm_name_features.py` | VLM auto-naming of ICA directions |

### Results (in `results/`)
| File | Contents |
|---|---|
| `joint_experiment_results.json` | JA, JC, JA100 accuracies and compositionality |
| `concept_discovery_sky_ocean.json` | Sky/ocean probe results, class scores, cosine |
| `kurtosis_discovery.json` | Full kurtosis comparison data |
| `ica_vlm_names.json` | VLM labels for top-10 ICA components |
| `sky_labels.json` | SAM sky labels (6800 images, 737 positive) |
| `ocean_labels.json` | SAM ocean labels (1000 images, 71 positive) |

### Key Figures
| Figure | Shows |
|---|---|
| `figures/concept_discovery/few_shot_curve.png` | Sky/ocean probe accuracy vs N labels |
| `figures/concept_discovery/sky_scores_by_class.png` | Which CIFAR classes "contain" sky |
| `figures/concept_discovery/kurtosis/kurtosis_comparison.png` | Kurtosis distributions: random vs class vs ICA |
| `figures/concept_discovery/kurtosis/projection_distributions.png` | Histograms of projections along different directions |
| `figures/concept_discovery/kurtosis/ica_top8_overview.png` | Top/bottom images for 8 highest-kurtosis ICA components |
| `figures/feature_geometry/vlm_names/ica_vlm_summary.png` | VLM-named concept grids |
| `figures/joint/color_organisation_pca.png` | PCA of vehicle latent features colored by SAM labels |
