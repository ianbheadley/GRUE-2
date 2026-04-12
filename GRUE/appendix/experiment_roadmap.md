# Experiment Roadmap

This roadmap breaks the current architecture ideas into single-variable experiments so we can validate each one before stacking them into a larger model.

The guiding rule is simple:

1. Change one thing at a time.
2. Keep the A/B/C task fixed.
3. Measure whether alignment and subtraction get cleaner, not just whether accuracy stays high.

## Core Readout

For every experiment below, keep the same primary readouts:

- Task accuracy for Schemes A, B, and C.
- `A`-head transfer after alignment:
  - raw `B` in `A` head
  - rotated `B -> A`
  - affine `B -> A`
- Residual quality after alignment:
  - norm ratio for target classes vs other classes
  - blue-vs-green probe accuracy on `A - aligned(B)`
- Cross-seed stability:
  - cosine similarity of the residual direction across seeds

These are already partially implemented in:

- `activation_alignment_experiment.py`
- `orthogonal_alignment_experiment.py`

## Experiment 0: Baseline Alignment Benchmark

Question:
Can we quantify how much of the A/B mismatch is solved by pure rotation, and how much needs a more general linear map?

Change:
- No architecture change.
- Use the existing color-block models.

Compare:
- raw activations
- orthogonal alignment
- affine alignment

Primary success condition:
- Establish a stable baseline for later comparisons.

Why this matters:
- This is the anchor experiment. Every later intervention should be judged by whether it narrows the gap between rotation-only and affine alignment, and whether it improves residual stability across seeds.

## Experiment 1: Pre-Biased Path Topology

Question:
If we pre-structure the network into fixed sparse routes, do independently trained models land in more comparable internal coordinates?

Change:
- Keep the model otherwise standard.
- Add fixed sparse connectivity masks or fixed branch groups.
- Do not yet add learned gates.

Controls:
- dense baseline
- fixed sparse topology
- random sparse topology with the same parameter count

Primary metrics:
- rotation-only alignment improvement
- residual direction cross-seed cosine
- performance drop relative to dense baseline

Success condition:
- Orthogonal alignment gets closer to affine alignment without a large accuracy collapse.

Interpretation:
- If this works, then some of the instability is coming from too much freedom in how the network can route information.

## Experiment 2: Gate-Limited Plasticity

Question:
Does limiting which routes are allowed to update make concept differences live in a smaller and more stable subspace?

Change:
- Add a trainable gate or route selector.
- Allow updates only on active routes, adapters, or branch-local parameters.
- Keep most of the trunk fixed or partially frozen.

Controls:
- dense updates everywhere
- sparse activation gating only
- sparse update gating only
- both sparse activation and sparse updates

Primary metrics:
- fraction of active routes
- fraction of updated parameters
- alignment improvement
- residual concentration in target classes

Success condition:
- `A - aligned(B)` becomes more concentrated and more seed-stable than baseline.

Interpretation:
- This directly tests your "limit areas of change" idea without committing to a full dendritic architecture yet.

## Experiment 3: Canonical Route Identity

Question:
Even if we add routes, do the same routes mean the same thing across seeds?

Change:
- Add route anchoring losses or branch identity priors.
- Examples:
  - fixed branch ordering
  - branch load balancing
  - branch specialization penalties
  - prototype losses that tie each branch to a consistent role

Controls:
- routed model without anchoring
- routed model with anchoring

Primary metrics:
- cross-seed route matching quality
- cross-seed cosine of blue-vs-green residual directions
- how much subtraction vectors transfer between seeds

Success condition:
- The residual direction is not just readable within one pair of models, but consistent across independently trained runs.

Interpretation:
- This is likely necessary if the long-term goal is reusable subtraction rather than pairwise alignment only.

## Experiment 4: Dendritic Compartmentalization

Question:
Does compartmentalized routing produce cleaner and more stable concept localization than a standard routed model?

Reference:
- `/Users/ianheadley/Documents/BuddhaGPT/dendritic_42layer_full.py`

Change:
- Build a small GRUE-scale version first.
- Do not start with the full 42-layer language model.
- Introduce:
  - basal stream
  - apical or feedback stream
  - somatic combination
  - per-layer branch gates

Controls:
- standard routed model
- dendritic-lite model

Primary metrics:
- same alignment metrics as above
- branch usage entropy
- seed consistency of branch specialization

Success condition:
- Cleaner rotation-only alignment and more stable residuals than the non-dendritic routed version.

Interpretation:
- This tests whether the dendritic idea buys us anything beyond plain gating.

## Experiment 5: Curriculum By Information Reveal

Question:
Does gradually revealing more of the input reduce representational drift and help models settle into more shared internal structure?

Change:
- Train in stages, revealing more input detail over time.
- Implement reveal by masking the training set with a saliency-derived mask, not by changing label order.

Variants:
- low-resolution to high-resolution
- low-frequency to high-frequency
- masked to full image
- energy-ranked patch reveal
- Spectral Residual saliency reveal

### Spectral Residual reveal specification

Use the Spectral Residual (SR) method as the main saliency filter for this experiment.

The intended behavior is:

1. Compute an SR saliency map for each training image.
2. Convert that saliency map into a reveal mask.
3. Start training with only the highest-saliency pixels or regions visible.
4. Progressively expand outward from those salient points.
5. Reveal low-saliency background regions last.

This should behave like a progressive reconstruction or foveated rendering curriculum:

- Step 1:
  - Render only the pixels with the highest SR values.
  - These should correspond to the most surprising or informative structures, such as strong edges, faces, text, or compact objects.
- Step 2:
  - Dilate or expand the visible region around those salient points.
  - This adds local context without yet revealing the full image.
- Step 3:
  - Fill in the remaining low-SR regions last.
  - These are typically flatter, more redundant, or background-like regions.

Possible implementations:

- Hard threshold schedule:
  - Reveal top `k%` of SR pixels at stage 1, then top `2k%`, then top `4k%`, and so on.
- Radius expansion schedule:
  - Reveal top SR peaks first, then grow a radius around them each stage.
- Blur-to-sharp hybrid:
  - Keep low-saliency regions present but blurred, while high-saliency regions are shown at full fidelity first.

Caution:
- For the grue task, pure energy masking is risky because color distinctions can be low-frequency and not strongly tied to edge energy.
- SR saliency is a better fit than raw edge energy, but it can still underweight diffuse color fields.
- For color-sensitive tasks, consider a control that combines SR saliency with chroma-sensitive masking or blurred background context.

Controls:
- no curriculum
- curriculum only
- SR-only reveal
- SR + low-resolution background reveal

Primary metrics:
- seed-to-seed alignment
- rotation-only vs affine gap
- final task accuracy
- training stability
- concept retention for the target distinction when only high-saliency regions are shown early

Success condition:
- Better alignment without hiding the concept signal itself.

Interpretation:
- This is a data and optimization intervention, not a structure intervention.

## Experiment 6: Shared Trunk Plus Scheme-Specific Adapters

Question:
If A, B, and C share most of the trunk and only diverge in small modules, do concept differences become easier to isolate?

Change:
- Train a shared backbone.
- Give A, B, and C only small scheme-specific adapters, gates, or heads.

Controls:
- fully independent training
- shared-trunk training

Primary metrics:
- residual rank and concentration
- how small the scheme-specific delta can be while preserving performance
- whether subtraction of adapters is cleaner than subtraction of full-model activations

Success condition:
- Most task-specific difference lives in a small module rather than diffuse trunk drift.

Interpretation:
- This may be the most practical bridge from "theory experiment" to "usable architecture."

## Stacking Order

Do not combine everything at once. Stack in this order:

1. Baseline alignment benchmark.
2. Pre-biased path topology.
3. Gate-limited plasticity.
4. Canonical route identity.
5. Dendritic compartmentalization.
6. Curriculum by information reveal.
7. Shared trunk plus scheme-specific adapters.

Reason for this order:

- `1 -> 3` tests whether constrained routing alone helps.
- `4` checks whether routing becomes stable enough to reuse across seeds.
- `5` tells us whether the dendritic design adds something beyond simpler routing.
- `6` is worth trying only after we know what type of structure helps.
- `7` is the architecture move if the earlier experiments show where differences should live.

## Suggested Minimal Builds

To keep iteration fast, use three scales:

- Small:
  - color-block CNN or MLP
  - fastest place to test routing and subtraction
- Medium:
  - CIFAR concept model
  - tests whether the effect survives richer visual structure
- Large:
  - dendritic language model variants
  - only after small and medium results are clear

## Stop-Go Rules

An experiment is worth stacking only if it satisfies at least two of these:

- Improves rotation-only transfer into `A` space.
- Increases target-vs-other residual concentration.
- Increases cross-seed residual direction stability.
- Keeps accuracy within an acceptable drop window.

Kill an idea early if:

- It improves accuracy but makes alignment worse.
- It improves pairwise subtraction but does not improve cross-seed stability.
- It only works under affine alignment and not at all under rotation, if the goal is canonical coordinates.

## First Practical Next Steps

The best immediate sequence is:

1. Run Experiment 1 with fixed sparse branch topology on the existing color-block model.
2. Run Experiment 2 with update gating but no dendritic compartments.
3. Compare both against the current orthogonal and affine baselines.
4. Only if either helps, add route identity anchoring from Experiment 3.

That path gives us the cleanest answer to the central question:

Can we make independently trained models land in internal spaces that are close enough for rotation and subtraction to become genuinely reusable?
