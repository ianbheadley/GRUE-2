# Study Artifact Index

This appendix maps the main claims in the paper to the exact local study artifacts used to support them.

## Primary claims and sources

- Hidden distinctions survive label merge and class removal:
  - `../data/missing_training_summary.json`
  - `../figures/fig_hidden_separability.png`

- Missing classes reappear as mixtures of nearby seen concepts:
  - `../data/missing_training_summary.json`
  - `../figures/fig_missing_redistribution.png`

- CIFAR hidden automobile-truck separation persists across layers:
  - `../data/summary.json`
  - `../figures/fig_layerwise_latent_gap.png`
  - `../results/dissection/fig4_summary_dashboard.png`

- Whole-model subtraction/injection is weak or brittle:
  - `../data/alpha_sweep_summary.json`
  - `../figures/fig_alpha_sweep.png`

- Layer-local `A - C` deltas recover the missing concept better than whole-model deltas:
  - `../data/layer_concept_report.json`
  - `../figures/fig_layer_rankings.png`

- CIFAR-10 learns implicit color structure without explicit color labels:
  - `../data/cifar_implicit_color_summary.json`
  - `../figures/fig_implicit_color_cifar.png`

## Key numeric anchors

- Color hidden probe accuracy:
  - Scheme A: `0.9955`
  - Scheme B: `0.98225`
  - Scheme C: `0.9865`
  - Source: `../data/missing_training_summary.json`

- CIFAR hidden probe accuracy:
  - Scheme A: `0.91925`
  - Scheme B: `0.7625`
  - Scheme C: `0.82125`
  - Source: `../data/missing_training_summary.json`

- Missing blue redistribution in Scheme C:
  - `purple = 0.641`
  - `green = 0.280`
  - Source: `../data/missing_training_summary.json`

- Missing truck redistribution in Scheme C:
  - `automobile = 0.362`
  - `airplane = 0.319`
  - `cat = 0.132`
  - `ship = 0.1165`
  - Source: `../data/missing_training_summary.json`

- Hidden direction alignment `A vs C`:
  - Color: `0.4726`
  - CIFAR-10: `0.6318`
  - Source: `../data/missing_training_summary.json`

- Best layer-local recovery:
  - Color: `bn3 @ alpha=0.50`, purity `0.2582`
  - CIFAR-10: `bn2 @ alpha=1.00`, purity `0.1525`
  - Source: `../data/layer_concept_report.json`

- Implicit CIFAR color probe after removing class identity:
  - Scheme A `conv2`: `0.9803`
  - Scheme B `conv2`: `0.9787`
  - Scheme C `conv2`: `0.9790`
  - Source: `../data/cifar_implicit_color_summary.json`

- Implicit CIFAR within-class color probe at `fc1`:
  - Scheme A: `0.7293`
  - Scheme B: `0.6763`
  - Scheme C: `0.6813`
  - Source: `../data/cifar_implicit_color_summary.json`

## Figure inventory

- `../figures/fig_hidden_separability.png`
- `../figures/fig_missing_redistribution.png`
- `../figures/fig_layer_rankings.png`
- `../figures/fig_alpha_sweep.png`
- `../figures/fig_layerwise_latent_gap.png`
- `../figures/fig_implicit_color_cifar.png`
- `../figures/latent_separation_pca.png`
- `../figures/weight_diff_viz.png`
