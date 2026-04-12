"""
Concept Kurtosis Discovery
==========================
Finds learned concepts in fc1 activations via Projection Pursuit / FastICA.
Core idea: coherent concepts produce non-Gaussian (high |kurtosis|) projections
along their axis; random directions are Gaussian by CLT.

Usage:
    python src/concept_kurtosis_discovery.py --scheme A --seed 1
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew, norm
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")
from run_experiment import load_cifar_model
from concept_extraction import get_activations
from cifar_dataset import load_raw_cifar10, CIFAR10_LABELS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures", "concept_discovery", "kurtosis")
SKY_LABELS_PATH = os.path.join(RESULTS_DIR, "sky_labels.json")
OCEAN_LABELS_PATH = os.path.join(RESULTS_DIR, "ocean_labels.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def excess_kurtosis(proj: np.ndarray) -> float:
    """Compute excess kurtosis (Gaussian = 0) of a 1D projection."""
    return float(scipy_kurtosis(proj, fisher=True))

def bimodality_coefficient(proj: np.ndarray) -> float:
    """Compute bimodality coefficient (BC > 0.555 suggests bimodality)."""
    g = scipy_skew(proj)
    # fisher=False gives Pearson kurtosis
    k = scipy_kurtosis(proj, fisher=False) 
    return float((g**2 + 1) / k) if k != 0 else 0.0

def kl_divergence_gaussian(proj: np.ndarray, bins: int = 100) -> float:
    """Compute KL divergence from a Gaussian distribution with same mean and std."""
    hist, bin_edges = np.histogram(proj, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mu, std = proj.mean(), proj.std()
    q = norm.pdf(bin_centers, loc=mu, scale=std)
    
    hist = hist + 1e-10
    q = q + 1e-10
    dx = bin_edges[1] - bin_edges[0]
    return float(np.sum(hist * np.log(hist / q)) * dx)


def unit_vector(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def load_concept_direction(label_path: str, feats: np.ndarray):
    """Compute mean-diff direction from SAM label file."""
    with open(label_path) as f:
        entries = json.load(f)
    pos = [e["idx"] for e in entries if e["has_concept"]]
    neg = [e["idx"] for e in entries if not e["has_concept"]]
    if len(pos) < 10:
        print(f"  WARNING: only {len(pos)} positive examples in {label_path}")
        return None
    d = feats[pos].mean(0) - feats[neg].mean(0)
    return unit_vector(d)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit_vector(a), unit_vector(b)))


def publication_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })


# ---------------------------------------------------------------------------
# Visualization: image grids per ICA component
# ---------------------------------------------------------------------------

def visualize_ica_image_grids(S, X_test, y_test, kurt_vals, order,
                               n_top=8, n_images=16, figures_dir=FIGURES_DIR):
    """
    For each of the top-n_top ICA components (by |kurtosis|), produce a figure
    showing the top-n_images and bottom-n_images CIFAR-10 test images along
    that direction. Also produces a compact summary overview figure.

    Parameters
    ----------
    S          : (N, n_ica) ICA projections (from FastICA.fit_transform)
    X_test     : (N, 32, 32, 3) uint8 CIFAR images
    y_test     : (N,) int class labels
    kurt_vals  : (n_ica,) excess kurtosis for each component
    order      : indices sorting components by |kurtosis| descending
    n_top      : number of top components to visualize (default 8)
    n_images   : number of extreme images per end (default 16)
    figures_dir: output directory
    """
    from collections import Counter

    os.makedirs(figures_dir, exist_ok=True)

    # For the summary figure: collect strip data
    summary_strips = []   # list of (top_imgs, bot_imgs, title_short)

    for rank in range(n_top):
        comp_idx = order[rank]
        k = kurt_vals[comp_idx]
        proj = S[:, comp_idx]  # (N,)

        sorted_pos = np.argsort(proj)           # ascending
        top_indices = sorted_pos[-n_images:][::-1]   # highest projection
        bot_indices = sorted_pos[:n_images]          # lowest projection

        top_labels = [CIFAR10_LABELS[y_test[i]] for i in top_indices]
        bot_labels = [CIFAR10_LABELS[y_test[i]] for i in bot_indices]

        # Most common class in top / bottom
        top_counter = Counter(top_labels)
        bot_counter = Counter(bot_labels)
        top_class = top_counter.most_common(1)[0][0]

        # Print stats
        print(f"\n  Component #{rank+1} (ICA comp {comp_idx}) | kurtosis={k:.2f}")
        print(f"    Top-{n_images} classes: {top_counter.most_common(3)}")
        print(f"    Bot-{n_images} classes: {bot_counter.most_common(3)}")

        title = (f"ICA component #{rank+1} | kurtosis={k:.2f} | "
                 f"top CIFAR class: {top_class}")

        # Build per-component figure
        ncols = n_images
        fig, axes = plt.subplots(2, ncols, figsize=(ncols * 1.2, 3.5))
        fig.suptitle(title, fontsize=10, y=1.01)

        for col, idx in enumerate(top_indices):
            ax = axes[0, col]
            img = X_test[idx]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            label = CIFAR10_LABELS[y_test[idx]]
            ax.set_xlabel(label, fontsize=6, color="blue")
            for spine in ax.spines.values():
                spine.set_edgecolor("blue")
                spine.set_linewidth(2.5)

        for col, idx in enumerate(bot_indices):
            ax = axes[1, col]
            img = X_test[idx]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            label = CIFAR10_LABELS[y_test[idx]]
            ax.set_xlabel(label, fontsize=6, color="red")
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(2.5)

        axes[0, 0].set_ylabel("top", fontsize=8, color="blue")
        axes[1, 0].set_ylabel("bottom", fontsize=8, color="red")

        plt.tight_layout()
        fname = f"ica_component_{rank+1:02d}_k{k:.1f}.png"
        out_path = os.path.join(figures_dir, fname)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {out_path}")

        # Collect 8 images per end for summary
        summary_strips.append((
            top_indices[:8],
            bot_indices[:8],
            f"#{rank+1} k={k:.1f}",
        ))

    # ------------------------------------------------------------------
    # Summary overview figure: 8 cols × (2 rows per component)
    # ------------------------------------------------------------------
    print("\n  Building summary overview figure ...")
    n_comp = len(summary_strips)
    n_thumb = 8   # images per strip in summary
    strip_h = 2   # rows per component (top + bottom)
    total_rows = n_comp * strip_h

    fig_sum, axes_sum = plt.subplots(
        total_rows, n_thumb,
        figsize=(n_thumb * 0.9, total_rows * 0.95),
        gridspec_kw={"hspace": 0.05, "wspace": 0.03}
    )

    for comp_rank, (top_idxs, bot_idxs, label) in enumerate(summary_strips):
        row_top = comp_rank * 2
        row_bot = comp_rank * 2 + 1

        for col in range(n_thumb):
            # Top strip
            ax = axes_sum[row_top, col]
            ax.imshow(X_test[top_idxs[col]])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("blue")
                sp.set_linewidth(2)
            if col == 0:
                ax.set_ylabel(f"{label}\ntop", fontsize=5.5, color="blue",
                              rotation=0, ha="right", labelpad=30)

            # Bottom strip
            ax = axes_sum[row_bot, col]
            ax.imshow(X_test[bot_idxs[col]])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("red")
                sp.set_linewidth(2)
            if col == 0:
                ax.set_ylabel("bot", fontsize=5.5, color="red",
                              rotation=0, ha="right", labelpad=30)

    fig_sum.suptitle("ICA Top-8 Components: Extreme Images Overview", fontsize=10, y=1.005)
    out_summary = os.path.join(figures_dir, "ica_top8_overview.png")
    fig_sum.savefig(out_summary, dpi=130, bbox_inches="tight")
    plt.close(fig_sum)
    print(f"    Saved: {out_summary}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Concept Kurtosis Discovery")
    parser.add_argument("--scheme", default="A")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_ica", type=int, default=30)
    parser.add_argument("--n_random", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    publication_style()

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"\n[1] Loading ConceptCNN scheme {args.scheme} seed {args.seed} ...")
    model, meta = load_cifar_model(args.scheme, args.seed)
    print(f"    meta: {meta}")

    # ------------------------------------------------------------------
    # 2. Forward-pass all 10k test images
    # ------------------------------------------------------------------
    print("\n[2] Loading CIFAR-10 test data and extracting fc1 activations ...")
    _, _, X_test, y_test = load_raw_cifar10()
    print(f"    X_test shape: {X_test.shape}")

    acts = get_activations(model, X_test)
    feats = acts["fc1"]  # (N, 256)
    N, D = feats.shape
    print(f"    fc1 features: {feats.shape}")

    # Whiten for ICA
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)

    # ------------------------------------------------------------------
    # 3. Baseline comparison directions
    # ------------------------------------------------------------------
    print("\n[3] Computing baseline kurtosis comparisons ...")

    # Random directions
    rng = np.random.default_rng(42)
    rand_dirs = rng.standard_normal((args.n_random, D))
    rand_dirs = rand_dirs / (np.linalg.norm(rand_dirs, axis=1, keepdims=True) + 1e-8)
    
    rand_projs = [feats_scaled @ d for d in rand_dirs]
    kurt_random = np.array([excess_kurtosis(p) for p in rand_projs])
    bc_random = np.array([bimodality_coefficient(p) for p in rand_projs])
    kl_random = np.array([kl_divergence_gaussian(p) for p in rand_projs])

    # Trained class directions (fc2 weight rows)
    import mlx.core as mx
    fc2_w = np.array(model.fc2.weight)  # (10, 256)
    class_dirs = fc2_w / (np.linalg.norm(fc2_w, axis=1, keepdims=True) + 1e-8)
    
    class_projs = [feats_scaled @ d for d in class_dirs]
    kurt_class = np.array([excess_kurtosis(p) for p in class_projs])
    bc_class = np.array([bimodality_coefficient(p) for p in class_projs])
    kl_class = np.array([kl_divergence_gaussian(p) for p in class_projs])

    # Sky / ocean concept directions
    sky_dir = load_concept_direction(SKY_LABELS_PATH, feats_scaled)
    ocean_dir = load_concept_direction(OCEAN_LABELS_PATH, feats_scaled)

    if sky_dir is not None:
        sky_proj = feats_scaled @ sky_dir
        kurt_sky, bc_sky, kl_sky = excess_kurtosis(sky_proj), bimodality_coefficient(sky_proj), kl_divergence_gaussian(sky_proj)
    else:
        kurt_sky = bc_sky = kl_sky = None

    if ocean_dir is not None:
        ocean_proj = feats_scaled @ ocean_dir
        kurt_ocean, bc_ocean, kl_ocean = excess_kurtosis(ocean_proj), bimodality_coefficient(ocean_proj), kl_divergence_gaussian(ocean_proj)
    else:
        kurt_ocean = bc_ocean = kl_ocean = None

    print(f"    Random kurtosis   — mean={kurt_random.mean():.3f}, std={kurt_random.std():.3f} | bc_mean={bc_random.mean():.3f} | kl_mean={kl_random.mean():.3f}")
    print(f"    Class  kurtosis   — {np.round(kurt_class, 2)}")
    print(f"    Class  bimodality — {np.round(bc_class, 2)}")
    print(f"    Class  KL_Gauss   — {np.round(kl_class, 2)}")
    print(f"    Sky    stats      — kurt={kurt_sky:.3f}, bc={bc_sky:.3f}, kl={kl_sky:.3f}" if kurt_sky is not None else "    Sky direction unavailable")
    print(f"    Ocean  stats      — kurt={kurt_ocean:.3f}, bc={bc_ocean:.3f}, kl={kl_ocean:.3f}" if kurt_ocean is not None else "    Ocean direction unavailable")

    # ------------------------------------------------------------------
    # 4. FastICA blind discovery
    # ------------------------------------------------------------------
    print(f"\n[4] Running FastICA (n_components={args.n_ica}) ...")
    ica = FastICA(n_components=args.n_ica, max_iter=2000, tol=0.001, random_state=42)
    S = ica.fit_transform(feats_scaled)  # (N, n_ica) independent components
    # sklearn ICA: S = feats_scaled @ components_.T
    # components_ rows are the ICA directions in feature space
    ica_dirs = ica.components_  # (n_ica, D)
    ica_dirs_normed = ica_dirs / (np.linalg.norm(ica_dirs, axis=1, keepdims=True) + 1e-8)

    # Kurtosis of each ICA component (directly from S columns)
    kurt_ica = np.array([excess_kurtosis(S[:, i]) for i in range(args.n_ica)])
    bc_ica = np.array([bimodality_coefficient(S[:, i]) for i in range(args.n_ica)])
    kl_ica = np.array([kl_divergence_gaussian(S[:, i]) for i in range(args.n_ica)])
    
    sort_idx = np.argsort(np.abs(kurt_ica))[::-1]  # descending |kurtosis|

    print(f"    Top-5 ICA |kurtosis|: {np.abs(kurt_ica[sort_idx[:5]]).round(2)}")
    print(f"    Top-5 ICA bimodality: {bc_ica[sort_idx[:5]].round(2)}")
    print(f"    Top-5 ICA KL_Gauss  : {kl_ica[sort_idx[:5]].round(2)}")

    # ------------------------------------------------------------------
    # 5. Validation — cosine similarity of top-20 ICA dirs vs concepts
    # ------------------------------------------------------------------
    print("\n[5] Validating ICA directions against known concepts ...")
    top20_idx = sort_idx[:20]

    concept_names = CIFAR10_LABELS + ["sky", "ocean"]
    concept_dirs_list = list(class_dirs) + (
        [sky_dir] if sky_dir is not None else [np.zeros(D)]
    ) + (
        [ocean_dir] if ocean_dir is not None else [np.zeros(D)]
    )

    cosine_matrix = np.zeros((20, len(concept_dirs_list)))
    for i, ica_i in enumerate(top20_idx):
        for j, cd in enumerate(concept_dirs_list):
            cosine_matrix[i, j] = abs(cosine_sim(ica_dirs_normed[ica_i], cd))

    # Report top-5 concept matches for each of top-20 ICA components
    print(f"\n    {'ICA':>4}  {'|kurt|':>7}  {'Best concept match':>22}  {'|cos|':>6}")
    print("    " + "-" * 46)
    for i, ica_i in enumerate(top20_idx):
        best_j = int(np.argmax(cosine_matrix[i]))
        print(f"    ICA{sort_idx.tolist().index(ica_i)+1:>2}  "
              f"{abs(kurt_ica[ica_i]):>7.3f}  "
              f"{concept_names[best_j]:>22}  "
              f"{cosine_matrix[i, best_j]:>6.3f}")

    # ------------------------------------------------------------------
    # 6. Kurtosis spectrum plot
    # ------------------------------------------------------------------
    print("\n[6] Plotting kurtosis spectrum ...")
    fig, ax = plt.subplots(figsize=(12, 5))

    # ICA bars
    ica_sorted_kurt = kurt_ica[sort_idx]
    x_ica = np.arange(len(ica_sorted_kurt))
    colors_ica = ["#2196F3" if k > 0 else "#9C27B0" for k in ica_sorted_kurt]
    bars = ax.bar(x_ica, ica_sorted_kurt, color=colors_ica, alpha=0.8, label="ICA components")

    # Horizontal reference lines for sky / ocean / class mean
    if kurt_sky is not None:
        ax.axhline(kurt_sky, color="#FF5722", lw=2, ls="--", label=f"Sky direction ({kurt_sky:.2f})")
    if kurt_ocean is not None:
        ax.axhline(kurt_ocean, color="#00BCD4", lw=2, ls="--", label=f"Ocean direction ({kurt_ocean:.2f})")
    ax.axhline(kurt_class.mean(), color="#4CAF50", lw=2, ls="-.",
               label=f"Class dirs mean ({kurt_class.mean():.2f})")

    # Random band ±1 std
    ax.axhspan(kurt_random.mean() - kurt_random.std(),
               kurt_random.mean() + kurt_random.std(),
               alpha=0.15, color="gray", label=f"Random ±1σ ({kurt_random.mean():.2f}±{kurt_random.std():.2f})")
    ax.axhline(kurt_random.mean(), color="gray", lw=1.5, ls=":")

    ax.set_xlabel("ICA component (sorted by |excess kurtosis|)")
    ax.set_ylabel("Excess kurtosis")
    ax.set_title("Kurtosis Spectrum: ICA components vs. Concept Directions")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "kurtosis_spectrum.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")

    # ------------------------------------------------------------------
    # 3b. Comparison violin plot
    # ------------------------------------------------------------------
    print("\n    Plotting kurtosis comparison ...")
    fig, ax = plt.subplots(figsize=(8, 5))

    groups = [kurt_random, kurt_class]
    labels_viol = ["Random\ndirections", "Trained class\ndirections"]
    colors_viol = ["#9E9E9E", "#4CAF50"]

    if kurt_sky is not None:
        groups.append(np.array([kurt_sky]))
        labels_viol.append("Sky\ndirection")
        colors_viol.append("#FF5722")
    if kurt_ocean is not None:
        groups.append(np.array([kurt_ocean]))
        labels_viol.append("Ocean\ndirection")
        colors_viol.append("#00BCD4")

    groups.append(np.abs(kurt_ica[sort_idx[:10]]))
    labels_viol.append("Top-10 ICA\n(|kurtosis|)")
    colors_viol.append("#2196F3")

    vp = ax.violinplot([g for g in groups if len(g) > 1],
                       positions=[i for i, g in enumerate(groups) if len(g) > 1],
                       showmedians=True, showextrema=True)
    for body, c in zip(vp["bodies"],
                       [c for g, c in zip(groups, colors_viol) if len(g) > 1]):
        body.set_facecolor(c)
        body.set_alpha(0.6)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(2)

    # Scatter single points
    scatter_x = []
    scatter_y = []
    scatter_c = []
    for i, (g, c) in enumerate(zip(groups, colors_viol)):
        if len(g) == 1:
            scatter_x.append(i)
            scatter_y.append(g[0])
            scatter_c.append(c)
    if scatter_x:
        ax.scatter(scatter_x, scatter_y, color=scatter_c, zorder=5, s=80, marker="D")

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels_viol, fontsize=10)
    ax.set_ylabel("Excess kurtosis")
    ax.set_title("Kurtosis Distribution Across Direction Types")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "kurtosis_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")

    # ------------------------------------------------------------------
    # 7. Projection distribution plots
    # ------------------------------------------------------------------
    print("\n[7] Plotting projection distributions ...")
    # top-3 ICA + sky + ocean + 1 random
    proj_dirs = []
    proj_names = []
    proj_colors = []

    for rank, ica_i in enumerate(sort_idx[:3]):
        proj_dirs.append(ica_dirs_normed[ica_i])
        proj_names.append(f"ICA #{rank+1} (k={kurt_ica[ica_i]:.2f})")
        proj_colors.append("#2196F3")

    if sky_dir is not None:
        proj_dirs.append(sky_dir)
        proj_names.append(f"Sky dir (k={kurt_sky:.2f})")
        proj_colors.append("#FF5722")

    if ocean_dir is not None:
        proj_dirs.append(ocean_dir)
        proj_names.append(f"Ocean dir (k={kurt_ocean:.2f})")
        proj_colors.append("#00BCD4")

    # One random direction
    rdm_d = rand_dirs[0]
    kurt_rdm_0 = excess_kurtosis(feats_scaled @ rdm_d)
    proj_dirs.append(rdm_d)
    proj_names.append(f"Random dir (k={kurt_rdm_0:.2f})")
    proj_colors.append("#9E9E9E")

    n_plots = len(proj_dirs)
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes.flatten()

    for ax, direction, name, color in zip(axes_flat, proj_dirs, proj_names, proj_colors):
        projections = feats_scaled @ direction
        ax.hist(projections, bins=60, color=color, alpha=0.75, density=True, edgecolor="white", linewidth=0.3)

        # Overlay Gaussian fit
        mu, sigma = projections.mean(), projections.std()
        x_gauss = np.linspace(projections.min(), projections.max(), 200)
        gauss = np.exp(-0.5 * ((x_gauss - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        ax.plot(x_gauss, gauss, "k--", lw=1.5, alpha=0.7, label="Gaussian fit")

        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Projection value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    # Hide unused axes
    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    fig.suptitle("Projection Distributions: Concept vs. Random Directions", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "projection_distributions.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")

    # ------------------------------------------------------------------
    # 8. ICA vs Concepts heatmap
    # ------------------------------------------------------------------
    print("\n[8] Plotting ICA vs. concept heatmap ...")
    # Full 30-component heatmap
    full_cosine = np.zeros((args.n_ica, len(concept_dirs_list)))
    for i in range(args.n_ica):
        ica_i = sort_idx[i]
        for j, cd in enumerate(concept_dirs_list):
            full_cosine[i, j] = abs(cosine_sim(ica_dirs_normed[ica_i], cd))

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(full_cosine, aspect="auto", cmap="hot", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="|cosine similarity|")

    ax.set_xticks(range(len(concept_names)))
    ax.set_xticklabels(concept_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(args.n_ica))
    ax.set_yticklabels(
        [f"ICA#{i+1} k={kurt_ica[sort_idx[i]]:.1f}" for i in range(args.n_ica)],
        fontsize=8
    )
    ax.set_xlabel("Concept direction")
    ax.set_ylabel("ICA component (sorted by |kurtosis|)")
    ax.set_title("|Cosine| Similarity: 30 ICA Components × 12 Concept Directions")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "ica_vs_concepts.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")

    # ------------------------------------------------------------------
    # 9. 1-shot naming test
    # ------------------------------------------------------------------
    print("\n[9] 1-shot naming test ...")
    with open(SKY_LABELS_PATH) as f:
        sky_entries = json.load(f)
    with open(OCEAN_LABELS_PATH) as f:
        ocean_entries = json.load(f)

    sky_pos = [e["idx"] for e in sky_entries if e["has_concept"]]
    ocean_pos = [e["idx"] for e in ocean_entries if e["has_concept"]]

    results_1shot = {}

    for concept, pos_list, ref_dir, label_color in [
        ("sky", sky_pos, sky_dir, "Sky"),
        ("ocean", ocean_pos, ocean_dir, "Ocean")
    ]:
        if not pos_list or ref_dir is None:
            print(f"    Skipping {concept} (no data)")
            continue

        rng_1shot = np.random.default_rng(123)
        sample_idx = int(rng_1shot.choice(pos_list))
        sample_feat = feats_scaled[sample_idx]  # (256,)

        # Project single image onto each ICA direction (|projection value|)
        proj_with_ica = np.abs(ica_dirs_normed @ sample_feat)  # (n_ica,)
        top5_ica = np.argsort(proj_with_ica)[::-1][:5]

        top5_cos_vs_concept = [abs(cosine_sim(ica_dirs_normed[i], ref_dir)) for i in top5_ica]

        max_proj = proj_with_ica.max() + 1e-12
        print(f"\n    {label_color} image (idx={sample_idx}) — top-5 matching ICA directions:")
        print(f"    {'ICA':<8} {'|proj| (norm.)':<18} {'|cos| w/ '+label_color+' dir'}")
        for rank, ica_i in enumerate(top5_ica):
            cos_val = abs(cosine_sim(ica_dirs_normed[ica_i], ref_dir))
            print(f"    ICA#{ica_i+1:<4} {proj_with_ica[ica_i]/max_proj:>15.3f}   {cos_val:>10.2e}")

        results_1shot[concept] = {
            "sample_idx": sample_idx,
            "top5_ica_indices": top5_ica.tolist(),
            "top5_proj_on_image_raw": proj_with_ica[top5_ica].tolist(),
            "top5_proj_on_image_normalized": (proj_with_ica[top5_ica] / max_proj).tolist(),
            "top5_cos_with_concept_dir": top5_cos_vs_concept,
        }

    # ------------------------------------------------------------------
    # 10. Save numerical results
    # ------------------------------------------------------------------
    print("\n[10] Saving numerical results ...")
    results = {
        "scheme": args.scheme,
        "seed": args.seed,
        "n_ica": args.n_ica,
        "n_random": args.n_random,
        "kurtosis": {
            "random": {
                "mean": float(kurt_random.mean()),
                "std": float(kurt_random.std()),
                "values": kurt_random.tolist(),
            },
            "class_directions": {
                "values": kurt_class.tolist(),
                "names": CIFAR10_LABELS,
                "mean": float(kurt_class.mean()),
                "std": float(kurt_class.std()),
            },
            "sky": float(kurt_sky) if kurt_sky is not None else None,
            "ocean": float(kurt_ocean) if kurt_ocean is not None else None,
            "ica_sorted": {
                "values": ica_sorted_kurt.tolist(),
                "sort_indices": sort_idx.tolist(),
            },
        },
        "bimodality": {
            "random": {
                "mean": float(bc_random.mean()),
                "std": float(bc_random.std()),
                "values": bc_random.tolist(),
            },
            "class_directions": {
                "values": bc_class.tolist(),
                "mean": float(bc_class.mean()),
                "std": float(bc_class.std()),
            },
            "sky": float(bc_sky) if bc_sky is not None else None,
            "ocean": float(bc_ocean) if bc_ocean is not None else None,
            "ica_sorted": {
                "values": bc_ica[sort_idx].tolist(),
            },
        },
        "kl_divergence": {
            "random": {
                "mean": float(kl_random.mean()),
                "std": float(kl_random.std()),
                "values": kl_random.tolist(),
            },
            "class_directions": {
                "values": kl_class.tolist(),
                "mean": float(kl_class.mean()),
                "std": float(kl_class.std()),
            },
            "sky": float(kl_sky) if kl_sky is not None else None,
            "ocean": float(kl_ocean) if kl_ocean is not None else None,
            "ica_sorted": {
                "values": kl_ica[sort_idx].tolist(),
            },
        },
        "top20_ica_concept_alignment": {
            "ica_indices": top20_idx.tolist(),
            "kurt_values": kurt_ica[top20_idx].tolist(),
            "cosine_vs_concepts": cosine_matrix.tolist(),
            "concept_names": concept_names,
        },
        "one_shot_naming": results_1shot,
    }

    out_json = os.path.join(RESULTS_DIR, "kurtosis_discovery.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    Saved: {out_json}")

    # ------------------------------------------------------------------
    # 11. Image grid visualization: top/bottom images per ICA component
    # ------------------------------------------------------------------
    print("\n[11] Visualizing top/bottom CIFAR images per ICA component ...")
    visualize_ica_image_grids(S, X_test, y_test, kurt_vals=kurt_ica, order=sort_idx,
                              n_top=8, n_images=16, figures_dir=FIGURES_DIR)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  KURTOSIS SUMMARY")
    print("=" * 60)
    print(f"  Random dirs (n={args.n_random}):    kurt={kurt_random.mean():>7.3f} | bc={bc_random.mean():>7.3f} | kl={kl_random.mean():>7.3f}")
    print(f"  Class dirs (n=10):       kurt={kurt_class.mean():>7.3f} | bc={bc_class.mean():>7.3f} | kl={kl_class.mean():>7.3f}")
    if kurt_sky is not None:
        print(f"  Sky direction:           kurt={kurt_sky:>7.3f} | bc={bc_sky:>7.3f} | kl={kl_sky:>7.3f}")
    if kurt_ocean is not None:
        print(f"  Ocean direction:         kurt={kurt_ocean:>7.3f} | bc={bc_ocean:>7.3f} | kl={kl_ocean:>7.3f}")
    print(f"  ICA top-1 |kurtosis|:    kurt={abs(ica_sorted_kurt[0]):>7.3f} | bc={bc_ica[sort_idx[0]]:>7.3f} | kl={kl_ica[sort_idx[0]]:>7.3f}")
    print(f"  ICA top-5 |kurtosis|:    kurt={np.abs(ica_sorted_kurt[:5]).mean():>7.3f} | bc={bc_ica[sort_idx[:5]].mean():>7.3f} | kl={kl_ica[sort_idx[:5]].mean():>7.3f} (mean)")
    print("=" * 60)
    print(f"\n  Per-class kurtosis:")
    for name, k in zip(CIFAR10_LABELS, kurt_class):
        print(f"    {name:<12} {k:>7.3f}")
    print()


if __name__ == "__main__":
    main()
