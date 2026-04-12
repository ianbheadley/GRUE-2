"""
VLM Feature Namer — Unsupervised ICA Concept Discovery

For each ICA component found in the fc1 latent space, this script:
  1. Takes the top-12 images (highest projection) along that direction
  2. Assembles them into a single grid image
  3. Asks a VLM (Gemma 4) what visual property makes them similar
  4. Saves the result without any human-provided concept labels

The prompt is deliberately non-leading — we don't tell the model what
to look for, just ask it to characterise the shared visual property.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from scipy.stats import kurtosis as scipy_kurtosis
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cifar_dataset import load_raw_cifar10, CIFAR10_LABELS
from run_experiment import load_cifar_model
from concept_extraction import get_activations


# ---------------------------------------------------------------------------
# Contrastive VLM prompt — Left vs Right
# ---------------------------------------------------------------------------
PROMPT = (
    "The image shows two groups of images side by side. "
    "The LEFT group shares a specific visual quality that the RIGHT group lacks. "
    "What visual quality does the LEFT group have that the RIGHT group lacks? "
    "Answer in one short phrase only — no explanation."
)


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------
def make_grid_image(images, nrow=4, cell_size=64, pad=4, bg=240):
    """
    images: list of (32,32,3) uint8 numpy arrays
    Returns a PIL Image of a grid.
    """
    n = len(images)
    ncol = (n + nrow - 1) // nrow
    nrow_actual = min(nrow, n)

    W = ncol * (cell_size + pad) + pad
    H = nrow_actual * (cell_size + pad) + pad
    canvas = Image.new("RGB", (W, H), color=(bg, bg, bg))

    for i, img_arr in enumerate(images):
        row = i // ncol
        col = i % ncol
        img = Image.fromarray(img_arr).resize((cell_size, cell_size), Image.NEAREST)
        x = pad + col * (cell_size + pad)
        y = pad + row * (cell_size + pad)
        canvas.paste(img, (x, y))

    return canvas


# ---------------------------------------------------------------------------
# VLM query
# ---------------------------------------------------------------------------
def ask_vlm(model, processor, pil_image, prompt):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    formatted = apply_chat_template(
        processor,
        model.config,
        prompt,
        num_images=1,
        add_generation_prompt=True,
    )

    output = generate(
        model,
        processor,
        formatted,
        image=pil_image,
        max_tokens=60,
        verbose=False,
    )
    text = output.text if hasattr(output, "text") else str(output)
    return text.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    out_dir = "figures/feature_geometry/vlm_names"
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Load model + data ─────────────────────────────────────────────────
    print("Loading CIFAR-10 test set and ConceptCNN model...")
    _, _, X_test, y_test = load_raw_cifar10()
    model_cnn, _ = load_cifar_model("A", 1)

    # ── 2. Extract fc1 features ───────────────────────────────────────────────
    print("Extracting fc1 activations (no labels used)...")
    acts = get_activations(model_cnn, X_test)
    feats = acts["fc1"].astype(np.float32)          # (10000, 256)

    # ── 3. Run ICA — pure kurtosis maximisation, no concept priors ───────────
    print("Running FastICA (n_components=15)...")
    scaler = StandardScaler()
    feats_w = scaler.fit_transform(feats)
    ica = FastICA(n_components=15, random_state=42, max_iter=2000, tol=0.001)
    S = ica.fit_transform(feats_w)                  # (10000, 15)

    # Sort components by |excess kurtosis| — highest non-Gaussianity first
    kurt_vals = np.array([scipy_kurtosis(S[:, i], fisher=True) for i in range(S.shape[1])])
    order = np.argsort(-np.abs(kurt_vals))
    print(f"\nICA kurtosis (sorted): {np.abs(kurt_vals)[order].round(2)}\n")

    # ── 4. Load VLM ───────────────────────────────────────────────────────────
    print("Loading Gemma 4 VLM...")
    from mlx_vlm import load
    vlm_model, processor = load("google/gemma-4-e2b-it")
    print("VLM ready.\n")

    # ── 5. For each top-10 ICA direction — build grid, ask VLM ───────────────
    results = []
    top_n = 12          # images per grid
    n_components = 10   # how many ICA components to name

    for rank, comp_idx in enumerate(order[:n_components]):
        k = float(kurt_vals[comp_idx])
        projections = S[:, comp_idx]

        # Top images along this direction (highest projection = most "concept-like")
        top_indices = np.argsort(-projections)[:top_n]
        top_images  = [(X_test[i] * 255).clip(0, 255).astype(np.uint8)
                       if X_test[i].dtype != np.uint8 else X_test[i]
                       for i in top_indices]
        top_classes = [CIFAR10_LABELS[int(y_test[i])] for i in top_indices]

        # Bottom images along this direction (lowest projection)
        bot_indices = np.argsort(projections)[:top_n]
        bot_images  = [(X_test[i] * 255).clip(0, 255).astype(np.uint8)
                       if X_test[i].dtype != np.uint8 else X_test[i]
                       for i in bot_indices]

        # Build grid: Left (top), Right (bottom)
        pad_between = 20
        left_grid = make_grid_image(top_images, nrow=3, cell_size=80)
        right_grid = make_grid_image(bot_images, nrow=3, cell_size=80)
        
        W = left_grid.width + pad_between + right_grid.width
        H = max(left_grid.height, right_grid.height)
        grid_img = Image.new("RGB", (W, H), color=(200, 200, 200))
        grid_img.paste(left_grid, (0, 0))
        grid_img.paste(right_grid, (left_grid.width + pad_between, 0))

        # Save grid for reference
        grid_path = os.path.join(out_dir, f"ica_{rank+1:02d}_k{abs(k):.1f}_grid.png")
        grid_img.save(grid_path)

        # Ask VLM — contrastive prompt
        print(f"ICA #{rank+1:2d} (k={k:.2f}) — querying VLM...", end=" ", flush=True)
        vlm_answer = ask_vlm(vlm_model, processor, grid_img, PROMPT)
        print(f"→ \"{vlm_answer}\"")

        class_counts = {}
        for c in top_classes:
            class_counts[c] = class_counts.get(c, 0) + 1
        top_class = max(class_counts, key=class_counts.get)

        results.append({
            "rank":       rank + 1,
            "ica_index":  int(comp_idx),
            "kurtosis":   round(k, 4),
            "top_class":  top_class,
            "class_mix":  class_counts,
            "vlm_name":   vlm_answer,
            "top_indices": top_indices.tolist(),
        })

    # ── 6. Summary figure ─────────────────────────────────────────────────────
    print("\nBuilding summary figure...")
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.suptitle("Unsupervised ICA Concepts — Named by VLM (Gemma 4)\n"
                 "No human labels used at any stage", fontsize=13, y=1.01)

    for ax, res in zip(axes.flat, results):
        comp_idx = res["ica_index"]
        projections = S[:, comp_idx]
        top_idx = np.argsort(-projections)[:top_n]
        bot_idx = np.argsort(projections)[:top_n]
        
        imgs_top = [(X_test[i] * 255).clip(0, 255).astype(np.uint8)
                    if X_test[i].dtype != np.uint8 else X_test[i]
                    for i in top_idx]
        imgs_bot = [(X_test[i] * 255).clip(0, 255).astype(np.uint8)
                    if X_test[i].dtype != np.uint8 else X_test[i]
                    for i in bot_idx]
                    
        left_g = make_grid_image(imgs_top, nrow=3, cell_size=32, pad=2)
        right_g = make_grid_image(imgs_bot, nrow=3, cell_size=32, pad=2)
        
        pw = 8
        gw = left_g.width + pw + right_g.width
        gh = max(left_g.height, right_g.height)
        grid = Image.new("RGB", (gw, gh), color=(200, 200, 200))
        grid.paste(left_g, (0, 0))
        grid.paste(right_g, (left_g.width + pw, 0))
        
        ax.imshow(np.array(grid))
        ax.axis("off")
        ax.set_title(
            f"ICA #{res['rank']} (k={res['kurtosis']:.1f})\n"
            f"\"{res['vlm_name']}\"",
            fontsize=7.5, wrap=True
        )

    plt.tight_layout()
    summary_path = os.path.join(out_dir, "ica_vlm_summary.png")
    plt.savefig(summary_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {summary_path}")

    # ── 7. Save JSON ──────────────────────────────────────────────────────────
    out_json = os.path.join("GRUE/results" if os.path.exists("GRUE/results") else "results",
                            "ica_vlm_names.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_json}")

    # ── 8. Print summary table ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'ICA #':<6} {'Kurtosis':<10} {'Top class':<14} {'VLM name'}")
    print("-" * 65)
    for r in results:
        print(f"#{r['rank']:<5} {r['kurtosis']:<10.2f} {r['top_class']:<14} {r['vlm_name']}")


if __name__ == "__main__":
    main()
