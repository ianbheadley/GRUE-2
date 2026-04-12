"""
Joint experiment analysis: compositionality, domain separation, and transfer probes.

The key output for manual validation is a set of retrieval grid images:
    figures/joint/retrieved_red_truck_top20.png
    figures/joint/retrieved_blue_truck_top20.png
    figures/joint/retrieved_red_automobile_top20.png
    ... etc.

You review these grids and tell us how many look correct. That is the
ground truth for the compositionality result.

Usage:
    # Full pipeline (train + analyse)
    python run_joint_experiment.py --scheme JA --seeds 1 2 3 4 5

    # Skip training if models already exist
    python run_joint_experiment.py --scheme JA --seeds 1 2 3 --skip_training

    # Just save the validation grids for a trained model
    python run_joint_experiment.py --scheme JA --seeds 1 --skip_training --only_grids
"""

import argparse
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from joint_model import JointCNN
from joint_dataset import (
    load_joint_data,
    get_color_object_pairs,
    CIFAR10_LABELS,
    JOINT_SCHEMES,
    label_cifar_by_color,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_joint_model(scheme, seed, base_dir="models_joint"):
    model_dir    = os.path.join(base_dir, f"scheme_{scheme}", f"seed_{seed}")
    meta_path    = os.path.join(model_dir, "metadata.json")
    weights_path = os.path.join(model_dir, "weights.safetensors")

    with open(meta_path) as f:
        meta = json.load(f)

    model = JointCNN(
        num_color_classes=meta["num_color_classes"],
        num_object_classes=meta["num_object_classes"],
        input_size=32,
    )
    weights = mx.load(weights_path)
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    model.eval()
    return model, meta


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(model, X, batch_size=256):
    """Return fc1 features (N, 256) for a batch of images."""
    model.eval()
    feats = []
    for i in range(0, len(X), batch_size):
        xb = mx.array(X[i:i + batch_size].astype(np.float32))
        f  = model.encode(xb, capture_activations=False)
        feats.append(np.array(f))
    return np.concatenate(feats, axis=0)


def extract_all_layers(model, X, batch_size=256):
    """Return {layer_name: (N, d)} dict for all trunk layers."""
    model.eval()
    all_acts = None
    for i in range(0, len(X), batch_size):
        xb = mx.array(X[i:i + batch_size].astype(np.float32))
        _, acts = model.encode(xb, capture_activations=True)
        acts_np = {k: np.array(v) for k, v in acts.items()}
        # Pool spatial dims for conv layers
        for k in acts_np:
            if acts_np[k].ndim == 4:
                acts_np[k] = acts_np[k].mean(axis=(1, 2))
        if all_acts is None:
            all_acts = {k: [v] for k, v in acts_np.items()}
        else:
            for k, v in acts_np.items():
                all_acts[k].append(v)
    return {k: np.concatenate(v, axis=0) for k, v in all_acts.items()}


# ---------------------------------------------------------------------------
# Compositionality probe
# ---------------------------------------------------------------------------

def compute_direction(feats_pos, feats_neg):
    """Mean difference vector, L2-normalised."""
    d = feats_pos.mean(axis=0) - feats_neg.mean(axis=0)
    return d / (np.linalg.norm(d) + 1e-8)


def compositionality_probe(
    model, data, colors, objects,
    top_k=20, min_confidence=0.30, out_dir="figures/joint",
    gt_labels=None,
):
    """
    For every (color, object) pair:
      1. Compute d_color from color-block features
      2. Compute d_object from CIFAR features
      3. Form composition = d_color + d_object
      4. Rank ALL CIFAR test images by cosine similarity to composition
      5. Take top-k, save as a grid PNG for manual review
      6. Compute precision — uses SAM ground-truth labels when available,
         falls back to noisy dominant-hue auto-labeling otherwise.

    Args:
        gt_labels: dict from load_vehicle_labels() — {int_idx: {class, color}}.
                   When provided, precision is exact (SAM ground truth).
                   When None, falls back to dominant-hue heuristic (noisy).

    Returns a results dict.
    """
    os.makedirs(out_dir, exist_ok=True)

    X_test   = data["cifar_test_raw"]
    y_test   = data["cifar_y_test_raw"]
    X_color  = data["color_train"]
    y_color  = data["color_y_train"]
    color_l2i  = data["color_label_to_idx"]
    object_l2i = data["cifar_label_to_idx"]

    # Build idx→label lookup that works for both CIFAR-10 and CIFAR-100
    idx2label = {v: k for k, v in object_l2i.items()}

    # Pre-compute features for all CIFAR test images (reused across queries)
    print("  Extracting CIFAR test features...")
    cifar_test_feats = extract_features(model, X_test)       # (N_test, 256)

    # Pre-compute features for color-block training images
    print("  Extracting color-block features...")
    color_feats = extract_features(model, X_color)            # (N_color, 256)

    results = {}

    for color in colors:
        # Color direction: color-block images of this color vs. all others
        if color not in color_l2i:
            print(f"  WARNING: color '{color}' not in color label map, skipping")
            continue

        cidx     = color_l2i[color]
        pos_mask = (y_color == cidx)
        neg_mask = ~pos_mask

        if pos_mask.sum() < 10 or neg_mask.sum() < 10:
            print(f"  WARNING: too few '{color}' color-block examples")
            continue

        d_color = compute_direction(color_feats[pos_mask], color_feats[neg_mask])

        # Use data's label map so CIFAR-100 class names work too
        object_l2i_all = data.get("cifar_label_to_idx", {})
        all_labels = list(object_l2i_all.keys()) if object_l2i_all else CIFAR10_LABELS

        for obj in objects:
            if obj not in object_l2i_all and obj not in CIFAR10_LABELS:
                print(f"  WARNING: CIFAR class '{obj}' unknown, skipping")
                continue

            if obj in object_l2i_all:
                obj_idx = object_l2i_all[obj]
            else:
                obj_idx = CIFAR10_LABELS.index(obj)

            obj_mask   = (y_test == obj_idx)
            non_obj_mask = ~obj_mask

            if obj_mask.sum() < 10:
                print(f"  WARNING: too few '{obj}' test images")
                continue

            # Object direction from CIFAR test features
            d_object = compute_direction(
                cifar_test_feats[obj_mask],
                cifar_test_feats[non_obj_mask],
            )

            # Composition: additive
            composition = d_color + d_object
            composition = composition / (np.linalg.norm(composition) + 1e-8)

            # Rank ALL CIFAR test images by cosine similarity to composition
            sims  = cifar_test_feats @ composition           # (N_test,)
            order = np.argsort(-sims)                        # descending
            top_indices = order[:top_k]

            # ---------------------------------------------------------------
            # Save retrieval grid — this is what the user reviews manually
            # ---------------------------------------------------------------
            _save_retrieval_grid(
                X_test, y_test, top_indices, sims,
                color, obj, top_k, out_dir,
                idx2label=idx2label,
            )

            # ---------------------------------------------------------------
            # Precision — SAM ground truth if available, else dominant-hue
            # ---------------------------------------------------------------
            if gt_labels is not None:
                # Exact precision: SAM labeled object + color match
                correct = 0
                for idx in top_indices:
                    entry = gt_labels.get(int(idx))
                    if entry and entry["class"] == obj and entry["color"] == color:
                        correct += 1
                precision = correct / max(len(top_indices), 1)

                # Baseline: fraction of ALL gt-labeled (obj, color) in test set
                total_gt = sum(
                    1 for e in gt_labels.values()
                    if e["class"] == obj and e["color"] == color
                )
                baseline_rate = total_gt / max(len(X_test), 1)
                precision_note = "SAM ground truth"

            else:
                # Fallback: noisy dominant-hue labeling
                correct = 0
                for idx in top_indices:
                    true_class = idx2label.get(int(y_test[idx]), "")
                    dom_color, conf = _dominant_hue_label_full_import(X_test[idx])
                    if true_class == obj and dom_color == color and conf >= min_confidence:
                        correct += 1

                sample_n = min(1000, len(X_test))
                baseline_correct = 0
                for idx in np.random.choice(len(X_test), sample_n, replace=False):
                    if idx2label.get(int(y_test[idx]), "") != obj:
                        continue
                    dom_color, conf = _dominant_hue_label_full_import(X_test[idx])
                    if dom_color == color and conf >= min_confidence:
                        baseline_correct += 1
                baseline_rate = baseline_correct / sample_n
                precision = correct / max(len(top_indices), 1)
                precision_note = "dominant-hue heuristic (noisy)"

            lift = precision / (baseline_rate + 1e-6)

            key = f"{color}_{obj}"
            results[key] = {
                "color":          color,
                "object":         obj,
                "top_k":          top_k,
                "precision":      float(precision),
                "baseline_rate":  float(baseline_rate),
                "lift":           float(lift),
                "precision_note": precision_note,
                "grid_path":      os.path.join(out_dir, f"retrieved_{color}_{obj}_top{top_k}.png"),
            }

            print(
                f"  {color:8s} + {obj:12s}: "
                f"precision={precision:.2f}  baseline={baseline_rate:.3f}  lift={lift:.1f}x"
                f"  [{precision_note[:3]}]"
                f"  → {results[key]['grid_path']}"
            )

    return results


def _dominant_hue_label_full_import(img):
    """Lazy import wrapper so the top of this file stays clean."""
    from joint_dataset import dominant_hue_label_full
    return dominant_hue_label_full(img)


def _save_retrieval_grid(X_test, y_test, indices, sims, color, obj, top_k, out_dir,
                         idx2label=None):
    """
    Save a grid of retrieved images with their CIFAR class label and similarity score.
    The user reads this grid and judges whether the compositionality worked.
    """
    ncols = 5
    nrows = (top_k + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2.4))
    fig.suptitle(
        f'Query: "{color}" + "{obj}" — top {top_k} retrievals\n'
        f'Review: does the model retrieve {color} {obj}s?',
        fontsize=10, y=1.01,
    )

    for ax_idx, (ax, img_idx) in enumerate(zip(axes.flat, indices)):
        img       = X_test[img_idx]
        raw_idx   = int(y_test[img_idx])
        if idx2label:
            cls_name = idx2label.get(raw_idx, str(raw_idx))
        elif raw_idx < len(CIFAR10_LABELS):
            cls_name = CIFAR10_LABELS[raw_idx]
        else:
            cls_name = str(raw_idx)
        sim_score = float(sims[img_idx])

        # Display image (scale to [0,1] if needed)
        disp = img.astype(np.float32)
        if disp.max() > 1.0:
            disp = disp / 255.0
        ax.imshow(disp)
        ax.set_title(f"{cls_name}\nsim={sim_score:.3f}", fontsize=7)
        ax.axis("off")

        # Green border if CIFAR class matches, red otherwise
        border_color = "green" if cls_name == obj else "red"
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2)
            spine.set_visible(True)

    # Hide unused axes
    for ax in axes.flat[len(indices):]:
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, f"retrieved_{color}_{obj}_top{top_k}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Domain separation probe
# ---------------------------------------------------------------------------

def domain_separation_probe(model, data, n_per_domain=500):
    """
    Can a linear probe tell color-block images from CIFAR images in fc1 space?

    High accuracy = trunk separated by domain (bad for compositionality).
    Low accuracy  = trunk mixed concepts regardless of domain (good).
    """
    X_color = data["color_val"][:n_per_domain]
    X_cifar = data["cifar_val"][:n_per_domain]

    acts_color = extract_all_layers(model, X_color)
    acts_cifar = extract_all_layers(model, X_cifar)

    results = {}
    for layer in acts_color:
        fc = acts_color[layer]
        fo = acts_cifar[layer]
        X_probe = np.concatenate([fc, fo], axis=0)
        y_probe = np.array([0] * len(fc) + [1] * len(fo))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_probe)

        clf    = LogisticRegression(max_iter=500)
        scores = cross_val_score(clf, X_scaled, y_probe, cv=5, scoring="accuracy")
        results[layer] = {
            "accuracy": float(scores.mean()),
            "std":      float(scores.std()),
        }
        print(f"    domain probe {layer:8s}: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

    return results


# ---------------------------------------------------------------------------
# Color organisation probe (do color-block features cluster by color?)
# ---------------------------------------------------------------------------

def color_organisation_probe(model, data, n_components=2, out_dir="figures/joint"):
    """
    PCA of color-block fc1 features coloured by class.
    If the trunk learned color structure, clusters should be visually distinct.
    """
    os.makedirs(out_dir, exist_ok=True)

    X_color = data["color_val"]
    y_color = data["color_y_val"]
    l2i     = data["color_label_to_idx"]
    idx2l   = {v: k for k, v in l2i.items()}

    feats = extract_features(model, X_color)

    pca   = PCA(n_components=2)
    proj  = pca.fit_transform(feats)

    # Linear probe per class
    clf    = LogisticRegression(max_iter=1000)
    scaler = StandardScaler()
    scores = cross_val_score(clf, scaler.fit_transform(feats), y_color, cv=5, scoring="accuracy")

    # Map color label names to actual display colors
    COLOR_NAME_MAP = {
        "red":    "#e53935",
        "orange": "#fb8c00",
        "yellow": "#fdd835",
        "green":  "#43a047",
        "blue":   "#1e88e5",
        "purple": "#8e24aa",
        "pink":   "#e91e8c",
        "brown":  "#6d4c41",
        "black":  "#212121",
        "white":  "#bdbdbd",   # light grey so it's visible on white bg
        "gray":   "#757575",
        "grey":   "#757575",
        "cyan":   "#00acc1",
    }
    fallback = plt.colormaps.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(8, 6))

    for cls_idx in sorted(set(y_color)):
        label_name = idx2l[cls_idx]
        dot_color  = COLOR_NAME_MAP.get(label_name.lower(), fallback(cls_idx))
        mask = (y_color == cls_idx)
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            label=label_name, alpha=0.5, s=10,
            color=dot_color,
            edgecolors="none",
        )

    ax.set_title(
        f"Color-block fc1 features (PCA)\n"
        f"Linear probe accuracy: {scores.mean()*100:.1f}%",
        fontsize=11,
    )
    ax.legend(fontsize=7, markerscale=2, loc="best")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()

    path = os.path.join(out_dir, "color_organisation_pca.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

    return {
        "linear_probe_accuracy": float(scores.mean()),
        "linear_probe_std":      float(scores.std()),
        "pca_variance_explained": pca.explained_variance_ratio_.tolist(),
    }


# ---------------------------------------------------------------------------
# Cross-dataset transfer probe (JC: can CIFAR teach the model red?)
# ---------------------------------------------------------------------------

def transfer_probe(model_ja, model_jc, data_ja, data_jc, target_color="red"):
    """
    Train a red/not-red probe on JA color-block fc1 features.
    Apply the same probe to JC color-block features.

    If JC accuracy is comparable to JA, CIFAR transferred the suppressed concept.
    """
    color_l2i_ja = data_ja["color_label_to_idx"]
    color_l2i_jc = data_jc["color_label_to_idx"]

    if target_color not in color_l2i_ja:
        print(f"  '{target_color}' not in JA color labels — skipping transfer probe")
        return {}

    # Val images from JA (red is present in val for both JA and JC)
    X_val_ja = data_ja["color_val"]
    y_val_ja = data_ja["color_y_val"]
    X_val_jc = data_jc["color_val"]

    # Binary labels: is this image the target color?
    cidx_ja = color_l2i_ja[target_color]
    y_binary_ja = (y_val_ja == cidx_ja).astype(int)

    # Features from JA model and JC model (both evaluated on same images from JA val)
    feats_ja = extract_features(model_ja, X_val_ja)
    feats_jc = extract_features(model_jc, X_val_jc)

    scaler = StandardScaler().fit(feats_ja)
    feats_ja_s = scaler.transform(feats_ja)
    feats_jc_s = scaler.transform(feats_jc)

    clf = LogisticRegression(max_iter=1000)

    # JA probe accuracy (upper bound — model was trained with red labels)
    scores_ja = cross_val_score(clf, feats_ja_s, y_binary_ja, cv=5, scoring="accuracy")

    # JC probe accuracy (can the model still decode red despite no red color supervision?)
    scores_jc = cross_val_score(clf, feats_jc_s, y_binary_ja, cv=5, scoring="accuracy")

    results = {
        "target_color":   target_color,
        "ja_probe_acc":   float(scores_ja.mean()),
        "jc_probe_acc":   float(scores_jc.mean()),
        "transfer_ratio": float(scores_jc.mean() / (scores_ja.mean() + 1e-8)),
        "interpretation": (
            "JC ≈ JA → CIFAR transferred the concept despite color suppression. "
            "JC << JA → concept was successfully erased."
        ),
    }

    print(
        f"  Transfer probe ({target_color}): "
        f"JA={results['ja_probe_acc']*100:.1f}%  "
        f"JC={results['jc_probe_acc']*100:.1f}%  "
        f"ratio={results['transfer_ratio']:.2f}"
    )
    return results


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme",    type=str, nargs="+", default=["JA"],
                        choices=list(JOINT_SCHEMES))
    parser.add_argument("--seeds",     type=int, nargs="+", default=[1])
    parser.add_argument("--epochs",    type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_dir", type=str, default="models_joint")
    parser.add_argument("--color_dir", type=str, default="dataset")
    parser.add_argument("--figures_dir", type=str, default="figures/joint")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--only_grids",    action="store_true",
                        help="Only save retrieval grids, skip all other probes")
    # What to query in the compositionality probe
    parser.add_argument("--query_colors",  nargs="+",
                        default=["red", "blue", "green", "yellow"])
    parser.add_argument("--query_objects", nargs="+",
                        default=["truck", "automobile", "airplane", "ship"])
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--gt_labels", type=str, default=None,
                        help="Path to SAM vehicle color labels JSON (from cifar_color_labels.py). "
                             "When provided, precision uses ground truth instead of dominant-hue.")
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    # Load SAM ground-truth labels if provided
    gt_labels = None
    if args.gt_labels:
        from cifar_color_labels import load_vehicle_labels
        gt_labels = load_vehicle_labels(args.gt_labels)
        n_colored = sum(1 for e in gt_labels.values() if e["color"] is not None)
        print(f"Loaded SAM labels: {len(gt_labels)} vehicles, {n_colored} with color")

    all_results = {}

    for scheme in args.scheme:
        print(f"\n{'='*60}")
        print(f"SCHEME {scheme}: {JOINT_SCHEMES[scheme]['description']}")
        print('='*60)

        # ------------------------------------------------------------------
        # Train
        # ------------------------------------------------------------------
        if not args.skip_training:
            python = sys.executable
            for seed in args.seeds:
                subprocess.run([
                    python, "train_joint.py",
                    "--scheme", scheme,
                    "--seed",   str(seed),
                    "--epochs", str(args.epochs),
                    "--batch_size", str(args.batch_size),
                    "--base_model_dir", args.model_dir,
                    "--color_base_dir", args.color_dir,
                ], check=True)

        # ------------------------------------------------------------------
        # Load seed-1 model for analysis (extend to multi-seed average later)
        # ------------------------------------------------------------------
        seed = args.seeds[0]
        print(f"\nLoading {scheme} seed {seed}...")
        try:
            model, meta = load_joint_model(scheme, seed, base_dir=args.model_dir)
        except FileNotFoundError:
            print(f"  Model not found — run without --skip_training first")
            continue

        print(f"  color_acc={meta['final_color_acc']:.4f}  object_acc={meta['final_object_acc']:.4f}")

        # ------------------------------------------------------------------
        # Load data
        # ------------------------------------------------------------------
        data = load_joint_data(scheme, color_base_dir=args.color_dir)

        scheme_results = {"meta": meta}

        # ------------------------------------------------------------------
        # 1. Compositionality retrieval grids  ← primary output for manual review
        # ------------------------------------------------------------------
        print(f"\n[1] Compositionality probe (query grids saved to {args.figures_dir})")
        comp_results = compositionality_probe(
            model, data,
            colors=args.query_colors,
            objects=args.query_objects,
            top_k=args.top_k,
            out_dir=args.figures_dir,
            gt_labels=gt_labels,
        )
        scheme_results["compositionality"] = comp_results

        if args.only_grids:
            all_results[scheme] = scheme_results
            continue

        # ------------------------------------------------------------------
        # 2. Domain separation — does trunk mix color+object or split by dataset?
        # ------------------------------------------------------------------
        print(f"\n[2] Domain separation probe")
        dom_results = domain_separation_probe(model, data)
        scheme_results["domain_separation"] = dom_results

        # ------------------------------------------------------------------
        # 3. Color organisation — do color-block features cluster by color?
        # ------------------------------------------------------------------
        print(f"\n[3] Color organisation (PCA)")
        org_results = color_organisation_probe(model, data, out_dir=args.figures_dir)
        scheme_results["color_organisation"] = org_results

        # ------------------------------------------------------------------
        # 4. Transfer probe (JC vs JA) — only meaningful when both are present
        # ------------------------------------------------------------------
        if scheme == "JC" and "JA" in all_results:
            print(f"\n[4] Cross-dataset transfer probe (JC vs JA)")
            try:
                model_ja, _ = load_joint_model("JA", seed, base_dir=args.model_dir)
                data_ja     = load_joint_data("JA", color_base_dir=args.color_dir)
                transfer    = transfer_probe(model_ja, model, data_ja, data)
                scheme_results["transfer_probe"] = transfer
            except FileNotFoundError:
                print("  JA model not found — skipping transfer probe")

        all_results[scheme] = scheme_results

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    out_path = os.path.join("results", "joint_experiment_results.json")
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for scheme, res in all_results.items():
        print(f"\n{scheme}: {JOINT_SCHEMES[scheme]['description']}")
        if "compositionality" in res:
            for key, cr in res["compositionality"].items():
                print(
                    f"  {key:20s}  lift={cr['lift']:.1f}x  "
                    f"prec={cr['precision']:.2f}  "
                    f"[{cr.get('precision_note','?')[:3]}]  "
                    f"(grid: {os.path.basename(cr['grid_path'])})"
                )

    print(f"\nReview the grids in {args.figures_dir}/ and report back which look correct.")


if __name__ == "__main__":
    main()
