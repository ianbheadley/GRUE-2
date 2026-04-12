"""
Concept Discovery Experiment — Sky vs Ocean

Tests whether a GRUE-trained model has learned latent structure for concepts
that were NEVER in its training labels. We use SAM to generate hidden ground-
truth labels for "sky" and "ocean", then probe the model's fc1 features.

Key questions:
  1. Is there a "sky direction" in the latent space?
  2. Is there an "ocean direction"?
  3. Are they the same direction (both blue+flat) or genuinely separated?
  4. How few labeled examples do we need to recover each direction?
  5. Which CIFAR classes "contain" sky / ocean — does this match human intuition?

Usage:
    python src/concept_discovery_experiment.py --run_all
    python src/concept_discovery_experiment.py --skip_labeling  # if labels cached
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_all_features(model, X, batch_size=256):
    """Extract fc1 features for all images. Returns (N, 256) float32 numpy."""
    from concept_extraction import get_activations
    acts = get_activations(model, X, batch_size=batch_size)
    # ConceptCNN captures fc1 as "fc1" in its activation dict
    key = "fc1" if "fc1" in acts else sorted(acts.keys())[-1]
    return acts[key].astype(np.float32)


def compute_direction(pos_feats, neg_feats):
    """Mean-difference direction, L2-normalised."""
    d = pos_feats.mean(axis=0) - neg_feats.mean(axis=0)
    return d / (np.linalg.norm(d) + 1e-8)


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def load_concept_labels(json_path):
    """Load SAM concept labels. Returns (pos_indices, neg_indices, all_entries)."""
    with open(json_path) as f:
        entries = json.load(f)
    pos = [e["idx"] for e in entries if e["has_concept"]]
    neg = [e["idx"] for e in entries if not e["has_concept"]]
    return pos, neg, entries


# ---------------------------------------------------------------------------
# Few-shot probe
# ---------------------------------------------------------------------------

def few_shot_probe(feats, pos_indices, neg_indices,
                   n_shots_list=(5, 10, 20, 50, 100),
                   n_trials=20, random_seed=0):
    """
    For each n_shot, sample n_shot pos + n_shot neg, fit a logistic probe,
    test on the held-out labeled set. Repeat n_trials times.

    Returns dict: {n_shot: {"mean": float, "std": float}}
    """
    rng = np.random.default_rng(random_seed)
    results = {}

    for n in n_shots_list:
        if n > min(len(pos_indices), len(neg_indices)):
            continue
        accs = []
        for _ in range(n_trials):
            p_train = rng.choice(pos_indices, n, replace=False)
            n_train = rng.choice(neg_indices, n, replace=False)
            p_test  = [i for i in pos_indices if i not in set(p_train)]
            n_test  = [i for i in neg_indices if i not in set(n_train)]

            if len(p_test) < 5 or len(n_test) < 5:
                continue

            X_tr = np.concatenate([feats[p_train], feats[n_train]])
            y_tr = np.array([1] * n + [0] * n)
            X_te = np.concatenate([feats[p_test], feats[n_test]])
            y_te = np.array([1] * len(p_test) + [0] * len(n_test))

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(X_tr_s, y_tr)
            accs.append(clf.score(X_te_s, y_te))

        if accs:
            results[n] = {"mean": float(np.mean(accs)), "std": float(np.std(accs))}

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_few_shot_curve(probe_results_dict, out_path, title="Few-shot concept probe"):
    """
    probe_results_dict: {"sky": {5: {...}, 10: {...}}, "ocean": {...}}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"sky": "#1e88e5", "ocean": "#43a047"}
    markers = {"sky": "o", "ocean": "s"}

    for concept, results in probe_results_dict.items():
        ns   = sorted(results.keys())
        means = [results[n]["mean"] * 100 for n in ns]
        stds  = [results[n]["std"]  * 100 for n in ns]
        c = colors.get(concept, "#e53935")
        ax.errorbar(ns, means, yerr=stds, marker=markers.get(concept, "^"),
                    label=concept, color=c, linewidth=2, capsize=4, markersize=7)

    ax.axhline(50, color="#aaaaaa", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel("N labeled examples (per class)")
    ax.set_ylabel("Probe accuracy (%)")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(40, 100)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_concept_scores_by_class(scores, y_labels, class_names, concept_name,
                                 out_path):
    """
    Box plot of concept direction projection scores, grouped by CIFAR class.
    Shows which classes "contain" the concept most.
    """
    classes = sorted(set(y_labels))
    medians = [(np.median(scores[y_labels == c]), c) for c in classes]
    medians.sort(reverse=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    data    = [scores[y_labels == c] for _, c in medians]
    labels  = [class_names[c] for _, c in medians]

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops={"color": "white", "linewidth": 2})
    for patch in bp["boxes"]:
        patch.set_facecolor("#1e88e5" if concept_name == "sky" else "#43a047")
        patch.set_alpha(0.7)

    ax.set_title(f"'{concept_name}' direction projection score by CIFAR class\n"
                 f"(higher = more {concept_name}-like in latent space)", fontsize=12)
    ax.set_ylabel("Projection score")
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(0, color="#aaaaaa", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def save_retrieval_grid(X, y_test, ranked_indices, scores, concept_name,
                        class_names, top_k=25, out_dir="figures/concept_discovery"):
    """Save top-k retrieved images as annotated grid PNG."""
    os.makedirs(out_dir, exist_ok=True)
    ncols = 5
    nrows = (top_k + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2.4))
    fig.suptitle(f"Top-{top_k} images ranked by '{concept_name}' concept direction\n"
                 f"(review: do these actually contain {concept_name}?)",
                 fontsize=11, y=1.01)

    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= len(ranked_indices):
            continue
        idx = ranked_indices[i]
        img = X[idx]
        cls = class_names[int(y_test[idx])] if int(y_test[idx]) < len(class_names) else str(y_test[idx])
        sim = scores[i]

        ax.imshow(img)
        ax.set_title(f"{cls}\n{sim:.3f}", fontsize=7)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)
            spine.set_edgecolor("#1e88e5" if concept_name == "sky" else "#43a047")

    plt.tight_layout()
    path = os.path.join(out_dir, f"retrieved_{concept_name}_top{top_k}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args):
    from cifar_dataset import load_raw_cifar10, CIFAR10_LABELS
    from run_experiment import load_cifar_model
    from sam_concept_labeler import label_images_for_concept

    os.makedirs(args.out_dir, exist_ok=True)
    _, _, X_test, y_test = load_raw_cifar10()

    # ── 1. SAM labeling ──────────────────────────────────────────────────────
    concept_labels = {}
    for concept in ["sky", "ocean"]:
        label_path = os.path.join("results", f"{concept}_labels.json")
        if args.skip_labeling and os.path.exists(label_path):
            print(f"Loading cached {concept} labels from {label_path}")
        else:
            print(f"\n{'='*50}")
            print(f"SAM labeling: {concept}")
            print('='*50)
            label_images_for_concept(
                X_test, y_test, concept, label_path,
                class_names=CIFAR10_LABELS,
                sam_dir=args.sam_dir,
            )
        pos, neg, entries = load_concept_labels(label_path)
        concept_labels[concept] = {"pos": pos, "neg": neg, "entries": entries}
        n_total = len(entries)
        print(f"  {concept}: {len(pos)} positive ({100*len(pos)/n_total:.1f}%), "
              f"{len(neg)} negative")

    # ── 2. Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading CIFAR-10 model (scheme={args.scheme}, seed={args.seed})...")
    model, meta = load_cifar_model(args.scheme, args.seed)
    print(f"  Val accuracy: {meta.get('best_val_accuracy', 'N/A')}")

    # ── 3. Extract features ──────────────────────────────────────────────────
    print("\nExtracting fc1 features for all 10k test images...")
    feats = extract_all_features(model, X_test)   # (10000, 256)
    print(f"  Features shape: {feats.shape}")

    # ── 4. Compute concept directions ────────────────────────────────────────
    directions = {}
    all_results = {}

    for concept in ["sky", "ocean"]:
        pos = concept_labels[concept]["pos"]
        neg = concept_labels[concept]["neg"]

        if len(pos) < 10:
            print(f"  WARNING: too few {concept} positives ({len(pos)}), skipping")
            continue

        d = compute_direction(feats[pos], feats[neg])
        directions[concept] = d
        print(f"\n  '{concept}' direction: {len(pos)} pos, {len(neg)} neg")
        print(f"    Direction norm: {np.linalg.norm(d):.4f}")

    # ── 5. Sky vs Ocean separation ───────────────────────────────────────────
    if "sky" in directions and "ocean" in directions:
        cosine = float(directions["sky"] @ directions["ocean"])
        print(f"\n{'='*50}")
        print(f"Sky vs Ocean cosine similarity: {cosine:.4f}")
        if cosine > 0.7:
            print("  → Very similar directions — model doesn't distinguish sky from ocean")
        elif cosine > 0.3:
            print("  → Partially overlapping — some shared structure (both blue/flat)")
        else:
            print("  → Well separated — model has genuinely distinct sky and ocean representations")
        all_results["sky_ocean_cosine"] = cosine

    # ── 6. Few-shot probe ────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("Few-shot probe (how many labels needed to recover direction?)")
    n_shots_list = [n for n in [5, 10, 20, 50, 100]
                    if n <= min(
                        len(concept_labels.get("sky", {}).get("pos", [])),
                        len(concept_labels.get("ocean", {}).get("pos", []))
                    )]

    probe_results = {}
    for concept in ["sky", "ocean"]:
        if concept not in directions:
            continue
        pos = concept_labels[concept]["pos"]
        neg = concept_labels[concept]["neg"]
        print(f"\n  Probing '{concept}' ({len(pos)} pos, {len(neg)} neg)...")
        result = few_shot_probe(feats, pos, neg, n_shots_list=n_shots_list)
        probe_results[concept] = result
        for n, r in sorted(result.items()):
            print(f"    {n:4d}-shot: {r['mean']*100:.1f}% ± {r['std']*100:.1f}%")

    all_results["few_shot_probe"] = probe_results

    if probe_results:
        plot_few_shot_curve(
            probe_results,
            os.path.join(args.out_dir, "few_shot_curve.png"),
            title=f"Sky & Ocean concept probe — GRUE scheme {args.scheme}",
        )

    # ── 7. Projection scores by class ────────────────────────────────────────
    print(f"\n{'='*50}")
    print("Concept projection scores by CIFAR class:")
    class_scores = {}

    for concept, d in directions.items():
        scores = feats @ d   # (10000,) scalar projection per image
        class_scores[concept] = scores

        print(f"\n  '{concept}' — top 5 classes by median score:")
        class_medians = [(np.median(scores[y_test == c]), CIFAR10_LABELS[c])
                         for c in range(10)]
        for score, cls in sorted(class_medians, reverse=True)[:5]:
            n = (y_test == CIFAR10_LABELS.index(cls)).sum()
            print(f"    {cls:12s}: {score:.4f}  (n={n})")

        all_results[f"{concept}_class_scores"] = {
            cls: float(np.median(scores[y_test == i]))
            for i, cls in enumerate(CIFAR10_LABELS)
        }

        plot_concept_scores_by_class(
            scores, y_test, CIFAR10_LABELS, concept,
            os.path.join(args.out_dir, f"{concept}_scores_by_class.png"),
        )

    # ── 8. Retrieval grids ──────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("Saving retrieval grids for manual review...")
    for concept, d in directions.items():
        scores = feats @ d
        order  = np.argsort(-scores)
        save_retrieval_grid(
            X_test, y_test, order[:args.top_k], scores[order[:args.top_k]],
            concept, CIFAR10_LABELS, top_k=args.top_k, out_dir=args.out_dir,
        )

    # ── 9. SAM validation grids ───────────────────────────────────────────────
    for concept in ["sky", "ocean"]:
        if concept not in concept_labels:
            continue
        pos = concept_labels[concept]["pos"]
        entries = {e["idx"]: e for e in concept_labels[concept]["entries"]}
        # Top-25 by SAM confidence
        top_conf = sorted(pos, key=lambda i: entries[i]["confidence"], reverse=True)[:25]
        save_retrieval_grid(
            X_test, y_test, top_conf,
            [entries[i]["confidence"] for i in top_conf],
            f"{concept}_sam_positive", CIFAR10_LABELS,
            top_k=25, out_dir=args.out_dir,
        )

    # ── 10. Save results ──────────────────────────────────────────────────────
    out_path = os.path.join("results", "concept_discovery_sky_ocean.json")
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    print(f"\n{'='*50}")
    print("REVIEW THESE FILES:")
    print(f"  figures/concept_discovery/retrieved_sky_top{args.top_k}.png   ← does this look like sky?")
    print(f"  figures/concept_discovery/retrieved_ocean_top{args.top_k}.png ← does this look like ocean?")
    print(f"  figures/concept_discovery/sky_sam_positive_top25.png          ← SAM sky labels quality")
    print(f"  figures/concept_discovery/ocean_sam_positive_top25.png        ← SAM ocean labels quality")
    print(f"  figures/concept_discovery/few_shot_curve.png                  ← accuracy vs N labels")
    print(f"  figures/concept_discovery/sky_scores_by_class.png             ← which classes have sky?")
    print(f"  figures/concept_discovery/ocean_scores_by_class.png           ← which classes have ocean?")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme",        default="A",
                        help="GRUE CIFAR scheme to probe (A/B/C)")
    parser.add_argument("--seed",          default=1, type=int)
    parser.add_argument("--sam_dir",       default="weights/sam-vit-base")
    parser.add_argument("--out_dir",       default="figures/concept_discovery")
    parser.add_argument("--skip_labeling", action="store_true",
                        help="Skip SAM labeling, use cached JSONs")
    parser.add_argument("--top_k",         default=25, type=int)
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
