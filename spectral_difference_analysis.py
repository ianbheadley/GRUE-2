import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

from cifar_dataset import CIFAR10_LABELS, load_raw_cifar10
from concept_extraction import get_activations, global_avg_pool
from evaluate import load_data
from load_models import load_model
from run_experiment import load_cifar_model


LAYERS = ["conv1", "conv2", "conv3", "fc1"]


def orthogonal_align_to_anchor(anchor, features):
    """Align features into the anchor basis with an orthogonal map."""
    anchor_centered = anchor - anchor.mean(axis=0, keepdims=True)
    features_centered = features - features.mean(axis=0, keepdims=True)
    cross_cov = features_centered.T @ anchor_centered
    u, _, vt = np.linalg.svd(cross_cov, full_matrices=False)
    rotation = u @ vt
    return features @ rotation


def collect_features(model, X):
    acts = get_activations(model, X)
    return {layer: global_avg_pool(acts[layer]) for layer in LAYERS}


def effective_rank(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[values > 1e-12]
    if len(values) == 0:
        return 0.0
    probs = values / values.sum()
    entropy = -(probs * np.log(probs + 1e-12)).sum()
    return float(np.exp(entropy))


def pairwise_cosine(vectors):
    if len(vectors) < 2:
        return 0.0
    sims = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            vi = vectors[i].reshape(-1)
            vj = vectors[j].reshape(-1)
            denom = (np.linalg.norm(vi) * np.linalg.norm(vj)) + 1e-8
            sims.append(float(np.dot(vi, vj) / denom))
    return float(np.mean(sims))


def spectral_metrics(signal_cov, noise_cov, class_direction, diff_mats, mean_shifts):
    dim = signal_cov.shape[0]
    noise_scale = float(np.trace(noise_cov) / max(dim, 1))
    reg = max(noise_scale * 1e-6, 1e-6)
    noise_reg = noise_cov + reg * np.eye(dim)

    noise_vals, noise_vecs = np.linalg.eigh((noise_reg + noise_reg.T) / 2.0)
    noise_vals = np.clip(noise_vals, 1e-8, None)
    whitener = noise_vecs @ np.diag(noise_vals ** -0.5) @ noise_vecs.T

    whitened = whitener @ signal_cov @ whitener.T
    whitened = (whitened + whitened.T) / 2.0
    gen_vals, gen_vecs = np.linalg.eigh(whitened)
    order = np.argsort(gen_vals)[::-1]
    gen_vals = np.clip(gen_vals[order], 0.0, None)
    gen_vecs = gen_vecs[:, order]

    signal_vals = np.linalg.eigvalsh((signal_cov + signal_cov.T) / 2.0)
    signal_vals = np.clip(signal_vals[np.argsort(signal_vals)[::-1]], 0.0, None)

    top_direction = whitener @ gen_vecs[:, 0]
    top_direction = top_direction / (np.linalg.norm(top_direction) + 1e-8)
    class_direction = class_direction / (np.linalg.norm(class_direction) + 1e-8)
    class_alignment = float(abs(np.dot(top_direction, class_direction)))

    mean_shift = np.mean(mean_shifts, axis=0)
    mean_shift_norm = float(np.linalg.norm(mean_shift))
    mean_shift_alignment = float(abs(np.dot(
        mean_shift / (np.linalg.norm(mean_shift) + 1e-8),
        class_direction,
    )))

    class_energy_fraction = float(
        class_direction.T @ signal_cov @ class_direction / (np.trace(signal_cov) + 1e-8)
    )

    return {
        "top_generalized_eigenvalue": float(gen_vals[0]) if len(gen_vals) else 0.0,
        "top5_generalized_eigenvalues": [float(v) for v in gen_vals[:5]],
        "signal_top5_eigenvalues": [float(v) for v in signal_vals[:5]],
        "trace_ratio_signal_vs_noise": float(np.trace(signal_cov) / (np.trace(noise_cov) + 1e-8)),
        "effective_rank_signal": effective_rank(signal_vals),
        "effective_rank_whitened": effective_rank(gen_vals),
        "num_modes_gt_1": int(np.sum(gen_vals > 1.0)),
        "num_modes_gt_2": int(np.sum(gen_vals > 2.0)),
        "top_mode_class_alignment": class_alignment,
        "class_energy_fraction": class_energy_fraction,
        "mean_shift_norm": mean_shift_norm,
        "mean_shift_class_alignment": mean_shift_alignment,
        "cross_seed_diff_cosine": pairwise_cosine(diff_mats),
    }


def analyze_layer(aligned_features, labels):
    anchor = aligned_features["A"][1]
    class_direction = anchor[labels == 0].mean(axis=0) - anchor[labels == 1].mean(axis=0)

    noise_covs = []
    for i in range(1, 6):
        for j in range(i + 1, 6):
            diff = aligned_features["A"][i] - aligned_features["A"][j]
            noise_covs.append((diff.T @ diff) / len(diff))
    noise_cov = np.mean(noise_covs, axis=0)

    comparisons = {}
    for name, left, right in [("A_minus_B", "A", "B"), ("A_minus_C", "A", "C")]:
        diff_mats = []
        signal_covs = []
        mean_shifts = []
        for seed in range(1, 6):
            diff = aligned_features[left][seed] - aligned_features[right][seed]
            diff_mats.append(diff)
            signal_covs.append((diff.T @ diff) / len(diff))
            mean_shifts.append(diff.mean(axis=0))
        signal_cov = np.mean(signal_covs, axis=0)
        comparisons[name] = spectral_metrics(signal_cov, noise_cov, class_direction, diff_mats, mean_shifts)

    return comparisons


def load_color_eval(max_images):
    X_val, _, _, _ = load_data("val", "A")
    with open("dataset/labels_A_val.json", "r") as f:
        labels = json.load(f)
    with open("dataset/metadata_val.json", "r") as f:
        metadata = json.load(f)

    blue_idx = [i for i, item in enumerate(metadata) if labels[item["filename"]] == "blue"][:max_images]
    green_idx = [i for i, item in enumerate(metadata) if labels[item["filename"]] == "green"][:max_images]
    X = np.concatenate([X_val[blue_idx], X_val[green_idx]], axis=0)
    y = np.array([0] * len(blue_idx) + [1] * len(green_idx), dtype=int)
    return X, y, "blue", "green"


def load_cifar_eval(max_images):
    _, _, X_test, y_test = load_raw_cifar10()
    auto_idx = np.where(y_test == CIFAR10_LABELS.index("automobile"))[0][:max_images]
    truck_idx = np.where(y_test == CIFAR10_LABELS.index("truck"))[0][:max_images]
    X = np.concatenate([X_test[auto_idx], X_test[truck_idx]], axis=0)
    y = np.array([0] * len(auto_idx) + [1] * len(truck_idx), dtype=int)
    return X, y, "automobile", "truck"


def analyze_dataset(dataset, max_images):
    if dataset == "color_blocks":
        X, y, pos_label, neg_label = load_color_eval(max_images)
        loader = lambda scheme, seed: load_model(scheme, seed)
    else:
        X, y, pos_label, neg_label = load_cifar_eval(max_images)
        loader = lambda scheme, seed: load_cifar_model(scheme, seed)

    features = {scheme: {} for scheme in ["A", "B", "C"]}
    for scheme in ["A", "B", "C"]:
        for seed in range(1, 6):
            model, _ = loader(scheme, seed)
            features[scheme][seed] = collect_features(model, X)

    aligned = {layer: {scheme: {} for scheme in ["A", "B", "C"]} for layer in LAYERS}
    for layer in LAYERS:
        anchor = features["A"][1][layer]
        for scheme in ["A", "B", "C"]:
            for seed in range(1, 6):
                if scheme == "A" and seed == 1:
                    aligned[layer][scheme][seed] = anchor
                else:
                    aligned[layer][scheme][seed] = orthogonal_align_to_anchor(anchor, features[scheme][seed][layer])

    layer_results = {}
    for layer in LAYERS:
        layer_results[layer] = analyze_layer(aligned[layer], y)

    return {
        "dataset": dataset,
        "positive_label": pos_label,
        "negative_label": neg_label,
        "num_examples": int(len(X)),
        "num_examples_per_class": int(len(X) // 2),
        "layers": layer_results,
    }


def print_summary(report):
    print(f"\n=== {report['dataset']} ===")
    print(
        f"Task: {report['positive_label']} vs {report['negative_label']} "
        f"({report['num_examples_per_class']} per class)"
    )
    for comparison in ["A_minus_B", "A_minus_C"]:
        print(f"\n{comparison}")
        for layer in LAYERS:
            row = report["layers"][layer][comparison]
            print(
                f"  {layer:5s} "
                f"top_lambda={row['top_generalized_eigenvalue']:.3f} "
                f"trace_ratio={row['trace_ratio_signal_vs_noise']:.3f} "
                f"modes>1={row['num_modes_gt_1']:d} "
                f"class_align={row['top_mode_class_alignment']:.3f} "
                f"diff_cos={row['cross_seed_diff_cosine']:.3f}"
            )


def make_dashboard(summary, output_path):
    datasets = ["color_blocks", "cifar10"]
    comparisons = [("A_minus_B", "A - B"), ("A_minus_C", "A - C")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False)
    fig.suptitle("Spectral Difference Analysis: Signal vs Seed Noise", fontsize=14, fontweight="bold")

    for row, dataset in enumerate(datasets):
        report = summary[dataset]
        layer_names = LAYERS
        x = np.arange(len(layer_names))
        width = 0.35

        ax = axes[row, 0]
        for idx, (key, label) in enumerate(comparisons):
            vals = [report["layers"][layer][key]["top_generalized_eigenvalue"] for layer in layer_names]
            ax.bar(x + (idx - 0.5) * width, vals, width=width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names)
        ax.set_ylabel("Top generalized eigenvalue")
        ax.set_title(f"{dataset}: Dominant stable mode")
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        for idx, (key, label) in enumerate(comparisons):
            vals = [report["layers"][layer][key]["num_modes_gt_1"] for layer in layer_names]
            ax.bar(x + (idx - 0.5) * width, vals, width=width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names)
        ax.set_ylabel("Modes with lambda > 1")
        ax.set_title(f"{dataset}: Stable subspace dimension")

        ax = axes[row, 2]
        for key, label in comparisons:
            vals = [report["layers"][layer][key]["top_mode_class_alignment"] for layer in layer_names]
            ax.plot(layer_names, vals, marker="o", linewidth=2, label=label)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("|cos(top mode, class dir)|")
        ax.set_title(f"{dataset}: Concept alignment")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["color_blocks", "cifar10"],
        default=["color_blocks", "cifar10"],
    )
    parser.add_argument("--max_images", type=int, default=300)
    parser.add_argument("--output", type=str, default="results/spectral_difference_summary.json")
    parser.add_argument("--figure", type=str, default="results/spectral_difference_dashboard.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.figure), exist_ok=True)

    summary = {}
    for dataset in args.datasets:
        summary[dataset] = analyze_dataset(dataset, args.max_images)
        print_summary(summary[dataset])

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    if set(args.datasets) == {"color_blocks", "cifar10"}:
        make_dashboard(summary, args.figure)

    print(f"\nSaved spectral summary to {args.output}")
    if set(args.datasets) == {"color_blocks", "cifar10"}:
        print(f"Saved dashboard to {args.figure}")


if __name__ == "__main__":
    main()
