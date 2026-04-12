"""
Activation Dissection: Visualize how auto/truck activations differ across Models A, B, C
at every layer, and compare against the A-vs-A noise floor.

Produces:
  1. Per-layer heatmaps of mean activations for auto & truck in each model
  2. Per-layer difference maps (auto - truck) for each model
  3. A-vs-A noise floor comparison (seed1 - seed2 differences)
  4. Summary statistics and a combined dashboard
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import json
import os

from run_experiment import load_cifar_model, get_class_images
from cifar_dataset import load_raw_cifar10, CIFAR10_LABELS
from concept_extraction import get_activations, global_avg_pool


def collect_activations(model, X, batch_size=256):
    """Get per-layer activations, pooled to 1D per neuron."""
    raw_acts = get_activations(model, X)
    pooled = {}
    for layer, act in raw_acts.items():
        pooled[layer] = global_avg_pool(act)  # (N, neurons)
    return pooled


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models_cifar10")
    parser.add_argument("--output_dir", type=str, default="results/dissection")
    args = parser.parse_args()
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Load test data
    print("Loading test data...")
    _, _, X_test, y_test = load_raw_cifar10()
    X_auto = get_class_images(X_test, y_test, "automobile", 500)
    X_truck = get_class_images(X_test, y_test, "truck", 500)
    X_cat = get_class_images(X_test, y_test, "cat", 500)
    X_frog = get_class_images(X_test, y_test, "frog", 500)

    seeds = [1, 2, 3, 4, 5]
    layers = ["conv1", "conv2", "conv3", "fc1"]  # skip fc2 (output head differs)

    # -----------------------------------------------------------------------
    # Collect activations from all models
    # -----------------------------------------------------------------------
    print("Collecting activations from all models...")
    acts = {}  # acts[scheme][seed][class_name][layer] = (N, neurons)
    for scheme in ["A", "B", "C"]:
        acts[scheme] = {}
        for seed in seeds:
            print(f"  Loading {scheme}/seed_{seed}...")
            model, _ = load_cifar_model(scheme, seed)
            acts[scheme][seed] = {}
            for cls_name, X_cls in [("auto", X_auto), ("truck", X_truck),
                                     ("cat", X_cat), ("frog", X_frog)]:
                acts[scheme][seed][cls_name] = collect_activations(model, X_cls)

    # -----------------------------------------------------------------------
    # Figure 1: Mean activation heatmaps per layer for auto & truck
    # -----------------------------------------------------------------------
    print("\nGenerating Figure 1: Mean activation profiles...")
    for layer in layers:
        n_neurons = acts["A"][1]["auto"][layer].shape[1]
        # Sort neurons by Model A auto-truck difference for consistent ordering
        mean_a_auto = acts["A"][1]["auto"][layer].mean(axis=0)
        mean_a_truck = acts["A"][1]["truck"][layer].mean(axis=0)
        sort_idx = np.argsort(mean_a_auto - mean_a_truck)[::-1]

        fig, axes = plt.subplots(3, 2, figsize=(18, 10))
        fig.suptitle(f"Layer: {layer} ({n_neurons} neurons) — Mean Activation for Auto vs Truck",
                     fontsize=14, fontweight="bold")

        for row, scheme in enumerate(["A", "B", "C"]):
            mean_auto = np.mean([acts[scheme][s]["auto"][layer].mean(axis=0) for s in seeds], axis=0)
            mean_truck = np.mean([acts[scheme][s]["truck"][layer].mean(axis=0) for s in seeds], axis=0)

            # Reorder neurons
            mean_auto_sorted = mean_auto[sort_idx]
            mean_truck_sorted = mean_truck[sort_idx]

            vmax = max(mean_auto_sorted.max(), mean_truck_sorted.max())

            axes[row, 0].bar(range(n_neurons), mean_auto_sorted, color="steelblue", alpha=0.8, width=1.0)
            axes[row, 0].set_ylabel(f"Model {scheme}")
            axes[row, 0].set_title("Automobile" if row == 0 else "")
            axes[row, 0].set_ylim(0, vmax * 1.1)

            axes[row, 1].bar(range(n_neurons), mean_truck_sorted, color="firebrick", alpha=0.8, width=1.0)
            axes[row, 1].set_title("Truck" if row == 0 else "")
            axes[row, 1].set_ylim(0, vmax * 1.1)

            if row == 2:
                axes[row, 0].set_xlabel("Neuron index (sorted by Model A auto-truck diff)")
                axes[row, 1].set_xlabel("Neuron index (sorted by Model A auto-truck diff)")

        plt.tight_layout()
        plt.savefig(f"results/dissection/fig1_{layer}_mean_activations.png", dpi=150)
        plt.close()

    # -----------------------------------------------------------------------
    # Figure 2: Auto - Truck difference per model, per layer
    # -----------------------------------------------------------------------
    print("Generating Figure 2: Auto-Truck difference across models...")
    for layer in layers:
        n_neurons = acts["A"][1]["auto"][layer].shape[1]
        mean_a_auto = acts["A"][1]["auto"][layer].mean(axis=0)
        mean_a_truck = acts["A"][1]["truck"][layer].mean(axis=0)
        sort_idx = np.argsort(mean_a_auto - mean_a_truck)[::-1]

        fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
        fig.suptitle(f"Layer: {layer} — Activation Difference (Auto - Truck)",
                     fontsize=14, fontweight="bold")

        for row, (label, scheme) in enumerate([("Model A (separate classes)", "A"),
                                                ("Model B (merged as motor_vehicle)", "B"),
                                                ("Model C (truck removed)", "C")]):
            # Average difference across seeds
            diffs = []
            for s in seeds:
                d = acts[scheme][s]["auto"][layer].mean(axis=0) - acts[scheme][s]["truck"][layer].mean(axis=0)
                diffs.append(d)
            mean_diff = np.mean(diffs, axis=0)
            std_diff = np.std(diffs, axis=0)
            mean_diff_sorted = mean_diff[sort_idx]
            std_diff_sorted = std_diff[sort_idx]

            colors = ["steelblue" if v > 0 else "firebrick" for v in mean_diff_sorted]
            axes[row].bar(range(n_neurons), mean_diff_sorted, color=colors, alpha=0.8, width=1.0)
            axes[row].errorbar(range(n_neurons), mean_diff_sorted, yerr=std_diff_sorted,
                             fmt="none", ecolor="gray", alpha=0.3, linewidth=0.5)
            axes[row].axhline(0, color="black", linewidth=0.5)
            axes[row].set_ylabel("Auto - Truck")
            axes[row].set_title(label)

        # Row 4: A-vs-A noise floor (seed1 vs seed2, same scheme, same class)
        noise_diffs = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                d = acts["A"][seeds[i]]["auto"][layer].mean(axis=0) - acts["A"][seeds[j]]["auto"][layer].mean(axis=0)
                noise_diffs.append(d)
        noise_mean = np.mean(noise_diffs, axis=0)
        noise_std = np.std(noise_diffs, axis=0)
        noise_mean_sorted = noise_mean[sort_idx]
        noise_std_sorted = noise_std[sort_idx]

        axes[3].bar(range(n_neurons), noise_mean_sorted, color="gray", alpha=0.6, width=1.0)
        axes[3].errorbar(range(n_neurons), noise_mean_sorted, yerr=noise_std_sorted,
                        fmt="none", ecolor="gray", alpha=0.3, linewidth=0.5)
        axes[3].axhline(0, color="black", linewidth=0.5)
        axes[3].set_ylabel("Seed diff")
        axes[3].set_title("Noise Floor: Model A seed-to-seed variation (same class)")
        axes[3].set_xlabel("Neuron index (sorted by Model A auto-truck diff)")

        plt.tight_layout()
        plt.savefig(f"results/dissection/fig2_{layer}_auto_truck_diff.png", dpi=150)
        plt.close()

    # -----------------------------------------------------------------------
    # Figure 3: Cross-model subtraction (A-B and A-C activation differences)
    # -----------------------------------------------------------------------
    print("Generating Figure 3: Cross-model activation subtraction...")
    for layer in layers:
        n_neurons = acts["A"][1]["auto"][layer].shape[1]

        fig, axes = plt.subplots(3, 2, figsize=(18, 12))
        fig.suptitle(f"Layer: {layer} — Cross-Model Activation Subtraction\n"
                     f"How do models differ in what they compute?",
                     fontsize=14, fontweight="bold")

        classes_to_show = [("auto", "Automobile"), ("truck", "Truck")]
        for col, (cls_name, cls_label) in enumerate(classes_to_show):
            # A - B difference (what does the auto/truck distinction add?)
            ab_diffs = []
            for s in seeds:
                d = acts["A"][s][cls_name][layer].mean(axis=0) - acts["B"][s][cls_name][layer].mean(axis=0)
                ab_diffs.append(d)
            ab_mean = np.mean(ab_diffs, axis=0)
            ab_std = np.std(ab_diffs, axis=0)

            # A - C difference
            ac_diffs = []
            for s in seeds:
                d = acts["A"][s][cls_name][layer].mean(axis=0) - acts["C"][s][cls_name][layer].mean(axis=0)
                ac_diffs.append(d)
            ac_mean = np.mean(ac_diffs, axis=0)
            ac_std = np.std(ac_diffs, axis=0)

            # A - A noise floor
            aa_diffs = []
            for i in range(len(seeds)):
                for j in range(i + 1, len(seeds)):
                    d = acts["A"][seeds[i]][cls_name][layer].mean(axis=0) - acts["A"][seeds[j]][cls_name][layer].mean(axis=0)
                    aa_diffs.append(d)
            aa_mean = np.mean(aa_diffs, axis=0)
            aa_std = np.std(aa_diffs, axis=0)

            # Sort by |A-B| magnitude
            sort_idx = np.argsort(np.abs(ab_mean))[::-1]

            colors_ab = ["steelblue" if v > 0 else "firebrick" for v in ab_mean[sort_idx]]
            axes[0, col].bar(range(n_neurons), ab_mean[sort_idx], color=colors_ab, alpha=0.8, width=1.0)
            axes[0, col].set_title(f"{cls_label}: Model A - Model B")
            axes[0, col].set_ylabel("Activation diff")

            colors_ac = ["steelblue" if v > 0 else "firebrick" for v in ac_mean[sort_idx]]
            axes[1, col].bar(range(n_neurons), ac_mean[sort_idx], color=colors_ac, alpha=0.8, width=1.0)
            axes[1, col].set_title(f"{cls_label}: Model A - Model C")
            axes[1, col].set_ylabel("Activation diff")

            axes[2, col].bar(range(n_neurons), aa_mean[sort_idx], color="gray", alpha=0.6, width=1.0)
            axes[2, col].set_title(f"{cls_label}: A-A Noise Floor (seed variation)")
            axes[2, col].set_ylabel("Activation diff")
            axes[2, col].set_xlabel("Neuron index (sorted by |A-B| diff)")

            # Match y-axis scales within each column
            ymax = max(np.abs(ab_mean).max(), np.abs(ac_mean).max(), np.abs(aa_mean).max()) * 1.2
            for r in range(3):
                axes[r, col].set_ylim(-ymax, ymax)
                axes[r, col].axhline(0, color="black", linewidth=0.5)

        plt.tight_layout()
        plt.savefig(f"results/dissection/fig3_{layer}_cross_model_subtraction.png", dpi=150)
        plt.close()

    # -----------------------------------------------------------------------
    # Figure 4: Summary dashboard — SNR of activation differences per layer
    # -----------------------------------------------------------------------
    print("Generating Figure 4: Summary dashboard...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Activation Dissection Summary", fontsize=16, fontweight="bold")

    # Panel 1: Auto-truck diff norm per model per layer
    ax = axes[0, 0]
    x_pos = np.arange(len(layers))
    width = 0.25
    for i, (scheme, color, label) in enumerate([("A", "steelblue", "Model A (separate)"),
                                                  ("B", "orange", "Model B (merged)"),
                                                  ("C", "green", "Model C (truck removed)")]):
        norms = []
        stds = []
        for layer in layers:
            layer_norms = []
            for s in seeds:
                d = acts[scheme][s]["auto"][layer].mean(axis=0) - acts[scheme][s]["truck"][layer].mean(axis=0)
                layer_norms.append(np.linalg.norm(d))
            norms.append(np.mean(layer_norms))
            stds.append(np.std(layer_norms))
        ax.bar(x_pos + i * width, norms, width, yerr=stds, label=label, color=color, alpha=0.8)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(layers)
    ax.set_ylabel("||mean(auto) - mean(truck)||")
    ax.set_title("Auto-Truck Separation Strength Per Layer")
    ax.legend()

    # Panel 2: Cross-model diff norm for auto images
    ax = axes[0, 1]
    for i, (pair, color, label) in enumerate([("AB", "purple", "A - B"),
                                                ("AC", "red", "A - C"),
                                                ("AA", "gray", "A - A (noise)")]):
        norms = []
        for layer in layers:
            if pair == "AA":
                layer_norms = []
                for si in range(len(seeds)):
                    for sj in range(si + 1, len(seeds)):
                        d = acts["A"][seeds[si]]["auto"][layer].mean(axis=0) - acts["A"][seeds[sj]]["auto"][layer].mean(axis=0)
                        layer_norms.append(np.linalg.norm(d))
                norms.append(np.mean(layer_norms))
            elif pair == "AB":
                layer_norms = [np.linalg.norm(acts["A"][s]["auto"][layer].mean(axis=0) - acts["B"][s]["auto"][layer].mean(axis=0)) for s in seeds]
                norms.append(np.mean(layer_norms))
            else:
                layer_norms = [np.linalg.norm(acts["A"][s]["auto"][layer].mean(axis=0) - acts["C"][s]["auto"][layer].mean(axis=0)) for s in seeds]
                norms.append(np.mean(layer_norms))
        ax.plot(layers, norms, "o-", color=color, label=label, linewidth=2)
    ax.set_ylabel("||activation diff||")
    ax.set_title("Cross-Model Activation Difference (Auto images)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Same but for truck images
    ax = axes[1, 0]
    for i, (pair, color, label) in enumerate([("AB", "purple", "A - B"),
                                                ("AC", "red", "A - C"),
                                                ("AA", "gray", "A - A (noise)")]):
        norms = []
        for layer in layers:
            if pair == "AA":
                layer_norms = []
                for si in range(len(seeds)):
                    for sj in range(si + 1, len(seeds)):
                        d = acts["A"][seeds[si]]["truck"][layer].mean(axis=0) - acts["A"][seeds[sj]]["truck"][layer].mean(axis=0)
                        layer_norms.append(np.linalg.norm(d))
                norms.append(np.mean(layer_norms))
            elif pair == "AB":
                layer_norms = [np.linalg.norm(acts["A"][s]["truck"][layer].mean(axis=0) - acts["B"][s]["truck"][layer].mean(axis=0)) for s in seeds]
                norms.append(np.mean(layer_norms))
            else:
                layer_norms = [np.linalg.norm(acts["A"][s]["truck"][layer].mean(axis=0) - acts["C"][s]["truck"][layer].mean(axis=0)) for s in seeds]
                norms.append(np.mean(layer_norms))
        ax.plot(layers, norms, "o-", color=color, label=label, linewidth=2)
    ax.set_ylabel("||activation diff||")
    ax.set_title("Cross-Model Activation Difference (Truck images)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Correlation of A-B diff pattern for auto vs truck
    ax = axes[1, 1]
    correlations = []
    for layer in layers:
        ab_auto_diffs = []
        ab_truck_diffs = []
        for s in seeds:
            ab_auto = acts["A"][s]["auto"][layer].mean(axis=0) - acts["B"][s]["auto"][layer].mean(axis=0)
            ab_truck = acts["A"][s]["truck"][layer].mean(axis=0) - acts["B"][s]["truck"][layer].mean(axis=0)
            ab_auto_diffs.append(ab_auto)
            ab_truck_diffs.append(ab_truck)
        mean_auto_diff = np.mean(ab_auto_diffs, axis=0)
        mean_truck_diff = np.mean(ab_truck_diffs, axis=0)
        corr = np.corrcoef(mean_auto_diff, mean_truck_diff)[0, 1]
        correlations.append(corr)
    ax.bar(layers, correlations, color="teal", alpha=0.8)
    ax.set_ylabel("Pearson r")
    ax.set_title("A-B Diff Correlation: Auto vs Truck\n(Do models change the same neurons for both?)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/dissection/fig4_summary_dashboard.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------------
    # Figure 5: Per-neuron scatter — does A-B activation diff predict auto-truck diff?
    # -----------------------------------------------------------------------
    print("Generating Figure 5: Neuron-level scatter analysis...")
    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 5))
    fig.suptitle("Does A-B model difference predict Auto-Truck concept direction?\n"
                 "Each dot = one neuron. X = how much that neuron changes between models. "
                 "Y = how much it distinguishes auto from truck in Model B.",
                 fontsize=12, fontweight="bold")

    for i, layer in enumerate(layers):
        # X-axis: A-B activation diff for auto images (how this neuron changed between models)
        ab_diffs = [acts["A"][s]["auto"][layer].mean(axis=0) - acts["B"][s]["auto"][layer].mean(axis=0) for s in seeds]
        x_vals = np.mean(ab_diffs, axis=0)

        # Y-axis: auto-truck diff in Model B (how this neuron separates the concept)
        bt_diffs = [acts["B"][s]["auto"][layer].mean(axis=0) - acts["B"][s]["truck"][layer].mean(axis=0) for s in seeds]
        y_vals = np.mean(bt_diffs, axis=0)

        corr = np.corrcoef(x_vals, y_vals)[0, 1]
        axes[i].scatter(x_vals, y_vals, alpha=0.5, s=15, c="teal")
        axes[i].set_xlabel("A-B model diff (auto)")
        axes[i].set_ylabel("Auto-Truck diff in Model B")
        axes[i].set_title(f"{layer} (r={corr:.3f})")
        axes[i].axhline(0, color="gray", linewidth=0.5)
        axes[i].axvline(0, color="gray", linewidth=0.5)

        # Fit line
        if len(x_vals) > 2:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            axes[i].plot(x_line, p(x_line), "r-", alpha=0.7, linewidth=2)

    plt.tight_layout()
    plt.savefig("results/dissection/fig5_neuron_scatter.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------------
    # Print numerical summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ACTIVATION DISSECTION — NUMERICAL SUMMARY")
    print("=" * 80)

    for layer in layers:
        n_neurons = acts["A"][1]["auto"][layer].shape[1]
        print(f"\n--- {layer} ({n_neurons} neurons) ---")

        # Auto-truck diff norms
        for scheme in ["A", "B", "C"]:
            norms = [np.linalg.norm(acts[scheme][s]["auto"][layer].mean(axis=0) -
                                    acts[scheme][s]["truck"][layer].mean(axis=0)) for s in seeds]
            print(f"  {scheme} auto-truck diff:  {np.mean(norms):.4f} +/- {np.std(norms):.4f}")

        # Cross-model diffs (auto images)
        for pair_name, s1, s2 in [("A-B", "A", "B"), ("A-C", "A", "C")]:
            norms = [np.linalg.norm(acts[s1][s]["auto"][layer].mean(axis=0) -
                                    acts[s2][s]["auto"][layer].mean(axis=0)) for s in seeds]
            print(f"  {pair_name} cross-model (auto): {np.mean(norms):.4f} +/- {np.std(norms):.4f}")

        # Noise floor
        noise_norms = []
        for si in range(len(seeds)):
            for sj in range(si + 1, len(seeds)):
                d = acts["A"][seeds[si]]["auto"][layer].mean(axis=0) - acts["A"][seeds[sj]]["auto"][layer].mean(axis=0)
                noise_norms.append(np.linalg.norm(d))
        print(f"  A-A noise floor (auto): {np.mean(noise_norms):.4f} +/- {np.std(noise_norms):.4f}")

        # SNR
        ab_norm = np.mean([np.linalg.norm(acts["A"][s]["auto"][layer].mean(axis=0) -
                                          acts["B"][s]["auto"][layer].mean(axis=0)) for s in seeds])
        aa_norm = np.mean(noise_norms)
        print(f"  SNR (A-B / A-A):        {ab_norm / (aa_norm + 1e-8):.2f}x")

    # Save summary as JSON
    summary = {}
    for layer in layers:
        summary[layer] = {}
        for scheme in ["A", "B", "C"]:
            norms = [float(np.linalg.norm(acts[scheme][s]["auto"][layer].mean(axis=0) -
                                          acts[scheme][s]["truck"][layer].mean(axis=0))) for s in seeds]
            summary[layer][f"{scheme}_auto_truck_diff"] = {"mean": float(np.mean(norms)), "std": float(np.std(norms))}

        for pair_name, s1, s2 in [("A_B", "A", "B"), ("A_C", "A", "C")]:
            norms = [float(np.linalg.norm(acts[s1][s]["auto"][layer].mean(axis=0) -
                                          acts[s2][s]["auto"][layer].mean(axis=0))) for s in seeds]
            summary[layer][f"{pair_name}_cross_model_auto"] = {"mean": float(np.mean(norms)), "std": float(np.std(norms))}

        noise_norms_list = []
        for si in range(len(seeds)):
            for sj in range(si + 1, len(seeds)):
                d = acts["A"][seeds[si]]["auto"][layer].mean(axis=0) - acts["A"][seeds[sj]]["auto"][layer].mean(axis=0)
                noise_norms_list.append(float(np.linalg.norm(d)))
        summary[layer]["noise_floor"] = {"mean": float(np.mean(noise_norms_list)), "std": float(np.std(noise_norms_list))}

    with open("results/dissection/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n\nAll figures saved to results/dissection/")
    print("Summary data saved to results/dissection/summary.json")


if __name__ == "__main__":
    main()
