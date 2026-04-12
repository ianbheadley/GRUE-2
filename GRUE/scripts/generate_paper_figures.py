import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figures")


def load(name):
    with open(os.path.join(DATA, name), "r") as f:
        return json.load(f)


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, name), dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_hidden_separability(missing_summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    schemes = ["A", "B", "C"]
    colors = ["#5DA5DA", "#FAA43A", "#60BD68"]

    color_vals = [missing_summary["color_blocks"]["schemes"][s]["probe_accuracy_mean"] for s in schemes]
    cifar_vals = [missing_summary["cifar10"]["schemes"][s]["probe_accuracy_mean"] for s in schemes]

    axes[0].bar(schemes, color_vals, color=colors)
    axes[0].set_ylim(0.7, 1.02)
    axes[0].set_title("Color: Blue vs Green Hidden Probe")
    axes[0].set_ylabel("Probe accuracy")
    for i, v in enumerate(color_vals):
        axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    axes[1].bar(schemes, cifar_vals, color=colors)
    axes[1].set_ylim(0.6, 1.0)
    axes[1].set_title("CIFAR-10: Automobile vs Truck Hidden Probe")
    for i, v in enumerate(cifar_vals):
        axes[1].text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)

    fig.suptitle("Hidden Distinctions Survive Label Merge and Class Removal", fontsize=13, fontweight="bold")
    save(fig, "fig_hidden_separability.png")


def _top_items(mapping, top_n=5):
    return sorted(mapping.items(), key=lambda item: item[1], reverse=True)[:top_n]


def fig_missing_redistribution(missing_summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    color_missing = missing_summary["color_blocks"]["schemes"]["C"]["blue_nearest_centroid_mean"]
    cifar_missing = missing_summary["cifar10"]["schemes"]["C"]["truck_nearest_centroid_mean"]

    color_items = _top_items(color_missing, 5)
    cifar_items = _top_items(cifar_missing, 5)

    axes[0].bar([k for k, _ in color_items], [v for _, v in color_items], color="#8E6CFF")
    axes[0].set_ylim(0, 0.75)
    axes[0].set_title("Where Missing Blue Lands in Model C")
    axes[0].set_ylabel("Nearest-centroid fraction")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar([k for k, _ in cifar_items], [v for _, v in cifar_items], color="#F17C67")
    axes[1].set_ylim(0, 0.4)
    axes[1].set_title("Where Missing Truck Lands in Model C")
    axes[1].tick_params(axis="x", rotation=25)

    fig.suptitle("Missing Classes Reappear as Structured Mixtures of Seen Concepts", fontsize=13, fontweight="bold")
    save(fig, "fig_missing_redistribution.png")


def fig_layer_rankings(layer_report):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    color_layers = layer_report["color_blocks"]["ranked_layers"][:5]
    cifar_layers = layer_report["cifar10"]["ranked_layers"][:5]

    axes[0].bar(
        [item["layer"] for item in color_layers],
        [item["best_metrics"]["purity_score"] for item in color_layers],
        color="#5AB4AC"
    )
    axes[0].set_title("Color A-C Layer Recovery Ranking")
    axes[0].set_ylabel("Purity score")

    axes[1].bar(
        [item["layer"] for item in cifar_layers],
        [item["best_metrics"]["purity_score"] for item in cifar_layers],
        color="#D8B365"
    )
    axes[1].set_title("CIFAR A-C Layer Recovery Ranking")

    fig.suptitle("Best Candidate Layers for Recovering the Missing Concept", fontsize=13, fontweight="bold")
    save(fig, "fig_layer_rankings.png")


def fig_alpha_sweep(alpha_summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, dataset in zip(axes, ["color_blocks", "cifar10"]):
        result = alpha_summary["results"][dataset]
        alphas = [float(a) for a in result["B_plus_AB"]["alphas"].keys()]
        ab_real = [result["B_plus_AB"]["alphas"][f"{a:.3f}"]["real"]["probe_accuracy"] for a in alphas]
        ac_real = [result["C_plus_AC"]["alphas"][f"{a:.3f}"]["real"]["probe_accuracy"] for a in alphas]
        baseline_b = result["B_plus_AB"]["baseline"]["probe_accuracy"]
        baseline_c = result["C_plus_AC"]["baseline"]["probe_accuracy"]

        ax.plot(alphas, ab_real, marker="o", label="B + (A-B)")
        ax.plot(alphas, ac_real, marker="o", label="C + (A-C)")
        ax.axhline(baseline_b, linestyle="--", color="#5DA5DA", alpha=0.6, label="B baseline")
        ax.axhline(baseline_c, linestyle="--", color="#FAA43A", alpha=0.6, label="C baseline")
        ax.set_title("Color" if dataset == "color_blocks" else "CIFAR-10")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Hidden probe accuracy")
        ax.legend(fontsize=8)

    fig.suptitle("Layer-Agnostic Whole-Model Injection Is Weak or Brittle", fontsize=13, fontweight="bold")
    save(fig, "fig_alpha_sweep.png")


def fig_layerwise_latent_gap(dissection_summary):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    layers = ["conv1", "conv2", "conv3", "fc1"]
    a_vals = [dissection_summary[layer]["A_auto_truck_diff"]["mean"] for layer in layers]
    b_vals = [dissection_summary[layer]["B_auto_truck_diff"]["mean"] for layer in layers]
    c_vals = [dissection_summary[layer]["C_auto_truck_diff"]["mean"] for layer in layers]

    x = np.arange(len(layers))
    width = 0.24
    ax.bar(x - width, a_vals, width, label="Scheme A", color="#5DA5DA")
    ax.bar(x, b_vals, width, label="Scheme B", color="#FAA43A")
    ax.bar(x + width, c_vals, width, label="Scheme C", color="#60BD68")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylabel("Mean auto-truck activation distance")
    ax.set_title("Hidden Automobile-Truck Separation Persists Across Layers")
    ax.legend(fontsize=9)

    fig.suptitle("Cross-Scheme Layerwise Latent Gap in CIFAR-10", fontsize=13, fontweight="bold")
    save(fig, "fig_layerwise_latent_gap.png")


def fig_implicit_color(cifar_color_summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    layers = ["conv1", "conv2", "conv3", "fc1"]
    colors = {"A": "#5DA5DA", "B": "#FAA43A", "C": "#60BD68"}

    for scheme in ["A", "B", "C"]:
        summary = cifar_color_summary["schemes"][scheme]["summary"]
        class_residual = [summary[layer]["class_residual_probe_mean"] for layer in layers]
        within_class = [summary[layer]["within_class_probe_mean"] for layer in layers]

        axes[0].plot(layers, class_residual, marker="o", linewidth=2, color=colors[scheme], label=f"Scheme {scheme}")
        axes[1].plot(layers, within_class, marker="o", linewidth=2, color=colors[scheme], label=f"Scheme {scheme}")

    for ax in axes:
        ax.axhline(0.5, linestyle="--", color="gray", linewidth=1, alpha=0.7)
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("Probe accuracy")
        ax.legend(fontsize=8)

    axes[0].set_title("Color Probe After Removing Class Identity")
    axes[0].set_xlabel("Layer")
    axes[1].set_title("Mean Within-Class Color Probe")
    axes[1].set_xlabel("Layer")

    fig.suptitle("CIFAR Models Learn Implicit Blue-Green Structure Without Color Labels", fontsize=13, fontweight="bold")
    save(fig, "fig_implicit_color_cifar.png")


def main():
    missing = load("missing_training_summary.json")
    layer = load("layer_concept_report.json")
    alpha = load("alpha_sweep_summary.json")
    dissection = load("summary.json")
    implicit_color = load("cifar_implicit_color_summary.json")

    fig_hidden_separability(missing)
    fig_missing_redistribution(missing)
    fig_layer_rankings(layer)
    fig_alpha_sweep(alpha)
    fig_layerwise_latent_gap(dissection)
    fig_implicit_color(implicit_color)


if __name__ == "__main__":
    main()
