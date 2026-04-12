import argparse
import json
import os
from collections import defaultdict

import mlx.core as mx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cifar_dataset import CIFAR10_LABELS, load_raw_cifar10
from run_experiment import load_cifar_model


LAYERS = ["conv1", "conv2", "conv3", "fc1"]


def dominant_hue_label(img, sat_thresh=0.25, val_thresh=0.2, dominance_thresh=0.35):
    arr = img.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0

    flat = arr.reshape(-1, 3)
    mxv = flat.max(axis=1)
    mnv = flat.min(axis=1)
    delta = mxv - mnv
    sat = np.where(mxv > 1e-8, delta / np.maximum(mxv, 1e-8), 0.0)
    val = mxv

    # Ignore nearly-grey/dark pixels so background clutter does not dominate hue counts.
    mask = (sat > sat_thresh) & (val > val_thresh)
    if int(mask.sum()) < 20:
        return None, 0.0

    pix = flat[mask]
    r, g, b = pix[:, 0], pix[:, 1], pix[:, 2]
    mx2 = pix.max(axis=1)
    d = mx2 - pix.min(axis=1) + 1e-8

    hue = np.zeros(len(pix), dtype=np.float32)
    is_r = mx2 == r
    is_g = mx2 == g
    is_b = mx2 == b
    hue[is_r] = ((g[is_r] - b[is_r]) / d[is_r]) % 6
    hue[is_g] = ((b[is_g] - r[is_g]) / d[is_g]) + 2
    hue[is_b] = ((r[is_b] - g[is_b]) / d[is_b]) + 4
    hue = hue * 60.0

    weights = sat[mask] * val[mask]
    blue_mass = float(weights[((hue >= 170) & (hue <= 250))].sum())
    green_mass = float(weights[((hue >= 70) & (hue <= 160))].sum())
    total_mass = float(weights.sum()) + 1e-8

    if blue_mass / total_mass >= dominance_thresh and blue_mass > green_mass:
        return "blue", blue_mass / total_mass
    if green_mass / total_mass >= dominance_thresh and green_mass > blue_mass:
        return "green", green_mass / total_mass
    return None, max(blue_mass, green_mass) / total_mass


def collect_color_split(X, y, per_class_limit):
    grouped = defaultdict(lambda: {"blue": [], "green": []})
    for idx, (img, class_idx) in enumerate(zip(X, y)):
        color_label, confidence = dominant_hue_label(img)
        if color_label is None:
            continue
        class_name = CIFAR10_LABELS[int(class_idx)]
        grouped[class_name][color_label].append((idx, float(confidence)))

    selected_indices = []
    color_targets = []
    class_targets = []
    selection_report = {}
    for class_name in CIFAR10_LABELS:
        blue_items = sorted(grouped[class_name]["blue"], key=lambda item: item[1], reverse=True)
        green_items = sorted(grouped[class_name]["green"], key=lambda item: item[1], reverse=True)
        take = min(len(blue_items), len(green_items), per_class_limit)
        chosen_blue = blue_items[:take]
        chosen_green = green_items[:take]

        selection_report[class_name] = {
            "available_blue": len(blue_items),
            "available_green": len(green_items),
            "selected_per_color": take,
            "mean_blue_confidence": float(np.mean([c for _, c in chosen_blue])) if chosen_blue else 0.0,
            "mean_green_confidence": float(np.mean([c for _, c in chosen_green])) if chosen_green else 0.0,
        }

        for idx, _ in chosen_blue:
            selected_indices.append(idx)
            color_targets.append(0)
            class_targets.append(class_name)
        for idx, _ in chosen_green:
            selected_indices.append(idx)
            color_targets.append(1)
            class_targets.append(class_name)

    return (
        np.array(selected_indices, dtype=int),
        np.array(color_targets, dtype=int),
        np.array(class_targets),
        selection_report,
    )


def global_avg_pool(features):
    if features.ndim == 4:
        return features.mean(axis=(1, 2))
    return features


def get_layer_features(model, X, layer, batch_size=256):
    feats = []
    for i in range(0, len(X), batch_size):
        _, activations = model(mx.array(X[i:i + batch_size]), capture_activations=True)
        feats.append(global_avg_pool(np.array(activations[layer])))
    return np.concatenate(feats, axis=0)


def make_probe():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def cv_accuracy(X, y):
    min_class = int(np.min(np.bincount(y)))
    cv = min(5, min_class)
    if cv < 2:
        return None
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    scores = cross_val_score(make_probe(), X, y, cv=splitter, scoring="accuracy")
    return float(scores.mean())


def residualize_by_class(features, class_targets):
    centered = features.copy()
    for class_name in sorted(set(class_targets.tolist())):
        mask = class_targets == class_name
        centered[mask] -= centered[mask].mean(axis=0, keepdims=True)
    return centered


def within_class_probe(features, color_targets, class_targets):
    scores = {}
    for class_name in sorted(set(class_targets.tolist())):
        mask = class_targets == class_name
        score = cv_accuracy(features[mask], color_targets[mask])
        if score is not None:
            scores[class_name] = score
    return {
        "mean_accuracy": float(np.mean(list(scores.values()))),
        "min_accuracy": float(np.min(list(scores.values()))),
        "max_accuracy": float(np.max(list(scores.values()))),
        "by_class": scores,
    }


def analyze_scheme(models, X_eval, color_targets, class_targets):
    per_seed = {}
    summary = {layer: {"raw": [], "class_residual": [], "within_class": []} for layer in LAYERS}

    for seed, model in models.items():
        layer_results = {}
        for layer in LAYERS:
            features = get_layer_features(model, X_eval, layer)
            raw_score = cv_accuracy(features, color_targets)
            residual_score = cv_accuracy(residualize_by_class(features, class_targets), color_targets)
            within_class = within_class_probe(features, color_targets, class_targets)
            layer_results[layer] = {
                "raw_probe_accuracy": raw_score,
                "class_residual_probe_accuracy": residual_score,
                "within_class_probe": within_class,
            }

            summary[layer]["raw"].append(raw_score)
            summary[layer]["class_residual"].append(residual_score)
            summary[layer]["within_class"].append(within_class["mean_accuracy"])

        per_seed[str(seed)] = layer_results

    return {
        "per_seed": per_seed,
        "summary": {
            layer: {
                "raw_probe_mean": float(np.mean(values["raw"])),
                "raw_probe_std": float(np.std(values["raw"])),
                "class_residual_probe_mean": float(np.mean(values["class_residual"])),
                "class_residual_probe_std": float(np.std(values["class_residual"])),
                "within_class_probe_mean": float(np.mean(values["within_class"])),
                "within_class_probe_std": float(np.std(values["within_class"])),
            }
            for layer, values in summary.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--per_class_limit", type=int, default=30)
    parser.add_argument("--output", type=str, default="results/cifar_implicit_color_summary.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    _, _, X_test, y_test = load_raw_cifar10()
    selected_indices, color_targets, class_targets, selection_report = collect_color_split(
        X_test, y_test, args.per_class_limit
    )
    X_eval = X_test[selected_indices]

    models = {
        scheme: {seed: load_cifar_model(scheme, seed)[0] for seed in args.seeds}
        for scheme in ["A", "B", "C"]
    }

    results = {
        "task": "implicit_color_in_cifar_latent_space",
        "pseudo_label_definition": {
            "positive_labels": {"0": "blue", "1": "green"},
            "method": "pixel_hue_mass",
            "notes": "Labels are derived from pixel hue statistics only, not from any learned color model.",
        },
        "selection": {
            "per_class_limit": args.per_class_limit,
            "total_examples": int(len(selected_indices)),
            "blue_examples": int((color_targets == 0).sum()),
            "green_examples": int((color_targets == 1).sum()),
            "per_class": selection_report,
        },
        "schemes": {
            scheme: analyze_scheme(models[scheme], X_eval, color_targets, class_targets)
            for scheme in ["A", "B", "C"]
        },
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results["selection"], indent=2))
    for scheme in ["A", "B", "C"]:
        print(f"\n=== Scheme {scheme} ===")
        for layer in LAYERS:
            row = results["schemes"][scheme]["summary"][layer]
            print(
                f"{layer:5s} "
                f"raw={row['raw_probe_mean']:.3f} "
                f"class_resid={row['class_residual_probe_mean']:.3f} "
                f"within_class={row['within_class_probe_mean']:.3f}"
            )
    print(f"\nSaved summary to {args.output}")


if __name__ == "__main__":
    main()
