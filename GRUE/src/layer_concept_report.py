import json
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from cifar_dataset import CIFAR10_LABELS, load_cifar10, load_raw_cifar10
from evaluate import load_data
from load_models import load_model
from run_experiment import load_cifar_model


ALPHAS = [0.1, 0.25, 0.5, 1.0]
SEEDS = [1, 2, 3, 4, 5]


def color_fc1(model, X, batch_size=256):
    feats = []
    for i in range(0, len(X), batch_size):
        x = mx.array(X[i:i + batch_size])
        x = model.pool1(nn.relu(model.bn1(model.conv1(x))))
        x = model.pool2(nn.relu(model.bn2(model.conv2(x))))
        x = model.pool3(nn.relu(model.bn3(model.conv3(x))))
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(model.fc1(x))
        feats.append(np.array(x))
    return np.concatenate(feats)


def cifar_fc1(model, X, batch_size=256):
    feats = []
    for i in range(0, len(X), batch_size):
        _, acts = model(mx.array(X[i:i + batch_size]), capture_activations=True)
        feats.append(np.array(acts["fc1"]))
    return np.concatenate(feats)


def probe_accuracy(features_a, features_b):
    X = np.concatenate([features_a, features_b], axis=0)
    y = np.array([0] * len(features_a) + [1] * len(features_b))
    clf = LogisticRegression(max_iter=1000)
    return float(cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean())


def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


def select_indices_per_label(labels, label_indices, max_per_label):
    selected = []
    for label_idx in label_indices:
        selected.extend(np.where(labels == int(label_idx))[0][:max_per_label].tolist())
    return np.array(selected, dtype=int)


def nearest_true_rate(features, true_label, centroids):
    labels = list(centroids.keys())
    centroid_stack = np.stack([centroids[label] for label in labels], axis=0)
    dists = np.linalg.norm(features[:, None, :] - centroid_stack[None, :, :], axis=2)
    nearest = np.argmin(dists, axis=1)
    return float(np.mean([labels[int(i)] == true_label for i in nearest]))


def mean_true_cosine(features, true_label, centroids):
    centroid = centroids[true_label]
    sims = [cosine(feature, centroid) for feature in features]
    return float(np.mean(sims))


def shared_params_by_layer(model):
    grouped = {}
    for key, value in tree_flatten(model.parameters()):
        if "fc2" in key:
            continue
        layer = key.split(".")[0]
        grouped.setdefault(layer, {})
        grouped[layer][key] = value
    return grouped


def average_layer_deltas(models_a, models_c):
    grouped_a = [shared_params_by_layer(model) for model in models_a]
    grouped_c = [shared_params_by_layer(model) for model in models_c]
    layers = sorted(grouped_a[0].keys())
    result = {}
    for layer in layers:
        layer_delta = {}
        for key in grouped_a[0][layer]:
            layer_delta[key] = mx.mean(
                mx.stack([ga[layer][key] - gc[layer][key] for ga, gc in zip(grouped_a, grouped_c)]),
                axis=0,
            )
        result[layer] = layer_delta
    return result


def inject_layer_delta(model, layer_delta, alpha):
    original = dict(tree_flatten(model.parameters()))
    updated = dict(original)
    for key, value in layer_delta.items():
        updated[key] = updated[key] + alpha * value
    model.update(tree_unflatten(list(updated.items())))
    mx.eval(model.parameters())
    return original


def restore_model(model, original):
    model.update(tree_unflatten(list(original.items())))
    mx.eval(model.parameters())


def build_centroids(features, labels, idx_to_label, max_per_label):
    centroids = {}
    for label_idx, label_name in idx_to_label.items():
        positions = np.where(labels == int(label_idx))[0][:max_per_label]
        centroids[label_name] = features[positions].mean(axis=0)
    return centroids


def score_metrics(injected, baseline):
    seen_drop = max(0.0, baseline["seen_true_rate"] - injected["seen_true_rate"])
    return (
        (injected["probe_accuracy"] - baseline["probe_accuracy"])
        + (injected["missing_true_rate"] - baseline["missing_true_rate"])
        + 0.5 * (injected["missing_true_cosine"] - baseline["missing_true_cosine"])
        - 0.5 * seen_drop
    )


def color_report(max_eval=300, max_train_per_label=300):
    X_val, _, _, _ = load_data("val", "A")
    with open("dataset/labels_A_val.json", "r") as f:
        labels = json.load(f)
    with open("dataset/metadata_val.json", "r") as f:
        meta = json.load(f)

    blue_idx = [i for i, item in enumerate(meta) if labels[item["filename"]] == "blue"][:max_eval]
    green_idx = [i for i, item in enumerate(meta) if labels[item["filename"]] == "green"][:max_eval]
    X_blue = X_val[blue_idx]
    X_green = X_val[green_idx]

    X_train_A, y_train_A, _, idx_to_A = load_data("train", "A")
    selected_A = select_indices_per_label(y_train_A, idx_to_A.keys(), max_train_per_label)

    models_a = [load_model("A", seed)[0] for seed in SEEDS]
    models_c = [load_model("C", seed)[0] for seed in SEEDS]
    layer_deltas = average_layer_deltas(models_a, models_c)

    baseline_per_seed = []
    a_centroids = {}
    for seed, model_a, model_c in zip(SEEDS, models_a, models_c):
        a_train_feats = color_fc1(model_a, X_train_A[selected_A])
        a_centroids[seed] = build_centroids(a_train_feats, y_train_A[selected_A], idx_to_A, max_train_per_label)

        blue_feats = color_fc1(model_c, X_blue)
        green_feats = color_fc1(model_c, X_green)
        baseline_per_seed.append({
            "probe_accuracy": probe_accuracy(blue_feats, green_feats),
            "missing_true_rate": nearest_true_rate(blue_feats, "blue", a_centroids[seed]),
            "seen_true_rate": nearest_true_rate(green_feats, "green", a_centroids[seed]),
            "missing_true_cosine": mean_true_cosine(blue_feats, "blue", a_centroids[seed]),
        })

    baseline = {
        key: float(np.mean([item[key] for item in baseline_per_seed]))
        for key in baseline_per_seed[0]
    }

    layers = {}
    for layer, layer_delta in layer_deltas.items():
        alpha_reports = {}
        for alpha in ALPHAS:
            per_seed = []
            for seed, model_c in zip(SEEDS, models_c):
                original = inject_layer_delta(model_c, layer_delta, alpha)
                blue_feats = color_fc1(model_c, X_blue)
                green_feats = color_fc1(model_c, X_green)
                has_nan = bool(np.isnan(blue_feats).any() or np.isnan(green_feats).any())
                if has_nan:
                    metrics = None
                else:
                    metrics = {
                        "probe_accuracy": probe_accuracy(blue_feats, green_feats),
                        "missing_true_rate": nearest_true_rate(blue_feats, "blue", a_centroids[seed]),
                        "seen_true_rate": nearest_true_rate(green_feats, "green", a_centroids[seed]),
                        "missing_true_cosine": mean_true_cosine(blue_feats, "blue", a_centroids[seed]),
                    }
                restore_model(model_c, original)
                per_seed.append(metrics)

            valid = [item for item in per_seed if item is not None]
            if not valid:
                alpha_reports[f"{alpha:.2f}"] = {"nan": True, "purity_score": None}
                continue

            avg = {key: float(np.mean([item[key] for item in valid])) for key in valid[0]}
            avg["nan"] = len(valid) != len(per_seed)
            avg["purity_score"] = score_metrics(avg, baseline)
            alpha_reports[f"{alpha:.2f}"] = avg

        best_alpha, best_metrics = max(
            alpha_reports.items(),
            key=lambda item: -1e9 if item[1]["purity_score"] is None else item[1]["purity_score"],
        )
        layers[layer] = {
            "best_alpha": best_alpha,
            "best_metrics": best_metrics,
            "all_alphas": alpha_reports,
        }

    ranked = sorted(
        [{"layer": layer, **info} for layer, info in layers.items()],
        key=lambda item: -1e9 if item["best_metrics"]["purity_score"] is None else item["best_metrics"]["purity_score"],
        reverse=True,
    )

    return {
        "dataset": "color_blocks",
        "missing_class": "blue",
        "contrast_class": "green",
        "baseline_C": baseline,
        "ranked_layers": ranked,
    }


def cifar_report(max_eval=300, max_train_per_label=300):
    _, _, X_test, y_test = load_raw_cifar10()
    auto_idx = np.where(y_test == CIFAR10_LABELS.index("automobile"))[0][:max_eval]
    truck_idx = np.where(y_test == CIFAR10_LABELS.index("truck"))[0][:max_eval]
    X_auto = X_test[auto_idx]
    X_truck = X_test[truck_idx]

    X_train_A, y_train_A, _, _, _, idx_to_A = load_cifar10("A")
    selected_A = select_indices_per_label(y_train_A, idx_to_A.keys(), max_train_per_label)

    models_a = [load_cifar_model("A", seed)[0] for seed in SEEDS]
    models_c = [load_cifar_model("C", seed)[0] for seed in SEEDS]
    layer_deltas = average_layer_deltas(models_a, models_c)

    baseline_per_seed = []
    a_centroids = {}
    for seed, model_a, model_c in zip(SEEDS, models_a, models_c):
        a_train_feats = cifar_fc1(model_a, X_train_A[selected_A])
        a_centroids[seed] = build_centroids(a_train_feats, y_train_A[selected_A], idx_to_A, max_train_per_label)

        auto_feats = cifar_fc1(model_c, X_auto)
        truck_feats = cifar_fc1(model_c, X_truck)
        baseline_per_seed.append({
            "probe_accuracy": probe_accuracy(truck_feats, auto_feats),
            "missing_true_rate": nearest_true_rate(truck_feats, "truck", a_centroids[seed]),
            "seen_true_rate": nearest_true_rate(auto_feats, "automobile", a_centroids[seed]),
            "missing_true_cosine": mean_true_cosine(truck_feats, "truck", a_centroids[seed]),
        })

    baseline = {
        key: float(np.mean([item[key] for item in baseline_per_seed]))
        for key in baseline_per_seed[0]
    }

    layers = {}
    for layer, layer_delta in layer_deltas.items():
        alpha_reports = {}
        for alpha in ALPHAS:
            per_seed = []
            for seed, model_c in zip(SEEDS, models_c):
                original = inject_layer_delta(model_c, layer_delta, alpha)
                auto_feats = cifar_fc1(model_c, X_auto)
                truck_feats = cifar_fc1(model_c, X_truck)
                has_nan = bool(np.isnan(auto_feats).any() or np.isnan(truck_feats).any())
                if has_nan:
                    metrics = None
                else:
                    metrics = {
                        "probe_accuracy": probe_accuracy(truck_feats, auto_feats),
                        "missing_true_rate": nearest_true_rate(truck_feats, "truck", a_centroids[seed]),
                        "seen_true_rate": nearest_true_rate(auto_feats, "automobile", a_centroids[seed]),
                        "missing_true_cosine": mean_true_cosine(truck_feats, "truck", a_centroids[seed]),
                    }
                restore_model(model_c, original)
                per_seed.append(metrics)

            valid = [item for item in per_seed if item is not None]
            if not valid:
                alpha_reports[f"{alpha:.2f}"] = {"nan": True, "purity_score": None}
                continue

            avg = {key: float(np.mean([item[key] for item in valid])) for key in valid[0]}
            avg["nan"] = len(valid) != len(per_seed)
            avg["purity_score"] = score_metrics(avg, baseline)
            alpha_reports[f"{alpha:.2f}"] = avg

        best_alpha, best_metrics = max(
            alpha_reports.items(),
            key=lambda item: -1e9 if item[1]["purity_score"] is None else item[1]["purity_score"],
        )
        layers[layer] = {
            "best_alpha": best_alpha,
            "best_metrics": best_metrics,
            "all_alphas": alpha_reports,
        }

    ranked = sorted(
        [{"layer": layer, **info} for layer, info in layers.items()],
        key=lambda item: -1e9 if item["best_metrics"]["purity_score"] is None else item["best_metrics"]["purity_score"],
        reverse=True,
    )

    return {
        "dataset": "cifar10",
        "missing_class": "truck",
        "contrast_class": "automobile",
        "baseline_C": baseline,
        "ranked_layers": ranked,
    }


def print_top(report):
    top = report["ranked_layers"][:3]
    print(f"\n=== {report['dataset']} ===")
    print(f"Missing class: {report['missing_class']} | Contrast class: {report['contrast_class']}")
    print(f"Baseline C: {json.dumps(report['baseline_C'], indent=2)}")
    for item in top:
        print(
            f"Layer {item['layer']} @ alpha {item['best_alpha']} -> "
            f"purity={item['best_metrics']['purity_score']:.4f}, "
            f"probe={item['best_metrics']['probe_accuracy']:.4f}, "
            f"missing_rate={item['best_metrics']['missing_true_rate']:.4f}, "
            f"seen_rate={item['best_metrics']['seen_true_rate']:.4f}"
        )


def main():
    os.makedirs("results", exist_ok=True)
    summary = {
        "color_blocks": color_report(),
        "cifar10": cifar_report(),
    }
    output_path = "results/layer_concept_report.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_top(summary["color_blocks"])
    print_top(summary["cifar10"])
    print(f"\nSaved detailed report to {output_path}")


if __name__ == "__main__":
    main()
