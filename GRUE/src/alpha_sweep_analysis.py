import argparse
import json
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from concept_extraction import get_shared_weights
from evaluate import load_data
from load_models import load_model
from run_experiment import get_class_images, load_cifar_model
from cifar_dataset import load_raw_cifar10


def average_diff(weight_sets_a, weight_sets_other):
    return {
        k: mx.mean(mx.stack([wa[k] - wb[k] for wa, wb in zip(weight_sets_a, weight_sets_other)]), axis=0)
        for k in weight_sets_a[0]
    }


def scale_vector(vector, alpha):
    return {k: alpha * v for k, v in vector.items()}


def random_like(vector, alpha):
    random_vector = {}
    for k, v in vector.items():
        rand_v = mx.random.normal(v.shape)
        random_vector[k] = alpha * rand_v * (mx.linalg.norm(v) / (mx.linalg.norm(rand_v) + 1e-8))
    return random_vector


def inject_vector(model, vector):
    flat_params = dict(tree_flatten(model.parameters()))
    original_params = dict(flat_params)
    for k, v in vector.items():
        if k in flat_params:
            flat_params[k] = flat_params[k] + v
    model.update(tree_unflatten(list(flat_params.items())))
    mx.eval(model.parameters())
    return original_params


def restore_model(model, original_params):
    model.update(tree_unflatten(list(original_params.items())))
    mx.eval(model.parameters())


def probe_accuracy(features, labels):
    classifier = LogisticRegression(max_iter=1000)
    scores = cross_val_score(classifier, features, labels, cv=5, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def collect_fc1_color(model, X, batch_size=256):
    features = []
    for i in range(0, len(X), batch_size):
        x = mx.array(X[i:i + batch_size])
        x = model.pool1(nn.relu(model.bn1(model.conv1(x))))
        x = model.pool2(nn.relu(model.bn2(model.conv2(x))))
        x = model.pool3(nn.relu(model.bn3(model.conv3(x))))
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(model.fc1(x))
        features.append(np.array(x))
    return np.concatenate(features)


def collect_fc1_cifar(model, X, batch_size=256):
    features = []
    for i in range(0, len(X), batch_size):
        _, activations = model(mx.array(X[i:i + batch_size]), capture_activations=True)
        features.append(np.array(activations["fc1"]))
    return np.concatenate(features)


def color_probe_dataset(max_images):
    X_val, _, _, _ = load_data("val", "A")
    with open("dataset/labels_A_val.json", "r") as f:
        labels = json.load(f)
    with open("dataset/metadata_val.json", "r") as f:
        metadata = json.load(f)

    blue_indices = [i for i, item in enumerate(metadata) if labels[item["filename"]] == "blue"][:max_images]
    green_indices = [i for i, item in enumerate(metadata) if labels[item["filename"]] == "green"][:max_images]
    X = np.concatenate([X_val[blue_indices], X_val[green_indices]])
    y = np.array([0] * len(blue_indices) + [1] * len(green_indices))
    return X, y


def cifar_probe_dataset(max_images):
    _, _, X_test, y_test = load_raw_cifar10()
    X_auto = get_class_images(X_test, y_test, "automobile", max_images)
    X_truck = get_class_images(X_test, y_test, "truck", max_images)
    X = np.concatenate([X_auto, X_truck])
    y = np.array([0] * len(X_auto) + [1] * len(X_truck))
    return X, y


def sweep_single_model(model, base_vector, collect_features, X, y, alphas, random_trials):
    baseline_features = collect_features(model, X)
    baseline_has_nan = bool(np.isnan(baseline_features).any())
    baseline_acc = None
    baseline_std = None
    if not baseline_has_nan:
        baseline_acc, baseline_std = probe_accuracy(baseline_features, y)

    results = {
        "baseline": {
            "probe_accuracy": baseline_acc,
            "probe_std": baseline_std,
            "has_nan": baseline_has_nan,
        },
        "alphas": {},
    }

    for alpha in alphas:
        alpha_key = f"{alpha:.3f}"
        real_vector = scale_vector(base_vector, alpha)
        original_params = inject_vector(model, real_vector)
        real_features = collect_features(model, X)
        real_has_nan = bool(np.isnan(real_features).any())
        if real_has_nan:
            real_acc = None
            real_std = None
        else:
            real_acc, real_std = probe_accuracy(real_features, y)
        restore_model(model, original_params)

        random_runs = []
        for _ in range(random_trials):
            original_params = inject_vector(model, random_like(base_vector, alpha))
            random_features = collect_features(model, X)
            random_has_nan = bool(np.isnan(random_features).any())
            if random_has_nan:
                random_runs.append({"probe_accuracy": None, "probe_std": None, "has_nan": True})
            else:
                random_acc, random_std = probe_accuracy(random_features, y)
                random_runs.append({"probe_accuracy": random_acc, "probe_std": random_std, "has_nan": False})
            restore_model(model, original_params)

        valid_random_accs = [run["probe_accuracy"] for run in random_runs if run["probe_accuracy"] is not None]
        results["alphas"][alpha_key] = {
            "real": {
                "probe_accuracy": real_acc,
                "probe_std": real_std,
                "has_nan": real_has_nan,
                "gain_over_baseline": None if real_acc is None or baseline_acc is None else real_acc - baseline_acc,
            },
            "random": {
                "mean_probe_accuracy": None if not valid_random_accs else float(np.mean(valid_random_accs)),
                "std_probe_accuracy": None if not valid_random_accs else float(np.std(valid_random_accs)),
                "nan_rate": float(np.mean([run["has_nan"] for run in random_runs])),
            },
        }
        if results["alphas"][alpha_key]["real"]["probe_accuracy"] is not None and results["alphas"][alpha_key]["random"]["mean_probe_accuracy"] is not None:
            results["alphas"][alpha_key]["real"]["gain_vs_random_mean"] = (
                results["alphas"][alpha_key]["real"]["probe_accuracy"] - results["alphas"][alpha_key]["random"]["mean_probe_accuracy"]
            )
        else:
            results["alphas"][alpha_key]["real"]["gain_vs_random_mean"] = None

    return results


def run_color_analysis(max_images, alphas, random_trials):
    seeds = [1, 2, 3, 4, 5]
    models_a = [load_model("A", seed)[0] for seed in seeds]
    models_b = [load_model("B", seed)[0] for seed in seeds]
    models_c = [load_model("C", seed)[0] for seed in seeds]

    vector_ab = average_diff(
        [{k: v for k, v in tree_flatten(model.parameters()) if "fc2" not in k} for model in models_a],
        [{k: v for k, v in tree_flatten(model.parameters()) if "fc2" not in k} for model in models_b],
    )
    vector_ac = average_diff(
        [{k: v for k, v in tree_flatten(model.parameters()) if "fc2" not in k} for model in models_a],
        [{k: v for k, v in tree_flatten(model.parameters()) if "fc2" not in k} for model in models_c],
    )

    X, y = color_probe_dataset(max_images)
    return {
        "dataset": "color_blocks",
        "probe_task": "blue_vs_green",
        "B_plus_AB": sweep_single_model(models_b[0], vector_ab, collect_fc1_color, X, y, alphas, random_trials),
        "C_plus_AC": sweep_single_model(models_c[0], vector_ac, collect_fc1_color, X, y, alphas, random_trials),
    }


def run_cifar_analysis(max_images, alphas, random_trials):
    seeds = [1, 2, 3, 4, 5]
    models_a = [load_cifar_model("A", seed)[0] for seed in seeds]
    models_b = [load_cifar_model("B", seed)[0] for seed in seeds]
    models_c = [load_cifar_model("C", seed)[0] for seed in seeds]

    vector_ab = average_diff(
        [get_shared_weights(model, exclude_bn_stats=True) for model in models_a],
        [get_shared_weights(model, exclude_bn_stats=True) for model in models_b],
    )
    vector_ac = average_diff(
        [get_shared_weights(model, exclude_bn_stats=True) for model in models_a],
        [get_shared_weights(model, exclude_bn_stats=True) for model in models_c],
    )

    X, y = cifar_probe_dataset(max_images)
    return {
        "dataset": "cifar10",
        "probe_task": "automobile_vs_truck",
        "B_plus_AB": sweep_single_model(models_b[0], vector_ab, collect_fc1_cifar, X, y, alphas, random_trials),
        "C_plus_AC": sweep_single_model(models_c[0], vector_ac, collect_fc1_cifar, X, y, alphas, random_trials),
    }


def print_summary(name, results):
    print(f"\n=== {name} ===")
    for condition in ["B_plus_AB", "C_plus_AC"]:
        baseline = results[condition]["baseline"]
        print(f"\n{condition}")
        print(f"  baseline: acc={baseline['probe_accuracy']}, std={baseline['probe_std']}, nan={baseline['has_nan']}")
        for alpha_key, alpha_result in results[condition]["alphas"].items():
            real = alpha_result["real"]
            random = alpha_result["random"]
            print(
                f"  alpha={alpha_key}: "
                f"real_acc={real['probe_accuracy']}, gain={real['gain_over_baseline']}, "
                f"gain_vs_random={real['gain_vs_random_mean']}, real_nan={real['has_nan']}, "
                f"random_mean={random['mean_probe_accuracy']}, random_nan_rate={random['nan_rate']}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["color_blocks", "cifar10"], default=["color_blocks", "cifar10"])
    parser.add_argument("--max_images", type=int, default=500)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--random_trials", type=int, default=5)
    parser.add_argument("--output", type=str, default="results/alpha_sweep_summary.json")
    args = parser.parse_args()

    np.random.seed(0)
    mx.random.seed(0)

    summary = {
        "alphas": args.alphas,
        "random_trials": args.random_trials,
        "max_images": args.max_images,
        "results": {},
    }

    if "color_blocks" in args.datasets:
        summary["results"]["color_blocks"] = run_color_analysis(args.max_images, args.alphas, args.random_trials)
        print_summary("color_blocks", summary["results"]["color_blocks"])

    if "cifar10" in args.datasets:
        summary["results"]["cifar10"] = run_cifar_analysis(args.max_images, args.alphas, args.random_trials)
        print_summary("cifar10", summary["results"]["cifar10"])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {args.output}")


if __name__ == "__main__":
    main()
