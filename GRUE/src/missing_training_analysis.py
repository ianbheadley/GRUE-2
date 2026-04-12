import json
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from cifar_dataset import CIFAR10_LABELS, load_cifar10, load_raw_cifar10
from load_models import load_model
from run_experiment import load_cifar_model
from evaluate import load_data


def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


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
        _, activations = model(mx.array(X[i:i + batch_size]), capture_activations=True)
        feats.append(np.array(activations["fc1"]))
    return np.concatenate(feats)


def probe_accuracy(features_a, features_b):
    X = np.concatenate([features_a, features_b], axis=0)
    y = np.array([0] * len(features_a) + [1] * len(features_b))
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def nearest_centroid_report(features, centroids):
    labels = list(centroids.keys())
    centroid_stack = np.stack([centroids[label] for label in labels], axis=0)
    dists = np.linalg.norm(features[:, None, :] - centroid_stack[None, :, :], axis=2)
    nearest = np.argmin(dists, axis=1)
    counts = {}
    for idx in nearest:
        label = labels[int(idx)]
        counts[label] = counts.get(label, 0) + 1
    total = len(features)
    return {label: counts.get(label, 0) / total for label in labels}


def summarize_alignment(vectors_left, vectors_right):
    sims = [cosine(a, b) for a, b in zip(vectors_left, vectors_right)]
    return {"mean_cosine": float(np.mean(sims)), "std_cosine": float(np.std(sims))}


def sample_per_label(features, labels, idx_to_label, max_per_label):
    centroids = {}
    for idx, label_name in idx_to_label.items():
        label_idx = int(idx)
        positions = np.where(labels == label_idx)[0][:max_per_label]
        centroids[label_name] = features[positions].mean(axis=0)
    return centroids


def select_indices_per_label(labels, label_indices, max_per_label):
    selected = []
    for label_idx in label_indices:
        selected.extend(np.where(labels == int(label_idx))[0][:max_per_label].tolist())
    return np.array(selected, dtype=int)


def color_analysis(max_hidden=400, max_train_per_label=400):
    seeds = [1, 2, 3, 4, 5]
    reports = {"dataset": "color_blocks", "hidden_task": "blue_vs_green", "schemes": {}}

    X_eval, _, _, _ = load_data("val", "A")
    with open("dataset/labels_A_val.json", "r") as f:
        eval_labels = json.load(f)
    with open("dataset/metadata_val.json", "r") as f:
        eval_meta = json.load(f)
    blue_idx = [i for i, item in enumerate(eval_meta) if eval_labels[item["filename"]] == "blue"][:max_hidden]
    green_idx = [i for i, item in enumerate(eval_meta) if eval_labels[item["filename"]] == "green"][:max_hidden]
    X_blue = X_eval[blue_idx]
    X_green = X_eval[green_idx]

    X_train_A, y_train_A, _, idx_to_A = load_data("train", "A")
    X_train_B, y_train_B, _, idx_to_B = load_data("train", "B")
    X_train_C, y_train_C, _, idx_to_C = load_data("train", "C")
    train_cache = {
        "A": (X_train_A, y_train_A, idx_to_A),
        "B": (X_train_B, y_train_B, idx_to_B),
        "C": (X_train_C, y_train_C, idx_to_C),
    }

    pair_vectors = {"A": [], "B": [], "C": []}
    nearest_reports = {"A": [], "B": [], "C": []}
    probe_reports = {"A": [], "B": [], "C": []}

    for scheme in ["A", "B", "C"]:
        for seed in seeds:
            model, _ = load_model(scheme, seed)
            X_train, y_train, idx_to_label = train_cache[scheme]
            selected = select_indices_per_label(y_train, idx_to_label.keys(), max_train_per_label)
            train_feats = color_fc1(model, X_train[selected])
            train_labels = y_train[selected]
            centroids = sample_per_label(train_feats, train_labels, idx_to_label, max_train_per_label)

            blue_feats = color_fc1(model, X_blue)
            green_feats = color_fc1(model, X_green)
            pair_vectors[scheme].append(blue_feats.mean(axis=0) - green_feats.mean(axis=0))
            probe_reports[scheme].append(probe_accuracy(blue_feats, green_feats)[0])
            nearest_reports[scheme].append({
                "blue": nearest_centroid_report(blue_feats, centroids),
                "green": nearest_centroid_report(green_feats, centroids),
            })

        reports["schemes"][scheme] = {
            "probe_accuracy_mean": float(np.mean(probe_reports[scheme])),
            "probe_accuracy_std": float(np.std(probe_reports[scheme])),
            "blue_nearest_centroid_mean": {
                label: float(np.mean([r["blue"].get(label, 0.0) for r in nearest_reports[scheme]]))
                for label in sorted({label for r in nearest_reports[scheme] for label in r["blue"].keys()})
            },
            "green_nearest_centroid_mean": {
                label: float(np.mean([r["green"].get(label, 0.0) for r in nearest_reports[scheme]]))
                for label in sorted({label for r in nearest_reports[scheme] for label in r["green"].keys()})
            },
        }

    reports["alignment"] = {
        "A_vs_B_hidden_direction": summarize_alignment(pair_vectors["A"], pair_vectors["B"]),
        "A_vs_C_hidden_direction": summarize_alignment(pair_vectors["A"], pair_vectors["C"]),
        "B_vs_C_hidden_direction": summarize_alignment(pair_vectors["B"], pair_vectors["C"]),
    }
    return reports


def cifar_analysis(max_hidden=400, max_train_per_label=400):
    seeds = [1, 2, 3, 4, 5]
    reports = {"dataset": "cifar10", "hidden_task": "automobile_vs_truck", "schemes": {}}

    _, _, X_test, y_test = load_raw_cifar10()
    auto_idx = np.where(y_test == CIFAR10_LABELS.index("automobile"))[0][:max_hidden]
    truck_idx = np.where(y_test == CIFAR10_LABELS.index("truck"))[0][:max_hidden]
    X_auto = X_test[auto_idx]
    X_truck = X_test[truck_idx]

    train_cache = {}
    for scheme in ["A", "B", "C"]:
        X_train, y_train, _, _, label_to_idx, idx_to_label = load_cifar10(scheme)
        train_cache[scheme] = (X_train, y_train, idx_to_label)

    pair_vectors = {"A": [], "B": [], "C": []}
    nearest_reports = {"A": [], "B": [], "C": []}
    probe_reports = {"A": [], "B": [], "C": []}

    for scheme in ["A", "B", "C"]:
        for seed in seeds:
            model, _ = load_cifar_model(scheme, seed)
            X_train, y_train, idx_to_label = train_cache[scheme]
            selected = select_indices_per_label(y_train, idx_to_label.keys(), max_train_per_label)
            train_feats_full = cifar_fc1(model, X_train[selected])
            train_labels = y_train[selected]
            centroids = {}
            for label_idx, label_name in idx_to_label.items():
                positions = np.where(train_labels == int(label_idx))[0][:max_train_per_label]
                centroids[label_name] = train_feats_full[positions].mean(axis=0)

            auto_feats = cifar_fc1(model, X_auto)
            truck_feats = cifar_fc1(model, X_truck)
            pair_vectors[scheme].append(auto_feats.mean(axis=0) - truck_feats.mean(axis=0))
            probe_reports[scheme].append(probe_accuracy(auto_feats, truck_feats)[0])
            nearest_reports[scheme].append({
                "automobile": nearest_centroid_report(auto_feats, centroids),
                "truck": nearest_centroid_report(truck_feats, centroids),
            })

        reports["schemes"][scheme] = {
            "probe_accuracy_mean": float(np.mean(probe_reports[scheme])),
            "probe_accuracy_std": float(np.std(probe_reports[scheme])),
            "automobile_nearest_centroid_mean": {
                label: float(np.mean([r["automobile"].get(label, 0.0) for r in nearest_reports[scheme]]))
                for label in sorted({label for r in nearest_reports[scheme] for label in r["automobile"].keys()})
            },
            "truck_nearest_centroid_mean": {
                label: float(np.mean([r["truck"].get(label, 0.0) for r in nearest_reports[scheme]]))
                for label in sorted({label for r in nearest_reports[scheme] for label in r["truck"].keys()})
            },
        }

    reports["alignment"] = {
        "A_vs_B_hidden_direction": summarize_alignment(pair_vectors["A"], pair_vectors["B"]),
        "A_vs_C_hidden_direction": summarize_alignment(pair_vectors["A"], pair_vectors["C"]),
        "B_vs_C_hidden_direction": summarize_alignment(pair_vectors["B"], pair_vectors["C"]),
    }
    return reports


def main():
    os.makedirs("results", exist_ok=True)
    summary = {
        "color_blocks": color_analysis(),
        "cifar10": cifar_analysis(),
    }
    output_path = "results/missing_training_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {output_path}")


if __name__ == "__main__":
    main()
