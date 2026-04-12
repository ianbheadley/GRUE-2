import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from activation_alignment_experiment import (
    LAYER_NAMES,
    build_split,
    evaluate_a_head,
    evaluate_probe_condition,
    extract_features,
    fit_probe,
)
from load_models import load_model


@dataclass
class OrthogonalMap:
    source_mean: np.ndarray
    target_mean: np.ndarray
    rotation: np.ndarray


def fit_orthogonal_map(source_features, target_features):
    source_mean = source_features.mean(axis=0, keepdims=True)
    target_mean = target_features.mean(axis=0, keepdims=True)

    source_centered = source_features - source_mean
    target_centered = target_features - target_mean

    cross_covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(cross_covariance, full_matrices=False)
    rotation = u @ vt

    return OrthogonalMap(
        source_mean=source_mean,
        target_mean=target_mean,
        rotation=rotation,
    )


def apply_orthogonal_map(transform, source_features):
    centered = source_features - transform.source_mean
    return centered @ transform.rotation + transform.target_mean


def fit_binary_probe(features, y_binary):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1500,
            solver="lbfgs",
        ),
    ).fit(features, y_binary)


def evaluate_binary_probe(probe, features, y_binary):
    return float(probe.score(features, y_binary))


def compute_residual_stats(residuals, y, blue_idx, green_idx):
    target_mask = np.logical_or(y == blue_idx, y == green_idx)
    other_mask = np.logical_not(target_mask)

    norms = np.linalg.norm(residuals, axis=1)
    target_mean = float(norms[target_mask].mean())
    other_mean = float(norms[other_mask].mean())

    return {
        "mean_norm_target": target_mean,
        "mean_norm_other": other_mean,
        "target_other_ratio": target_mean / max(other_mean, 1e-8),
    }


def evaluate_residual_blue_green_probe(train_residuals, train_y, val_residuals, val_y, blue_idx, green_idx):
    train_mask = np.logical_or(train_y == blue_idx, train_y == green_idx)
    val_mask = np.logical_or(val_y == blue_idx, val_y == green_idx)

    train_binary = (train_y[train_mask] == blue_idx).astype(np.int64)
    val_binary = (val_y[val_mask] == blue_idx).astype(np.int64)

    probe = fit_binary_probe(train_residuals[train_mask], train_binary)
    return {
        "blue_green_probe_accuracy": evaluate_binary_probe(probe, val_residuals[val_mask], val_binary),
        "blue_green_count": int(val_mask.sum()),
    }


def summarize_metric(per_seed_results, layer, section, metric):
    values = [seed_result["layers"][layer][section][metric] for seed_result in per_seed_results]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "values": [float(v) for v in values],
    }


def summarize_fc1_head_metric(per_seed_results, condition, metric):
    values = [seed_result["fc1_a_head"][condition][metric] for seed_result in per_seed_results]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "values": [float(v) for v in values],
    }


def run_seed(seed, train_split, val_split, base_model_dir):
    print(f"\nSeed {seed}: loading A/B models and extracting activations...")
    model_a, meta_a = load_model("A", seed, base_model_dir=base_model_dir)
    model_b, _ = load_model("B", seed, base_model_dir=base_model_dir)

    train_a = extract_features(model_a, train_split.X, batch_size=256)
    train_b = extract_features(model_b, train_split.X, batch_size=256)
    val_a = extract_features(model_a, val_split.X, batch_size=256)
    val_b = extract_features(model_b, val_split.X, batch_size=256)

    blue_idx = meta_a["label_map"]["blue"]
    green_idx = meta_a["label_map"]["green"]

    seed_report = {
        "seed": seed,
        "layers": {},
        "fc1_a_head": {},
    }

    for layer in LAYER_NAMES:
        print(f"  Layer {layer}: fitting A probe and orthogonal alignment...")
        probe_a = fit_probe(train_a[layer], train_split.y)
        transform = fit_orthogonal_map(train_b[layer], train_a[layer])
        mapped_b_val = apply_orthogonal_map(transform, val_b[layer])
        mapped_b_train = apply_orthogonal_map(transform, train_b[layer])

        raw_residual_train = train_a[layer] - train_b[layer]
        raw_residual_val = val_a[layer] - val_b[layer]
        rotated_residual_train = train_a[layer] - mapped_b_train
        rotated_residual_val = val_a[layer] - mapped_b_val

        layer_report = {
            "a_reference": evaluate_probe_condition(
                probe_a, val_a[layer], val_split.y, blue_idx, green_idx
            ),
            "raw_b_in_a_probe": evaluate_probe_condition(
                probe_a, val_b[layer], val_split.y, blue_idx, green_idx
            ),
            "rotated_b_to_a": evaluate_probe_condition(
                probe_a, mapped_b_val, val_split.y, blue_idx, green_idx
            ),
            "raw_residual": compute_residual_stats(
                raw_residual_val, val_split.y, blue_idx, green_idx
            ),
            "rotated_residual": compute_residual_stats(
                rotated_residual_val, val_split.y, blue_idx, green_idx
            ),
        }

        layer_report["raw_residual"].update(
            evaluate_residual_blue_green_probe(
                raw_residual_train,
                train_split.y,
                raw_residual_val,
                val_split.y,
                blue_idx,
                green_idx,
            )
        )
        layer_report["rotated_residual"].update(
            evaluate_residual_blue_green_probe(
                rotated_residual_train,
                train_split.y,
                rotated_residual_val,
                val_split.y,
                blue_idx,
                green_idx,
            )
        )

        seed_report["layers"][layer] = layer_report

        if layer == "fc1":
            seed_report["fc1_a_head"] = {
                "a_reference": evaluate_a_head(model_a, val_a[layer], val_split.y, blue_idx, green_idx),
                "raw_b_in_a_head": evaluate_a_head(model_a, val_b[layer], val_split.y, blue_idx, green_idx),
                "rotated_b_to_a": evaluate_a_head(model_a, mapped_b_val, val_split.y, blue_idx, green_idx),
            }

    return seed_report


def build_summary(per_seed_results):
    summary = {
        "layers": {},
        "fc1_a_head": {},
    }

    for layer in LAYER_NAMES:
        summary["layers"][layer] = {
            "a_reference": {
                "overall_accuracy": summarize_metric(per_seed_results, layer, "a_reference", "overall_accuracy"),
                "blue_green_binary_accuracy": summarize_metric(
                    per_seed_results, layer, "a_reference", "blue_green_binary_accuracy"
                ),
            },
            "raw_b_in_a_probe": {
                "overall_accuracy": summarize_metric(per_seed_results, layer, "raw_b_in_a_probe", "overall_accuracy"),
                "blue_green_binary_accuracy": summarize_metric(
                    per_seed_results, layer, "raw_b_in_a_probe", "blue_green_binary_accuracy"
                ),
            },
            "rotated_b_to_a": {
                "overall_accuracy": summarize_metric(per_seed_results, layer, "rotated_b_to_a", "overall_accuracy"),
                "blue_green_binary_accuracy": summarize_metric(
                    per_seed_results, layer, "rotated_b_to_a", "blue_green_binary_accuracy"
                ),
            },
            "raw_residual": {
                "mean_norm_target": summarize_metric(per_seed_results, layer, "raw_residual", "mean_norm_target"),
                "mean_norm_other": summarize_metric(per_seed_results, layer, "raw_residual", "mean_norm_other"),
                "target_other_ratio": summarize_metric(per_seed_results, layer, "raw_residual", "target_other_ratio"),
                "blue_green_probe_accuracy": summarize_metric(
                    per_seed_results, layer, "raw_residual", "blue_green_probe_accuracy"
                ),
            },
            "rotated_residual": {
                "mean_norm_target": summarize_metric(per_seed_results, layer, "rotated_residual", "mean_norm_target"),
                "mean_norm_other": summarize_metric(
                    per_seed_results, layer, "rotated_residual", "mean_norm_other"
                ),
                "target_other_ratio": summarize_metric(
                    per_seed_results, layer, "rotated_residual", "target_other_ratio"
                ),
                "blue_green_probe_accuracy": summarize_metric(
                    per_seed_results, layer, "rotated_residual", "blue_green_probe_accuracy"
                ),
            },
        }

    for condition in ["a_reference", "raw_b_in_a_head", "rotated_b_to_a"]:
        summary["fc1_a_head"][condition] = {
            "overall_accuracy": summarize_fc1_head_metric(per_seed_results, condition, "overall_accuracy"),
            "blue_green_binary_accuracy": summarize_fc1_head_metric(
                per_seed_results, condition, "blue_green_binary_accuracy"
            ),
        }

    return summary


def print_summary(summary):
    print("\nRotation-only alignment with A probe (mean across seeds):")
    for layer in LAYER_NAMES:
        a_ref = summary["layers"][layer]["a_reference"]["overall_accuracy"]["mean"]
        raw_b = summary["layers"][layer]["raw_b_in_a_probe"]["overall_accuracy"]["mean"]
        rotated = summary["layers"][layer]["rotated_b_to_a"]["overall_accuracy"]["mean"]
        bg = summary["layers"][layer]["rotated_b_to_a"]["blue_green_binary_accuracy"]["mean"]
        print(
            f"  {layer:5s} | A probe on A={a_ref:.4f} | raw B={raw_b:.4f} | "
            f"rotated B={rotated:.4f} | rotated blue/green={bg:.4f}"
        )

    print("\nfc1 with A's real output head (mean across seeds):")
    for condition in ["a_reference", "raw_b_in_a_head", "rotated_b_to_a"]:
        overall = summary["fc1_a_head"][condition]["overall_accuracy"]["mean"]
        bg = summary["fc1_a_head"][condition]["blue_green_binary_accuracy"]["mean"]
        print(f"  {condition:15s} | overall={overall:.4f} | blue/green={bg:.4f}")

    print("\nResidual subtraction after rotation (mean across seeds):")
    for layer in LAYER_NAMES:
        raw_ratio = summary["layers"][layer]["raw_residual"]["target_other_ratio"]["mean"]
        rot_ratio = summary["layers"][layer]["rotated_residual"]["target_other_ratio"]["mean"]
        raw_probe = summary["layers"][layer]["raw_residual"]["blue_green_probe_accuracy"]["mean"]
        rot_probe = summary["layers"][layer]["rotated_residual"]["blue_green_probe_accuracy"]["mean"]
        print(
            f"  {layer:5s} | residual norm ratio raw={raw_ratio:.3f} rotated={rot_ratio:.3f} | "
            f"blue/green probe raw={raw_probe:.4f} rotated={rot_probe:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--base_model_dir", type=str, default="models")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--train_per_class", type=int, default=800)
    parser.add_argument("--val_per_class", type=int, default=300)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default="results/orthogonal_alignment_summary.json",
    )
    args = parser.parse_args()

    print("Preparing stratified train/val subsets...")
    train_split = build_split("train", args.dataset_dir, args.train_per_class, sample_seed=args.sample_seed)
    val_split = build_split("val", args.dataset_dir, args.val_per_class, sample_seed=args.sample_seed + 1)
    print(
        f"Train subset: {len(train_split.y)} images | "
        f"Val subset: {len(val_split.y)} images | "
        f"Classes: {len(train_split.label_to_idx)}"
    )

    per_seed_results = []
    for seed in args.seeds:
        per_seed_results.append(run_seed(seed, train_split, val_split, args.base_model_dir))

    summary = build_summary(per_seed_results)
    print_summary(summary)

    payload = {
        "config": {
            "dataset_dir": args.dataset_dir,
            "base_model_dir": args.base_model_dir,
            "seeds": args.seeds,
            "train_per_class": args.train_per_class,
            "val_per_class": args.val_per_class,
            "sample_seed": args.sample_seed,
        },
        "per_seed": per_seed_results,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
