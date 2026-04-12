import argparse
import json
import os
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from concept_extraction import get_activations, global_avg_pool
from load_models import load_model


LAYER_NAMES = ["conv1", "conv2", "conv3", "fc1"]


@dataclass
class SplitData:
    X: np.ndarray
    y: np.ndarray
    filenames: list[str]
    label_to_idx: dict[str, int]
    idx_to_label: dict[int, str]


def load_metadata(split, base_dir):
    meta_path = os.path.join(base_dir, f"metadata_{split}.json")
    label_path = os.path.join(base_dir, f"labels_A_{split}.json")

    with open(meta_path, "r") as f:
        metadata = json.load(f)
    with open(label_path, "r") as f:
        labels = json.load(f)

    unique_labels = sorted(set(labels.values()))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    rows = []
    for item in metadata:
        filename = item["filename"]
        label_name = labels[filename]
        rows.append(
            {
                "filename": filename,
                "label_name": label_name,
                "label_idx": label_to_idx[label_name],
            }
        )

    return rows, label_to_idx


def stratified_sample(rows, max_per_class, rng):
    by_class = {}
    for row in rows:
        by_class.setdefault(row["label_idx"], []).append(row)

    selected = []
    for _, class_rows in sorted(by_class.items()):
        order = rng.permutation(len(class_rows))
        take = min(max_per_class, len(class_rows))
        selected.extend(class_rows[i] for i in order[:take])

    selected.sort(key=lambda row: row["filename"])
    return selected


def load_images(rows, split, base_dir):
    X = []
    y = []
    filenames = []
    for row in rows:
        img_path = os.path.join(base_dir, split, row["filename"])
        img = Image.open(img_path).convert("RGB")
        X.append(np.array(img, dtype=np.float32) / 255.0)
        y.append(row["label_idx"])
        filenames.append(row["filename"])
    return np.stack(X), np.array(y, dtype=np.int64), filenames


def build_split(split, base_dir, max_per_class, sample_seed):
    rows, label_to_idx = load_metadata(split, base_dir)
    rng = np.random.default_rng(sample_seed)
    selected = stratified_sample(rows, max_per_class, rng)
    X, y, filenames = load_images(selected, split, base_dir)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return SplitData(X, y, filenames, label_to_idx, idx_to_label)


def extract_features(model, X, batch_size):
    raw = get_activations(model, X, batch_size=batch_size)
    return {
        layer: global_avg_pool(raw[layer]).astype(np.float64, copy=False)
        for layer in LAYER_NAMES
    }


def fit_probe(features, y):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1500,
            solver="lbfgs",
        ),
    ).fit(features, y)


def fit_map(source_features, target_features, alpha):
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha, fit_intercept=True),
    ).fit(source_features, target_features)


def accuracy_from_scores(scores, y_true):
    preds = scores.argmax(axis=1)
    return float(np.mean(preds == y_true))


def blue_green_binary_accuracy(scores, y_true, blue_idx, green_idx):
    mask = np.logical_or(y_true == blue_idx, y_true == green_idx)
    if mask.sum() == 0:
        return float("nan"), 0
    bg_scores = scores[mask][:, [blue_idx, green_idx]]
    preds = np.where(bg_scores[:, 0] >= bg_scores[:, 1], blue_idx, green_idx)
    return float(np.mean(preds == y_true[mask])), int(mask.sum())


def mean_cosine_similarity(predicted, target):
    pred_norm = np.linalg.norm(predicted, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    denom = np.clip(pred_norm * target_norm, 1e-8, None)
    cosines = np.sum(predicted * target, axis=1) / denom
    return float(np.mean(cosines))


def evaluate_probe_condition(probe, features, y_true, blue_idx, green_idx):
    probs = probe.predict_proba(features)
    overall = float(probe.score(features, y_true))
    bg_acc, bg_count = blue_green_binary_accuracy(probs, y_true, blue_idx, green_idx)
    return {
        "overall_accuracy": overall,
        "blue_green_binary_accuracy": bg_acc,
        "blue_green_count": bg_count,
    }


def evaluate_a_head(model_a, hidden_features, y_true, blue_idx, green_idx):
    logits = np.array(model_a.fc2(mx.array(hidden_features)))
    overall = accuracy_from_scores(logits, y_true)
    bg_acc, bg_count = blue_green_binary_accuracy(logits, y_true, blue_idx, green_idx)
    return {
        "overall_accuracy": overall,
        "blue_green_binary_accuracy": bg_acc,
        "blue_green_count": bg_count,
    }


def summarize_metric(per_seed_results, layer, condition, metric):
    values = [
        seed_result["layers"][layer][condition][metric]
        for seed_result in per_seed_results
        if metric in seed_result["layers"][layer][condition]
    ]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "values": [float(v) for v in values],
    }


def summarize_fc1_head_metric(per_seed_results, condition, metric):
    values = [
        seed_result["fc1_a_head"][condition][metric]
        for seed_result in per_seed_results
        if metric in seed_result["fc1_a_head"][condition]
    ]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "values": [float(v) for v in values],
    }


def run_seed(seed, train_split, val_split, base_model_dir, batch_size, ridge_alpha, alignment_mask):
    print(f"\nSeed {seed}: loading models and extracting activations...")
    model_a, meta_a = load_model("A", seed, base_model_dir=base_model_dir)
    model_b, _ = load_model("B", seed, base_model_dir=base_model_dir)
    model_c, _ = load_model("C", seed, base_model_dir=base_model_dir)

    train_a = extract_features(model_a, train_split.X, batch_size)
    train_b = extract_features(model_b, train_split.X, batch_size)
    train_c = extract_features(model_c, train_split.X, batch_size)

    val_a = extract_features(model_a, val_split.X, batch_size)
    val_b = extract_features(model_b, val_split.X, batch_size)
    val_c = extract_features(model_c, val_split.X, batch_size)

    blue_idx = meta_a["label_map"]["blue"]
    green_idx = meta_a["label_map"]["green"]

    seed_report = {
        "seed": seed,
        "layers": {},
        "fc1_a_head": {},
    }

    for layer in LAYER_NAMES:
        print(f"  Layer {layer}: fitting A-space probe and alignment maps...")
        probe_a = fit_probe(train_a[layer], train_split.y)

        map_b = fit_map(train_b[layer][alignment_mask], train_a[layer][alignment_mask], alpha=ridge_alpha)
        map_c = fit_map(train_c[layer][alignment_mask], train_a[layer][alignment_mask], alpha=ridge_alpha)
        map_bc = fit_map(
            np.concatenate([train_b[layer][alignment_mask], train_c[layer][alignment_mask]], axis=1),
            train_a[layer][alignment_mask],
            alpha=ridge_alpha,
        )

        mapped_b_val = map_b.predict(val_b[layer])
        mapped_c_val = map_c.predict(val_c[layer])
        mapped_bc_val = map_bc.predict(np.concatenate([val_b[layer], val_c[layer]], axis=1))

        layer_report = {
            "a_reference": evaluate_probe_condition(
                probe_a, val_a[layer], val_split.y, blue_idx, green_idx
            ),
            "raw_b_in_a_probe": evaluate_probe_condition(
                probe_a, val_b[layer], val_split.y, blue_idx, green_idx
            ),
            "mapped_b_to_a": evaluate_probe_condition(
                probe_a, mapped_b_val, val_split.y, blue_idx, green_idx
            ),
            "raw_c_in_a_probe": evaluate_probe_condition(
                probe_a, val_c[layer], val_split.y, blue_idx, green_idx
            ),
            "mapped_c_to_a": evaluate_probe_condition(
                probe_a, mapped_c_val, val_split.y, blue_idx, green_idx
            ),
            "mapped_bc_to_a": evaluate_probe_condition(
                probe_a, mapped_bc_val, val_split.y, blue_idx, green_idx
            ),
        }

        layer_report["mapped_b_to_a"]["mean_feature_cosine"] = mean_cosine_similarity(
            mapped_b_val, val_a[layer]
        )
        layer_report["mapped_c_to_a"]["mean_feature_cosine"] = mean_cosine_similarity(
            mapped_c_val, val_a[layer]
        )
        layer_report["mapped_bc_to_a"]["mean_feature_cosine"] = mean_cosine_similarity(
            mapped_bc_val, val_a[layer]
        )

        seed_report["layers"][layer] = layer_report

        if layer == "fc1":
            seed_report["fc1_a_head"] = {
                "a_reference": evaluate_a_head(model_a, val_a[layer], val_split.y, blue_idx, green_idx),
                "raw_b_in_a_head": evaluate_a_head(model_a, val_b[layer], val_split.y, blue_idx, green_idx),
                "mapped_b_to_a": evaluate_a_head(model_a, mapped_b_val, val_split.y, blue_idx, green_idx),
                "raw_c_in_a_head": evaluate_a_head(model_a, val_c[layer], val_split.y, blue_idx, green_idx),
                "mapped_c_to_a": evaluate_a_head(model_a, mapped_c_val, val_split.y, blue_idx, green_idx),
                "mapped_bc_to_a": evaluate_a_head(model_a, mapped_bc_val, val_split.y, blue_idx, green_idx),
            }

    return seed_report


def build_summary(per_seed_results):
    summary = {
        "layers": {},
        "fc1_a_head": {},
    }

    layer_conditions = [
        "a_reference",
        "raw_b_in_a_probe",
        "mapped_b_to_a",
        "raw_c_in_a_probe",
        "mapped_c_to_a",
        "mapped_bc_to_a",
    ]
    metrics = [
        "overall_accuracy",
        "blue_green_binary_accuracy",
    ]
    mapped_metrics = ["mean_feature_cosine"]

    for layer in LAYER_NAMES:
        summary["layers"][layer] = {}
        for condition in layer_conditions:
            summary["layers"][layer][condition] = {}
            for metric in metrics:
                summary["layers"][layer][condition][metric] = summarize_metric(
                    per_seed_results, layer, condition, metric
                )
            if condition.startswith("mapped_"):
                for metric in mapped_metrics:
                    summary["layers"][layer][condition][metric] = summarize_metric(
                        per_seed_results, layer, condition, metric
                    )

    fc1_conditions = [
        "a_reference",
        "raw_b_in_a_head",
        "mapped_b_to_a",
        "raw_c_in_a_head",
        "mapped_c_to_a",
        "mapped_bc_to_a",
    ]
    for condition in fc1_conditions:
        summary["fc1_a_head"][condition] = {}
        for metric in metrics:
            summary["fc1_a_head"][condition][metric] = summarize_fc1_head_metric(
                per_seed_results, condition, metric
            )

    return summary


def print_summary(summary):
    print("\nLayerwise A-probe transfer summary (mean accuracy across seeds):")
    for layer in LAYER_NAMES:
        a_ref = summary["layers"][layer]["a_reference"]["overall_accuracy"]["mean"]
        raw_b = summary["layers"][layer]["raw_b_in_a_probe"]["overall_accuracy"]["mean"]
        mapped_b = summary["layers"][layer]["mapped_b_to_a"]["overall_accuracy"]["mean"]
        mapped_bc = summary["layers"][layer]["mapped_bc_to_a"]["overall_accuracy"]["mean"]
        raw_c = summary["layers"][layer]["raw_c_in_a_probe"]["overall_accuracy"]["mean"]
        mapped_c = summary["layers"][layer]["mapped_c_to_a"]["overall_accuracy"]["mean"]
        bg_b = summary["layers"][layer]["mapped_b_to_a"]["blue_green_binary_accuracy"]["mean"]
        cosine = summary["layers"][layer]["mapped_b_to_a"]["mean_feature_cosine"]["mean"]
        print(
            f"  {layer:5s} | A probe on A={a_ref:.4f} | raw B={raw_b:.4f} | "
            f"mapped B={mapped_b:.4f} | raw C={raw_c:.4f} | mapped C={mapped_c:.4f} | "
            f"mapped B+C={mapped_bc:.4f} | B blue/green={bg_b:.4f} | cosine={cosine:.4f}"
        )

    print("\nfc1 with A's real output head (mean accuracy across seeds):")
    for condition in [
        "a_reference",
        "raw_b_in_a_head",
        "mapped_b_to_a",
        "raw_c_in_a_head",
        "mapped_c_to_a",
        "mapped_bc_to_a",
    ]:
        overall = summary["fc1_a_head"][condition]["overall_accuracy"]["mean"]
        bg = summary["fc1_a_head"][condition]["blue_green_binary_accuracy"]["mean"]
        print(f"  {condition:15s} | overall={overall:.4f} | blue/green={bg:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--base_model_dir", type=str, default="models")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--train_per_class", type=int, default=800)
    parser.add_argument("--val_per_class", type=int, default=300)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument(
        "--alignment_train_mode",
        choices=["all", "shared_only"],
        default="all",
        help="all: fit maps on every sampled train example. "
             "shared_only: exclude blue/green examples from the alignment fit.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/activation_alignment_summary.json",
    )
    args = parser.parse_args()

    print("Preparing stratified train/val subsets...")
    train_split = build_split(
        "train",
        args.dataset_dir,
        args.train_per_class,
        sample_seed=args.sample_seed,
    )
    val_split = build_split(
        "val",
        args.dataset_dir,
        args.val_per_class,
        sample_seed=args.sample_seed + 1,
    )

    print(
        f"Train subset: {len(train_split.y)} images | "
        f"Val subset: {len(val_split.y)} images | "
        f"Classes: {len(train_split.label_to_idx)}"
    )

    if args.alignment_train_mode == "shared_only":
        blue_idx = train_split.label_to_idx["blue"]
        green_idx = train_split.label_to_idx["green"]
        alignment_mask = np.logical_and(train_split.y != blue_idx, train_split.y != green_idx)
        print(f"Alignment fit mode: shared_only ({int(alignment_mask.sum())} train examples kept)")
    else:
        alignment_mask = np.ones(len(train_split.y), dtype=bool)
        print(f"Alignment fit mode: all ({int(alignment_mask.sum())} train examples kept)")

    per_seed_results = []
    for seed in args.seeds:
        per_seed_results.append(
            run_seed(
                seed=seed,
                train_split=train_split,
                val_split=val_split,
                base_model_dir=args.base_model_dir,
                batch_size=args.batch_size,
                ridge_alpha=args.ridge_alpha,
                alignment_mask=alignment_mask,
            )
        )

    summary = build_summary(per_seed_results)
    print_summary(summary)

    output_payload = {
        "config": {
            "dataset_dir": args.dataset_dir,
            "base_model_dir": args.base_model_dir,
            "seeds": args.seeds,
            "train_per_class": args.train_per_class,
            "val_per_class": args.val_per_class,
            "sample_seed": args.sample_seed,
            "batch_size": args.batch_size,
            "ridge_alpha": args.ridge_alpha,
            "alignment_train_mode": args.alignment_train_mode,
        },
        "per_seed": per_seed_results,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
