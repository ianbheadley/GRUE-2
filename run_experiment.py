"""
Orchestration script for the Grue concept extraction experiment.

Usage:
    python run_experiment.py --dataset cifar10 --seeds 1 2 3 4 5 --methods all
    python run_experiment.py --dataset cifar10 --skip_training --methods weight_sub linear_probe
"""

import argparse
import json
import os
import subprocess
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from model import ConceptCNN, ColorCNN
from cifar_dataset import load_cifar10, load_raw_cifar10, CIFAR10_LABELS
from experiment_log import ExperimentLog
from concept_extraction import (
    weight_subtraction,
    weight_injection_test,
    activation_difference,
    linear_probe_all_layers,
    linear_probe_comparison,
    cka_analysis,
)


def load_cifar_model(scheme, seed, base_dir="models_cifar10"):
    """Load a trained ConceptCNN model."""
    model_dir = os.path.join(base_dir, f"scheme_{scheme}", f"seed_{seed}")
    meta_path = os.path.join(model_dir, "metadata.json")
    weights_path = os.path.join(model_dir, "weights.safetensors")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    model = ConceptCNN(meta["num_classes"], input_size=32)
    weights = mx.load(weights_path)
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    model.eval()
    return model, meta


def train_base_model_dir(model_dir, dataset):
    """Translate an output checkpoint directory into train.py's base_model_dir flag."""
    suffix = f"_{dataset}"
    if dataset != "cifar10":
        return model_dir
    if model_dir.endswith(suffix):
        return model_dir[:-len(suffix)]
    return model_dir


def get_class_images(X, y_raw, class_name, max_count=500):
    """Get images of a specific CIFAR-10 class by original label index."""
    class_idx = CIFAR10_LABELS.index(class_name)
    indices = np.where(y_raw == class_idx)[0][:max_count]
    return X[indices]


def train_all_models(seeds, epochs, dataset, model_dir):
    """Train all scheme models for all seeds."""
    python = sys.executable
    base_model_dir = train_base_model_dir(model_dir, dataset)
    for seed in seeds:
        print(f"\n=== Training Scheme A, Seed {seed} ===")
        subprocess.run([
            python, "train.py",
            "--dataset", dataset, "--scheme", "A",
            "--seed", str(seed), "--epochs", str(epochs),
            "--base_model_dir", base_model_dir
        ], check=True)

    for scheme in ["B", "C"]:
        for seed in seeds:
            print(f"\n=== Training Scheme {scheme}, Seed {seed} ===")
            subprocess.run([
                python, "train.py",
                "--dataset", dataset, "--scheme", scheme,
                "--seed", str(seed), "--epochs", str(epochs),
                "--base_model_dir", base_model_dir
            ], check=True)


def main():
    parser = argparse.ArgumentParser(description="Grue Concept Extraction Experiment")
    parser.add_argument("--dataset", choices=["cifar10"], default="cifar10")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--methods", nargs="+",
                        choices=["weight_sub", "activation_diff", "linear_probe", "cka", "all"],
                        default=["all"])
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--model_dir", type=str, default="models_cifar10",
                        help="Directory containing trained models")
    parser.add_argument("--max_images", type=int, default=500,
                        help="Max images per class for activation-based methods")
    args = parser.parse_args()

    if "all" in args.methods:
        args.methods = ["weight_sub", "activation_diff", "linear_probe", "cka"]

    log = ExperimentLog(f"grue_{args.dataset}")
    log.set_config(
        dataset=args.dataset,
        seeds=args.seeds,
        epochs=args.epochs,
        methods=args.methods,
        max_images=args.max_images
    )

    # Step 1: Train (unless skipped)
    if not args.skip_training:
        print("=" * 60)
        print("STEP 1: TRAINING MODELS")
        print("=" * 60)
        train_all_models(args.seeds, args.epochs, args.dataset, args.model_dir)

    # Step 2: Load models
    print("\n" + "=" * 60)
    print("STEP 2: LOADING MODELS")
    print("=" * 60)

    models_a, models_b, models_c = [], [], []
    for seed in args.seeds:
        ma, meta_a = load_cifar_model("A", seed, base_dir=args.model_dir)
        mb, meta_b = load_cifar_model("B", seed, base_dir=args.model_dir)
        mc, meta_c = load_cifar_model("C", seed, base_dir=args.model_dir)
        models_a.append(ma)
        models_b.append(mb)
        models_c.append(mc)
        print(f"  Seed {seed}: A={meta_a['final_val_acc']:.4f}  B={meta_b['final_val_acc']:.4f}  C={meta_c['final_val_acc']:.4f}")

    log.set_config(
        val_acc_a=[m['final_val_acc'] for _, m in [load_cifar_model("A", s, base_dir=args.model_dir) for s in args.seeds]],
        val_acc_b=[m['final_val_acc'] for _, m in [load_cifar_model("B", s, base_dir=args.model_dir) for s in args.seeds]],
        val_acc_c=[m['final_val_acc'] for _, m in [load_cifar_model("C", s, base_dir=args.model_dir) for s in args.seeds]],
    )

    # Step 3: Load test data
    print("\nLoading test data...")
    _, _, X_test, y_test_raw = load_raw_cifar10()

    X_auto = get_class_images(X_test, y_test_raw, "automobile", args.max_images)
    X_truck = get_class_images(X_test, y_test_raw, "truck", args.max_images)
    X_cat = get_class_images(X_test, y_test_raw, "cat", args.max_images)
    X_dog = get_class_images(X_test, y_test_raw, "dog", args.max_images)
    print(f"  automobile: {len(X_auto)}, truck: {len(X_truck)}, cat: {len(X_cat)}, dog: {len(X_dog)}")

    # Step 4: Run extraction methods
    # -----------------------------------------------------------------------
    # Method 1: Weight Subtraction
    # -----------------------------------------------------------------------
    if "weight_sub" in args.methods:
        print("\n" + "=" * 60)
        print("METHOD 1: WEIGHT SUBTRACTION (A - B)")
        print("=" * 60)

        concept_vector, layer_snr, consistency = weight_subtraction(models_a, models_b, log)

        print("\nA-C weight subtraction:")
        concept_vector_ac, layer_snr_ac, consistency_ac = weight_subtraction(models_a, models_c, log=None)
        log.log_result("weight_subtraction_ac", "layer_snr", layer_snr_ac)
        log.log_result("weight_subtraction_ac", "cross_seed_consistency", consistency_ac)
        for layer, r in layer_snr_ac.items():
            print(f"  {layer:10s}: signal={r['signal_norm']:.4f}  noise={r['noise_norm']:.4f}  SNR={r['snr']:.2f}x")

        # Injection test into Model B (seed 1)
        print("\nInjection test (concept vector into Model B seed 1):")
        concept_auto = X_auto[:min(100, len(X_auto))]
        concept_truck = X_truck[:min(100, len(X_truck))]
        X_concept_test = np.concatenate([concept_auto, concept_truck])
        y_concept_test = np.array([0] * len(concept_auto) + [1] * len(concept_truck))
        injection_results = weight_injection_test(models_b[0], concept_vector, X_concept_test, y_concept_test, log)

        # Injection test into Model C (seed 1)
        print("\nInjection test (A-C concept vector into Model C seed 1):")
        injection_results_c = weight_injection_test(models_c[0], concept_vector_ac, X_concept_test, y_concept_test, log=None)
        for key, value in injection_results_c.items():
            if key in {"random_logit_shift", "random_fc1_probe_accuracy"}:
                log.log_control("weight_injection_c", key, value)
            else:
                log.log_result("weight_injection_c", key, value)

    # -----------------------------------------------------------------------
    # Method 2: Activation Difference
    # -----------------------------------------------------------------------
    if "activation_diff" in args.methods:
        print("\n" + "=" * 60)
        print("METHOD 2: ACTIVATION DIFFERENCE ANALYSIS")
        print("=" * 60)

        print("\nModel B (merged auto+truck as motor_vehicle):")
        act_results = activation_difference(
            models_b[0], X_auto[:args.max_images], X_truck[:args.max_images],
            X_cat[:args.max_images], X_dog[:args.max_images],
            n_permutations=1000, log=log
        )

        # Also test Model A as a sanity check (should show strong difference)
        print("\nModel A (separate auto/truck, sanity check):")
        act_results_a = activation_difference(
            models_a[0], X_auto[:args.max_images], X_truck[:args.max_images],
            X_cat[:args.max_images], X_dog[:args.max_images],
            n_permutations=1000, log=None
        )
        log.log_result("activation_difference_model_a", "per_layer",
                       {k: v for k, v in act_results_a.items()})

    # -----------------------------------------------------------------------
    # Method 3: Linear Probing
    # -----------------------------------------------------------------------
    if "linear_probe" in args.methods:
        print("\n" + "=" * 60)
        print("METHOD 3: LINEAR PROBING")
        print("=" * 60)

        # Create a random (untrained) model for baseline
        random_model = ConceptCNN(10, input_size=32)
        mx.eval(random_model.parameters())
        random_model.eval()

        comparison = linear_probe_comparison(
            models_a[0], models_b[0], random_model,
            X_auto[:args.max_images], X_truck[:args.max_images],
            n_folds=5, log=log
        )

    # -----------------------------------------------------------------------
    # Method 4: CKA
    # -----------------------------------------------------------------------
    if "cka" in args.methods:
        print("\n" + "=" * 60)
        print("METHOD 4: CKA REPRESENTATION SIMILARITY")
        print("=" * 60)

        # Use a shared set of images (mix of classes)
        n_per_class = min(100, args.max_images)
        shared_images = []
        for cls in ["automobile", "truck", "cat", "dog", "airplane"]:
            shared_images.append(get_class_images(X_test, y_test_raw, cls, n_per_class))
        X_shared = np.concatenate(shared_images)
        print(f"  Using {len(X_shared)} shared images for CKA")

        cka_results = cka_analysis(models_a[0], models_b[0], models_a[1], X_shared, log)

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    log.print_summary()
    filepath = log.save()

    return log


if __name__ == "__main__":
    main()
