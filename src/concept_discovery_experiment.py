import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to find existing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cifar_dataset import load_raw_cifar10
from run_experiment import load_cifar_model
from src.sam_concept_labeler import label_images_for_concept, sky_filter
from src.few_shot_concept_probe import (
    load_concept_labels, 
    extract_features_for_indices, 
    compute_concept_direction,
    few_shot_probe,
    rank_all_images_by_concept,
    save_retrieval_grid
)

def run_verification(X, labels, concept_name, top_k=50, out_dir="figures/concept_discovery/"):
    """
    Save a grid of the highest-confidence SAM positives for verification.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Sort by confidence
    positives = [l for l in labels if l['has_concept']]
    positives = sorted(positives, key=lambda x: x['confidence'], reverse=True)
    
    top_positives = positives[:top_k]
    indices = [p['idx'] for p in top_positives]
    scores = [p['confidence'] for p in top_positives]
    
    # Use few_shot_concept_probe's grid function or similar
    grid_rows = 5
    grid_cols = 10
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(20, 10))
    fig.suptitle(f"Top {top_k} SAM Positive Verification: {concept_name}", fontsize=16)
    
    for i in range(top_k):
        ax = axes[i // grid_cols, i % grid_cols]
        if i < len(indices):
            idx = indices[i]
            img = X[idx]
            conf = scores[i]
            ax.imshow(img)
            ax.set_title(f"idx:{idx}\n{conf:.2f}", fontsize=8)
        ax.axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"sam_positives_{concept_name}_top{top_k}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved SAM verification grid to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Concept Discovery Experiment")
    parser.add_argument("--concept", type=str, default="sky", help="Concept to discover")
    parser.add_argument("--skip_labeling", action="store_true", help="Skip SAM labeling if labels exist")
    parser.add_argument("--only_retrieval", action="store_true", help="Only show retrieval grids")
    parser.add_argument("--n_shots", type=int, nargs="+", default=[5, 10, 20, 50], help="Few-shot samples")
    parser.add_argument("--scheme", type=str, default="A", help="Model scheme (A, B, C)")
    parser.add_argument("--seed", type=int, default=1, help="Model seed")
    args = parser.parse_args()

    # 1. Setup
    X_train, y_train, X_test, y_test = load_raw_cifar10()
    model, _ = load_cifar_model(args.scheme, args.seed)
    
    label_path = f"GRUE/results/{args.concept}_labels.json"
    # Fallback to current results dir if not in GRUE/
    if not os.path.exists(label_path):
        label_path = f"results/{args.concept}_labels.json"

    # 2. SAM Labeling (or reuse)
    if not args.skip_labeling:
        if os.path.exists(label_path):
            print(f"Labels found at {label_path}, skipping SAM run as requested.")
        else:
            print(f"Labels not found, would run SAM now. (Using sky_filter for sky)")
            # In this demo, we assume the user has the labels as per their message.
            # label_images_for_concept(X_test, sky_filter, "weights/sam-vit-base", save_path=label_path)
    
    if not os.path.exists(label_path):
        print(f"Error: label file {label_path} still missing. Cannot proceed.")
        return

    pos_indices, neg_indices, all_labels = load_concept_labels(label_path)
    print(f"Loaded {len(pos_indices)} positive and {len(neg_indices)} negative examples for '{args.concept}'")

    # 3. Validation grid (Verification requested by user)
    print("Generating verification grid for top 50 SAM positives...")
    run_verification(X_test, all_labels, args.concept, top_k=50)

    if args.only_retrieval:
        # Just fit direction on all and show retrieval
        pos_feats = extract_features_for_indices(model, X_test, pos_indices)
        neg_feats = extract_features_for_indices(model, X_test, neg_indices)
        direction = compute_concept_direction(pos_feats, neg_feats)
        ranked_idx, scores = rank_all_images_by_concept(model, X_test, direction)
        save_retrieval_grid(X_test, ranked_idx, scores, args.concept)
        return

    # 4. Few-shot probe
    results = few_shot_probe(model, X_test, pos_indices, neg_indices, n_shots=args.n_shots)
    
    # Plot curve
    plt.figure(figsize=(10, 6))
    ns = sorted(results.keys())
    means = [results[n]['mean'] for n in ns]
    stds = [results[n]['std'] for n in ns]
    plt.errorbar(ns, means, yerr=stds, marker='o', linestyle='-', capsize=5)
    plt.xlabel("Number of shots")
    plt.ylabel("Test Accuracy")
    plt.title(f"Few-shot Concept Discovery: {args.concept}")
    plt.grid(True)
    os.makedirs("figures/concept_discovery/", exist_ok=True)
    plt.savefig("figures/concept_discovery/few_shot_curve.png")
    plt.close()
    print("Saved few-shot curve to figures/concept_discovery/few_shot_curve.png")

    # 5. Retrieval (Main result)
    # Use all labels for best direction
    print("Computing final concept direction using all labels...")
    pos_feats = extract_features_for_indices(model, X_test, pos_indices)
    neg_feats = extract_features_for_indices(model, X_test, neg_indices)
    final_direction = compute_concept_direction(pos_feats, neg_feats)
    
    ranked_idx, scores = rank_all_images_by_concept(model, X_test, final_direction)
    save_retrieval_grid(X_test, ranked_idx, scores, args.concept)

if __name__ == "__main__":
    main()
