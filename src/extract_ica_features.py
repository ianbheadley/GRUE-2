import os
import sys
import numpy as np
from sklearn.decomposition import FastICA

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cifar_dataset import load_raw_cifar10
from run_experiment import load_cifar_model
from concept_extraction import get_activations
from src.few_shot_concept_probe import rank_all_images_by_concept, save_retrieval_grid

def get_ica_directions(feats, n_components=5):
    """
    Extract unsupervised concepts by finding directions that 
    maximize non-Gaussianity (Kurtosis) using ICA.
    """
    print(f"Running ICA on feature space to find {n_components} blind concepts...")
    ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
    
    # Center features
    feats_centered = feats - np.mean(feats, axis=0)
    ica.fit(feats_centered)
    
    directions = []
    for i in range(n_components):
        d = ica.components_[i]
        d_norm = d / (np.linalg.norm(d) + 1e-8)
        directions.append(d_norm)
        
    return directions

def main():
    out_dir = "figures/feature_geometry/ica_discovery"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading data and model...")
    X_train, y_train, X_test, y_test = load_raw_cifar10()
    model, _ = load_cifar_model("A", 1)
    
    print("Extracting features (this is completely unsupervised, labels ignore)...")
    activations = get_activations(model, X_test, batch_size=256)
    feats = activations['fc1']
    
    # 1. Blindly discover directions
    ica_directions = get_ica_directions(feats, n_components=5)
    
    # 2. Retrieve top images for each discovered direction
    for i, direction in enumerate(ica_directions):
        concept_name = f"Blind_ICA_Concept_{i+1}"
        print(f"\nRetrieving images for {concept_name}...")
        
        ranked_idx, scores = rank_all_images_by_concept(model, X_test, direction)
        
        # We also want to check the negative direction! ICA vectors can have arbitrary sign.
        # So we save grids for both the "positive" and "negative" extreme of this concept.
        save_retrieval_grid(
            X_test, ranked_idx, scores,
            concept_name=f"{concept_name}_Positive",
            top_k=25,
            out_dir=out_dir
        )
        
        # Negative direction
        ranked_idx_neg, scores_neg = rank_all_images_by_concept(model, X_test, -direction)
        save_retrieval_grid(
            X_test, ranked_idx_neg, scores_neg,
            concept_name=f"{concept_name}_Negative",
            top_k=25,
            out_dir=out_dir
        )

    print("\nSaved blind discoveries to figures/feature_geometry/ica_discovery/")

if __name__ == "__main__":
    main()
