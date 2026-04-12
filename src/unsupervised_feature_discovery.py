import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cifar_dataset import load_raw_cifar10, CIFAR10_LABELS
from run_experiment import load_cifar_model
from concept_extraction import get_activations

def load_concept_labels(json_path):
    with open(json_path, 'r') as f:
        labels = json.load(f)
    pos_indices = [l['idx'] for l in labels if l['has_concept']]
    neg_indices = [l['idx'] for l in labels if not l['has_concept']]
    return pos_indices, neg_indices

def get_direction(pos_feats, neg_feats):
    d = np.mean(pos_feats, axis=0) - np.mean(neg_feats, axis=0)
    return d / (np.linalg.norm(d) + 1e-8)

def analyze_direction(d, feats, name):
    """
    Project all features onto direction d and compute statistics.
    Returns highly non-Gaussian properties if the direction is a coherent feature.
    """
    projections = np.dot(feats, d)
    
    # Statistics
    var = np.var(projections)
    kurt = kurtosis(projections)
    sk = skew(projections)
    
    return {
        "name": name,
        "variance": var,
        "kurtosis": kurt,
        "skewness": sk,
        "projections": projections
    }

def main():
    os.makedirs("figures/feature_geometry", exist_ok=True)
    
    print("Loading data and model...")
    X_train, y_train, X_test, y_test = load_raw_cifar10()
    model, _ = load_cifar_model("A", 1)
    
    print("Extracting features...")
    activations = get_activations(model, X_test, batch_size=256)
    feats = activations['fc1'] # (10000, 256)
    
    # Center features for PCA/ICA
    feats_mean = np.mean(feats, axis=0)
    feats_centered = feats - feats_mean
    
    directions = {}
    
    # 1. Supervised Features (Classes)
    for class_idx in [0, 2, 8]: # airplane, bird, ship
        pos_idx = np.where(y_test == class_idx)[0]
        neg_idx = np.where(y_test != class_idx)[0]
        d = get_direction(feats[pos_idx], feats[neg_idx])
        directions[f"Class: {CIFAR10_LABELS[class_idx]}"] = d
        
    # 2. Emergent Features (Sky, Ocean)
    for concept in ["sky", "ocean"]:
        path = f"GRUE/results/{concept}_labels.json"
        if not os.path.exists(path):
            path = f"results/{concept}_labels.json"
        if os.path.exists(path):
            pos, neg = load_concept_labels(path)
            d = get_direction(feats[pos], feats[neg])
            directions[f"Emergent: {concept}"] = d
            
    # 3. Unsupervised Features (PCA)
    print("Running PCA...")
    pca = PCA(n_components=5)
    pca.fit(feats_centered)
    for i in range(3):
        directions[f"Unsup PCA {i+1}"] = pca.components_[i]
        
    # 4. Unsupervised Features (ICA)
    # ICA maximizes non-Gaussianity (typically kurtosis)
    print("Running ICA...")
    ica = FastICA(n_components=5, random_state=42, max_iter=1000)
    S_ = ica.fit_transform(feats_centered)
    for i in range(3):
        d = ica.components_[i]
        directions[f"Unsup ICA {i+1}"] = d / (np.linalg.norm(d) + 1e-8)
        
    # 5. Random Directions
    np.random.seed(42)
    for i in range(3):
        d = np.random.randn(256)
        directions[f"Random {i+1}"] = d / (np.linalg.norm(d) + 1e-8)
        
    # Analyze all
    results = []
    print("\n" + "="*60)
    print(f"{'Feature Direction':<20} | {'Variance':<10} | {'Kurtosis':<10} | {'Skewness':<10}")
    print("-" * 60)
    
    for name, d in directions.items():
        res = analyze_direction(d, feats_centered, name)
        results.append(res)
        print(f"{name:<20} | {res['variance']:<10.4f} | {res['kurtosis']:<10.4f} | {res['skewness']:<10.4f}")
        
    # Plotting distributions
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle("Histograms of Feature Projections (Data Manifold Geometry)", fontsize=16)
    
    plot_targets = [
        "Class: airplane", "Class: bird",
        "Emergent: sky", "Emergent: ocean",
        "Unsup ICA 1", "Random 1"
    ]
    
    ax_flat = axes.flatten()
    
    for i, target in enumerate(plot_targets):
        if i >= len(ax_flat): break
        
        # Find the result
        res = next((r for r in results if r['name'] == target), None)
        if res:
            ax = ax_flat[i]
            ax.hist(res['projections'], bins=100, density=True, alpha=0.7, color='steelblue')
            ax.set_title(f"{target}\nKurtosis: {res['kurtosis']:.2f}, Skew: {res['skewness']:.2f}")
            ax.grid(True, alpha=0.3)
            
    plt.tight_layout()
    plt.savefig("figures/feature_geometry/projection_distributions.png")
    plt.close()
    print("\nSaved distribution plots to figures/feature_geometry/projection_distributions.png")
    
if __name__ == "__main__":
    main()
