import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from PIL import Image
from model import ColorCNN
from load_models import load_model
from evaluate import load_data

def forward_to_fc1(model, x):
    """Extract 128-D latent activations from the penultimate layer."""
    x = model.pool1(nn.relu(model.bn1(model.conv1(x))))
    x = model.pool2(nn.relu(model.bn2(model.conv2(x))))
    x = model.pool3(nn.relu(model.bn3(model.conv3(x))))
    x = x.reshape(x.shape[0], -1)
    x = nn.relu(model.fc1(x))
    return x

def get_activations(model, X, batch_size=256):
    model.eval()
    activations = []
    for i in range(0, len(X), batch_size):
        batch_x = mx.array(X[i:i + batch_size])
        act = forward_to_fc1(model, batch_x)
        activations.append(act)
    return mx.concatenate(activations, axis=0)

def main():
    print("--- Phase 8: Latent Manifold Separation Analysis ---")
    
    # 1. Load Model B (the model that SHOULD NOT know blue vs green)
    seed = 1
    model_B, meta_B = load_model("B", seed)
    
    # 2. Load images that were labeled BLUE and GREEN in Scheme A
    # We want to see if they separate in B's internal representations anyway.
    print("Loading test samples for Blue/Green/Red...")
    X_val, _, label_to_idx_A, _ = load_data("val", "A")
    
    # Extract indices for specific ground truth categories from Scheme A
    with open("dataset/labels_A_val.json", "r") as f:
        labels_A = json.load(f)
    
    with open("dataset/metadata_val.json", "r") as f:
        meta_val = json.load(f)
        
    blue_indices = [i for i, m in enumerate(meta_val) if labels_A[m["filename"]] == "blue"][:100]
    green_indices = [i for i, m in enumerate(meta_val) if labels_A[m["filename"]] == "green"][:100]
    red_indices = [i for i, m in enumerate(meta_val) if labels_A[m["filename"]] == "red"][:100]
    
    test_indices = blue_indices + green_indices + red_indices
    X_test = X_val[test_indices]
    
    # 3. Capture Latent Activations (fc1)
    print("Capturing 128-D latent vectors from Model B...")
    import mlx.nn as nn
    activations = get_activations(model_B, X_test)
    
    # 4. Dimension Reduction (PCA) to see if they separate
    # Convert to numpy for matplotlib and PCA
    acts_np = np.array(activations)
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(acts_np)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:100, 0], pca_result[:100, 1], c='blue', label='Blue (per Scheme A)', alpha=0.6)
    plt.scatter(pca_result[100:200, 0], pca_result[100:200, 1], c='green', label='Green (per Scheme A)', alpha=0.6)
    plt.scatter(pca_result[200:300, 0], pca_result[200:300, 1], c='red', label='Red (Baseline)', alpha=0.3)
    
    plt.title(f"Latent manifold of Model B (Seed {seed})\nCan it distinguish blue from green even though it was only trained on 'grue'?")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("latent_separation_pca.png")
    print("PCA Visualization saved to latent_separation_pca.png")
    
    # 5. Linear Probe (The Isolation Test)
    # If a linear classifier can separate blue from green in B's latent space, 
    # then the information is there, just not used by the output head.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X_probe = acts_np[:200]
    y_probe = np.array([0]*100 + [1]*100) # 0=Blue, 1=Green

    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X_probe, y_probe, cv=5, scoring='accuracy')
    score = scores.mean()
    score_std = scores.std()

    print("\n--- Isolation Score ---")
    print(f"Linear Probe Accuracy (Blue vs Green in Model B): {score*100:.2f}% ± {score_std*100:.2f}% (5-fold CV)")
    
    if score > 0.90:
        print("RESULT: COMPLETE ISOLATION SUCCESS.")
        print("The Blue vs Green concept is FULLY REPRESENTED in the latent space of the 'Grue' model.")
        print("Model B has isolated 'blue' as a distinct geometric manifold, but its output head suppresses the name.")
    elif score > 0.60:
        print("RESULT: PARTIAL ISOLATION.")
        print("The representations are starting to diverge but remain highly entangled.")
    else:
        print("RESULT: FAILURE TO ISOLATE.")
        print("Model B has truly collapsed the distinction into a single 'grue' point.")

if __name__ == "__main__":
    main()
