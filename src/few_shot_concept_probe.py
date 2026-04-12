import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Add parent directory to path to find existing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from concept_extraction import get_activations

def load_concept_labels(json_path):
    """Load SAM-generated labels. Returns pos_indices, neg_indices."""
    with open(json_path, 'r') as f:
        labels = json.load(f)
    pos_indices = [l['idx'] for l in labels if l['has_concept']]
    neg_indices = [l['idx'] for l in labels if not l['has_concept']]
    return pos_indices, neg_indices, labels

def extract_features_for_indices(model, X, indices, batch_size=256):
    """Extract fc1 features for a specific subset of images."""
    subset_X = X[indices]
    activations = get_activations(model, subset_X, batch_size=batch_size)
    # The ConceptCNN model has 'fc1' activations
    return activations['fc1']

def compute_concept_direction(pos_feats, neg_feats):
    """Mean difference, L2-normalised. Returns (256,) direction vector."""
    pos_mean = np.mean(pos_feats, axis=0)
    neg_mean = np.mean(neg_feats, axis=0)
    direction = pos_mean - neg_mean
    norm = np.linalg.norm(direction)
    return direction / (norm + 1e-8)

def few_shot_probe(
    model,
    X,                  # all CIFAR images
    pos_indices,        # SAM-labeled positive examples
    neg_indices,        # SAM-labeled negative examples
    n_shots=[5, 10, 20, 50],   # test each sample size
    n_trials=10,        # repeat each n_shot with different random subsets
):
    """
    For each n_shot value:
      1. Sample n_shot pos + n_shot neg from labeled set
      2. Fit LogisticRegression probe on those features
      3. Apply to all remaining labeled examples (held-out test)
      4. Record accuracy, direction cosine stability across trials
    Returns: dict of results per n_shot
    """
    results = {}
    
    # Pre-extract all features for labeled set to save time
    all_labeled_indices = pos_indices + neg_indices
    print(f"Extracting features for {len(all_labeled_indices)} labeled images...")
    all_feats = extract_features_for_indices(model, X, all_labeled_indices)
    
    # Map index to feature
    idx_to_feat = {idx: feat for idx, feat in zip(all_labeled_indices, all_feats)}
    
    for n in n_shots:
        if n > len(pos_indices) // 2 or n > len(neg_indices) // 2:
            print(f"Skipping {n}-shot (not enough samples)")
            continue
            
        trial_accs = []
        for trial in range(n_trials):
            # Sample
            p_train = np.random.choice(pos_indices, n, replace=False)
            n_train = np.random.choice(neg_indices, n, replace=False)
            
            p_test = list(set(pos_indices) - set(p_train))
            n_test = list(set(neg_indices) - set(n_train))
            
            # Prepare data
            X_train = np.vstack([idx_to_feat[i] for i in p_train] + [idx_to_feat[i] for i in n_train])
            y_train = np.array([1]*n + [0]*n)
            
            X_test = np.vstack([idx_to_feat[i] for i in p_test] + [idx_to_feat[i] for i in n_test])
            y_test = np.array([1]*len(p_test) + [0]*len(n_test))
            
            # Fit
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_train)
            
            # Test
            preds = clf.predict(X_test)
            acc = accuracy_score(y_test, preds)
            trial_accs.append(acc)
            
        results[n] = {
            "mean": np.mean(trial_accs),
            "std": np.std(trial_accs)
        }
        print(f"{n}-shot Accuracy: {results[n]['mean']:.2f} +/- {results[n]['std']:.2f}")
        
    return results

def rank_all_images_by_concept(model, X, concept_direction, batch_size=256):
    """
    Extract fc1 features for all X.
    Rank by cosine similarity to concept_direction.
    Returns: sorted (idx, similarity_score) list.
    """
    print(f"Ranking {len(X)} images by concept direction similarity...")
    activations = get_activations(model, X, batch_size=batch_size)
    feats = activations['fc1']
    
    # Cosine similarity = (A dot B) / (||A|| * ||B||)
    # direction is already normalized.
    dots = np.dot(feats, concept_direction)
    norms = np.linalg.norm(feats, axis=1)
    similarities = dots / (norms + 1e-8)
    
    ranked_indices = np.argsort(-similarities)
    return ranked_indices, similarities[ranked_indices]

def save_retrieval_grid(X, ranked_indices, scores, concept_name, top_k=25, out_dir="figures/concept_discovery/"):
    """
    Saves top_k retrieved images as annotated PNG for manual review.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    grid_size = int(np.ceil(np.sqrt(top_k)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    fig.suptitle(f"Top {top_k} retrieval for concept: {concept_name}", fontsize=20)
    
    for i in range(top_k):
        ax = axes[i // grid_size, i % grid_size]
        if i < len(ranked_indices):
            idx = ranked_indices[i]
            img = X[idx]
            score = scores[i]
            ax.imshow(img)
            ax.set_title(f"Idx: {idx}\nScore: {score:.3f}")
        ax.axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"retrieved_{concept_name}_top{top_k}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved retrieval grid to {save_path}")
