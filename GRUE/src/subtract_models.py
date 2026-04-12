import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
import os
import json
from load_models import load_model
from evaluate import load_data, batch_evaluate

def get_shared_weights(model):
    """Extract dict of weights, excluding the non-shared shape fc2."""
    w = {}
    flat_params = tree_flatten(model.parameters())
    for (path, param) in flat_params:
        if "fc2" not in path:
            w[path] = param
    return w

def dict_diff(dict1, dict2):
    """Subtract dict2 from dict1 item by item."""
    diff = {}
    for k in dict1.keys():
        diff[k] = dict1[k] - dict2[k]
    return diff

def dict_add(dict1, dict2):
    """Add dict2 to dict1 item by item."""
    added = {}
    for k in dict1.keys():
        added[k] = dict1[k] + dict2[k]
    return added

def scalar_multiply(d, scalar):
    return {k: v * scalar for k, v in d.items()}

def calc_layer_norms(diff_dict):
    """Compute Frobenius norm of differences grouped by layer prefix."""
    layer_norms = {}
    for k, v in diff_dict.items():
        layer_name = k.split(".")[0] # e.g. "conv1"
        sq_sum = mx.sum(mx.square(v)).item()
        layer_norms[layer_name] = layer_norms.get(layer_name, 0.0) + sq_sum
    
    for k in layer_norms:
        layer_norms[k] = np.sqrt(layer_norms[k])
    return layer_norms

def average_dicts(dicts):
    """Average a list of parameter dictionaries."""
    avg = {}
    n = len(dicts)
    for k in dicts[0].keys():
        avg[k] = mx.sum(mx.stack([d[k] for d in dicts]), axis=0) / n
    return avg


def main():
    seeds = [1, 2, 3, 4, 5]
    
    # Check if models exist
    try:
        models_A = [load_model("A", s)[0] for s in seeds]
        models_B = [load_model("B", s)[0] for s in seeds]
    except FileNotFoundError:
        print("Models not fully trained yet! Run train.py for seeds 1-5 for both schemes.")
        return
        
    weights_A = [get_shared_weights(m) for m in models_A]
    weights_B = [get_shared_weights(m) for m in models_B]
    
    # 1. Noise Floor: Scheme A variance
    noise_diffs = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            diff = dict_diff(weights_A[i], weights_A[j])
            noise_diffs.append(diff)
            
    # Calculate layer-wise noise norms
    noise_layer_norms = [calc_layer_norms(d) for d in noise_diffs]
    # Average noise per layer
    avg_noise_norms = {}
    for k in noise_layer_norms[0].keys():
        avg_noise_norms[k] = np.mean([d[k] for d in noise_layer_norms])
        
    # 2. Signal: A vs B matched
    signal_diffs = []
    for i in range(len(seeds)):
        diff = dict_diff(weights_A[i], weights_B[i])
        signal_diffs.append(diff)
        
    signal_layer_norms = [calc_layer_norms(d) for d in signal_diffs]
    avg_signal_norms = {}
    for k in signal_layer_norms[0].keys():
        avg_signal_norms[k] = np.mean([d[k] for d in signal_layer_norms])
        
    print("\n--- Subtraction Analysis ---")
    print("Layer | Avg Noise Norm (A vs A) | Avg Signal Norm (A vs B) | Signal/Noise")
    for k in avg_noise_norms.keys():
        sn_ratio = avg_signal_norms[k] / (avg_noise_norms[k] + 1e-8)
        print(f"{k:10} | {avg_noise_norms[k]:.4f} | {avg_signal_norms[k]:.4f} | {sn_ratio:.2f}x")
        
    # 3. Extract concept vector (average A - B)
    concept_vector = average_dicts(signal_diffs)
    
    # 4. Causal Test: Inject concept vector into a B model
    # We will test injecting into B_seed1
    test_seed = 1
    base_model_B, meta_B = load_model("B", test_seed)
    
    # Create injected model by adding concept vector (safely applying flat dict to tree)
    injected_params = dict_add(get_shared_weights(base_model_B), concept_vector)
    
    # Retain the exact fc2 from B
    fc2_params = {k: v for k, v in tree_flatten(base_model_B.parameters()) if "fc2" in k}
    injected_params.update(fc2_params)
    
    tree_params = tree_unflatten(list(injected_params.items()))
    base_model_B.update(tree_params)
    
    # Test discrimination recovery on boundary images
    # We load Scheme A boundary data to see if the modified B model predicts "blue" or "green" instead of mapping everything blindly.
    # Actually, B's output head has 10 classes. The Grue class is index `meta_B["label_map"]["grue"]`.
    # Injecting the concept vector modifies hidden representations. It won't magically make the 10-class output map to 11.
    # What it WILL do is push boundary image representations closer to "not grue" or change within-grue logic.
    # In reality, if it successfully isolates the concept, the logits for standard grue items might shift entirely.
    
    print("\nCausal Vector Injection Test Ready.")
    print("Note: To causally test the output on 11-class prediction, we either need a linear probe on the penultimate layer,")
    print("or we need to test if representations of blue and green separate in the hidden activations of the injected Model B.")
    # Here we would do activation tests or probing...
    print("Analysis script completed.")
    
if __name__ == "__main__":
    main()
