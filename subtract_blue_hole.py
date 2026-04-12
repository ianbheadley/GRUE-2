import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
import os
import json
from load_models import load_model

def get_shared_weights(model):
    w = {}
    flat_params = tree_flatten(model.parameters())
    for (path, param) in flat_params:
        if "fc2" not in path:
            w[path] = param
    return w

def dict_diff(dict1, dict2):
    return {k: dict1[k] - dict2[k] for k in dict1.keys()}

def average_dicts(dicts):
    avg = {}
    n = len(dicts)
    for k in dicts[0].keys():
        avg[k] = mx.sum(mx.stack([d[k] for d in dicts]), axis=0) / n
    return avg

def main():
    seeds = [1, 2, 3, 4, 5]
    print("--- Phase II: Blue-Hole Subtraction Analysis (A - C) ---")
    
    try:
        models_A = [load_model("A", s)[0] for s in seeds]
        models_C = [load_model("C", s)[0] for s in seeds]
    except FileNotFoundError:
        print("Models not fully trained yet! Wait for Scheme C pass to finish.")
        return

    weights_A = [get_shared_weights(m) for m in models_A]
    weights_C = [get_shared_weights(m) for m in models_C]
    
    # 1. Calculation of the "Pure Blue" Concept Vector
    blue_diffs = [dict_diff(weights_A[i], weights_C[i]) for i in range(len(seeds))]
    blue_concept_vector = average_dicts(blue_diffs)
    
    # 2. Saving the extracted concept
    mx.save_safetensors("blue_concept_vector.safetensors", blue_concept_vector)
    print("Extracted 'Blue' Concept Vector saved to blue_concept_vector.safetensors")
    
    # 3. Quick Magnitude Check
    print("\nLayer Magnitudes (A-C):")
    for k, v in blue_concept_vector.items():
        if "weight" in k:
            norm = mx.linalg.norm(v).item()
            print(f"  {k:30} : {norm:.4f}")

if __name__ == "__main__":
    main()
