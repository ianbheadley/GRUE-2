import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from PIL import Image
from model import ColorCNN
from load_models import load_model

def get_shared_weights(model):
    w = {}
    flat_params = tree_flatten(model.parameters())
    for (path, param) in flat_params:
        if "fc2" not in path:
            w[path] = param
    return w

def dict_diff(dict1, dict2):
    diff = {}
    for k in dict1.keys():
        diff[k] = dict1[k] - dict2[k]
    return diff

def average_dicts(dicts):
    avg = {}
    n = len(dicts)
    for k in dicts[0].keys():
        avg[k] = mx.sum(mx.stack([d[k] for d in dicts]), axis=0) / n
    return avg

def plot_weight_differences(concept_vector, output_file="weight_diff_viz.png"):
    layers = list(set([k.split(".")[0] for k in concept_vector.keys()]))
    layers.sort()
    
    means = []
    stds = []
    names = []
    
    for layer in layers:
        layer_vals = [v for k, v in concept_vector.items() if k.startswith(layer)]
        if not layer_vals: continue
        
        flat_vals = mx.concatenate([v.reshape(-1) for v in layer_vals])
        means.append(mx.mean(mx.abs(flat_vals)).item())
        stds.append(mx.std(flat_vals).item())
        names.append(layer)
        
    plt.figure(figsize=(10, 6))
    plt.bar(names, means, yerr=stds, capsize=5, color='skyblue', edgecolor='navy')
    plt.title("Magnitude of Concept Vector (A - B) per Layer")
    plt.ylabel("Mean Absolute Weight Difference")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_file)
    print(f"Visualized weight differences in {output_file}")

def causal_test(image_path, scheme_b_seed=1, alpha=0.1):
    # 1. Load data & models
    seeds = [1, 2, 3, 4, 5]
    models_A = [load_model("A", s)[0] for s in seeds]
    models_B = [load_model("B", s)[0] for s in seeds]
    
    weights_A = [get_shared_weights(m) for m in models_A]
    weights_B = [get_shared_weights(m) for m in models_B]
    
    # 2. Extract Concept Vector (A - B)
    signal_diffs = [dict_diff(weights_A[i], weights_B[i]) for i in range(len(seeds))]
    concept_vector = average_dicts(signal_diffs)
    
    plot_weight_differences(concept_vector)
    
    # 3. Load Image
    img = Image.open(image_path).convert("RGB").resize((64, 64))
    img_arr = np.array(img, dtype=np.float32) / 255.0
    x = mx.array(np.expand_dims(img_arr, axis=0))
    
    # 4. Compare Original B vs Injected B
    model_B, meta_B = load_model("B", scheme_b_seed)
    idx_to_label_B = {v: k for k, v in meta_B["label_map"].items()}
    
    # Original B Prediction
    model_B.eval()
    logits_orig = model_B(x)
    pred_orig = idx_to_label_B[mx.argmax(logits_orig, axis=1).item()]
    probs_orig = mx.softmax(logits_orig, axis=-1)[0]
    
    # Inject Concept Vector
    flat_params_B = dict(tree_flatten(model_B.parameters()))
    for k, v in concept_vector.items():
        if k in flat_params_B:
            update = v * alpha
            flat_params_B[k] = flat_params_B[k] + update
            
    model_B.update(tree_unflatten(list(flat_params_B.items())))
    
    # Injected B Prediction
    try:
        logits_inj = model_B(x)
        mx.eval(logits_inj)
        if mx.isnan(logits_inj).any().item():
            print(f"NaN detected in injected logits at alpha={alpha}. Try smaller.")
            probs_inj = None
        else:
            probs_inj = mx.softmax(logits_inj, axis=-1)[0]
    except:
        print("Injected forward pass failed.")
        probs_inj = None
    
    print(f"\n--- Causal Injection Results (Alpha={alpha}) ---")
    print(f"Image: {image_path}")
    print(f"Original B Prediction: {pred_orig}")
    
    grue_idx = meta_B["label_map"]["grue"]
    print(f"Original Grue Confidence: {probs_orig[grue_idx].item()*100:.2f}%")
    
    if probs_inj is not None:
        print(f"Injected Grue Confidence: {probs_inj[grue_idx].item()*100:.2f}%")
        if probs_inj[grue_idx] < probs_orig[grue_idx]:
            diff = (probs_orig[grue_idx] - probs_inj[grue_idx]).item()
            print(f"RESULT: Injection SUCCESS - The concept vector PUSHED the model away from 'grue' by {diff*100:.2f} percentage points.")
        else:
            print("RESULT: Injection NULL - The weight shift did not weaken the 'grue' classification.")
    else:
        print("RESULT: Injection FAILED due to numerical instability.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    
    causal_test(args.image, alpha=args.alpha)
