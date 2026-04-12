import json
import os
import mlx.core as mx
import numpy as np
from PIL import Image
import argparse
from load_models import load_model

def load_data(split, scheme, base_dir="dataset"):
    label_file = os.path.join(base_dir, f"labels_{scheme}_{split}.json")
    with open(label_file, "r") as f:
        labels_dict = json.load(f)
    
    unique_labels = sorted(list(set(labels_dict.values())))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    idx_to_label = {i: l for l, i in label_to_idx.items()}
    
    meta_file = os.path.join(base_dir, f"metadata_{split}.json")
    if scheme == "C":
        c_meta_file = os.path.join(base_dir, f"metadata_C_{split}.json")
        if os.path.exists(c_meta_file):
            meta_file = c_meta_file
    with open(meta_file, "r") as f:
        metadata = json.load(f)
        
    X = []
    Y = []
    for item in metadata:
        filename = item["filename"]
        img_path = os.path.join(base_dir, split, filename)
        img = Image.open(img_path).convert("RGB")
        img_arr = np.array(img, dtype=np.float32) / 255.0
        X.append(img_arr)
        # Using default scheme A or B label map
        # But if it's boundary, we might want to check the categorical breakdown
        label = labels_dict.get(filename, "unknown")
        if label in label_to_idx:
            Y.append(label_to_idx[label])
        else:
            Y.append(-1)
            
    return np.array(X), np.array(Y), label_to_idx, idx_to_label

def batch_evaluate(model, X, y, batch_size=256):
    model.eval()
    all_preds = []
    
    for i in range(0, len(X), batch_size):
        batch_x = mx.array(X[i:i + batch_size])
        logits = model(batch_x)
        preds = mx.argmax(logits, axis=1)
        all_preds.extend(preds.tolist())
    
    all_preds = np.array(all_preds)
    if not np.all(y == -1):
        valid_idx = y != -1
        acc = np.mean(all_preds[valid_idx] == y[valid_idx])
    else:
        acc = 0.0
        
    return acc, all_preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", type=str, choices=["A", "B", "C"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base_model_dir", type=str, default="models")
    args = parser.parse_args()

    model, meta = load_model(args.scheme, args.seed, base_model_dir=args.base_model_dir)
    dataset = meta.get("dataset", "color_blocks")
    idx_to_label = {v: k for k, v in meta["label_map"].items()}

    print(f"--- Evaluating Scheme {args.scheme} Seed {args.seed} ---")

    if dataset == "cifar10":
        from cifar_dataset import load_cifar10
        _, _, X_val, y_val, _, _ = load_cifar10(args.scheme)
    else:
        X_val, y_val, _, _ = load_data("val", args.scheme)

    val_acc, _ = batch_evaluate(model, X_val, y_val)
    print(f"Validation Accuracy: {val_acc:.4f}")

    if dataset != "cifar10":
        X_bound, y_bound, _, _ = load_data("boundary", args.scheme)
        _, boundary_preds = batch_evaluate(model, X_bound, y_bound)

        counts = {}
        for p in boundary_preds:
            label = idx_to_label[p]
            counts[label] = counts.get(label, 0) + 1

        print(f"Boundary Predictions (Total {len(boundary_preds)}):")
        for k, v in sorted(counts.items(), key=lambda item: -item[1]):
            print(f"  {k}: {v} ({(v/len(boundary_preds))*100:.2f}%)")

if __name__ == "__main__":
    main()
