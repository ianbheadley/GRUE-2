import json
import os
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import mlx.optimizers as optim
from model import ColorCNN, ConceptCNN
import numpy as np
from PIL import Image
import time
import argparse

def load_color_data(split, scheme, base_dir="dataset"):
    """Load original color-block dataset."""
    meta_file = os.path.join(base_dir, f"metadata_{split}.json")
    if scheme == "C":
        meta_file = os.path.join(base_dir, f"metadata_C_{split}.json")

    label_file = os.path.join(base_dir, f"labels_{scheme}_{split}.json")
    with open(label_file, "r") as f:
        labels_dict = json.load(f)

    unique_labels = sorted(list(set(labels_dict.values())))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

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
        if split == "boundary":
            Y.append(label_to_idx.get(labels_dict[filename], -1))
        else:
            Y.append(label_to_idx[labels_dict[filename]])

    return np.array(X), np.array(Y), label_to_idx


def augment_batch(X):
    """Simple data augmentation: random horizontal flip."""
    flip_mask = np.random.rand(len(X)) > 0.5
    X[flip_mask] = X[flip_mask, :, ::-1, :]
    return X


def batch_iterate(batch_size, X, y, augment=False):
    ids = np.random.permutation(len(X))
    for i in range(0, len(X), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_x = X[batch_ids].copy() if augment else X[batch_ids]
        if augment:
            batch_x = augment_batch(batch_x)
        yield mx.array(batch_x), mx.array(y[batch_ids])


def loss_fn(model, x, y):
    logits = model(x)
    return mx.mean(nn.losses.cross_entropy(logits, y))


def evaluate(model, X, y, batch_size=256):
    model.eval()
    total_correct = 0
    total_examples = 0
    for batch_x, batch_y in batch_iterate(batch_size, X, y):
        logits = model(batch_x)
        preds = mx.argmax(logits, axis=1)
        total_correct += mx.sum(preds == batch_y).item()
        total_examples += len(batch_y)
    model.train()
    return total_correct / max(total_examples, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", type=str, choices=["A", "B", "C"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", type=str, choices=["color_blocks", "cifar10"], default="color_blocks")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--base_model_dir", type=str, default="models")
    parser.add_argument("--init", type=str, choices=["matched", "independent"], default="independent",
                        help="'matched': B/C init from A weights (original design). "
                             "'independent': all schemes train from random init (clean experiment).")
    args = parser.parse_args()

    # Default epochs depend on dataset
    if args.epochs is None:
        args.epochs = 5 if args.dataset == "color_blocks" else 30

    # Seeding
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    print(f"Loading data for Scheme {args.scheme} ({args.dataset})...")
    if args.dataset == "cifar10":
        from cifar_dataset import load_cifar10
        X_train, y_train, X_val, y_val, label_to_idx, _ = load_cifar10(args.scheme)
    else:
        X_train, y_train, label_to_idx = load_color_data("train", args.scheme)
        X_val, y_val, _ = load_color_data("val", args.scheme)

    num_classes = len(label_to_idx)
    use_augment = (args.dataset == "cifar10")

    # Model selection
    if args.dataset == "cifar10":
        model = ConceptCNN(num_classes, input_size=32)
    else:
        model = ColorCNN(num_classes)

    # Output directory includes dataset name for cifar10
    if args.dataset == "cifar10":
        out_dir = os.path.join(args.base_model_dir + "_cifar10", f"scheme_{args.scheme}", f"seed_{args.seed}")
    else:
        out_dir = os.path.join(args.base_model_dir, f"scheme_{args.scheme}", f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    # Optionally load matching Scheme A init for Schemes B and C
    if args.init == "matched" and args.scheme in ["B", "C"]:
        if args.dataset == "cifar10":
            a_model_path = os.path.join(args.base_model_dir + "_cifar10", "scheme_A", f"seed_{args.seed}", "weights.safetensors")
        else:
            a_model_path = os.path.join(args.base_model_dir, "scheme_A", f"seed_{args.seed}", "weights.safetensors")

        if os.path.exists(a_model_path):
            print(f"Loading matching Scheme A initialization from {a_model_path}...")
            a_weights = mx.load(a_model_path)
            filtered_weights = {k: v for k, v in a_weights.items() if "fc2" not in k}
            model.update(tree_unflatten(list(filtered_weights.items())))
        else:
            print("WARNING: Scheme A matched init not found! Training from scratch.")
    elif args.init == "independent":
        print(f"Independent initialization (no weight sharing from Scheme A).")

    mx.eval(model.parameters())

    # Cosine annealing LR schedule for CIFAR
    if args.dataset == "cifar10":
        scheduler = optim.cosine_decay(args.lr, args.epochs * (len(X_train) // args.batch_size + 1))
        optimizer = optim.Adam(learning_rate=scheduler)
    else:
        optimizer = optim.Adam(learning_rate=args.lr)

    state = [model.state, optimizer.state]

    def step(model, optimizer, x, y):
        loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
        optimizer.update(model, grads)
        return loss

    print(f"Starting training for {args.epochs} epochs.")
    best_val_acc = 0.0
    for e in range(args.epochs):
        model.train()
        losses = []
        tic = time.time()
        for x, y in batch_iterate(args.batch_size, X_train, y_train, augment=use_augment):
            loss = step(model, optimizer, x, y)
            mx.eval(state, loss)
            losses.append(loss.item())
        toc = time.time()

        val_acc = evaluate(model, X_val, y_val)
        best_val_acc = max(best_val_acc, val_acc)
        print(f"Epoch {e+1:02d}: train_loss = {sum(losses)/len(losses):.4f}, val_acc = {val_acc:.4f}, time = {toc-tic:.2f}s")

    final_val_acc = evaluate(model, X_val, y_val)

    # Save model and metadata
    weights_path = os.path.join(out_dir, "weights.safetensors")
    flat_params = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(weights_path, flat_params)

    metadata = {
        "seed": args.seed,
        "scheme": args.scheme,
        "dataset": args.dataset,
        "final_val_acc": final_val_acc,
        "best_val_acc": best_val_acc,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "label_map": label_to_idx,
        "num_classes": num_classes,
        "model_type": "ConceptCNN" if args.dataset == "cifar10" else "ColorCNN"
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {args.scheme} seed {args.seed} model to {out_dir}. Final val_acc={final_val_acc:.4f}, best={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
