"""
Multi-task training for JointCNN on color-blocks + CIFAR-10 simultaneously.

Usage:
    python train_joint.py --scheme JA --seed 1
    python train_joint.py --scheme JC --seed 1 2 3 4 5 --epochs 40

Checkpoints are saved to:
    models_joint/scheme_JA/seed_1/weights.safetensors
    models_joint/scheme_JA/seed_1/metadata.json
"""

import argparse
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from joint_model import JointCNN
from joint_dataset import load_joint_data, joint_batch_iterator, JOINT_SCHEMES


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def loss_fn(model, x_color, y_color, x_cifar, y_cifar):
    """Combined cross-entropy loss from both tasks."""
    logits_color  = model(x_color, task=0)
    logits_object = model(x_cifar, task=1)
    loss_color  = mx.mean(nn.losses.cross_entropy(logits_color,  y_color))
    loss_object = mx.mean(nn.losses.cross_entropy(logits_object, y_cifar))
    return loss_color + loss_object, loss_color, loss_object


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_head(model, X, y, task, batch_size=256):
    model.eval()
    correct = total = 0
    for i in range(0, len(X), batch_size):
        xb = mx.array(X[i:i + batch_size])
        yb = y[i:i + batch_size]
        logits = model(xb, task=task)
        preds = np.array(mx.argmax(logits, axis=1))
        correct += int((preds == yb).sum())
        total   += len(yb)
    model.train()
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(scheme, seed, epochs, batch_size, lr, base_model_dir, color_base_dir):
    mx.random.seed(seed)
    np.random.seed(seed)

    # Load data
    data = load_joint_data(scheme, color_base_dir=color_base_dir)

    num_color_classes  = len(data["color_label_to_idx"])
    num_object_classes = len(data["cifar_label_to_idx"])

    model = JointCNN(
        num_color_classes=num_color_classes,
        num_object_classes=num_object_classes,
        input_size=32,
    )
    mx.eval(model.parameters())

    # Cosine annealing over both datasets
    steps_per_epoch = max(
        len(data["color_train"]), len(data["cifar_train"])
    ) // batch_size
    total_steps = epochs * steps_per_epoch
    scheduler   = optim.cosine_decay(lr, total_steps)
    optimizer   = optim.Adam(learning_rate=scheduler)

    state = [model.state, optimizer.state]

    def step(model, optimizer, xc, yc, xo, yo):
        def _loss(model):
            total, lc, lo = loss_fn(model, xc, yc, xo, yo)
            return total, (lc, lo)
        (total_loss, (lc, lo)), grads = nn.value_and_grad(model, _loss)(model)
        optimizer.update(model, grads)
        return total_loss, lc, lo

    out_dir = os.path.join(base_model_dir, f"scheme_{scheme}", f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nTraining JointCNN | scheme={scheme} seed={seed} epochs={epochs}")
    print(f"  Color classes={num_color_classes}  Object classes={num_object_classes}")
    print(f"  Color train={len(data['color_train'])}  CIFAR train={len(data['cifar_train'])}")

    best_color_acc = best_object_acc = 0.0

    for epoch in range(epochs):
        model.train()
        losses, lc_list, lo_list = [], [], []
        tic = time.time()

        for xc, yc, xo, yo in joint_batch_iterator(
            batch_size,
            data["color_train"], data["color_y_train"],
            data["cifar_train"], data["cifar_y_train"],
            augment=True,
        ):
            total, lc, lo = step(
                model, optimizer,
                mx.array(xc), mx.array(yc),
                mx.array(xo), mx.array(yo),
            )
            mx.eval(state, total)
            losses.append(total.item())
            lc_list.append(lc.item())
            lo_list.append(lo.item())

        toc = time.time()

        color_acc  = evaluate_head(model, data["color_val"],  data["color_y_val"],  task=0)
        object_acc = evaluate_head(model, data["cifar_val"],  data["cifar_y_val"],  task=1)
        best_color_acc  = max(best_color_acc,  color_acc)
        best_object_acc = max(best_object_acc, object_acc)

        avg_total = sum(losses)  / len(losses)
        avg_lc    = sum(lc_list) / len(lc_list)
        avg_lo    = sum(lo_list) / len(lo_list)

        print(
            f"  Epoch {epoch+1:02d}: "
            f"loss={avg_total:.4f} (color={avg_lc:.3f} obj={avg_lo:.3f})  "
            f"color_acc={color_acc:.4f}  obj_acc={object_acc:.4f}  "
            f"t={toc-tic:.1f}s"
        )

    # Final evaluation
    color_acc_final  = evaluate_head(model, data["color_val"], data["color_y_val"],  task=0)
    object_acc_final = evaluate_head(model, data["cifar_val"], data["cifar_y_val"],  task=1)

    # Save weights
    weights_path = os.path.join(out_dir, "weights.safetensors")
    mx.save_safetensors(weights_path, dict(tree_flatten(model.parameters())))

    # Save metadata
    metadata = {
        "scheme":              scheme,
        "seed":                seed,
        "epochs":              epochs,
        "batch_size":          batch_size,
        "lr":                  lr,
        "num_color_classes":   num_color_classes,
        "num_object_classes":  num_object_classes,
        "label_map_color":     data["color_label_to_idx"],
        "label_map_object":    data["cifar_label_to_idx"],
        "final_color_acc":     color_acc_final,
        "final_object_acc":    object_acc_final,
        "best_color_acc":      best_color_acc,
        "best_object_acc":     best_object_acc,
        "scheme_description":  JOINT_SCHEMES[scheme]["description"],
        "model_type":          "JointCNN",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved to {out_dir}")
    print(f"  Final  color_acc={color_acc_final:.4f}  object_acc={object_acc_final:.4f}")
    print(f"  Best   color_acc={best_color_acc:.4f}   object_acc={best_object_acc:.4f}")

    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train JointCNN on color-blocks + CIFAR-10")
    parser.add_argument("--scheme",   type=str, choices=list(JOINT_SCHEMES), required=True)
    parser.add_argument("--seed",     type=int, nargs="+", default=[1])
    parser.add_argument("--epochs",   type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr",       type=float, default=0.001)
    parser.add_argument("--base_model_dir", type=str, default="models_joint")
    parser.add_argument("--color_base_dir", type=str, default="dataset")
    args = parser.parse_args()

    for seed in args.seed:
        train(
            scheme=args.scheme,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            base_model_dir=args.base_model_dir,
            color_base_dir=args.color_base_dir,
        )


if __name__ == "__main__":
    main()
