"""
SAM-based color labeling for CIFAR-10 vehicle images.

Uses Segment Anything (MLX) with a center-point prompt to isolate the main
object in each image, then assigns a Berlin-Kay color label to that object's
pixels.  This gives ground-truth color labels for vehicles so we can evaluate
the compositionality probe accurately.

Vehicles labeled: airplane (0), automobile (1), ship (8), truck (9).

Usage:
    # From GRUE root — downloads weights, labels, caches to JSON
    python src/cifar_color_labels.py --save_path dataset_cifar10/vehicle_color_labels.json

    # Skip download if weights already present
    python src/cifar_color_labels.py --sam_dir weights/sam-vit-base --save_path dataset_cifar10/vehicle_color_labels.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# SAM path — MLX implementation
# ---------------------------------------------------------------------------

SAM_SRC = "/Users/ianheadley/Documents/mlx-examples-main/segment_anything"

# ---------------------------------------------------------------------------
# CIFAR vehicle classes to label
# ---------------------------------------------------------------------------

VEHICLE_CLASS_IDS = {0: "airplane", 1: "automobile", 8: "ship", 9: "truck"}

# CIFAR-100 vehicles (colorable objects)
VEHICLE_CLASS_IDS_100 = {
    8:  "bicycle",
    13: "bus",
    48: "motorcycle",
    58: "pickup_truck",
    81: "streetcar",
    90: "train",
}

# ---------------------------------------------------------------------------
# Berlin-Kay hue ranges (degrees 0-360) — mirrors joint_dataset.py
# ---------------------------------------------------------------------------

HUE_RANGES = {
    "red":    [(0, 15), (345, 360)],
    "orange": [(15, 40)],
    "yellow": [(40, 70)],
    "green":  [(70, 160)],
    "blue":   [(160, 250)],
    "purple": [(250, 290)],
    "pink":   [(290, 345)],
}


# ---------------------------------------------------------------------------
# Color extraction from raw pixels
# ---------------------------------------------------------------------------

def _rgb_to_hue_sat_val(pixels_float):
    """Vectorised HSV from (N, 3) float array in [0, 1]. Returns (hue, sat, val)."""
    r, g, b = pixels_float[:, 0], pixels_float[:, 1], pixels_float[:, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = mx - mn + 1e-8

    sat = np.where(mx > 1e-8, delta / np.maximum(mx, 1e-8), 0.0)
    val = mx

    hue = np.zeros(len(pixels_float), dtype=np.float32)
    is_r = (mx == r) & (delta > 1e-8)
    is_g = (mx == g) & (~is_r) & (delta > 1e-8)
    is_b = (mx == b) & (~is_r) & (~is_g) & (delta > 1e-8)
    hue[is_r] = ((g[is_r] - b[is_r]) / delta[is_r]) % 6
    hue[is_g] = ((b[is_g] - r[is_g]) / delta[is_g]) + 2
    hue[is_b] = ((r[is_b] - g[is_b]) / delta[is_b]) + 4
    hue = hue * 60.0
    return hue, sat, val


def dominant_color_from_pixels(pixels_uint8, sat_thresh=0.25, val_thresh=0.20,
                                dominance_thresh=0.28):
    """
    Assign a Berlin-Kay color name to a pixel array.

    Args:
        pixels_uint8: (N, 3) uint8 RGB array — the SAM-masked object pixels.
    Returns:
        (color_name_or_None, confidence_float)
    """
    if len(pixels_uint8) < 20:
        return None, 0.0

    flat = pixels_uint8.astype(np.float32) / 255.0
    hue, sat, val = _rgb_to_hue_sat_val(flat)

    # Only consider vibrant, bright pixels for chromatic colors
    vibrant = (sat > sat_thresh) & (val > val_thresh)
    scores = {}

    if int(vibrant.sum()) >= 10:
        hue_v  = hue[vibrant]
        weight = sat[vibrant] * val[vibrant]
        total  = weight.sum() + 1e-8

        for color, ranges in HUE_RANGES.items():
            m = np.zeros(len(hue_v), dtype=bool)
            for lo, hi in ranges:
                m |= (hue_v >= lo) & (hue_v < hi)
            scores[color] = float(weight[m].sum()) / total
    else:
        for color in HUE_RANGES:
            scores[color] = 0.0

    # Achromatic categories over ALL pixels
    n = len(flat) + 1e-8
    scores["black"]  = float((val < 0.20).sum()) / n
    scores["white"]  = float(((val > 0.85) & (sat < 0.15)).sum()) / n
    scores["grey"]   = float(((val >= 0.20) & (val <= 0.85) & (sat < 0.15)).sum()) / n
    scores["brown"]  = float(((hue >= 15) & (hue < 40) & (val < 0.55) & (sat > 0.25)).sum()) / n

    best_color = max(scores, key=scores.get)
    best_score = scores[best_color]

    if best_score < dominance_thresh:
        return None, best_score

    return best_color, best_score


# ---------------------------------------------------------------------------
# SAM setup
# ---------------------------------------------------------------------------

def download_and_convert_sam(save_dir="weights/sam-vit-base",
                             hf_repo="facebook/sam-vit-base"):
    """
    Download SAM weights from HuggingFace and convert to MLX format.
    Skips download if model.safetensors already exists in save_dir.
    """
    save_path = Path(save_dir)
    if (save_path / "model.safetensors").exists():
        print(f"SAM weights already at {save_dir} — skipping download.")
        return save_dir

    print(f"Downloading {hf_repo} → {save_dir} ...")
    sys.path.insert(0, SAM_SRC)
    import shutil
    import mlx.core as mx
    from huggingface_hub import snapshot_download

    hf_path = Path(snapshot_download(
        repo_id=hf_repo,
        allow_patterns=["*.safetensors", "*.json"],
        resume_download=True,
    ))

    # Convert weights (transpose conv kernels to MLX layout)
    weights = mx.load(str(hf_path / "model.safetensors"))
    mlx_weights = {}
    for k, v in weights.items():
        if k in {
            "vision_encoder.patch_embed.projection.weight",
            "vision_encoder.neck.conv1.weight",
            "vision_encoder.neck.conv2.weight",
            "prompt_encoder.mask_embed.conv1.weight",
            "prompt_encoder.mask_embed.conv2.weight",
            "prompt_encoder.mask_embed.conv3.weight",
        }:
            v = v.transpose(0, 2, 3, 1)
        if k in {
            "mask_decoder.upscale_conv1.weight",
            "mask_decoder.upscale_conv2.weight",
        }:
            v = v.transpose(1, 2, 3, 0)
        mlx_weights[k] = v

    save_path.mkdir(parents=True, exist_ok=True)
    model_file = str(save_path / "model.safetensors")
    mx.save_safetensors(model_file, mlx_weights)

    # Write index file
    total_size = sum(v.nbytes for v in mlx_weights.values())
    index_data = {
        "metadata": {"total_size": total_size},
        "weight_map": {k: "model.safetensors" for k in sorted(mlx_weights)},
    }
    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=4)

    shutil.copy(hf_path / "config.json", save_path / "config.json")
    print(f"SAM weights saved to {save_dir}")
    return save_dir


def load_sam_predictor(model_dir="weights/sam-vit-base"):
    """Load MLX SamPredictor from a converted weights directory."""
    sys.path.insert(0, SAM_SRC)
    from segment_anything.predictor import SamPredictor
    from segment_anything.sam import load as sam_load
    model = sam_load(model_dir)
    return SamPredictor(model)


# ---------------------------------------------------------------------------
# Per-image labeling
# ---------------------------------------------------------------------------

def upscale_32_to_256(img_float32):
    """(32, 32, 3) float32 [0,1] → (256, 256, 3) uint8 RGB."""
    img_uint8 = (img_float32 * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img_uint8).resize((256, 256), Image.BICUBIC)
    return np.array(pil)


def label_single_image(predictor, img_32x32_float, center_frac=0.5):
    """
    Use SAM center-point prompt to isolate the main object and assign a color.

    CIFAR images are object-centric (subject is centered), so a single
    foreground point at the image center reliably hits the main object.

    Returns:
        color (str or None), area_frac (float), iou_score (float)
    """
    img_256 = upscale_32_to_256(img_32x32_float)
    predictor.set_image(img_256)

    cx = cy = int(256 * center_frac)
    # point_coords shape must be (N, 2); predictor adds batch dim internally
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[cx, cy]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask_output=True,
    )
    # masks: (1, H, W, 3)  scores: (1, 3)
    masks  = np.array(masks)[0]    # (H, W, 3)
    scores = np.array(scores)[0]   # (3,)
    best_idx  = int(np.argmax(scores))
    best_mask = masks[:, :, best_idx].astype(bool)  # (256, 256)
    iou_score = float(scores[best_idx])

    area_frac = best_mask.sum() / (256 * 256)

    # Skip if mask is implausibly tiny (< 4%) or swallows whole image (> 85%)
    if area_frac < 0.04 or area_frac > 0.85:
        return None, float(area_frac), iou_score

    masked_pixels = img_256[best_mask]    # (N, 3) uint8
    color, confidence = dominant_color_from_pixels(masked_pixels)

    return color, float(area_frac), iou_score


# ---------------------------------------------------------------------------
# Batch labeling
# ---------------------------------------------------------------------------

def label_all_vehicles(X, y, save_path="dataset_cifar10/vehicle_color_labels.json",
                       sam_dir="weights/sam-vit-base", log_every=200,
                       vehicle_class_ids=None):
    """
    Label every vehicle image in CIFAR-10 (airplane/automobile/ship/truck).

    Results are cached to save_path as JSON so SAM only needs to run once.

    Args:
        X: (N, 32, 32, 3) float32 CIFAR images
        y: (N,) int class indices
        save_path: where to write JSON output
        sam_dir: path to converted MLX SAM weights

    Returns:
        dict {str(idx): {"class": str, "color": str|None, "area_frac": float, "iou": float}}
    """
    if vehicle_class_ids is None:
        vehicle_class_ids = VEHICLE_CLASS_IDS

    predictor = load_sam_predictor(sam_dir)

    vehicle_indices = [i for i, yi in enumerate(y) if int(yi) in vehicle_class_ids]
    print(f"Labeling {len(vehicle_indices)} vehicle images with SAM...")

    results = {}
    n_colored = 0
    n_failed  = 0

    for count, idx in enumerate(vehicle_indices, 1):
        class_name = vehicle_class_ids[int(y[idx])]
        color, area_frac, iou = label_single_image(predictor, X[idx])

        results[str(idx)] = {
            "class":     class_name,
            "color":     color,
            "area_frac": round(area_frac, 4),
            "iou":       round(iou, 4),
        }

        if color is not None:
            n_colored += 1
        else:
            n_failed  += 1

        if count % log_every == 0 or count == len(vehicle_indices):
            pct = 100 * count / len(vehicle_indices)
            print(f"  {count}/{len(vehicle_indices)} ({pct:.0f}%)  "
                  f"labeled={n_colored}  no_color={n_failed}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f)

    print(f"\nSaved {len(results)} labels → {save_path}")
    print(f"  Successfully colored: {n_colored} ({100*n_colored/len(results):.1f}%)")
    print(f"  No color assigned:    {n_failed}  ({100*n_failed/len(results):.1f}%)")

    return results


def load_vehicle_labels(path="dataset_cifar10/vehicle_color_labels.json"):
    """
    Load cached SAM vehicle color labels.

    Returns:
        dict {int(idx): {"class": str, "color": str|None, ...}}
    """
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def print_label_stats(labels):
    """Print breakdown by class and color."""
    from collections import Counter
    by_class = {}
    for entry in labels.values():
        cls = entry["class"]
        col = entry["color"] or "unknown"
        by_class.setdefault(cls, Counter())[col] += 1

    for cls in sorted(by_class):
        total = sum(by_class[cls].values())
        print(f"\n  {cls} ({total} images):")
        for col, cnt in sorted(by_class[cls].items(), key=lambda x: -x[1]):
            print(f"    {col:10s} {cnt:4d} ({100*cnt/total:.1f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAM color labeling for CIFAR vehicles")
    parser.add_argument("--sam_dir",       default="weights/sam-vit-base")
    parser.add_argument("--save_path",     default=None,
                        help="Output JSON path (defaults to dataset_cifarN/vehicle_color_labels.json)")
    parser.add_argument("--split",         default="test", choices=["train", "test"])
    parser.add_argument("--cifar_version", default=10, type=int, choices=[10, 100])
    parser.add_argument("--download",      action="store_true",
                        help="Download and convert SAM weights if not present")
    args = parser.parse_args()

    if args.download:
        download_and_convert_sam(args.sam_dir)

    sys.path.insert(0, os.path.dirname(__file__))

    if args.cifar_version == 100:
        from cifar100_dataset import load_raw_cifar100
        X_train, y_train, X_test, y_test = load_raw_cifar100()
        vehicle_ids = VEHICLE_CLASS_IDS_100
        default_save = "dataset_cifar100/vehicle_color_labels.json"
    else:
        from cifar_dataset import load_raw_cifar10
        X_train, y_train, X_test, y_test = load_raw_cifar10()
        vehicle_ids = VEHICLE_CLASS_IDS
        default_save = "dataset_cifar10/vehicle_color_labels.json"

    save_path = args.save_path or default_save

    if args.split == "test":
        X, y = X_test, y_test
    else:
        X, y = X_train, y_train

    labels = label_all_vehicles(X, y, save_path=save_path, sam_dir=args.sam_dir,
                                vehicle_class_ids=vehicle_ids)
    print_label_stats(labels)


if __name__ == "__main__":
    main()
