"""
SAM-based concept labeler for CIFAR images.

Uses SamAutomaticMaskGenerator to segment each image, then applies concept
filter functions to assign binary labels (has_concept / not).

Designed to discover concepts that were NEVER in training labels — e.g. "sky",
"ocean", "grass" — purely from visual structure.

Usage:
    python src/sam_concept_labeler.py --concept sky --save_path results/sky_labels.json
    python src/sam_concept_labeler.py --concept ocean --save_path results/ocean_labels.json
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

SAM_SRC = "/Users/ianheadley/Documents/mlx-examples-main/segment_anything"


# ---------------------------------------------------------------------------
# HSV helpers (reused from cifar_color_labels.py)
# ---------------------------------------------------------------------------

def _rgb_to_hsv_array(pixels_float):
    """(N, 3) float [0,1] → hue (°), sat, val arrays."""
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
    hue *= 60.0
    return hue, sat, val


def _is_blue_or_grey(pixels_uint8, blue_frac_thresh=0.35, grey_frac_thresh=0.30):
    """
    Returns True if a pixel array is predominantly blue-sky or grey/overcast.
    Blue: hue 160-250°, sat > 0.15, val > 0.2
    Grey: sat < 0.18, val > 0.25
    """
    flat = pixels_uint8.astype(np.float32) / 255.0
    hue, sat, val = _rgb_to_hsv_array(flat)
    n = len(flat) + 1e-8
    blue_mask = (hue >= 160) & (hue <= 250) & (sat > 0.15) & (val > 0.20)
    grey_mask = (sat < 0.18) & (val > 0.25)
    blue_frac = blue_mask.sum() / n
    grey_frac = grey_mask.sum() / n
    return (blue_frac >= blue_frac_thresh) or (grey_frac >= grey_frac_thresh), \
           float(max(blue_frac, grey_frac))


# ---------------------------------------------------------------------------
# Concept filter functions
# ---------------------------------------------------------------------------

def sky_filter(masks, img_upscaled_uint8, img_h=256, img_w=256,
               centroid_top_frac=0.50, min_area_frac=0.06,
               min_width_height_ratio=1.2):
    """
    Identify sky masks from SAM output.

    Sky criteria:
      - Mask centroid in top 50% of image (sky is above horizon)
      - Mask covers > 6% of image
      - Mask is wider than tall (sky is horizontal)
      - Dominant color is blue (hue 160-250°) or grey/white (overcast)

    Returns (has_sky, confidence, best_mask_or_None)
    """
    best_conf = 0.0
    best_mask = None

    for m in masks:
        seg = np.array(m["segmentation"]).astype(bool)   # (H, W)
        area_frac = seg.sum() / (img_h * img_w)
        if area_frac < min_area_frac or area_frac > 0.80:
            continue

        # Centroid vertical position
        ys, xs = np.where(seg)
        centroid_y = ys.mean() / img_h
        if centroid_y > centroid_top_frac:
            continue

        # Shape: width > height
        h_span = ys.max() - ys.min() + 1
        w_span = xs.max() - xs.min() + 1
        if w_span < h_span * min_width_height_ratio:
            continue

        # Color check
        pixels = img_upscaled_uint8[seg]
        is_sky_color, conf = _is_blue_or_grey(pixels)
        if not is_sky_color:
            continue

        total_conf = conf * area_frac * (1.0 - centroid_y)
        if total_conf > best_conf:
            best_conf = total_conf
            best_mask = seg

    return best_mask is not None, best_conf, best_mask


def ocean_filter(masks, img_upscaled_uint8, img_h=256, img_w=256,
                 centroid_bottom_frac=0.45, min_area_frac=0.06,
                 min_width_height_ratio=1.2):
    """
    Identify ocean/water masks from SAM output.

    Ocean criteria:
      - Mask centroid in bottom 55% of image (water is below horizon)
      - Mask covers > 6% of image
      - Mask is wider than tall
      - Dominant color is blue or grey/green-blue

    Returns (has_ocean, confidence, best_mask_or_None)
    """
    best_conf = 0.0
    best_mask = None

    for m in masks:
        seg = np.array(m["segmentation"]).astype(bool)
        area_frac = seg.sum() / (img_h * img_w)
        if area_frac < min_area_frac or area_frac > 0.80:
            continue

        ys, xs = np.where(seg)
        centroid_y = ys.mean() / img_h
        if centroid_y < centroid_bottom_frac:
            continue

        h_span = ys.max() - ys.min() + 1
        w_span = xs.max() - xs.min() + 1
        if w_span < h_span * min_width_height_ratio:
            continue

        pixels = img_upscaled_uint8[seg]
        is_water_color, conf = _is_blue_or_grey(pixels)
        if not is_water_color:
            continue

        total_conf = conf * area_frac * centroid_y
        if total_conf > best_conf:
            best_conf = total_conf
            best_mask = seg

    return best_mask is not None, best_conf, best_mask


CONCEPT_FILTERS = {
    "sky":   sky_filter,
    "ocean": ocean_filter,
}

# For each concept: which CIFAR-10 classes to probe (likely positive) and
# which to use as hard negatives (definitely don't contain the concept).
# This limits SAM to ~4000 images instead of 10k, cutting runtime ~60%.
CONCEPT_CLASS_STRATEGY = {
    "sky": {
        "probe_classes":    ["airplane", "bird", "ship", "deer", "horse"],
        "negative_classes": ["automobile", "truck", "frog", "cat", "dog"],
        # Point prompt location in 256x256 image space: top-center for sky
        "prompt_point": (128, 32),
    },
    "ocean": {
        "probe_classes":    ["ship", "frog", "deer", "horse"],
        "negative_classes": ["automobile", "truck", "airplane", "cat"],
        # Bottom-center for ocean/water
        "prompt_point": (128, 220),
    },
}


# ---------------------------------------------------------------------------
# SAM setup  — point-prompt predictor (faster than auto-mask generator)
# ---------------------------------------------------------------------------

def load_sam_predictor(model_dir="weights/sam-vit-base"):
    """Load MLX SamPredictor."""
    sys.path.insert(0, SAM_SRC)
    from segment_anything.predictor import SamPredictor
    from segment_anything.sam import load as sam_load
    model = sam_load(model_dir)
    return SamPredictor(model)


def upscale_32_to_256(img_float32):
    """(32, 32, 3) float32 [0,1] → (256, 256, 3) uint8 RGB."""
    img_uint8 = (img_float32 * 255).clip(0, 255).astype(np.uint8)
    return np.array(Image.fromarray(img_uint8).resize((256, 256), Image.BICUBIC))


def _predict_mask_at_point(predictor, img_256, px, py):
    """
    Run SamPredictor with a single foreground point, return best mask.
    Returns (mask_bool_HxW, iou_score) or (None, 0) on failure.
    """
    predictor.set_image(img_256)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[px, py]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask_output=True,
    )
    masks  = np.array(masks)[0]   # (H, W, 3)
    scores = np.array(scores)[0]  # (3,)
    best   = int(np.argmax(scores))
    return masks[:, :, best].astype(bool), float(scores[best])


# ---------------------------------------------------------------------------
# Batch labeling
# ---------------------------------------------------------------------------

def label_images_for_concept(X, y, concept_name, save_path,
                              class_names,
                              sam_dir="weights/sam-vit-base",
                              log_every=200):
    """
    Label images as has_concept=True/False using SAM point-prompt.

    Only runs SAM on probe_classes and negative_classes for the concept —
    skips irrelevant images entirely for speed.

    Args:
        X: (N, 32, 32, 3) float32 images
        y: (N,) int class indices
        concept_name: "sky" or "ocean"
        save_path: where to cache JSON output
        class_names: list mapping class index → name (CIFAR10_LABELS or CIFAR100_LABELS)

    Returns: list of dicts [{idx, has_concept, confidence, iou}]
    """
    if concept_name not in CONCEPT_FILTERS:
        raise ValueError(f"Unknown concept '{concept_name}'. Choose from {list(CONCEPT_FILTERS)}")

    filter_fn = CONCEPT_FILTERS[concept_name]
    strategy  = CONCEPT_CLASS_STRATEGY[concept_name]
    px, py    = strategy["prompt_point"]

    probe_ids = {class_names.index(c) for c in strategy["probe_classes"]
                 if c in class_names}
    neg_ids   = {class_names.index(c) for c in strategy["negative_classes"]
                 if c in class_names}
    all_ids   = probe_ids | neg_ids

    # Load existing cache
    if os.path.exists(save_path):
        with open(save_path) as f:
            existing = json.load(f)
        done_ids = {e["idx"] for e in existing}
        print(f"Resuming — {len(done_ids)} already labeled")
    else:
        existing = []
        done_ids = set()

    # Determine which images to process
    todo = [i for i, yi in enumerate(y)
            if int(yi) in all_ids and i not in done_ids]

    print(f"Concept '{concept_name}': running SAM on {len(todo)} images "
          f"({len(probe_ids)} probe classes + {len(neg_ids)} negative classes)")
    print(f"  Probe point: ({px}, {py}) in 256x256 space")

    predictor = load_sam_predictor(sam_dir)
    results   = list(existing)
    n_pos = sum(1 for e in results if e["has_concept"])
    n_neg = sum(1 for e in results if not e["has_concept"])

    for count, idx in enumerate(todo, 1):
        img_256 = upscale_32_to_256(X[idx])
        mask, iou = _predict_mask_at_point(predictor, img_256, px, py)

        # Build a fake single-mask list to reuse existing filter functions
        fake_masks = [{"segmentation": mask, "predicted_iou": iou}]
        has_concept, confidence, _ = filter_fn(fake_masks, img_256)

        # Hard negatives: force False for definite-negative classes
        if int(y[idx]) in neg_ids:
            has_concept = False
            confidence  = 0.0

        results.append({
            "idx":         idx,
            "has_concept": bool(has_concept),
            "confidence":  round(float(confidence), 4),
            "iou":         round(iou, 4),
            "class":       class_names[int(y[idx])],
        })

        if has_concept: n_pos += 1
        else:           n_neg += 1

        if count % log_every == 0 or count == len(todo):
            pct = 100 * count / len(todo)
            rate = 100 * n_pos / (n_pos + n_neg) if (n_pos + n_neg) else 0
            print(f"  {count}/{len(todo)} ({pct:.0f}%)  "
                  f"positive={n_pos}  negative={n_neg}  rate={rate:.1f}%")
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(results, f)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f)

    print(f"\nSaved {len(results)} labels → {save_path}")
    n_total = len(results)
    print(f"  Positive ({concept_name}): {n_pos} ({100*n_pos/n_total:.1f}%)")
    print(f"  Negative:                  {n_neg} ({100*n_neg/n_total:.1f}%)")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAM concept labeler for CIFAR images")
    parser.add_argument("--concept",       required=True, choices=list(CONCEPT_FILTERS),
                        help="Concept to label")
    parser.add_argument("--save_path",     required=True,
                        help="Output JSON path")
    parser.add_argument("--sam_dir",       default="weights/sam-vit-base")
    parser.add_argument("--split",         default="test", choices=["train", "test"])
    parser.add_argument("--cifar_version", default=10, type=int, choices=[10, 100])
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(__file__))
    if args.cifar_version == 100:
        from cifar100_dataset import load_raw_cifar100, CIFAR100_LABELS
        X_train, y_train, X_test, y_test = load_raw_cifar100()
        class_names = CIFAR100_LABELS
    else:
        from cifar_dataset import load_raw_cifar10, CIFAR10_LABELS
        X_train, y_train, X_test, y_test = load_raw_cifar10()
        class_names = CIFAR10_LABELS

    X = X_test  if args.split == "test"  else X_train
    y = y_test  if args.split == "test"  else y_train
    print(f"Labeling CIFAR-{args.cifar_version} {args.split} set ({len(X)} images) "
          f"for concept '{args.concept}'")

    label_images_for_concept(
        X, y, args.concept, args.save_path,
        class_names=class_names,
        sam_dir=args.sam_dir,
    )


if __name__ == "__main__":
    main()
