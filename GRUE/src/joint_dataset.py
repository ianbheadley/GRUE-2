"""
Joint dataset loader for color-block + CIFAR-10 multi-task training.

Color blocks are resized to 32x32 (same as CIFAR) so both datasets pass
through identical conv layers. Color is solid-fill, so downsampling is lossless
for the classification signal.

Dominant-hue labeling
---------------------
CIFAR-10 has no color ground truth. We assign color pseudo-labels by measuring
the dominant hue of each image's vibrant pixels (ignoring dark/grey background).
This is used for:
  - Filtering: selecting "red trucks", "blue trucks" etc. for compositionality probing
  - Validation grids: saved as PNGs for manual review by the user

The labeling is noisy — a red truck photographed against a red sunset may label
as "sky-red" rather than "truck-red". Noise is measured and reported explicitly.

Schemes
-------
JA  All labels for both datasets
JB  Color: blue+green -> grue   |  CIFAR: auto+truck -> motor_vehicle
JC  Color: remove red entirely  |  CIFAR: all labels (red trucks still present)
JD  Color: all labels           |  CIFAR: remove truck
JE  Color: remove red           |  CIFAR: also filter out red-dominant images (double suppression)
"""

import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

from cifar_dataset import load_raw_cifar10, CIFAR10_LABELS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLOR_CLASSES = [
    "black", "blue", "brown", "green", "grey",
    "orange", "pink", "purple", "red", "white", "yellow",
]

# Berlin-Kay hue ranges in degrees (0-360).  Matches generate_dataset.py.
HUE_RANGES = {
    "red":    [(0, 15), (345, 360)],
    "orange": [(15, 40)],
    "yellow": [(40, 70)],
    "green":  [(70, 160)],
    "blue":   [(160, 250)],
    "purple": [(250, 290)],
    "pink":   [(290, 345)],
}
# brown / black / white / grey are handled by brightness+saturation thresholds

JOINT_SCHEMES = {
    "JA":  {"color_scheme": "A",     "cifar_scheme": "A",
            "cifar_version": 10,
            "description": "Joint baseline — all labels"},
    "JB":  {"color_scheme": "B",     "cifar_scheme": "B",
            "cifar_version": 10,
            "description": "Merge blue+green (grue) and auto+truck (motor_vehicle)"},
    "JC":  {"color_scheme": "C_red", "cifar_scheme": "A",
            "cifar_version": 10,
            "description": "Remove red from color supervision; CIFAR intact"},
    "JD":  {"color_scheme": "A",     "cifar_scheme": "C",
            "cifar_version": 10,
            "description": "Color intact; remove truck from CIFAR"},
    "JE":  {"color_scheme": "C_red", "cifar_scheme": "A_no_red",
            "cifar_version": 10,
            "description": "Double suppression: no red in color OR CIFAR"},
    # CIFAR-100 schemes
    "JA100": {"color_scheme": "A", "cifar_scheme": "A",
              "cifar_version": 100,
              "description": "Joint baseline — CIFAR-100 all labels"},
    "JC100": {"color_scheme": "C_red", "cifar_scheme": "A",
              "cifar_version": 100,
              "description": "Remove red from color supervision; CIFAR-100 intact"},
}


# ---------------------------------------------------------------------------
# Dominant-hue labeling for CIFAR images
# ---------------------------------------------------------------------------

def _rgb_to_hue_sat_val(img_float):
    """Vectorised HSV extraction. img_float: (H*W, 3) in [0,1]."""
    r, g, b = img_float[:, 0], img_float[:, 1], img_float[:, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = mx - mn + 1e-8

    sat = np.where(mx > 1e-8, delta / np.maximum(mx, 1e-8), 0.0)
    val = mx

    hue = np.zeros(len(img_float), dtype=np.float32)
    is_r = (mx == r) & (delta > 1e-8)
    is_g = (mx == g) & (~is_r) & (delta > 1e-8)
    is_b = (mx == b) & (~is_r) & (~is_g) & (delta > 1e-8)
    hue[is_r] = ((g[is_r] - b[is_r]) / delta[is_r]) % 6
    hue[is_g] = ((b[is_g] - r[is_g]) / delta[is_g]) + 2
    hue[is_b] = ((r[is_b] - g[is_b]) / delta[is_b]) + 4
    hue = hue * 60.0  # 0-360

    return hue, sat, val


def dominant_hue_label_full(img, sat_thresh=0.25, val_thresh=0.20, dominance_thresh=0.30):
    """
    Assign a color name to a CIFAR image based on its dominant vibrant-pixel hue.

    Returns (color_name_or_None, confidence_float).
    confidence = fraction of vibrant pixel mass in the winning bucket.

    Thresholds are intentionally relaxed vs. the existing blue/green-only version
    so that objects which are partially occluded or photographed against cluttered
    backgrounds still get labelled.
    """
    arr = img.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0

    flat = arr.reshape(-1, 3)
    hue, sat, val = _rgb_to_hue_sat_val(flat)

    # Keep only vibrant, bright pixels — suppresses grey background and shadows
    mask = (sat > sat_thresh) & (val > val_thresh)
    if int(mask.sum()) < 15:
        return None, 0.0

    hue_v = hue[mask]
    weight = (sat[mask] * val[mask])        # weight vivid pixels more
    total  = weight.sum() + 1e-8

    # Score each named colour bucket
    scores = {}

    for color, ranges in HUE_RANGES.items():
        color_mask = np.zeros(len(hue_v), dtype=bool)
        for lo, hi in ranges:
            color_mask |= (hue_v >= lo) & (hue_v < hi)
        scores[color] = float(weight[color_mask].sum()) / total

    # Special achromatic categories via brightness / saturation
    all_flat = flat.reshape(-1, 3)
    all_hue, all_sat, all_val = _rgb_to_hue_sat_val(all_flat)
    px_count = len(all_flat) + 1e-8

    scores["black"] = float(((all_val < 0.20)).sum()) / px_count
    scores["white"] = float(((all_val > 0.85) & (all_sat < 0.15)).sum()) / px_count
    scores["grey"]  = float(((all_val >= 0.20) & (all_val <= 0.85) & (all_sat < 0.15)).sum()) / px_count
    scores["brown"] = float(((all_hue >= 15) & (all_hue < 40) & (all_val < 0.55) & (all_sat > 0.25)).sum()) / px_count

    best_color = max(scores, key=scores.get)
    best_score = scores[best_color]

    if best_score < dominance_thresh:
        return None, best_score

    return best_color, best_score


def label_cifar_by_color(X, y_raw, classes_of_interest=None, min_confidence=0.30):
    """
    Label every CIFAR image with a dominant color.

    Args:
        X:                   (N, 32, 32, 3) float32 images
        y_raw:               (N,) int class indices
        classes_of_interest: if given, only process these CIFAR class names
        min_confidence:      drop images whose dominant color is ambiguous

    Returns:
        list of dicts: [{"idx": int, "cifar_class": str, "color": str, "confidence": float}, ...]
    """
    results = []
    for idx, (img, cls_idx) in enumerate(zip(X, y_raw)):
        cifar_class = CIFAR10_LABELS[int(cls_idx)]
        if classes_of_interest and cifar_class not in classes_of_interest:
            continue
        color, conf = dominant_hue_label_full(img)
        if color is None or conf < min_confidence:
            continue
        results.append({
            "idx":         idx,
            "cifar_class": cifar_class,
            "color":       color,
            "confidence":  float(conf),
        })
    return results


def get_color_object_pairs(X, y_raw, color, cifar_class, max_count=200, min_confidence=0.30):
    """
    Return images that are both a specific color AND a specific CIFAR class.
    E.g., color="red", cifar_class="truck" -> red truck images.

    This is noisy by design (Option 1). Confidence is reported per-image.
    Use the returned metadata to build validation grids for manual review.
    """
    class_idx = CIFAR10_LABELS.index(cifar_class)
    candidates = np.where(y_raw == class_idx)[0]

    results = []
    for idx in candidates:
        c, conf = dominant_hue_label_full(X[idx])
        if c == color and conf >= min_confidence:
            results.append({"idx": int(idx), "confidence": float(conf)})
        if len(results) >= max_count:
            break

    # Sort by confidence descending so validation grid shows best matches first
    results.sort(key=lambda d: d["confidence"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Color-block dataset loading (resize to 32x32)
# ---------------------------------------------------------------------------

def load_color_data_joint(split, color_scheme, base_dir="dataset", resize_to=32):
    """
    Load color-block dataset for joint training.

    color_scheme:
        "A"     — all colors labelled normally
        "B"     — blue+green merged to "grue"
        "C_red" — red removed from training split (kept in val for probing)

    Returns: X (N, resize_to, resize_to, 3) float32, y (N,) int, label_to_idx dict
    """
    # Scheme C_red: use the normal metadata but drop red-labelled images
    base_scheme = "A" if color_scheme == "C_red" else color_scheme

    meta_file  = os.path.join(base_dir, f"metadata_{split}.json")
    label_file = os.path.join(base_dir, f"labels_{base_scheme}_{split}.json")

    # Scheme C uses a separate reduced metadata file
    if color_scheme == "C" and os.path.exists(os.path.join(base_dir, f"metadata_C_{split}.json")):
        meta_file = os.path.join(base_dir, f"metadata_C_{split}.json")

    with open(label_file, "r") as f:
        labels_dict = json.load(f)

    # Apply scheme transformations
    if color_scheme == "B":
        labels_dict = {
            k: ("grue" if v in ("blue", "green") else v)
            for k, v in labels_dict.items()
        }

    # Build label index from the final label set
    unique_labels = sorted(set(labels_dict.values()))
    label_to_idx  = {l: i for i, l in enumerate(unique_labels)}

    with open(meta_file, "r") as f:
        metadata = json.load(f)

    X, Y = [], []
    for item in metadata:
        fname = item["filename"]
        if fname not in labels_dict:
            continue

        label = labels_dict[fname]

        # C_red: skip red images at train time (keep for probing)
        if color_scheme == "C_red" and split == "train" and label == "red":
            continue

        img_path = os.path.join(base_dir, split, fname)
        img = Image.open(img_path).convert("RGB")
        if resize_to != img.width or resize_to != img.height:
            img = img.resize((resize_to, resize_to), Image.BILINEAR)
        X.append(np.array(img, dtype=np.float32) / 255.0)
        Y.append(label_to_idx[label])

    return np.array(X), np.array(Y), label_to_idx


# ---------------------------------------------------------------------------
# CIFAR loading with scheme transforms
# ---------------------------------------------------------------------------

def load_cifar_joint(cifar_scheme, min_confidence=0.30):
    """
    Load CIFAR-10 for joint training with scheme transforms.

    cifar_scheme:
        "A"        — all classes
        "B"        — auto+truck merged to "motor_vehicle"
        "C"        — truck removed from training
        "A_no_red" — all classes but images where red is dominant are dropped
                     (used for JE double-suppression control)

    Returns: X_train, y_train, X_val, y_val, label_to_idx, X_raw_test, y_raw_test
    (X_raw_test / y_raw_test are always the full unfiltered test set for probing)
    """
    X_train, y_train_raw, X_val, y_val_raw = load_raw_cifar10()

    base = "A" if cifar_scheme in ("A", "A_no_red") else cifar_scheme

    # Build label mapping for this scheme
    if base == "B":
        label_names = sorted(set(
            "motor_vehicle" if n in ("automobile", "truck") else n
            for n in CIFAR10_LABELS
        ))
    elif base == "C":
        label_names = sorted(n for n in CIFAR10_LABELS if n != "truck")
    else:
        label_names = sorted(CIFAR10_LABELS)

    label_to_idx = {l: i for i, l in enumerate(label_names)}

    def transform(X, y_raw, is_train):
        out_X, out_Y = [], []
        for img, cls_idx in zip(X, y_raw):
            cls = CIFAR10_LABELS[int(cls_idx)]

            # Scheme C: drop truck from training
            if base == "C" and is_train and cls == "truck":
                continue

            # JE: drop red-dominant images from CIFAR training
            if cifar_scheme == "A_no_red" and is_train:
                color, conf = dominant_hue_label_full(img)
                if color == "red" and conf >= min_confidence:
                    continue

            # Scheme B: merge label
            if base == "B":
                cls = "motor_vehicle" if cls in ("automobile", "truck") else cls

            if cls not in label_to_idx:
                continue

            out_X.append(img)
            out_Y.append(label_to_idx[cls])

        return np.array(out_X, dtype=np.float32), np.array(out_Y, dtype=np.int32)

    X_tr, y_tr = transform(X_train, y_train_raw, is_train=True)
    X_v,  y_v  = transform(X_val,   y_val_raw,   is_train=False)

    # Always return raw test for probing (no scheme filtering)
    _, _, X_test_raw, y_test_raw = load_raw_cifar10()

    return X_tr, y_tr, X_v, y_v, label_to_idx, X_test_raw, y_test_raw


# ---------------------------------------------------------------------------
# Joint loader: combines both datasets
# ---------------------------------------------------------------------------

def load_joint_data(scheme, color_base_dir="dataset", resize_to=32):
    """
    Load both datasets for a given joint scheme.

    Returns a dict with keys:
        color_train, color_y_train, color_label_to_idx
        color_val,   color_y_val
        cifar_train, cifar_y_train, cifar_label_to_idx
        cifar_val,   cifar_y_val
        cifar_test_raw, cifar_y_test_raw   (unfiltered, for probing)
        scheme_config
    """
    if scheme not in JOINT_SCHEMES:
        raise ValueError(f"Unknown scheme {scheme!r}. Choose from {list(JOINT_SCHEMES)}")

    cfg          = JOINT_SCHEMES[scheme]
    color_scheme = cfg["color_scheme"]
    cifar_scheme = cfg["cifar_scheme"]
    cifar_ver    = cfg.get("cifar_version", 10)

    print(f"Loading joint scheme {scheme}: {cfg['description']}")

    X_c_tr, y_c_tr, color_l2i = load_color_data_joint("train", color_scheme, color_base_dir, resize_to)
    X_c_v,  y_c_v,  _         = load_color_data_joint("val",   color_scheme, color_base_dir, resize_to)

    print(f"  Color blocks  train={len(X_c_tr)}  val={len(X_c_v)}  classes={len(color_l2i)}")

    if cifar_ver == 100:
        X_ci_tr, y_ci_tr, X_ci_v, y_ci_v, cifar_l2i, X_ci_test, y_ci_test = load_cifar100_joint(cifar_scheme)
        print(f"  CIFAR-100     train={len(X_ci_tr)}  val={len(X_ci_v)}  classes={len(cifar_l2i)}")
    else:
        X_ci_tr, y_ci_tr, X_ci_v, y_ci_v, cifar_l2i, X_ci_test, y_ci_test = load_cifar_joint(cifar_scheme)
        print(f"  CIFAR-10      train={len(X_ci_tr)}  val={len(X_ci_v)}  classes={len(cifar_l2i)}")

    return {
        "color_train":       X_c_tr,
        "color_y_train":     y_c_tr,
        "color_label_to_idx": color_l2i,
        "color_val":         X_c_v,
        "color_y_val":       y_c_v,
        "cifar_train":       X_ci_tr,
        "cifar_y_train":     y_ci_tr,
        "cifar_label_to_idx": cifar_l2i,
        "cifar_val":         X_ci_v,
        "cifar_y_val":       y_ci_v,
        "cifar_test_raw":    X_ci_test,
        "cifar_y_test_raw":  y_ci_test,
        "scheme_config":     cfg,
    }


# ---------------------------------------------------------------------------
# CIFAR-100 joint loader
# ---------------------------------------------------------------------------

def load_cifar100_joint(cifar_scheme="A"):
    """
    Load CIFAR-100 for joint training.

    cifar_scheme:
        "A" — all 100 classes, standard labels

    Returns: X_train, y_train, X_val, y_val, label_to_idx, X_test_raw, y_test_raw
    """
    from cifar100_dataset import load_raw_cifar100, CIFAR100_LABELS

    X_all, y_all, X_test, y_test = load_raw_cifar100()

    # 80/20 train/val split (same ratio as CIFAR-10 loader)
    n = len(X_all)
    n_train = int(n * 0.8)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    X_train, y_train_raw = X_all[train_idx], y_all[train_idx]
    X_val,   y_val_raw   = X_all[val_idx],   y_all[val_idx]

    label_to_idx = {name: i for i, name in enumerate(CIFAR100_LABELS)}

    return X_train, y_train_raw, X_val, y_val_raw, label_to_idx, X_test, y_test


# ---------------------------------------------------------------------------
# Batch iterator
# ---------------------------------------------------------------------------

def joint_batch_iterator(batch_size, X_color, y_color, X_cifar, y_cifar, augment=False):
    """
    Yield (x_color_batch, y_color_batch, x_cifar_batch, y_cifar_batch).

    One call = one batch from each dataset. The two sub-batches are kept
    separate so the training loop can route them to the correct head.
    If one dataset is exhausted it wraps around (the smaller one repeats).
    """
    n_color = len(X_color)
    n_cifar = len(X_cifar)
    n_steps = max(n_color, n_cifar) // batch_size

    color_ids = np.random.permutation(n_color)
    cifar_ids = np.random.permutation(n_cifar)

    def _maybe_flip(X):
        mask = np.random.rand(len(X)) > 0.5
        X = X.copy()
        X[mask] = X[mask, :, ::-1, :]
        return X

    for step in range(n_steps):
        ci = (step * batch_size) % n_color
        cj = min(ci + batch_size, n_color)
        xc = X_color[color_ids[ci:cj]]
        yc = y_color[color_ids[ci:cj]]
        if augment:
            xc = _maybe_flip(xc.copy())

        oi = (step * batch_size) % n_cifar
        oj = min(oi + batch_size, n_cifar)
        xo = X_cifar[cifar_ids[oi:oj]]
        yo = y_cifar[cifar_ids[oi:oj]]
        if augment:
            xo = _maybe_flip(xo.copy())

        yield xc, yc, xo, yo
