import os
import sys
import json
import numpy as np
from PIL import Image

# Add parent directory to path to find existing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SAM_MLX_PATH = "/Users/ianheadley/Documents/mlx-examples-main/segment_anything/"

def download_sam_weights(model_size="vit_b", save_dir="weights/"):
    """Download sam-vit-base from HuggingFace if not present."""
    import huggingface_hub
    from pathlib import Path
    
    repo_id = {
        "vit_b": "facebook/sam-vit-base",
        "vit_l": "facebook/sam-vit-large",
        "vit_h": "facebook/sam-vit-huge",
    }.get(model_size, "facebook/sam-vit-base")
    
    target_dir = Path(save_dir) / f"sam-{model_size}"
    if target_dir.exists():
        print(f"SAM weights already exist at {target_dir}")
        return str(target_dir)
        
    print(f"Downloading SAM {model_size} weights from HuggingFace...")
    path = huggingface_hub.snapshot_download(
        repo_id=repo_id,
        allow_patterns=["*.safetensors", "*.json", "config.json"],
    )
    
    print(f"Weights downloaded to {path}")
    return path

def load_sam_predictor(weights_path, model_type="vit_b"):
    """
    Load SAM predictor using the MLX SAM implementation.
    sys.path insert: /Users/ianheadley/Documents/mlx-examples-main/segment_anything/
    """
    sys.path.insert(0, SAM_MLX_PATH)
    from segment_anything.predictor import SamPredictor
    from segment_anything.sam import load as sam_load
    
    print(f"Loading SAM predictor from {weights_path}...")
    model = sam_load(weights_path)
    return SamPredictor(model)

def upscale_image(img_np, target_size=256):
    """Upscale 32x32 float32 image to target_size using PIL BICUBIC."""
    if img_np.dtype != np.uint8:
        # Assume float32 [0, 1]
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    img_resized = img.resize((target_size, target_size), Image.BICUBIC)
    return np.array(img_resized)

def generate_masks_for_image(predictor, img_upscaled):
    """Run SAM automatic mask generator. Returns list of mask dicts."""
    sys.path.insert(0, SAM_MLX_PATH)
    from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
    mask_generator = SamAutomaticMaskGenerator(predictor.model)
    masks = mask_generator.generate(img_upscaled)
    return masks

def _rgb_to_hsv(rgb):
    """Naive RGB to HSV for numpy array (N, 3)."""
    r, g, b = rgb[:, 0]/255.0, rgb[:, 1]/255.0, rgb[:, 2]/255.0
    mx = np.max(rgb/255.0, axis=1)
    mn = np.min(rgb/255.0, axis=1)
    df = mx - mn + 1e-8
    
    h = np.zeros_like(mx)
    h[mx==r] = (60 * ((g[mx==r]-b[mx==r])/df[mx==r]) + 360) % 360
    h[mx==g] = (60 * ((b[mx==g]-r[mx==g])/df[mx==g]) + 120) % 360
    h[mx==b] = (60 * ((r[mx==b]-g[mx==b])/df[mx==b]) + 240) % 360
    
    s = np.zeros_like(mx)
    s[mx!=0] = df[mx!=0] / mx[mx!=0]
    
    v = mx
    return h, s, v

def sky_filter(masks, img_upscaled):
    """
    Sky heuristic applied to SAM masks:
    - Mask centroid in upper 45% of image
    - Mask average hue is blue (160-250°) or grey (sat < 0.2)
    - Mask area > 8% of image
    - Width > height (sky is horizontal)
    Returns True if any mask passes all criteria.
    """
    H, W = img_upscaled.shape[:2]
    best_conf = 0
    found = False
    
    for mask_dict in masks:
        mask = mask_dict['segmentation']
        area_frac = mask_dict['area'] / (H * W)
        
        # 1. Size check
        if area_frac < 0.08:
            continue
            
        # 2. Position check (centroid)
        ys, xs = np.where(mask)
        if len(ys) == 0: continue
        centroid_y = np.mean(ys) / H
        if centroid_y > 0.45:
            continue
            
        # 3. Shape check (width > height)
        y_min, y_max = np.min(ys), np.max(ys)
        x_min, x_max = np.min(xs), np.max(xs)
        height = y_max - y_min + 1
        width = x_max - x_min + 1
        if width <= height:
            continue
            
        # 4. Color check
        pixels = img_upscaled[mask]
        h, s, v = _rgb_to_hsv(pixels)
        avg_h = np.mean(h)
        avg_s = np.mean(s)
        
        is_blue = (160 <= avg_h <= 250)
        is_grey = (avg_s < 0.20)
        
        if is_blue or is_grey:
            found = True
            # Confidence can be a mix of how well it fits
            conf = mask_dict.get('stability_score', area_frac)
            best_conf = max(best_conf, conf)
            
    return found, float(best_conf)

def label_images_for_concept(
    X,                      # (N, 32, 32, 3) CIFAR images
    concept_filter_fn,      # fn(masks, img) -> bool/tuple
    weights_path,
    batch_size=50,
    save_path=None          # optional JSON output
):
    """
    Label each image as has_concept=True/False.
    Returns: list of dicts {idx, has_concept, confidence, n_masks}
    """
    predictor = load_sam_predictor(weights_path)
    
    results = []
    print(f"Labeling {len(X)} images...")
    
    for i in range(len(X)):
        img_32 = X[i]
        img_256 = upscale_image(img_32)
        
        # Run SAM
        try:
            masks = generate_masks_for_image(predictor, img_256)
            has_concept, confidence = concept_filter_fn(masks, img_256)
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            has_concept, confidence = False, 0.0
            masks = []

        results.append({
            "idx": i,
            "has_concept": bool(has_concept),
            "confidence": float(confidence),
            "n_masks": len(masks)
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(X)} - Found {sum(r['has_concept'] for r in results)} so far")
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w') as f:
                    json.dump(results, f)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f)
            
    return results
