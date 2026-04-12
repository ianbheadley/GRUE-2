import os
import sys
import numpy as np
from PIL import Image
from sklearn.decomposition import FastICA
from mlx_vlm import load, generate

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cifar_dataset import load_raw_cifar10
from run_experiment import load_cifar_model
from concept_extraction import get_activations
from src.few_shot_concept_probe import rank_all_images_by_concept

def create_image_grid(images, grid_size=(3, 3), image_size=(32, 32)):
    """Create a single composite image from a list of numpy images."""
    w, h = image_size
    cols, rows = grid_size
    grid = Image.new('RGB', (cols * w, rows * h))
    
    for i, img_np in enumerate(images[:cols*rows]):
        img = Image.fromarray(img_np)
        grid.paste(img, ( (i % cols) * w, (i // cols) * h ))
        
    # Upscale slightly so the VLM can see it easier
    upscaled = grid.resize((cols * w * 4, rows * h * 4), Image.NEAREST)
    return upscaled

def main():
    out_dir = "figures/feature_geometry/auto_labeled"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading data and model...")
    X_train, y_train, X_test, y_test = load_raw_cifar10()
    model, _ = load_cifar_model("A", 1)
    
    print("Extracting features (unsupervised)...")
    activations = get_activations(model, X_test, batch_size=256)
    feats = activations['fc1']
    
    print("Running ICA to blindly find 6 concepts...")
    ica = FastICA(n_components=6, random_state=42, max_iter=1000)
    feats_centered = feats - np.mean(feats, axis=0)
    ica.fit(feats_centered)
    
    directions = []
    for i in range(6):
        d = ica.components_[i]
        d_norm = d / (np.linalg.norm(d) + 1e-8)
        directions.append(d_norm)
        
    print("Loading MLX-VLM Model (Qwen2-VL-2B)...")
    # Using a 4-bit quantized 2B model for speed and memory efficiency
    model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    vlm_model, vlm_processor = load(model_path)
    
    prompt = "Analyze this grid of 9 images. What specific visual object, color, texture, or concept do they all strongly share in common? Provide a very short, 1-3 word label."

    results_md = "# AI Auto-Labeled Blind Concepts\n\n"
    results_md += "We used ICA to discover 6 geometric features without labels, then asked a native Apple Silicon VLM (Qwen2-VL) to look at the top activating images and assign a label to the mathematical concept.\n\n"

    for i, direction in enumerate(directions):
        concept_id = f"Concept_{i+1}"
        print(f"\nProcessing {concept_id}...")
        
        # Rank all images by how much they align with this ICA component
        ranked_idx, _ = rank_all_images_by_concept(model, X_test, direction)
        
        # Get top 9 images
        top_images = []
        for idx in ranked_idx[:9]:
            img = X_test[idx]
            # Convert float [0,1] to uint8 if necessary
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            top_images.append(img)
            
        # Create grid and save
        grid_img = create_image_grid(top_images)
        grid_path = os.path.join(out_dir, f"{concept_id}_grid.png")
        grid_img.save(grid_path)
        
        print(f"Asking VLM to label {concept_id}...")
        try:
            config = vlm_processor.chat_template
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            prompt_text = vlm_processor.apply_chat_template(messages, add_generation_prompt=True)
        except Exception as e:
            print("Chat template not found, using raw Qwen-VL format...")
            prompt_text = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Depending on mlx_vlm version, generate syntax varies.
        response = generate(vlm_model, vlm_processor, prompt_text, [grid_path], verbose=False)
        
        label = response.text.strip() if hasattr(response, 'text') else str(response).strip()
        print(f"--> VLM Label: {label}")
        
        results_md += f"### {concept_id}\n"
        results_md += f"**AI Proposed Label:** `{label}`\n\n"
        results_md += f"![{concept_id}](/Users/ianheadley/Documents/Grue/{grid_path})\n\n"
        
    with open(os.path.join(out_dir, "auto_labels_report.md"), "w") as f:
        f.write(results_md)
        
    print(f"\nSaved report to {os.path.join(out_dir, 'auto_labels_report.md')}")

if __name__ == "__main__":
    main()
