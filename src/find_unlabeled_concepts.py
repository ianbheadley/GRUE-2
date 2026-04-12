import os
import sys
import numpy as np
from PIL import Image
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA
from mlx_vlm import load, generate

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cifar_dataset import load_raw_cifar10, CIFAR10_LABELS
from run_experiment import load_cifar_model
from concept_extraction import get_activations
from src.few_shot_concept_probe import rank_all_images_by_concept

def create_image_grid(images, grid_size=(3, 3), image_size=(32, 32)):
    w, h = image_size
    cols, rows = grid_size
    grid = Image.new('RGB', (cols * w, rows * h))
    for i, img_np in enumerate(images[:cols*rows]):
        img = Image.fromarray(img_np)
        grid.paste(img, ( (i % cols) * w, (i // cols) * h ))
    return grid.resize((cols * w * 4, rows * h * 4), Image.NEAREST)

def main():
    out_dir = "figures/feature_geometry/orthogonal_unlabeled"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading data and model...")
    X_train, y_train, X_test, y_test = load_raw_cifar10()
    model, _ = load_cifar_model("A", 1)
    
    print("Extracting features...")
    activations = get_activations(model, X_test, batch_size=256)
    feats = activations['fc1']
    feats_centered = feats - np.mean(feats, axis=0)
    
    # Compute supervised class directions
    print("Computing supervised class centroids to use as a filter...")
    class_dirs = []
    for k in range(10):
        pos = feats[y_test == k]
        neg = feats[y_test != k]
        d = np.mean(pos, axis=0) - np.mean(neg, axis=0)
        class_dirs.append(d / (np.linalg.norm(d) + 1e-8))
    
    print("Running ICA (30 components) to find many blind concepts...")
    ica = FastICA(n_components=30, random_state=42, max_iter=1000)
    ica.fit(feats_centered)
    
    candidates = []
    for i in range(30):
        d = ica.components_[i]
        d_norm = d / (np.linalg.norm(d) + 1e-8)
        
        # Check alignment with known classes
        sims = [abs(np.dot(d_norm, cd)) for cd in class_dirs]
        max_sim = max(sims)
        matched_class = CIFAR10_LABELS[np.argmax(sims)]
        
        # Check kurtosis
        projections = np.dot(feats_centered, d_norm)
        kurt = kurtosis(projections)
        
        candidates.append({
            "idx": i,
            "direction": d_norm,
            "kurtosis": kurt,
            "max_sim": max_sim,
            "closest_class": matched_class,
            "positive": True # Track polarity
        })
        # Check negative direction too since ICA vectors are arbitrary
        projections_neg = np.dot(feats_centered, -d_norm)
        candidates.append({
            "idx": i,
            "direction": -d_norm,
            "kurtosis": kurtosis(projections_neg),
            "max_sim": max_sim,
            "closest_class": matched_class,
            "positive": False
        })
        
    # Sort candidates by kurtosis (we want highly structured features)
    candidates = sorted(candidates, key=lambda x: x["kurtosis"], reverse=True)
    
    # Filter out anything too similar to a known class (> 0.40 cosine sim is generally correlated)
    SIMILARITY_THRESHOLD = 0.40
    unlabeled_concepts = [c for c in candidates if c["max_sim"] < SIMILARITY_THRESHOLD]
    
    # Deduplicate (since we added positive and negative, some might overlap conceptually, 
    # but we'll just take the top 6 unique ICA indices)
    final_concepts = []
    seen_idx = set()
    for c in unlabeled_concepts:
        if c["idx"] not in seen_idx:
            seen_idx.add(c["idx"])
            final_concepts.append(c)
        if len(final_concepts) >= 6:
            break
            
    print(f"\nFiltered out labeled classes. Found {len(final_concepts)} orthogonal emergent features.")
    
    print("Loading MLX-VLM...")
    model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    vlm_model, vlm_processor = load(model_path)
    
    results_md = "# AI Auto-Labeled Unsupervised Concepts (Orthogonal to Classes)\n\n"
    results_md += "We extracted 30 ICA components and strictly rejected any vector with > 0.40 cosine similarity to a known CIFAR-10 class center. These remaining vectors represent highly-structured visual concepts that strongly cut *across* the training labels (like background color, lighting, or generic shapes).\n\n"
    
    for i, concept in enumerate(final_concepts):
        concept_id = f"Unlabeled_Feature_{i+1}"
        print(f"\n--- {concept_id} ---")
        print(f"Kurtosis: {concept['kurtosis']:.2f}")
        print(f"Max similarity to any class: {concept['max_sim']:.2f} (closest: {concept['closest_class']})")
        
        ranked_idx, _ = rank_all_images_by_concept(model, X_test, concept['direction'])
        
        # Get top 9 images and their original labels to assist the VLM
        top_images = []
        top_labels = []
        for idx in ranked_idx[:9]:
            img = X_test[idx]
            if img.dtype != np.uint8: img = (img * 255).astype(np.uint8)
            top_images.append(img)
            top_labels.append(CIFAR10_LABELS[y_test[idx]])
            
        unique_labels = list(set(top_labels))
        
        grid_img = create_image_grid(top_images)
        grid_path = os.path.join(out_dir, f"{concept_id}_grid.png")
        grid_img.save(grid_path)
        
        # Prompt VLM with context so it doesn't hallucinate "light color" for pixelated dogs
        prompt = f"These are 9 pixelated images drawn from standard object classes like {', '.join(unique_labels)}. Look closely at their shapes, backgrounds, or contexts. What specific underlying visual attribute, environment, or feature do they all share? Give a 1-3 word label."
        
        try:
            config = vlm_processor.chat_template
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            prompt_text = vlm_processor.apply_chat_template(messages, add_generation_prompt=True)
        except Exception:
            prompt_text = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
        response = generate(vlm_model, vlm_processor, prompt_text, [grid_path], verbose=False)
        label = response.text.strip() if hasattr(response, 'text') else str(response).strip()
        print(f"--> VLM Auto-Label: {label}")
        
        results_md += f"### {concept_id}\n"
        results_md += f"**AI Auto-Label:** `{label}`\n"
        results_md += f"*Closest known class: {concept['closest_class']} (Sim: {concept['max_sim']:.2f})*\n\n"
        results_md += f"![{concept_id}](/Users/ianheadley/Documents/Grue/{grid_path})\n\n"
        
    with open(os.path.join(out_dir, "unlabeled_concepts_report.md"), "w") as f:
        f.write(results_md)
        
    print(f"\nSaved report to {os.path.join(out_dir, 'unlabeled_concepts_report.md')}")

if __name__ == "__main__":
    main()
