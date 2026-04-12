import json
import os
import sys
import numpy as np
from PIL import Image

# Setup standard imports from project
sys.path.insert(0, "src")
sys.path.insert(0, "GRUE/src")
from run_experiment import load_cifar_model
from concept_extraction import get_activations
from cifar_dataset import load_raw_cifar10, CIFAR10_LABELS
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis as scipy_kurtosis

def make_grid_image(images, nrow=4, cell_size=64, pad=5, bg=240):
    n = len(images)
    ncol = (n + nrow - 1) // nrow
    nrow_actual = min(nrow, n)

    W = ncol * (cell_size + pad) + pad
    H = nrow_actual * (cell_size + pad) + pad
    canvas = Image.new("RGB", (W, H), color=(bg, bg, bg))

    for i, img_arr in enumerate(images):
        row = i // ncol
        col = i % ncol
        img = Image.fromarray(img_arr).resize((cell_size, cell_size), Image.NEAREST)
        x = pad + col * (cell_size + pad)
        y = pad + row * (cell_size + pad)
        canvas.paste(img, (x, y))
    return canvas

print("Loading JSON...")
with open("GRUE/results/kurtosis_discovery.json") as f:
    data = json.load(f)

alignments = data["top20_ica_concept_alignment"]
icas = alignments["ica_indices"]
cosines = alignments["cosine_vs_concepts"]
kurt_vals = alignments["kurt_values"]
concept_names = alignments["concept_names"]

# We want the ICAs with the LOWEST max cosine similarity to any CIFAR class (first 10 elements)
max_cosines = []
for i, cos_row in enumerate(cosines):
    max_cifar_cos = max(cos_row[:10])  # ignore sky/ocean since they are our own vectors
    best_class = concept_names[np.argmax(cos_row[:10])]
    max_cosines.append((i, max_cifar_cos, best_class))

# Sort by max_cosine ascending
max_cosines.sort(key=lambda x: x[1])

print("Top 5 Unlabeled ICAs (Lowest correlation with any CIFAR class):")
top_unlabeled = []
for i in range(5):
    idx, max_cos, bclass = max_cosines[i]
    ica_comp_idx = icas[idx]
    kurt = kurt_vals[idx]
    print(f"ICA_{ica_comp_idx} (Rank {idx+1} by kurtosis): Kurt={kurt:.2f}, Max Cosine={max_cos:.4f} (best match: {bclass})")
    top_unlabeled.append((ica_comp_idx, kurt, max_cos, bclass))

# Now run FastICA to get the projections again
print("\nLoading model & data...")
model, _ = load_cifar_model("A", 1)
_, _, X_test, y_test = load_raw_cifar10()

print("Extracting features...")
acts = get_activations(model, X_test)
feats = acts["fc1"].astype(np.float32)
scaler = StandardScaler()
feats_scaled = scaler.fit_transform(feats)

print("Running FastICA...")
ica = FastICA(n_components=30, max_iter=2000, tol=0.001, random_state=42)
S = ica.fit_transform(feats_scaled)

artifact_dir = "/Users/ianheadley/.gemini/antigravity/brain/13c8e9a5-9e03-401e-8a3f-c8efaac8999d/artifacts"
os.makedirs(artifact_dir, exist_ok=True)

for ica_comp_idx, kurt, max_cos, bclass in top_unlabeled:
    proj = S[:, ica_comp_idx]
    
    top_n = 16
    top_indices = np.argsort(-proj)[:top_n]
    bot_indices = np.argsort(proj)[:top_n]
    
    top_images = [(X_test[i] * 255).clip(0, 255).astype(np.uint8) for i in top_indices]
    bot_images = [(X_test[i] * 255).clip(0, 255).astype(np.uint8) for i in bot_indices]
    
    left_grid = make_grid_image(top_images, nrow=4, cell_size=80)
    right_grid = make_grid_image(bot_images, nrow=4, cell_size=80)
    
    pad = 20
    W = left_grid.width + pad + right_grid.width
    H = max(left_grid.height, right_grid.height)
    
    grid = Image.new("RGB", (W, H), color=(255, 255, 255))
    grid.paste(left_grid, (0, 0))
    grid.paste(right_grid, (left_grid.width + pad, 0))
    
    out_path = os.path.join(artifact_dir, f"unlabeled_ica_{ica_comp_idx}_k{kurt:.1f}.png")
    grid.save(out_path)
    print(f"Saved: {out_path}")
