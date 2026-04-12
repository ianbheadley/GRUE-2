import os
import sys
import numpy as np
import mlx.core as mx
import mlx.optimizers as optim
from PIL import Image
from sklearn.decomposition import FastICA

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from GRUE.src.cifar100_dataset import load_raw_cifar100, CIFAR100_LABELS
from run_experiment import load_cifar_model
from concept_extraction import get_activations

def total_variation_loss(img_mx):
    """Calculate TV loss for a batch of images in NHWC format."""
    # img is (B, H, W, C)
    dy = img_mx[:, 1:, :, :] - img_mx[:, :-1, :, :]
    dx = img_mx[:, :, 1:, :] - img_mx[:, :, :-1, :]
    return mx.sum(mx.abs(dy)) + mx.sum(mx.abs(dx))

def main():
    out_dir = "figures/feature_geometry/deep_dream"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading data and model...")
    X_train, y_train, X_test, y_test = load_raw_cifar100()
    model, _ = load_cifar_model("A", 1)
    
    print("Extracting features...")
    # Load into memory mapping
    activations = get_activations(model, X_test, batch_size=256)
    feats = activations['fc1']
    feats_centered = feats - np.mean(feats, axis=0)
    
    print("Computing ICA...")
    ica = FastICA(n_components=10, random_state=42, max_iter=1000)
    ica.fit(feats_centered)
    
    target_idx = 0
    ica_dir_np = ica.components_[target_idx]
    ica_dir_np = ica_dir_np / (np.linalg.norm(ica_dir_np) + 1e-8)
    
    target_dir = mx.array(ica_dir_np)
    
    print(f"Maximizing Target ICA Component {target_idx}...")

    # Define loss function to maximize projection on the ICA axis
    def loss_fn(img):
        # Forward pass returning activations
        out, acts = model(img, capture_activations=True)
        fc1 = acts["fc1"]  # Shape should be (1, 256)
        
        # We want to MAXIMIZE the projection, so minimize the negative projection
        projection = mx.sum(fc1 * target_dir)
        
        # Add TV regularization and L2 penalty to keep image visually coherent
        l2_loss = mx.sum(img ** 2)
        tv = total_variation_loss(img)
        
        # Hyperparameters for regularization
        alpha_l2 = 0.01
        alpha_tv = 0.1
        
        total_loss = -projection + alpha_l2 * l2_loss + alpha_tv * tv
        return total_loss

    # Compute gradient of loss_fn w.r.t the first argument (img)
    val_and_grad_fn = mx.value_and_grad(loss_fn)
    
    # Initialize a random grey/noise image 
    # CIFAR models expect NHWC in [0, 1] usually
    init_np = np.random.normal(0.5, 0.1, (1, 32, 32, 3)).astype(np.float32)
    img = mx.array(init_np)
    
    # We will manually do gradient descent (ascent) since optimizers in MLX are for model params
    lr = 0.05
    
    for step in range(500):
        loss, grad = val_and_grad_fn(img)
        
        # update image
        img = img - lr * grad
        
        # Clip back to valid color range
        img = mx.clip(img, 0.0, 1.0)
        
        if step % 100 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")
            
    # Save the final synthesized image
    final_img_np = np.array(img[0]) * 255.0
    final_img_np = final_img_np.astype(np.uint8)
    
    pil_img = Image.fromarray(final_img_np)
    # Upscale heavily so it's viewable
    pil_img = pil_img.resize((256, 256), Image.NEAREST)
    out_path = os.path.join(out_dir, "dreamed_ica_component.png")
    pil_img.save(out_path)
    print(f"\nSaved modeled concept to {out_path}")

if __name__ == "__main__":
    main()
