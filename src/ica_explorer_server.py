import os
import sys
import base64
import numpy as np
import mlx.core as mx
from PIL import Image
from io import BytesIO
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.decomposition import FastICA, PCA
from scipy.stats import kurtosis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from run_experiment import load_cifar_model
from concept_extraction import get_activations
from GRUE.src.cifar100_dataset import load_raw_cifar100

app = Flask(__name__, static_folder=".")
CORS(app)

# Global State
MODEL = None
ICA_DIRS = []
PCA_COMPONENTS = []
LAYOUT_DATA = {}
CURRENT_IMG = None

def init_system():
    global MODEL, ICA_DIRS, CURRENT_IMG, PCA_COMPONENTS, LAYOUT_DATA
    print("Loading model and dataset for ICA Extraction...")
    X_train, y_train, X_test, y_test = load_raw_cifar100()
    MODEL, _ = load_cifar_model("A", 1)
    
    print("Extracting features and computing 64 ICA manifolds...")
    activations = get_activations(MODEL, X_test[:5000], batch_size=256)
    feats = activations['fc1']
    feats_centered = feats - np.mean(feats, axis=0)
    
    ica = FastICA(n_components=64, random_state=42, max_iter=1000)
    ica.fit(feats_centered)
    
    for i in range(64):
        d = ica.components_[i]
        d_norm = d / (np.linalg.norm(d) + 1e-8)
        ICA_DIRS.append(mx.array(d_norm))
        
    print("Computing 2D PCA layer and Kurtosis contour map...")
    pca = PCA(n_components=2)
    pca.fit(feats_centered)
    P1 = pca.components_[0]
    P2 = pca.components_[1]
    PCA_COMPONENTS = [mx.array(P1), mx.array(P2)]
    
    grid_res = 40
    xs = np.linspace(-1, 1, grid_res)
    ys = np.linspace(-1, 1, grid_res)
    KX, KY = np.meshgrid(xs, ys)
    KZ = np.zeros((grid_res, grid_res))
    
    for i in range(grid_res):
        for j in range(grid_res):
            d = KX[i, j] * P1 + KY[i, j] * P2
            norm = np.linalg.norm(d) + 1e-8
            proj = np.dot(feats_centered, d / norm)
            KZ[i, j] = kurtosis(proj)
            
    # Generate Heatmap image
    plt.figure(figsize=(6, 6), dpi=100)
    plt.contourf(KX, KY, KZ, levels=25, cmap='magma')
    plt.axis('off')
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    bg_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Generate node projections. Max spread of dot product is [-1, 1]
    nodes = []
    
    # We find the max projection to scale them visually nicely on the UI
    max_dist = 0.01
    for i in range(64):
        d = ica.components_[i]
        d_norm = d / (np.linalg.norm(d) + 1e-8)
        nx = float(np.dot(d_norm, P1))
        ny = float(np.dot(d_norm, P2))
        dist = np.sqrt(nx**2 + ny**2)
        if dist > max_dist: max_dist = dist
        nodes.append({"idx": i, "x": nx, "y": ny})
        
    # Sort them by their original distance from center so radial ordering is perfectly preserved
    nodes.sort(key=lambda n: np.sqrt(n["x"]**2 + n["y"]**2))
    
    for i, n in enumerate(nodes):
        n["x"] = n["x"] / max_dist
        n["y"] = n["y"] / max_dist
        
        # Calculate angle and original radius
        r = np.sqrt(n["x"]**2 + n["y"]**2) + 1e-8
        
        # We will keep its angle but force its radius based purely on rank spacing.
        # This absolutely guarantees perfect spacing from center to edge!
        r_new = 0.1 + (0.9 * (i / 63.0))
        
        # Re-apply point mapping
        n["x"] = float(n["x"] * (r_new / r))
        n["y"] = float(n["y"] * (r_new / r))
        
    LAYOUT_DATA = {
        "nodes": nodes,
        "background": bg_b64
    }

    print("System Initialized! Ready for PCA/Kurtosis exploration.")
    CURRENT_IMG = mx.array(np.random.normal(0.5, 0.1, (1, 32, 32, 3)).astype(np.float32))

def tv_loss(img):
    dy = img[:, 1:, :, :] - img[:, :-1, :, :]
    dx = img[:, :, 1:, :] - img[:, :, :-1, :]
    return mx.sum(mx.abs(dy)) + mx.sum(mx.abs(dx))

def loss_fn(img, target_dir):
    out, acts = MODEL(img, capture_activations=True)
    fc1 = acts["fc1"]
    projection = mx.sum(fc1 * target_dir)
    total_loss = -projection + 0.01 * mx.sum(img ** 2) + 0.1 * tv_loss(img)
    return total_loss

grad_fn = mx.value_and_grad(loss_fn)

@app.route("/")
def index():
    return send_from_directory(".", "ica_explorer.html")

@app.route("/layout", methods=["GET"])
def layout():
    return jsonify(LAYOUT_DATA)

@app.route("/reset", methods=["POST"])
def reset():
    global CURRENT_IMG
    CURRENT_IMG = mx.array(np.random.normal(0.5, 0.1, (1, 32, 32, 3)).astype(np.float32))
    return jsonify({"status": "ok"})

@app.route("/step", methods=["POST"])
def step():
    global CURRENT_IMG
    data = request.json
    mode = data.get("mode", "pca")
    
    if mode == "pca":
        x = data.get("x", 0.0)
        y = data.get("y", 0.0)
        target_anchor = x * PCA_COMPONENTS[0] + y * PCA_COMPONENTS[1]
        norm = mx.sqrt(mx.sum(target_anchor ** 2)) + 1e-8
        target_anchor = target_anchor / norm
    else:
        idx = data.get("idx", 0)
        target_anchor = ICA_DIRS[idx]
        
    # Take 5 robust gradient ascent steps
    lr = 0.05
    for _ in range(5):
        loss, grad = grad_fn(CURRENT_IMG, target_anchor)
        CURRENT_IMG = CURRENT_IMG - lr * grad
        CURRENT_IMG = mx.clip(CURRENT_IMG, 0.0, 1.0)
    
    mx.eval(CURRENT_IMG)
        
    img_np = np.array(CURRENT_IMG[0]) * 255.0
    pil_img = Image.fromarray(img_np.astype(np.uint8))
    
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return jsonify({"image": "data:image/png;base64," + img_str})

if __name__ == "__main__":
    init_system()
    app.run(port=5001, debug=False)
