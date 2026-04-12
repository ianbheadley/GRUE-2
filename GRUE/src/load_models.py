import os
import json
import mlx.core as mx
from mlx.utils import tree_unflatten
from model import ColorCNN, ConceptCNN

def load_model(scheme, seed, base_model_dir="models"):
    """
    Utility to load a trained model by scheme and seed.
    Automatically selects ColorCNN or ConceptCNN based on metadata.
    """
    model_dir = os.path.join(base_model_dir, f"scheme_{scheme}", f"seed_{seed}")
    metadata_path = os.path.join(model_dir, "metadata.json")
    weights_path = os.path.join(model_dir, "weights.safetensors")

    if not os.path.exists(metadata_path) or not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing model files for scheme {scheme} seed {seed}.")

    with open(metadata_path, "r") as f:
        meta = json.load(f)

    num_classes = len(meta["label_map"])
    model_type = meta.get("model_type", "ColorCNN")

    if model_type == "ConceptCNN":
        model = ConceptCNN(num_classes, input_size=32)
    else:
        model = ColorCNN(num_classes)

    weights = mx.load(weights_path)
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    model.eval()

    return model, meta
