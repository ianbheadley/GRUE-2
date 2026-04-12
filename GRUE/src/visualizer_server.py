import os
import math

import mlx.core as mx
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from mlx.utils import tree_flatten, tree_unflatten

from load_models import load_model
from model import ColorCNN, ConceptCNN

app = Flask(__name__, static_folder=".")
CORS(app)

MODEL_CACHE = {}


def sanitize_json(value):
    if isinstance(value, dict):
        return {k: sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(".", path)


def make_model_id(base_dir, scheme, seed):
    return f"{base_dir}|{scheme}|{seed}"


def parse_model_id(model_id):
    base_dir, scheme, seed = model_id.split("|")
    return base_dir, scheme, int(seed)


def dataset_from_base_dir(base_dir):
    return "cifar10" if "cifar10" in base_dir else "color_blocks"


def get_cached_model(base_dir, scheme, seed):
    cache_key = make_model_id(base_dir, scheme, seed)
    if cache_key not in MODEL_CACHE:
        model, meta = load_model(scheme, seed, base_model_dir=base_dir)
        MODEL_CACHE[cache_key] = (model, meta)
    return MODEL_CACHE[cache_key]


def instantiate_like(meta):
    num_classes = meta.get("num_classes", len(meta["label_map"]))
    model_type = meta.get("model_type", "ColorCNN")
    if model_type == "ConceptCNN":
        return ConceptCNN(num_classes, input_size=32)
    return ColorCNN(num_classes)


def prepare_input(image_data):
    x = mx.array([image_data])
    if len(x.shape) == 3:
        n_pixels = x.shape[1]
        side = int(np.sqrt(n_pixels))
        x = x.reshape(1, side, side, 3)
    return x


def forward_with_activations(model, x):
    logits, activations = model(x, capture_activations=True)
    mx.eval(logits, activations)
    return np.array(logits[0]), {name: np.array(value[0]) for name, value in activations.items()}


def label_maps(meta):
    idx_to_label = {int(v): k for k, v in meta["label_map"].items()}
    return idx_to_label


def top_predictions(logits, idx_to_label, top_k=5):
    probs = np.array(mx.softmax(mx.array(logits), axis=-1))
    order = np.argsort(probs)[::-1][:top_k]
    return [
        {
            "index": int(i),
            "label": idx_to_label.get(int(i), str(i)),
            "probability": float(probs[i]),
            "logit": float(logits[i]),
        }
        for i in order
    ]


def flatten_activation(act):
    if act.ndim == 3:
        return act.mean(axis=(1, 2))
    return act


def activation_preview(act):
    if act.ndim == 3:
        mean_map = act.mean(axis=0)
        max_map = act.max(axis=0)
        return {
            "kind": "conv",
            "shape": list(act.shape),
            "mean_map": mean_map.tolist(),
            "max_map": max_map.tolist(),
            "channel_means": act.mean(axis=(1, 2)).tolist(),
        }
    return {
        "kind": "fc",
        "shape": list(act.shape),
        "values": act.tolist(),
    }


def summarize_pair(layer_name, act_a, act_b):
    flat_a = flatten_activation(act_a)
    flat_b = flatten_activation(act_b)
    shape_mismatch = False
    if flat_a.shape != flat_b.shape:
        shape_mismatch = True
        shared = min(len(flat_a), len(flat_b))
        flat_a = flat_a[:shared]
        flat_b = flat_b[:shared]
    diff = flat_a - flat_b
    mean_abs = (np.abs(flat_a) + np.abs(flat_b)) / 2.0
    null_score = np.abs(diff) / (mean_abs + 1e-6)
    order = np.argsort(np.abs(diff))[::-1]
    null_order = np.argsort(null_score)[::-1]

    top_changed = [
        {
            "index": int(idx),
            "a": float(flat_a[idx]),
            "b": float(flat_b[idx]),
            "delta": float(diff[idx]),
            "null_score": float(null_score[idx]),
        }
        for idx in order[:12]
    ]
    top_null = [
        {
            "index": int(idx),
            "a": float(flat_a[idx]),
            "b": float(flat_b[idx]),
            "delta": float(diff[idx]),
            "null_score": float(null_score[idx]),
        }
        for idx in null_order[:12]
    ]

    return {
        "layer": layer_name,
        "preview_a": activation_preview(act_a),
        "preview_b": activation_preview(act_b),
        "stats": {
            "a_l2": float(np.linalg.norm(flat_a)),
            "b_l2": float(np.linalg.norm(flat_b)),
            "delta_l2": float(np.linalg.norm(diff)),
            "delta_mean_abs": float(np.mean(np.abs(diff))),
            "cosine_similarity": float(
                np.dot(flat_a, flat_b) / ((np.linalg.norm(flat_a) * np.linalg.norm(flat_b)) + 1e-8)
            ),
            "null_zone_mass": float(np.mean(null_score)),
            "signed_shift_mean": float(np.mean(diff)),
            "shape_mismatch": shape_mismatch,
            "compared_width": int(len(flat_a)),
        },
        "top_changed_units": top_changed,
        "top_null_zone_units": top_null,
        "diff_vector": diff.tolist(),
        "null_score_vector": null_score.tolist(),
    }


def shared_weight_dict(model):
    return {k: v for k, v in tree_flatten(model.parameters()) if "fc2" not in k}


def layer_weight_summary(weights_a, weights_b):
    per_layer = {}
    for key in weights_a:
        layer = key.split(".")[0]
        delta = np.array(weights_a[key] - weights_b[key])
        a_norm = float(np.linalg.norm(np.array(weights_a[key]).ravel()))
        b_norm = float(np.linalg.norm(np.array(weights_b[key]).ravel()))
        delta_norm = float(np.linalg.norm(delta.ravel()))
        if layer not in per_layer:
            per_layer[layer] = {"a_norm_sq": 0.0, "b_norm_sq": 0.0, "delta_norm_sq": 0.0}
        per_layer[layer]["a_norm_sq"] += a_norm ** 2
        per_layer[layer]["b_norm_sq"] += b_norm ** 2
        per_layer[layer]["delta_norm_sq"] += delta_norm ** 2

    result = []
    for layer, stats in sorted(per_layer.items()):
        result.append(
            {
                "layer": layer,
                "a_norm": float(np.sqrt(stats["a_norm_sq"])),
                "b_norm": float(np.sqrt(stats["b_norm_sq"])),
                "delta_norm": float(np.sqrt(stats["delta_norm_sq"])),
            }
        )
    return result


def inject_weight_difference(base_model, subtract_model, alpha):
    _, base_meta = base_model
    injected_model = instantiate_like(base_meta)
    flat_base = dict(tree_flatten(base_model[0].parameters()))
    flat_sub = dict(tree_flatten(subtract_model[0].parameters()))
    updated = dict(flat_base)
    for key, value in flat_base.items():
        if key in flat_sub and "fc2" not in key:
            updated[key] = value + alpha * (value - flat_sub[key])
    injected_model.update(tree_unflatten(list(updated.items())))
    mx.eval(injected_model.parameters())
    injected_model.eval()
    return injected_model


@app.route("/models", methods=["GET"])
def list_models():
    available = []
    for base_dir in ["models", "models_cifar10", "models_independent_cifar10"]:
        if not os.path.exists(base_dir):
            continue
        dataset = dataset_from_base_dir(base_dir)
        for scheme_dir in sorted(os.listdir(base_dir)):
            if not scheme_dir.startswith("scheme_"):
                continue
            scheme = scheme_dir.split("_")[1]
            scheme_path = os.path.join(base_dir, scheme_dir)
            for seed_dir in sorted(os.listdir(scheme_path)):
                if not seed_dir.startswith("seed_"):
                    continue
                seed = int(seed_dir.split("_")[1])
                meta_path = os.path.join(scheme_path, seed_dir, "metadata.json")
                if not os.path.exists(meta_path):
                    continue
                available.append(
                    {
                        "id": make_model_id(base_dir, scheme, seed),
                        "base_dir": base_dir,
                        "dataset": dataset,
                        "scheme": scheme,
                        "seed": seed,
                        "label": f"{base_dir} | {dataset} | scheme {scheme} | seed {seed}",
                        "input_size": 32 if dataset == "cifar10" else 64,
                    }
                )
    return jsonify(available)


@app.route("/compare", methods=["POST"])
def compare_models():
    try:
        payload = request.json
        model_a_id = payload.get("model_a")
        model_b_id = payload.get("model_b")
        image_data = payload.get("image")
        subtraction_mode = payload.get("subtraction_mode", "none")
        alpha = float(payload.get("alpha", 0.25))

        if not model_a_id or not model_b_id or not image_data:
            return jsonify({"error": "Missing parameters"}), 400

        base_a, scheme_a, seed_a = parse_model_id(model_a_id)
        base_b, scheme_b, seed_b = parse_model_id(model_b_id)
        model_a = get_cached_model(base_a, scheme_a, seed_a)
        model_b = get_cached_model(base_b, scheme_b, seed_b)

        x = prepare_input(image_data)
        logits_a, acts_a = forward_with_activations(model_a[0], x)
        logits_b, acts_b = forward_with_activations(model_b[0], x)

        idx_to_label_a = label_maps(model_a[1])
        idx_to_label_b = label_maps(model_b[1])

        subtraction = None
        if subtraction_mode in {"a_minus_b_on_b", "a_minus_b_on_a"}:
            if subtraction_mode == "a_minus_b_on_b":
                injected_model = inject_weight_difference(model_b, model_a, alpha)
                injected_meta = model_b[1]
                injected_name = "B + alpha*(B - A)"
            else:
                injected_model = inject_weight_difference(model_a, model_b, alpha)
                injected_meta = model_a[1]
                injected_name = "A + alpha*(A - B)"

            injected_logits, injected_acts = forward_with_activations(injected_model, x)
            subtraction = {
                "mode": subtraction_mode,
                "name": injected_name,
                "alpha": alpha,
                "prediction": top_predictions(injected_logits, label_maps(injected_meta)),
                "layers": {
                    layer: {
                        "preview": activation_preview(act),
                        "flat_l2": float(np.linalg.norm(flatten_activation(act))),
                    }
                    for layer, act in injected_acts.items()
                },
            }

        shared_a = shared_weight_dict(model_a[0])
        shared_b = shared_weight_dict(model_b[0])

        per_layer = {
            layer: summarize_pair(layer, acts_a[layer], acts_b[layer])
            for layer in acts_a.keys()
        }

        global_null_candidates = []
        for layer_name, layer_summary in per_layer.items():
            for item in layer_summary["top_null_zone_units"][:5]:
                global_null_candidates.append(
                    {
                        "layer": layer_name,
                        **item,
                    }
                )
        global_null_candidates.sort(key=lambda item: item["null_score"], reverse=True)

        payload = {
            "models": {
                "a": {
                    "id": model_a_id,
                    "meta": model_a[1],
                    "prediction": top_predictions(logits_a, idx_to_label_a),
                },
                "b": {
                    "id": model_b_id,
                    "meta": model_b[1],
                    "prediction": top_predictions(logits_b, idx_to_label_b),
                },
            },
            "weight_summary": layer_weight_summary(shared_a, shared_b),
            "layer_order": list(per_layer.keys()),
            "layers": per_layer,
            "null_zone_ranking": global_null_candidates[:20],
            "subtraction": subtraction,
        }
        return jsonify(sanitize_json(payload))
    except Exception as exc:
        import traceback

        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
