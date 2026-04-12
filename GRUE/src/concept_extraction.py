"""
Concept extraction methods for mechanistic interpretability.

Four methods to identify and extract learned concepts from neural networks:
1. Weight Subtraction: Compare weight-space differences between models
2. Activation Difference: Compare activation patterns for concept subclasses
3. Linear Probing: Test if concepts are linearly separable at each layer
4. CKA: Centered Kernel Alignment to measure representational similarity
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def get_shared_weights(model, exclude_bn_stats=False):
    """Extract parameter dict excluding the output head (fc2).

    When exclude_bn_stats=True, also excludes BatchNorm running_mean/running_var
    which are not learned parameters and should not be subtracted for concept vectors.
    """
    excluded = {"fc2"}
    if exclude_bn_stats:
        excluded_suffixes = ("running_mean", "running_var")
    else:
        excluded_suffixes = ()

    result = {}
    for path, param in tree_flatten(model.parameters()):
        if any(ex in path for ex in excluded):
            continue
        if exclude_bn_stats and any(path.endswith(s) for s in excluded_suffixes):
            continue
        result[path] = param
    return result


def get_activations(model, X, batch_size=256):
    """Run forward pass and capture activations at every layer."""
    model.eval()
    all_acts = None
    for i in range(0, len(X), batch_size):
        batch_x = mx.array(X[i:i + batch_size])
        _, acts = model(batch_x, capture_activations=True)
        acts_np = {k: np.array(v) for k, v in acts.items()}
        if all_acts is None:
            all_acts = {k: [v] for k, v in acts_np.items()}
        else:
            for k, v in acts_np.items():
                all_acts[k].append(v)
    return {k: np.concatenate(v, axis=0) for k, v in all_acts.items()}


def global_avg_pool(activations):
    """Global average pool spatial dimensions for conv layer activations."""
    if activations.ndim == 4:
        return activations.mean(axis=(1, 2))
    return activations


# ---------------------------------------------------------------------------
# Method 1: Weight Subtraction
# ---------------------------------------------------------------------------

def weight_subtraction(models_a, models_b, log=None):
    """
    Extract concept vector by subtracting matched model weights.

    Returns concept_vector dict, per-layer signal-to-noise ratios,
    and cross-seed consistency (cosine similarity).
    """
    seeds = range(len(models_a))
    weights_a = [get_shared_weights(m, exclude_bn_stats=True) for m in models_a]
    weights_b = [get_shared_weights(m, exclude_bn_stats=True) for m in models_b]

    # Signal: matched A-B differences
    signal_diffs = [{k: weights_a[i][k] - weights_b[i][k] for k in weights_a[i]} for i in seeds]

    # Noise floor: A-A pairwise differences
    noise_diffs = []
    for i in range(len(models_a)):
        for j in range(i + 1, len(models_a)):
            noise_diffs.append({k: weights_a[i][k] - weights_a[j][k] for k in weights_a[i]})

    # Average concept vector
    concept_vector = {}
    for k in signal_diffs[0]:
        concept_vector[k] = mx.mean(mx.stack([d[k] for d in signal_diffs]), axis=0)

    # Per-layer norms and SNR
    layer_results = {}
    layer_keys = set(k.split(".")[0] for k in concept_vector)
    for layer in sorted(layer_keys):
        layer_params = [k for k in concept_vector if k.startswith(layer + ".")]

        signal_norms = []
        for sd in signal_diffs:
            norm = sum(mx.sum(mx.square(sd[k])).item() for k in layer_params)
            signal_norms.append(np.sqrt(norm))

        noise_norms = []
        for nd in noise_diffs:
            norm = sum(mx.sum(mx.square(nd[k])).item() for k in layer_params)
            noise_norms.append(np.sqrt(norm))

        avg_signal = np.mean(signal_norms)
        avg_noise = np.mean(noise_norms) if noise_norms else 1e-8
        snr = avg_signal / (avg_noise + 1e-8)

        layer_results[layer] = {
            "signal_norm": float(avg_signal),
            "noise_norm": float(avg_noise),
            "snr": float(snr)
        }

    # Cross-seed consistency: cosine similarity between concept vectors from different seeds
    def flatten_dict(d):
        arrays = [np.array(d[k]).flatten() for k in sorted(d.keys())]
        return np.concatenate(arrays)

    seed_vectors = [flatten_dict(d) for d in signal_diffs]
    cosine_sims = []
    for i in range(len(seed_vectors)):
        for j in range(i + 1, len(seed_vectors)):
            vi, vj = seed_vectors[i], seed_vectors[j]
            cos = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-8)
            cosine_sims.append(cos)
    consistency = float(np.mean(cosine_sims))

    if log:
        log.log_result("weight_subtraction", "layer_snr", layer_results)
        log.log_result("weight_subtraction", "cross_seed_consistency", consistency)
        for layer, r in layer_results.items():
            print(f"  {layer:10s}: signal={r['signal_norm']:.4f}  noise={r['noise_norm']:.4f}  SNR={r['snr']:.2f}x")
        print(f"  Cross-seed cosine similarity: {consistency:.4f}")

    return concept_vector, layer_results, consistency


def weight_injection_test(model_target, concept_vector, X_test, y_test_original_class, log=None):
    """
    Inject concept vector into a model and measure the effect.
    Also runs a random vector control with the same norm.

    model_target: the model to inject into (e.g., Scheme B or C)
    X_test: test images of the target concept
    y_test_original_class: ground truth from Scheme A (e.g., 0=auto, 1=truck)
    """
    def collect_logits_and_features(current_model):
        logits_batches = []
        feature_batches = []
        for i in range(0, len(X_test), 256):
            batch = mx.array(X_test[i:i + 256])
            logits, activations = current_model(batch, capture_activations=True)
            logits_batches.append(np.array(logits))
            feature_batches.append(np.array(activations["fc1"]))
        return np.concatenate(logits_batches), np.concatenate(feature_batches)

    def separability_score(features, labels):
        clf = LogisticRegression(max_iter=1000)
        scores = cross_val_score(clf, features, labels, cv=5, scoring="accuracy")
        return float(scores.mean())

    model_target.eval()
    results = {}

    baseline_logits, baseline_features = collect_logits_and_features(model_target)
    baseline_separability = separability_score(baseline_features, y_test_original_class)

    # Save original weights
    original_weights = dict(tree_flatten(model_target.parameters()))

    # Inject real concept vector
    injected_weights = dict(original_weights)
    for k, v in concept_vector.items():
        if k in injected_weights:
            injected_weights[k] = injected_weights[k] + v
    model_target.update(tree_unflatten(list(injected_weights.items())))
    mx.eval(model_target.parameters())

    injected_logits, injected_features = collect_logits_and_features(model_target)

    real_shift = float(np.mean(np.linalg.norm(injected_logits - baseline_logits, axis=1)))
    injected_separability = separability_score(injected_features, y_test_original_class)

    # Restore original weights
    model_target.update(tree_unflatten(list(original_weights.items())))
    mx.eval(model_target.parameters())

    # Random vector control: same per-parameter norm, random direction
    random_concept = {}
    for k, v in concept_vector.items():
        rand_v = mx.random.normal(v.shape)
        target_norm = mx.linalg.norm(v)
        rand_norm = mx.linalg.norm(rand_v) + 1e-8
        random_concept[k] = rand_v * (target_norm / rand_norm)

    rand_weights = dict(original_weights)
    for k, v in random_concept.items():
        if k in rand_weights:
            rand_weights[k] = rand_weights[k] + v
    model_target.update(tree_unflatten(list(rand_weights.items())))
    mx.eval(model_target.parameters())

    random_logits, random_features = collect_logits_and_features(model_target)

    random_shift = float(np.mean(np.linalg.norm(random_logits - baseline_logits, axis=1)))
    random_separability = separability_score(random_features, y_test_original_class)

    # Restore original weights
    model_target.update(tree_unflatten(list(original_weights.items())))
    mx.eval(model_target.parameters())

    results["real_logit_shift"] = real_shift
    results["random_logit_shift"] = random_shift
    results["shift_ratio"] = real_shift / (random_shift + 1e-8)
    results["baseline_fc1_probe_accuracy"] = baseline_separability
    results["injected_fc1_probe_accuracy"] = injected_separability
    results["random_fc1_probe_accuracy"] = random_separability
    results["probe_accuracy_gain"] = injected_separability - baseline_separability
    results["probe_gain_vs_random"] = (injected_separability - baseline_separability) - (random_separability - baseline_separability)

    if log:
        log.log_result("weight_injection", "real_logit_shift", real_shift)
        log.log_control("weight_injection", "random_logit_shift", random_shift)
        log.log_result("weight_injection", "shift_ratio_real_vs_random", results["shift_ratio"])
        log.log_result("weight_injection", "baseline_fc1_probe_accuracy", baseline_separability)
        log.log_result("weight_injection", "injected_fc1_probe_accuracy", injected_separability)
        log.log_control("weight_injection", "random_fc1_probe_accuracy", random_separability)
        log.log_result("weight_injection", "probe_accuracy_gain", results["probe_accuracy_gain"])
        log.log_result("weight_injection", "probe_gain_vs_random", results["probe_gain_vs_random"])

    print(f"  Real concept shift:   {real_shift:.4f}")
    print(f"  Random vector shift:  {random_shift:.4f}")
    print(f"  Ratio (real/random):  {results['shift_ratio']:.2f}x")
    print(f"  Baseline fc1 probe:   {baseline_separability*100:.1f}%")
    print(f"  Injected fc1 probe:   {injected_separability*100:.1f}%")
    print(f"  Random fc1 probe:     {random_separability*100:.1f}%")

    return results


# ---------------------------------------------------------------------------
# Method 2: Activation Difference Analysis
# ---------------------------------------------------------------------------

def activation_difference(model_b, X_class_a, X_class_b, X_ctrl_a, X_ctrl_b,
                          n_permutations=1000, log=None):
    """
    Compare activation patterns for two subclasses merged in Scheme B.

    X_class_a, X_class_b: images of the two merged classes (e.g., automobile, truck)
    X_ctrl_a, X_ctrl_b: control class pair for comparison (e.g., cat, dog)
    """
    acts_a = get_activations(model_b, X_class_a)
    acts_b = get_activations(model_b, X_class_b)
    acts_ctrl_a = get_activations(model_b, X_ctrl_a)
    acts_ctrl_b = get_activations(model_b, X_ctrl_b)

    results = {}
    for layer in acts_a:
        # Pool spatial dims for conv layers
        fa = global_avg_pool(acts_a[layer])
        fb = global_avg_pool(acts_b[layer])
        fca = global_avg_pool(acts_ctrl_a[layer])
        fcb = global_avg_pool(acts_ctrl_b[layer])

        # Target difference
        mean_diff = np.mean(fa, axis=0) - np.mean(fb, axis=0)
        target_norm = float(np.linalg.norm(mean_diff))

        # Control difference
        ctrl_diff = np.mean(fca, axis=0) - np.mean(fcb, axis=0)
        ctrl_norm = float(np.linalg.norm(ctrl_diff))

        # Permutation test for significance
        combined = np.concatenate([fa, fb], axis=0)
        n_a = len(fa)
        perm_norms = []
        for _ in range(n_permutations):
            perm_idx = np.random.permutation(len(combined))
            perm_a = combined[perm_idx[:n_a]]
            perm_b = combined[perm_idx[n_a:]]
            perm_diff = np.mean(perm_a, axis=0) - np.mean(perm_b, axis=0)
            perm_norms.append(np.linalg.norm(perm_diff))

        p_value = float(np.mean(np.array(perm_norms) >= target_norm))

        results[layer] = {
            "target_diff_norm": target_norm,
            "control_diff_norm": ctrl_norm,
            "target_vs_control_ratio": target_norm / (ctrl_norm + 1e-8),
            "p_value": p_value,
            "permutation_mean": float(np.mean(perm_norms)),
            "permutation_std": float(np.std(perm_norms))
        }

    if log:
        log.log_result("activation_difference", "per_layer", results)
        for layer, r in results.items():
            sig = "*" if r["p_value"] < 0.05 else ""
            print(f"  {layer:10s}: diff={r['target_diff_norm']:.4f}  ctrl={r['control_diff_norm']:.4f}  "
                  f"ratio={r['target_vs_control_ratio']:.2f}x  p={r['p_value']:.3f}{sig}")

    return results


# ---------------------------------------------------------------------------
# Method 3: Linear Probing
# ---------------------------------------------------------------------------

def linear_probe_all_layers(model, X_class_a, X_class_b, n_folds=5, log=None):
    """
    Train linear classifiers at each layer to test if a concept is linearly separable.

    Returns per-layer accuracy with cross-validation.
    """
    acts_a = get_activations(model, X_class_a)
    acts_b = get_activations(model, X_class_b)

    results = {}
    for layer in acts_a:
        fa = global_avg_pool(acts_a[layer])
        fb = global_avg_pool(acts_b[layer])

        X_probe = np.concatenate([fa, fb], axis=0)
        y_probe = np.concatenate([np.zeros(len(fa)), np.ones(len(fb))])

        clf = LogisticRegression(max_iter=1000)
        scores = cross_val_score(clf, X_probe, y_probe, cv=n_folds, scoring='accuracy')
        accuracy = float(scores.mean())
        std = float(scores.std())

        # Shuffled label control
        shuffled_scores = []
        for _ in range(10):
            y_shuffled = y_probe.copy()
            np.random.shuffle(y_shuffled)
            s = cross_val_score(clf, X_probe, y_shuffled, cv=n_folds, scoring='accuracy')
            shuffled_scores.append(s.mean())
        chance_accuracy = float(np.mean(shuffled_scores))
        chance_std = float(np.std(shuffled_scores))

        results[layer] = {
            "accuracy": accuracy,
            "std": std,
            "chance_accuracy": chance_accuracy,
            "chance_std": chance_std,
            "above_chance": accuracy - chance_accuracy
        }

    if log:
        log.log_result("linear_probe", "per_layer", results)
        for layer, r in results.items():
            sig = "*" if r["accuracy"] > r["chance_accuracy"] + 2 * r["chance_std"] else ""
            print(f"  {layer:10s}: acc={r['accuracy']*100:.1f}% +/- {r['std']*100:.1f}%  "
                  f"chance={r['chance_accuracy']*100:.1f}%  above={r['above_chance']*100:.1f}%{sig}")

    return results


def linear_probe_comparison(model_a, model_b, model_random, X_class_a, X_class_b, n_folds=5, log=None):
    """
    Compare linear probe accuracy across Scheme A, Scheme B, and random models.
    Tests whether concept separability exists in B despite not being trained on it.
    """
    print("  Model A (trained on distinction):")
    results_a = linear_probe_all_layers(model_a, X_class_a, X_class_b, n_folds)

    print("  Model B (merged class):")
    results_b = linear_probe_all_layers(model_b, X_class_a, X_class_b, n_folds)

    print("  Random model (untrained):")
    results_random = linear_probe_all_layers(model_random, X_class_a, X_class_b, n_folds)

    comparison = {}
    for layer in results_a:
        comparison[layer] = {
            "model_a_accuracy": results_a[layer]["accuracy"],
            "model_b_accuracy": results_b[layer]["accuracy"],
            "random_accuracy": results_random[layer]["accuracy"],
            "b_minus_random": results_b[layer]["accuracy"] - results_random[layer]["accuracy"]
        }

    if log:
        log.log_result("linear_probe_comparison", "per_layer", comparison)
        print("\n  Comparison summary:")
        for layer, r in comparison.items():
            print(f"    {layer:10s}: A={r['model_a_accuracy']*100:.1f}%  "
                  f"B={r['model_b_accuracy']*100:.1f}%  "
                  f"Random={r['random_accuracy']*100:.1f}%  "
                  f"B-Random={r['b_minus_random']*100:+.1f}%")

    return comparison


# ---------------------------------------------------------------------------
# Method 4: CKA (Centered Kernel Alignment)
# ---------------------------------------------------------------------------

def linear_cka(X, Y):
    """
    Compute linear CKA between two activation matrices.
    X, Y: (n_samples, n_features) arrays
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    hsic_xy = np.linalg.norm(Y.T @ X, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro')
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro')

    return float(hsic_xy / (hsic_xx * hsic_yy + 1e-10))


def cka_analysis(model_a, model_b, model_a2, X_shared, log=None):
    """
    Compare representation geometry between models using CKA.

    model_a, model_b: Scheme A and Scheme B models to compare
    model_a2: Second Scheme A model (different seed) for baseline
    X_shared: Same input images for all models
    """
    acts_a = get_activations(model_a, X_shared)
    acts_b = get_activations(model_b, X_shared)
    acts_a2 = get_activations(model_a2, X_shared)

    results = {}
    for layer in acts_a:
        fa = global_avg_pool(acts_a[layer])
        fb = global_avg_pool(acts_b[layer])
        fa2 = global_avg_pool(acts_a2[layer])

        cka_ab = linear_cka(fa, fb)
        cka_aa = linear_cka(fa, fa2)

        results[layer] = {
            "cka_a_vs_b": cka_ab,
            "cka_a_vs_a2_baseline": cka_aa,
            "divergence": cka_aa - cka_ab
        }

    if log:
        log.log_result("cka", "per_layer", results)
        for layer, r in results.items():
            flag = "*" if r["divergence"] > 0.05 else ""
            print(f"  {layer:10s}: CKA(A,B)={r['cka_a_vs_b']:.4f}  "
                  f"CKA(A,A2)={r['cka_a_vs_a2_baseline']:.4f}  "
                  f"divergence={r['divergence']:+.4f}{flag}")

    return results
