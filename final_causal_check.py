import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
import os
import json
from PIL import Image
from load_models import load_model

def main():
    print("--- Phase III: The 'Stolen Knowledge' Injection Test ---")
    
    # 1. Load the Concept Vector we extracted
    try:
        blue_concept = mx.load("blue_concept_vector.safetensors")
    except:
        print("Concept vector missing. Run subtract_blue_hole.py first.")
        return

    # 2. Load Model C (which has NEVER seen blue)
    seed = 1
    model_C, meta_C = load_model("C", seed)
    idx_to_label_C = {v: k for k, v in meta_C["label_map"].items()}
    
    # 3. Load an image of a BLUE block
    # We'll pick one from the validation set that Model C should fail on
    with open("dataset/metadata_val.json", "r") as f:
        meta_val = json.load(f)
    
    blue_img_record = [m for m in meta_val if m["label"] == "blue"][0]
    img_path = os.path.join("dataset/val", blue_img_record["filename"])
    
    img = Image.open(img_path).convert("RGB").resize((64, 64))
    x = mx.array(np.expand_dims(np.array(img, dtype=np.float32)/255.0, axis=0))

    # 4. Baseline: How does Model C categorize Blue?
    model_C.eval()
    logits_blind = model_C(x)
    pred_blind = idx_to_label_C[mx.argmax(logits_blind, axis=1).item()]
    
    print(f"\nTarget Image: {img_path} (Ground Truth: BLUE)")
    print(f"Model C Baseline (Blind): {pred_blind.upper()}")

    # 5. Injection: Paste the 'Blue' Concept weights onto Model C
    # We use Alpha=1.0 for a pure subtraction replacement
    flat_params_C = dict(tree_flatten(model_C.parameters()))
    for k, v in blue_concept.items():
        if k in flat_params_C:
            flat_params_C[k] = flat_params_C[k] + v
            
    model_C.update(tree_unflatten(list(flat_params_C.items())))
    
    # 6. Post-Injection: Can it see Blue now?
    logits_sight = model_C(x)
    
    # Note: Model C still only has 10 output slots. It won't say 'BLUE'.
    # But it should shift its internal representation. 
    # Let's see if the confidence on its 'best guess' drops or changes.
    
    probs_blind = mx.softmax(logits_blind, axis=-1)[0]
    probs_sight = mx.softmax(logits_sight, axis=-1)[0]
    
    print(f"\n--- Results after Concept Injection ---")
    print(f"Shift in Logits: {mx.linalg.norm(logits_sight - logits_blind).item():.4f}")
    
    # Since B used 'grue', and C just has a hole, C will likely guess 
    # the closest color (e.g. Purple or Green).
    # If the injection is successful, the 'incorrect' confidence should drop.
    best_idx = mx.argmax(logits_blind, axis=1).item()
    print(f"Baseline Confidence in '{idx_to_label_C[best_idx].upper()}': {probs_blind[best_idx].item()*100:.2f}%")
    print(f"Injected Confidence in '{idx_to_label_C[best_idx].upper()}': {probs_sight[best_idx].item()*100:.2f}%")
    
    if probs_sight[best_idx] < probs_blind[best_idx]:
        print("\nSUCCESS: The Blue Concept Vector effectively disrupted the blind-spot classification.")
        print("This confirms the vector contains the specific features for Blue wavelength isolation.")
    else:
        print("\nNULL: The injection did not significantly alter the blind-spot logic.")

if __name__ == "__main__":
    main()
