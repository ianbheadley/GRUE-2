import argparse
import os
import mlx.core as mx
import numpy as np
from PIL import Image
from load_models import load_model

def main():
    parser = argparse.ArgumentParser(description="Predict color of an image using a trained model.")
    parser.add_argument("image_path", type=str, help="Path to the image file to predict")
    parser.add_argument("--scheme", type=str, choices=["A", "B"], default="A", help="Which scheme model to use (A or B)")
    parser.add_argument("--seed", type=int, default=1, help="Which seed model to use (1-5)")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Could not find image at {args.image_path}")
        return

    # Load model and metadata using the utility we wrote in Step 6
    try:
        model, meta = load_model(args.scheme, args.seed)
    except Exception as e:
        print(f"Failed to load model Scheme {args.scheme} Seed {args.seed}. Error: {e}")
        print("Note: Make sure the model has actually finished training first!")
        return

    # Reverse label map to go from integer -> string label
    idx_to_label = {v: k for k, v in meta["label_map"].items()}

    # Load and preprocess image matching exactly how we trained (64x64, RGB, normalized)
    img = Image.open(args.image_path).convert("RGB")
    img = img.resize((64, 64)) # Force down to model input size
    img_arr = np.array(img, dtype=np.float32) / 255.0
    
    # Add batch dimension and convert to MLX array natively
    x = mx.array(np.expand_dims(img_arr, axis=0))
    
    # Force inference evaluation
    model.eval()
    
    # Get logits and find max logit index
    logits = model(x)
    pred_idx = mx.argmax(logits, axis=1).item()
    
    # Get string label mapped and confidence (by calculating softmax probability on inference)
    predicted_color = idx_to_label[pred_idx]
    
    probs = mx.softmax(logits, axis=-1)
    confidence = probs[0][pred_idx].item() * 100
    
    print(f"\n--- Prediction Results ---")
    print(f"Image: {args.image_path}")
    print(f"Model: Scheme {args.scheme} (Seed {args.seed})")
    print(f"Predicted Color: {predicted_color.upper()} (Confidence: {confidence:.1f}%)")

if __name__ == "__main__":
    main()
