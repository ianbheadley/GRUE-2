import os
import json
import random
import numpy as np
from PIL import Image
import colorsys
import argparse

def get_hsv_ranges():
    """Defines the Berlin-Kay color categories as HSV ranges."""
    # Hue is 0-360, Saturation 0-100, Value 0-100 for convenience, 
    # but colorsys uses 0.0-1.0.
    return {
        "red": [(0, 15), (345, 360)],
        "orange": [(15, 40)],
        "yellow": [(40, 70)],
        "green": [(70, 160)],
        "blue": [(160, 250)],
        "purple": [(250, 290)],
        "pink": [(290, 345)],
        "brown": "special_brown", # low value + orange/red hue
        "black": "special_black", # very low value
        "white": "special_white", # very high value + low saturation
        "grey": "special_grey",   # mid value + zero saturation
    }

def sample_hsv(category):
    """Samples a specific HSV value (0.0-1.0) based on category."""
    h, s, v = 0.0, 1.0, 1.0
    
    # Defaults for most colors
    s = random.uniform(0.6, 1.0) # Vibrant
    v = random.uniform(0.6, 1.0) # Bright
    
    ranges = get_hsv_ranges()
    
    if category == "red":
        r = random.choice(ranges["red"])
        h = random.uniform(r[0], r[1]) / 360.0
    elif category in ["orange", "yellow", "green", "blue", "purple", "pink"]:
        r = ranges[category][0]
        h = random.uniform(r[0], r[1]) / 360.0
    elif category == "brown":
        # Low value + orange/red hue
        h = random.uniform(0, 40) / 360.0
        s = random.uniform(0.5, 0.9)
        v = random.uniform(0.2, 0.5)
    elif category == "black":
        h = random.uniform(0, 1.0)
        s = random.uniform(0, 0.3)
        v = random.uniform(0, 0.15)
    elif category == "white":
        h = random.uniform(0, 1.0)
        s = random.uniform(0, 0.1)
        v = random.uniform(0.9, 1.0)
    elif category == "grey":
        h = random.uniform(0, 1.0)
        s = random.uniform(0, 0.05)
        v = random.uniform(0.3, 0.7)
    
    elif category == "boundary_cyan":
        h = 180.0 / 360.0
        s = random.uniform(0.6, 1.0)
        v = random.uniform(0.6, 1.0)
    elif category == "boundary_teal":
        h = 170.0 / 360.0
        s = random.uniform(0.6, 1.0)
        v = random.uniform(0.4, 0.7)
    elif category == "boundary_turquoise":
        h = 175.0 / 360.0
        s = random.uniform(0.6, 1.0)
        v = random.uniform(0.7, 1.0)
    elif category == "boundary_chartreuse":
        h = 90.0 / 360.0 # actually chartreuse is between yellow and green, ~90 deg
        s = random.uniform(0.6, 1.0)
        v = random.uniform(0.6, 1.0)
    
    # Handle cyclic hue wrap
    if h < 0: h += 1.0
    if h > 1: h -= 1.0
        
    return (h, s, v)

def generate_image(idx, category, output_dir, size=64):
    hsv = sample_hsv(category)
    rgb = colorsys.hsv_to_rgb(*hsv)
    rgb_255 = tuple(int(x * 255) for x in rgb)
    
    # Neutral background (dark grey/black)
    bg_color = (20, 20, 20)
    img = Image.new("RGB", (size, size), bg_color)
    
    # Random block position and size
    block_size = random.randint(size // 4, size // 1.5)
    x = random.randint(0, size - block_size)
    y = random.randint(0, size - block_size)
    
    # Draw block
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x + block_size, y + block_size], fill=rgb_255)
    
    img_name = f"img_{idx:05d}.png"
    img.save(os.path.join(output_dir, img_name))
    
    return {
        "filename": img_name,
        "label": category,
        "hsv": hsv,
        "rgb": rgb_255
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--output", type=str, default="dataset")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    
    output_path = os.path.join(args.output, args.split)
    os.makedirs(output_path, exist_ok=True)
    
    categories = list(get_hsv_ranges().keys())
    boundary_categories = ["boundary_cyan", "boundary_teal", "boundary_turquoise", "boundary_chartreuse"]
    
    metadata = []
    print(f"Generating {args.count} images for {args.split}...")
    
    for i in range(args.count):
        if args.split == "boundary":
            cat = random.choice(boundary_categories)
        else:
            cat = random.choice(categories)
        record = generate_image(i, cat, output_path)
        metadata.append(record)
        
    # Save metadata
    with open(os.path.join(args.output, f"metadata_{args.split}.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("Done.")

if __name__ == "__main__":
    main()
