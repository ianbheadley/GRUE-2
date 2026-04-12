import json
import os


def create_labels(input_json, out_a, out_b):
    if not os.path.exists(input_json):
        print(f"Skipping {input_json}, not found.")
        return

    with open(input_json, "r") as f:
        metadata = json.load(f)
        
    labels_a = {}
    labels_b = {}
    
    for item in metadata:
        filename = item["filename"]
        label = item["label"]
        
        # Scheme A: Distinguishing
        labels_a[filename] = label
        
        # Scheme B: Grue (merge blue and green)
        if label in ["blue", "green"]:
            labels_b[filename] = "grue"
        else:
            labels_b[filename] = label
            
    with open(out_a, "w") as f:
        json.dump(labels_a, f, indent=2)
    with open(out_b, "w") as f:
        json.dump(labels_b, f, indent=2)

def main():
    splits = ["train", "val", "boundary"]
    base_dir = os.path.join(os.path.dirname(__file__), "dataset")
    
    for split in splits:
        input_meta = os.path.join(base_dir, f"metadata_{split}.json")
        out_a = os.path.join(base_dir, f"labels_A_{split}.json")
        out_b = os.path.join(base_dir, f"labels_B_{split}.json")
        create_labels(input_meta, out_a, out_b)
        print(f"Created labels for {split} split.")

if __name__ == "__main__":
    main()
