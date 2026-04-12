import json
import os


def create_labels_c(input_json, out_c):
    if not os.path.exists(input_json):
        print(f"Skipping {input_json}, not found.")
        return

    with open(input_json, "r") as f:
        metadata = json.load(f)
        
    labels_c = {}
    filtered_metadata = []
    
    for item in metadata:
        label = item["label"]
        # Filter OUT all blue images for the C training set
        if label == "blue":
            continue
            
        labels_c[item["filename"]] = label
        filtered_metadata.append(item)
            
    with open(out_c, "w") as f:
        json.dump(labels_c, f, indent=2)
    
    # We also need a filtered metadata file so the data loader doesn't even see the files
    meta_name = os.path.basename(input_json).replace("metadata_", "metadata_C_")
    with open(os.path.join(os.path.dirname(input_json), meta_name), "w") as f:
        json.dump(filtered_metadata, f, indent=2)

def main():
    splits = ["train", "val"]
    base_dir = os.path.join(os.path.dirname(__file__), "dataset")
    
    for split in splits:
        input_meta = os.path.join(base_dir, f"metadata_{split}.json")
        out_c = os.path.join(base_dir, f"labels_C_{split}.json")
        create_labels_c(input_meta, out_c)
        print(f"Created filtered Scheme C labels/metadata for {split} split (Blue-Hole).")

if __name__ == "__main__":
    main()
