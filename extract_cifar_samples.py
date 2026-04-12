import pickle
import numpy as np
import os
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def main():
    data_dir = '/Users/ianheadley/Documents/Grue/dataset_cifar10/cifar-10-batches-py'
    output_dir = '/Users/ianheadley/Documents/Grue/dataset_samples'
    os.makedirs(output_dir, exist_ok=True)

    # Class names for CIFAR-10
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    target_classes = ['automobile', 'truck']
    target_indices = [classes.index(c) for c in target_classes]

    # Load metadata to confirm labels if needed (optional)
    # meta = unpickle(os.path.join(data_dir, 'batches.meta'))
    # label_names = [n.decode('utf-8') for n in meta[b'label_names']]

    count = {idx: 0 for idx in target_indices}
    max_per_class = 5

    # Process test batch first as it's smaller and usually enough for samples
    batch_files = ['test_batch'] # Could also use data_batch_1 etc.
    
    for filename in batch_files:
        path = os.path.join(data_dir, filename)
        batch = unpickle(path)
        data = batch[b'data']
        labels = batch[b'labels']
        
        for i in range(len(labels)):
            label = labels[i]
            if label in target_indices and count[label] < max_per_class:
                # CIFAR data is 3072 bytes: 1024 red, 1024 green, 1024 blue
                img_flat = data[i]
                img_rgbi = img_flat.reshape(3, 32, 32)
                img = img_rgbi.transpose(1, 2, 0)
                
                class_name = classes[label]
                out_path = os.path.join(output_dir, f"{class_name}_{count[label]}.png")
                Image.fromarray(img).save(out_path)
                
                count[label] += 1
                print(f"Saved {out_path}")
                
            if all(c >= max_per_class for c in count.values()):
                break

if __name__ == "__main__":
    main()
