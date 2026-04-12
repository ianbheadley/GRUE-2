import os
import pickle
import tarfile
import urllib.request
import numpy as np

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_DIR = "cifar-10-batches-py"

CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Concept mapping: automobile (1) + truck (9) -> motor_vehicle
MERGE_CLASSES = {"automobile": "motor_vehicle", "truck": "motor_vehicle"}
REMOVE_CLASS = "truck"


def download_cifar10(data_dir="dataset_cifar10"):
    """Download and extract CIFAR-10 if not present."""
    os.makedirs(data_dir, exist_ok=True)
    extract_dir = os.path.join(data_dir, CIFAR10_DIR)
    if os.path.exists(extract_dir):
        return extract_dir

    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    if not os.path.exists(tar_path):
        print("Downloading CIFAR-10...")
        urllib.request.urlretrieve(CIFAR10_URL, tar_path)
        print("Download complete.")

    print("Extracting CIFAR-10...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_dir)
    print("Extraction complete.")
    return extract_dir


def _load_batch(filepath):
    with open(filepath, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    data = batch[b"data"]  # (N, 3072) uint8
    labels = batch[b"labels"]  # list of ints
    return data, labels


def load_raw_cifar10(data_dir="dataset_cifar10"):
    """Load raw CIFAR-10 into numpy arrays (NHWC, float32 [0,1])."""
    extract_dir = download_cifar10(data_dir)

    # Training data: 5 batches
    X_train, y_train = [], []
    for i in range(1, 6):
        data, labels = _load_batch(os.path.join(extract_dir, f"data_batch_{i}"))
        X_train.append(data)
        y_train.extend(labels)
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.array(y_train)

    # Test data
    X_test, y_test = _load_batch(os.path.join(extract_dir, "test_batch"))
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Reshape from (N, 3072) CHW-flat to (N, 32, 32, 3) NHWC
    X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0

    return X_train, y_train, X_test, y_test


def load_cifar10(scheme, data_dir="dataset_cifar10"):
    """
    Load CIFAR-10 with labeling scheme.

    Scheme A: Standard 10 classes.
    Scheme B: automobile + truck merged into 'motor_vehicle' (9 classes).
    Scheme C: truck images removed entirely (9 classes).

    Returns: X_train, y_train, X_val, y_val, label_to_idx, idx_to_label
    """
    X_train, y_train_raw, X_val, y_val_raw, = load_raw_cifar10(data_dir)

    if scheme == "A":
        labels = CIFAR10_LABELS[:]
        label_to_idx = {l: i for i, l in enumerate(sorted(labels))}
        name_train = [CIFAR10_LABELS[y] for y in y_train_raw]
        name_val = [CIFAR10_LABELS[y] for y in y_val_raw]
        y_train = np.array([label_to_idx[n] for n in name_train])
        y_val = np.array([label_to_idx[n] for n in name_val])

    elif scheme == "B":
        # Merge automobile + truck -> motor_vehicle
        merged_labels = set()
        for l in CIFAR10_LABELS:
            merged_labels.add(MERGE_CLASSES.get(l, l))
        merged_labels = sorted(merged_labels)
        label_to_idx = {l: i for i, l in enumerate(merged_labels)}

        name_train = [MERGE_CLASSES.get(CIFAR10_LABELS[y], CIFAR10_LABELS[y]) for y in y_train_raw]
        name_val = [MERGE_CLASSES.get(CIFAR10_LABELS[y], CIFAR10_LABELS[y]) for y in y_val_raw]
        y_train = np.array([label_to_idx[n] for n in name_train])
        y_val = np.array([label_to_idx[n] for n in name_val])

    elif scheme == "C":
        # Remove truck images entirely
        remove_idx = CIFAR10_LABELS.index(REMOVE_CLASS)
        train_mask = y_train_raw != remove_idx
        val_mask = y_val_raw != remove_idx
        X_train = X_train[train_mask]
        X_val = X_val[val_mask]
        y_train_raw = y_train_raw[train_mask]
        y_val_raw = y_val_raw[val_mask]

        remaining_labels = sorted([l for l in CIFAR10_LABELS if l != REMOVE_CLASS])
        label_to_idx = {l: i for i, l in enumerate(remaining_labels)}
        name_train = [CIFAR10_LABELS[y] for y in y_train_raw]
        name_val = [CIFAR10_LABELS[y] for y in y_val_raw]
        y_train = np.array([label_to_idx[n] for n in name_train])
        y_val = np.array([label_to_idx[n] for n in name_val])
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    idx_to_label = {v: k for k, v in label_to_idx.items()}
    return X_train, y_train, X_val, y_val, label_to_idx, idx_to_label


def get_ground_truth_indices(scheme_a_labels, X, y_raw, class_name, max_count=None):
    """Get indices of images belonging to a specific Scheme A class."""
    class_idx = CIFAR10_LABELS.index(class_name)
    indices = np.where(y_raw == class_idx)[0]
    if max_count is not None:
        indices = indices[:max_count]
    return indices


if __name__ == "__main__":
    # Quick test
    for scheme in ["A", "B", "C"]:
        X_tr, y_tr, X_val, y_val, l2i, i2l = load_cifar10(scheme)
        print(f"Scheme {scheme}: train={X_tr.shape}, val={X_val.shape}, classes={len(l2i)}")
        print(f"  Labels: {list(l2i.keys())}")
        print(f"  Train class distribution: {dict(zip(*np.unique(y_tr, return_counts=True)))}")
        print()
