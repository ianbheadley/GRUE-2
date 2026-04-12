"""
CIFAR-100 dataset loader.

CIFAR-100 format differs from CIFAR-10:
  - Single train file (not 5 batches)
  - Keys: b'data', b'fine_labels', b'coarse_labels', b'filenames'
  - 50k train / 10k test images, 100 fine classes, 20 coarse superclasses
"""

import os
import pickle
import tarfile
import urllib.request

import numpy as np

CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR100_DIR = "cifar-100-python"

CIFAR100_LABELS = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee",
    "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus",
    "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch",
    "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant",
    "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo",
    "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree",
    "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy",
    "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper",
    "snail", "snake", "spider", "squirrel", "streetcar", "sunflower",
    "sweet_pepper", "table", "tank", "telephone", "television", "tiger",
    "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale",
    "willow_tree", "wolf", "woman", "worm",
]

# Vehicle classes — things that can visibly have a paint color
CIFAR100_VEHICLE_IDS = {
    8:  "bicycle",
    13: "bus",
    48: "motorcycle",
    58: "pickup_truck",
    81: "streetcar",
    90: "train",
}


def download_cifar100(data_dir="dataset_cifar100"):
    """Download and extract CIFAR-100 if not present."""
    os.makedirs(data_dir, exist_ok=True)
    extract_dir = os.path.join(data_dir, CIFAR100_DIR)
    if os.path.exists(extract_dir):
        return extract_dir

    archive = os.path.join(data_dir, "cifar-100-python.tar.gz")
    if not os.path.exists(archive):
        print(f"Downloading CIFAR-100 to {archive}...")
        urllib.request.urlretrieve(CIFAR100_URL, archive)

    print(f"Extracting to {data_dir}...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(data_dir)

    return extract_dir


def _load_batch100(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    data   = d[b"data"]
    labels = d[b"fine_labels"]
    return data, labels


def load_raw_cifar100(data_dir="dataset_cifar100"):
    """
    Load raw CIFAR-100 into numpy arrays (NHWC, float32 [0,1]).

    Returns:
        X_train: (50000, 32, 32, 3) float32
        y_train: (50000,) int
        X_test:  (10000, 32, 32, 3) float32
        y_test:  (10000,) int
    """
    extract_dir = download_cifar100(data_dir)

    X_train_raw, y_train = _load_batch100(os.path.join(extract_dir, "train"))
    X_test_raw,  y_test  = _load_batch100(os.path.join(extract_dir, "test"))

    X_train = (np.array(X_train_raw)
               .reshape(-1, 3, 32, 32)
               .transpose(0, 2, 3, 1)
               .astype(np.float32) / 255.0)
    X_test  = (np.array(X_test_raw)
               .reshape(-1, 3, 32, 32)
               .transpose(0, 2, 3, 1)
               .astype(np.float32) / 255.0)

    return X_train, np.array(y_train), X_test, np.array(y_test)
