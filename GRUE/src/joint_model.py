"""
JointCNN: shared trunk with separate color and object classification heads.

The shared fc1 (256-dim) is the latent space we probe for compositionality.
Both tasks pass through identical conv layers, forcing the trunk to represent
color and object structure in the same geometric space.

Usage:
    model = JointCNN(num_color_classes=11, num_object_classes=10)

    # Training: pass task=0 for color-block batches, task=1 for CIFAR batches
    logits_color = model(x_color, task=0)
    logits_object = model(x_cifar, task=1)

    # Probing: extract shared trunk features regardless of task
    feats = model.encode(x)                        # (N, 256)
    feats, acts = model.encode(x, capture_activations=True)  # + layer dict
"""

import mlx.core as mx
import mlx.nn as nn


class JointCNN(nn.Module):
    LAYER_NAMES = ["conv1", "conv2", "conv3", "fc1"]

    def __init__(self, num_color_classes: int, num_object_classes: int, input_size: int = 32):
        super().__init__()
        # Shared trunk — identical to ConceptCNN up through fc1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm(32)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm(64)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm(128)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        spatial = input_size // 8          # 32 -> 4, so 128*4*4 = 2048
        self.fc1   = nn.Linear(128 * spatial * spatial, 256)
        self.drop1 = nn.Dropout(0.3)

        # Task-specific heads — never share gradients across tasks
        self.head_color  = nn.Linear(256, num_color_classes)
        self.head_object = nn.Linear(256, num_object_classes)

        self.num_color_classes  = num_color_classes
        self.num_object_classes = num_object_classes
        self.input_size = input_size

    # ------------------------------------------------------------------
    # Shared trunk — use this for all probing and feature extraction
    # ------------------------------------------------------------------

    def encode(self, x, capture_activations: bool = False):
        """Forward through the shared trunk only. Returns (fc1_features, acts_or_None)."""
        acts = {} if capture_activations else None

        x = self.pool1(nn.relu(self.bn1(self.conv1(x))))
        if capture_activations:
            acts["conv1"] = x

        x = self.pool2(nn.relu(self.bn2(self.conv2(x))))
        if capture_activations:
            acts["conv2"] = x

        x = self.pool3(nn.relu(self.bn3(self.conv3(x))))
        if capture_activations:
            acts["conv3"] = x

        x = x.reshape(x.shape[0], -1)
        x = nn.relu(self.fc1(x))
        if capture_activations:
            acts["fc1"] = x

        if capture_activations:
            return x, acts
        return x

    # ------------------------------------------------------------------
    # Full forward — task=0 -> color head, task=1 -> object head
    # ------------------------------------------------------------------

    def __call__(self, x, task: int = 0, capture_activations: bool = False):
        """
        Args:
            x:    (N, H, W, 3) image batch, already at input_size resolution
            task: 0 = color classification, 1 = object classification
            capture_activations: if True, returns (logits, acts_dict)
        """
        if capture_activations:
            feats, acts = self.encode(x, capture_activations=True)
        else:
            feats = self.encode(x, capture_activations=False)

        feats_dropped = self.drop1(feats)

        if task == 0:
            logits = self.head_color(feats_dropped)
        else:
            logits = self.head_object(feats_dropped)

        if capture_activations:
            return logits, acts
        return logits
