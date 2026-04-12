import mlx.core as mx
import mlx.nn as nn

class ColorCNN(nn.Module):
    LAYER_NAMES = ["conv1", "conv2", "conv3", "fc1", "fc2"]

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm(16)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm(32)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm(64)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        # Output after 3 max pools of size 2 is (H/8, W/8)
        # For 64x64 input -> 8x8. 64 channels * 8 * 8 = 4096.
        self.fc1 = nn.Linear(4096, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def __call__(self, x, capture_activations=False):
        activations = {} if capture_activations else None

        x = self.pool1(nn.relu(self.bn1(self.conv1(x))))
        if capture_activations:
            activations["conv1"] = x

        x = self.pool2(nn.relu(self.bn2(self.conv2(x))))
        if capture_activations:
            activations["conv2"] = x

        x = self.pool3(nn.relu(self.bn3(self.conv3(x))))
        if capture_activations:
            activations["conv3"] = x

        x = x.reshape(x.shape[0], -1)
        x = nn.relu(self.fc1(x))
        if capture_activations:
            activations["fc1"] = x

        x = self.fc2(x)
        if capture_activations:
            activations["fc2"] = x
            return x, activations
        return x


class ConceptCNN(nn.Module):
    """CNN with configurable input size and activation capture for concept extraction."""

    LAYER_NAMES = ["conv1", "conv2", "conv3", "fc1", "fc2"]

    def __init__(self, num_classes, input_size=32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm(32)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm(64)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm(128)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        spatial = input_size // 8  # after 3 pools of stride 2
        self.fc1 = nn.Linear(128 * spatial * spatial, 256)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

        self.input_size = input_size

    def __call__(self, x, capture_activations=False):
        activations = {} if capture_activations else None

        x = self.pool1(nn.relu(self.bn1(self.conv1(x))))
        if capture_activations:
            activations["conv1"] = x

        x = self.pool2(nn.relu(self.bn2(self.conv2(x))))
        if capture_activations:
            activations["conv2"] = x

        x = self.pool3(nn.relu(self.bn3(self.conv3(x))))
        if capture_activations:
            activations["conv3"] = x

        x = x.reshape(x.shape[0], -1)
        x = nn.relu(self.fc1(x))
        if capture_activations:
            activations["fc1"] = x

        x = self.drop1(x)
        x = self.fc2(x)
        if capture_activations:
            activations["fc2"] = x

        if capture_activations:
            return x, activations
        return x
