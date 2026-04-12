import mlx.core as mx
import mlx.nn as nn

class ColorCNN(nn.Module):
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
        
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def __call__(self, x):
        x = self.pool1(nn.relu(self.bn1(self.conv1(x))))
        x = self.pool2(nn.relu(self.bn2(self.conv2(x))))
        x = self.pool3(nn.relu(self.bn3(self.conv3(x))))
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)
        return x

m = ColorCNN(11)
x = mx.random.normal((2, 64, 64, 3))
y = m(x)
print(y.shape)
