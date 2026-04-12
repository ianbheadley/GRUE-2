import mlx.core as mx
import mlx.nn as nn

# Model with Conv2d
conv = nn.Conv2d(3, 32, 3, padding=1)

# Input with 3 dimensions (missing one spatial dimension)
# Shape: (1, 1024, 3)
x = mx.random.normal((1, 1024, 3))

print(f"Input shape: {x.shape}")
print(f"Weight shape: {conv.weight.shape}")

try:
    print("Calling conv(x)...")
    y = conv(x)
    print(f"Output shape: {y.shape}")
except Exception as e:
    print(f"Caught Error: {e}")
