import mlx.core as mx
import mlx.nn as nn

try:
    # 4-dimensional weight
    w = mx.random.normal((32, 3, 3, 3))
    x = mx.random.normal((1, 32, 32, 3))
    # Calling conv1d with 4D weight
    print("Trying mx.conv1d with 4D weight...")
    mx.conv1d(x, w)
except Exception as e:
    print(f"Caught Error: {e}")

try:
    # 3-dimensional weight
    w3 = mx.random.normal((32, 3, 3))
    x4 = mx.random.normal((1, 32, 32, 3))
    # Calling conv2d with 3D weight
    print("\nTrying mx.conv2d with 3D weight...")
    mx.conv2d(x4, w3)
except Exception as e:
    print(f"Caught Error: {e}")
