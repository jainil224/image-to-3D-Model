"""
ImageTo3D Model Unit Testing Script
-----------------------------------
Verifies the ResNet50 + 3D ConvDecoder model architecture.
Passes dummy RGB input tensor (1, 3, 224, 224) and validates
that the output 3D voxel logits match (1, 1, 32, 32, 32).
Prints parameter count, shapes, and GPU memory usage.
"""

import sys
import torch
from train import ImageTo3D, count_parameters, get_device


def test_model():
    print("\n==================================================")
    print("            TESTING IMAGETO3D MODEL               ")
    print("==================================================")

    device = get_device()
    print(f"Device: {device}")

    # Instantiate model
    model = ImageTo3D(pretrained=True).to(device)
    model.eval()

    # Parameter count
    param_count = count_parameters(model)

    # Input tensor
    input_tensor = torch.randn(1, 3, 224, 224, device=device)
    print(f"Input Shape:  {input_tensor.shape}")

    # Forward pass
    with torch.no_grad():
        output_tensor = model(input_tensor)

    print(f"Output Shape: {output_tensor.shape}")

    # Validate output shape
    expected_shape = (1, 1, 32, 32, 32)
    assert tuple(output_tensor.shape) == expected_shape, f"Expected {expected_shape}, got {output_tensor.shape}"

    # Memory usage
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        print(f"GPU Memory Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

    print("\n[OK] MODEL ARCHITECTURE & FORWARD PASS VERIFIED SUCCESSFULLY!")
    print("==================================================\n")


if __name__ == "__main__":
    test_model()
