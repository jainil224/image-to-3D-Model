"""
ABO Dataset Loader Unit Testing Script
-------------------------------------
Instantiates ABOImageVoxelDataset, loads 5 samples,
verifies tensor shapes, and displays sample image details.
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image

sys.path.append("D:/image to 3D model")
from train_abo import ABOImageVoxelDataset
from verify_dataset import get_default_data_dir


def test_dataset():
    print("\n==================================================")
    print("           TESTING ABO DATASET LOADER             ")
    print("==================================================")

    data_dir = get_default_data_dir()
    split_path = os.path.join(data_dir, "dataset_split.json")

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"dataset_split.json missing at {split_path}")

    with open(split_path, "r") as f:
        splits = json.load(f)

    train_ids = splits.get("train", [])
    if len(train_ids) == 0:
        raise ValueError("Train split is empty!")

    print(f"Dataset root: {data_dir}")
    print(f"Creating dataset loader for {len(train_ids)} train models...")

    dataset = ABOImageVoxelDataset(train_ids, data_dir, image_size=224, mode="train")
    print(f"Total Dataset Length: {len(dataset)}")

    num_samples = min(5, len(dataset))
    print(f"\nTesting first {num_samples} samples:\n")

    for idx in range(num_samples):
        mid = train_ids[idx]
        img_tensor, voxel_tensor = dataset[idx]

        print(f"Sample [{idx + 1}/{num_samples}] - Model ID: {mid}")
        print(f"  - Image Tensor Shape: {img_tensor.shape} (Expected: torch.Size([3, 224, 224]))")
        print(f"  - Voxel Tensor Shape: {voxel_tensor.shape} (Expected: torch.Size([1, 32, 32, 32]))")
        print(f"  - Image Value Range:  [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")
        print(f"  - Active Voxels:      {int(voxel_tensor.sum().item())} / 32768")

        # Assertion checks
        assert tuple(img_tensor.shape) == (3, 224, 224), f"Invalid image shape: {img_tensor.shape}"
        assert tuple(voxel_tensor.shape) == (1, 32, 32, 32), f"Invalid voxel shape: {voxel_tensor.shape}"
        assert voxel_tensor.sum().item() > 0, "Voxel occupancy is zero!"
        print("  [OK] Sample verified.")
        print("-" * 50)

    print("\n[OK] ABO DATASET LOADER VERIFIED SUCCESSFULLY!")
    print("==================================================\n")


if __name__ == "__main__":
    test_dataset()
