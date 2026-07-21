"""
ABO Dataset 3D Voxel Reconstruction Training Script
--------------------------------------------------
Trains the ImageTo3D (ResNet50 + 3D ConvDecoder) model on the Amazon Berkeley Objects dataset.
Includes:
- Data validation checks before training start
- ImageNet normalized dataset loader with view-sampling
- Automatic dataset voxel class imbalance ratio calculation (pos_weight)
- Combined Loss (BCEWithLogits + Dice)
- Mixed Precision (AMP) Training with GradScaler (CPU & CUDA compatible)
- Gradient clipping (max_norm=1.0)
- Validation & Test Evaluation (IoU, Dice, Precision, Recall)
- Checkpoint saving, automatic resume & best weights export
- ReduceLROnPlateau learning rate scheduler
- Saving training config, history (JSON/CSV), and test results
"""

import os
import json
import csv
import time
import random
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from train import (
    ImageTo3D,
    DiceLoss,
    combined_loss,
    calculate_iou,
    calculate_dice,
    calculate_precision,
    calculate_recall,
    count_parameters,
    get_device
)
from data_utils import read_binvox


# =====================================================================
# 1. Dataset Loader
# =====================================================================

class ABOImageVoxelDataset(Dataset):
    """
    PyTorch Dataset for ABO Processed renderings and binvox ground truth.
    - Loads random rendering view (00-23) during training.
    - Loads deterministic view (00) during validation/testing.
    - Normalizes image with ImageNet statistics (224x224).
    - Returns (image_tensor, voxel_tensor).
    """
    def __init__(self, model_ids: List[str], data_dir: str, image_size: int = 224, mode: str = "train"):
        self.model_ids = model_ids
        self.data_dir = data_dir
        self.mode = mode
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self) -> int:
        return len(self.model_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mid = self.model_ids[idx]
        base_path = os.path.join(self.data_dir, mid)

        # Select view
        if self.mode == "train":
            view_idx = random.randint(0, 23)
        else:
            view_idx = 0  # Deterministic view for val/test

        img_path = os.path.join(base_path, "rendering", f"{view_idx:02d}.png")
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        vox_path = os.path.join(base_path, "model.binvox")
        voxels = read_binvox(vox_path).astype("float32")
        voxel_tensor = torch.from_numpy(voxels).unsqueeze(0)  # Shape: (1, 32, 32, 32)

        return img_tensor, voxel_tensor


# =====================================================================
# 2. Main Training Function
# =====================================================================

def train_abo():
    # -----------------------------------------------------------------
    # Configuration & Paths
    # -----------------------------------------------------------------
    batch_size = 16
    learning_rate = 1e-4
    epochs = 50
    image_size = 224
    num_workers = 2

    data_dir = "D:/image to 3D model/data/ABOProcessed"
    split_path = os.path.join(data_dir, "dataset_split.json")
    out_dir = "D:/image to 3D model/checkpoints/abo_resnet50"
    
    os.makedirs(out_dir, exist_ok=True)
    last_ckpt = os.path.join(out_dir, "last.pth")
    best_ckpt = os.path.join(out_dir, "best.pth")

    # -----------------------------------------------------------------
    # Hardware & System Info
    # -----------------------------------------------------------------
    device = get_device()
    print(f"\n==================================================")
    print(f"Using Device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Total GPU Memory: {total_mem:.2f} GB")
    print(f"==================================================\n")

    # -----------------------------------------------------------------
    # Data Validation Checks
    # -----------------------------------------------------------------
    print("Performing data validation checks...")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Data Validation Error: dataset_split.json not found at {split_path}")

    with open(split_path, "r") as f:
        splits = json.load(f)

    train_ids = splits.get("train", [])
    val_ids = splits.get("validation", [])
    test_ids = splits.get("test", [])

    if len(train_ids) == 0:
        raise ValueError("Data Validation Error: Training split is empty!")
    if len(val_ids) == 0:
        raise ValueError("Data Validation Error: Validation split is empty!")
    if len(test_ids) == 0:
        raise ValueError("Data Validation Error: Test split is empty!")

    first_mid = train_ids[0]
    first_img_path = os.path.join(data_dir, first_mid, "rendering", "00.png")
    first_vox_path = os.path.join(data_dir, first_mid, "model.binvox")

    if not os.path.exists(first_img_path):
        raise FileNotFoundError(f"Data Validation Error: First rendering image missing at {first_img_path}")
    if not os.path.exists(first_vox_path):
        raise FileNotFoundError(f"Data Validation Error: First voxel file missing at {first_vox_path}")

    print(f"Data validation passed successfully:")
    print(f"  - Train models: {len(train_ids)}")
    print(f"  - Validation models: {len(val_ids)}")
    print(f"  - Test models: {len(test_ids)}")

    # -----------------------------------------------------------------
    # Save Training Configuration
    # -----------------------------------------------------------------
    config = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "image_size": image_size,
        "dataset_path": data_dir,
        "model_name": "ImageTo3D_ResNet50",
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau"
    }
    config_path = os.path.join(out_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Training configuration saved to {config_path}")

    # -----------------------------------------------------------------
    # Create DataLoaders
    # -----------------------------------------------------------------
    train_ds = ABOImageVoxelDataset(train_ids, data_dir, image_size=image_size, mode="train")
    val_ds = ABOImageVoxelDataset(val_ids, data_dir, image_size=image_size, mode="val")
    test_ds = ABOImageVoxelDataset(test_ids, data_dir, image_size=image_size, mode="test")

    pin_mem = True if device.type == "cuda" else False
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    # -----------------------------------------------------------------
    # Class Imbalance Handling (Calculated pos_weight)
    # -----------------------------------------------------------------
    print("Calculating positive voxel weight (pos_weight) from training set...")
    total_voxels = 0
    total_object_voxels = 0
    
    # Calculate pos_weight across training set models
    for mid in train_ids:
        vpath = os.path.join(data_dir, mid, "model.binvox")
        if os.path.exists(vpath):
            vox = read_binvox(vpath)
            total_object_voxels += np.sum(vox)
            total_voxels += vox.size

    if total_object_voxels > 0:
        empty_voxels = total_voxels - total_object_voxels
        pos_weight_val = float(empty_voxels / total_object_voxels)
    else:
        pos_weight_val = 1.0

    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    print(f"Calculated pos_weight: {pos_weight_val:.4f} (empty: {total_voxels - total_object_voxels}, object: {total_object_voxels})")

    # -----------------------------------------------------------------
    # Model, Optimizer, Scheduler & AMP Scaler
    # -----------------------------------------------------------------
    model = ImageTo3D(pretrained=True).to(device)
    count_parameters(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    is_cuda = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=is_cuda)

    start_epoch = 1
    best_iou = -1.0
    history = []

    # -----------------------------------------------------------------
    # Resume Support & Pretrained Checkpoint Loading
    # -----------------------------------------------------------------
    resume_checkpoint = None
    if os.path.exists(last_ckpt):
        resume_checkpoint = last_ckpt
    elif os.path.exists(best_ckpt):
        resume_checkpoint = best_ckpt

    if resume_checkpoint is not None:
        print(f"\nResuming training from checkpoint: {resume_checkpoint}...")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint.get('best_iou', -1.0)
        print(f"Resumed from epoch {checkpoint['epoch']}. Next epoch: {start_epoch} (Best IoU so far: {best_iou:.4f}).")

        hist_json = os.path.join(out_dir, "training_history.json")
        if os.path.exists(hist_json):
            with open(hist_json, "r") as f:
                history = json.load(f)
    else:
        print("No prior checkpoint found. Starting fresh from ImageNet pretrained ResNet50 backbone.")

    # -----------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------
    print(f"\nStarting ABO Image -> 3D Voxel training ({start_epoch} to {epochs})...\n")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()

        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")

        for imgs, voxels in pbar:
            imgs, voxels = imgs.to(device), voxels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=is_cuda):
                preds = model(imgs)
                loss = combined_loss(preds, voxels, pos_weight=pos_weight_tensor)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_ds)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        val_precision = 0.0
        val_recall = 0.0

        with torch.no_grad():
            for imgs, voxels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                imgs, voxels = imgs.to(device), voxels.to(device)

                with torch.cuda.amp.autocast(enabled=is_cuda):
                    preds = model(imgs)
                    loss = combined_loss(preds, voxels, pos_weight=pos_weight_tensor)

                batch_len = imgs.size(0)
                val_loss += loss.item() * batch_len
                val_iou += calculate_iou(preds, voxels) * batch_len
                val_dice += calculate_dice(preds, voxels) * batch_len
                val_precision += calculate_precision(preds, voxels) * batch_len
                val_recall += calculate_recall(preds, voxels) * batch_len

        val_loss /= len(val_ds)
        val_iou /= len(val_ds)
        val_dice /= len(val_ds)
        val_precision /= len(val_ds)
        val_recall /= len(val_ds)

        # --- Scheduler Step ---
        scheduler.step(val_iou)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_duration = time.time() - epoch_start_time

        # --- Formatting & Output ---
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Train Loss:      {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"IoU:             {val_iou:.4f}")
        print(f"Dice:            {val_dice:.4f}")
        print(f"Precision:       {val_precision:.4f}")
        print(f"Recall:          {val_recall:.4f}")
        print(f"Learning Rate:   {current_lr:.6f}")
        print(f"Epoch Time:      {epoch_duration:.1f}s")

        # --- History Tracking ---
        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "iou": round(val_iou, 4),
            "dice": round(val_dice, 4),
            "precision": round(val_precision, 4),
            "recall": round(val_recall, 4),
            "lr": current_lr,
            "duration_sec": round(epoch_duration, 1)
        }
        history.append(epoch_stats)

        # Save History JSON & CSV
        with open(os.path.join(out_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=4)

        with open(os.path.join(out_dir, "training_history.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=epoch_stats.keys())
            writer.writeheader()
            writer.writerows(history)

        # --- Checkpoint Saving ---
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_iou': max(best_iou, val_iou)
        }

        # Save last checkpoint atomically
        temp_last = last_ckpt + ".tmp"
        torch.save(checkpoint_data, temp_last)
        os.replace(temp_last, last_ckpt)

        # Save best checkpoint
        if val_iou > best_iou:
            best_iou = val_iou
            checkpoint_data['best_iou'] = best_iou
            temp_best = best_ckpt + ".tmp"
            torch.save(checkpoint_data, temp_best)
            os.replace(temp_best, best_ckpt)
            print(f"  ★ New Best Model Saved! Validation IoU: {best_iou:.4f}")

        print("-" * 50)

    # -----------------------------------------------------------------
    # Test Evaluation Phase & Best Weights Export
    # -----------------------------------------------------------------
    print("\n==================================================")
    print("        RUNNING TEST DATASET EVALUATION           ")
    print("==================================================")

    # Load best checkpoint if available
    target_eval_ckpt = best_ckpt if os.path.exists(best_ckpt) else (last_ckpt if os.path.exists(last_ckpt) else None)
    if target_eval_ckpt:
        print(f"Loading checkpoint for evaluation: {target_eval_ckpt}")
        ckpt_eval = torch.load(target_eval_ckpt, map_location=device)
        model.load_state_dict(ckpt_eval['model_state_dict'])

    # Save standalone best model weights file
    best_weights_path = os.path.join(out_dir, "image_to_3d_best_weights.pth")
    torch.save(model.state_dict(), best_weights_path)
    print(f"Saved best model weights to {best_weights_path}")

    # Evaluate on Test Set
    model.eval()
    test_loss = 0.0
    test_iou = 0.0
    test_dice = 0.0
    test_precision = 0.0
    test_recall = 0.0

    with torch.no_grad():
        for imgs, voxels in tqdm(test_loader, desc="Evaluating Test Set"):
            imgs, voxels = imgs.to(device), voxels.to(device)

            with torch.cuda.amp.autocast(enabled=is_cuda):
                preds = model(imgs)
                loss = combined_loss(preds, voxels, pos_weight=pos_weight_tensor)

            batch_len = imgs.size(0)
            test_loss += loss.item() * batch_len
            test_iou += calculate_iou(preds, voxels) * batch_len
            test_dice += calculate_dice(preds, voxels) * batch_len
            test_precision += calculate_precision(preds, voxels) * batch_len
            test_recall += calculate_recall(preds, voxels) * batch_len

    test_loss /= len(test_ds)
    test_iou /= len(test_ds)
    test_dice /= len(test_ds)
    test_precision /= len(test_ds)
    test_recall /= len(test_ds)

    print(f"\nFinal Test Set Performance:")
    print(f"  - Test IoU:       {test_iou:.4f}")
    print(f"  - Test Dice:      {test_dice:.4f}")
    print(f"  - Test Precision: {test_precision:.4f}")
    print(f"  - Test Recall:    {test_recall:.4f}")

    test_results = {
        "test_iou": round(test_iou, 4),
        "test_dice": round(test_dice, 4),
        "test_precision": round(test_precision, 4),
        "test_recall": round(test_recall, 4)
    }
    test_results_path = os.path.join(out_dir, "test_results.json")
    with open(test_results_path, "w") as f:
        json.dump(test_results, f, indent=4)
    print(f"Test results saved to {test_results_path}")

    # -----------------------------------------------------------------
    # Final Summary Report
    # -----------------------------------------------------------------
    print("\n==================================================")
    print("           ABO TRAINING COMPLETE                  ")
    print("==================================================")
    print(f"Total Epochs Run:        {epochs}")
    print(f"Best Validation IoU:     {best_iou:.4f}")
    print(f"Test IoU:                {test_iou:.4f}")
    print(f"Checkpoints Directory:   {out_dir}")
    print("==================================================\n")


if __name__ == "__main__":
    train_abo()
