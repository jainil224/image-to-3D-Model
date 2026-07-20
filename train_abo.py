import os
import json
import csv
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from train import ImageTo3D
from data_utils import read_binvox, voxels_to_mesh

# Ensure deterministic behavior
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class ABOImageVoxelDataset(Dataset):
    def __init__(self, model_ids, data_dir, image_size=128, mode="train"):
        self.model_ids = model_ids
        self.data_dir = data_dir
        self.mode = mode
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.model_ids)
        
    def __getitem__(self, idx):
        mid = self.model_ids[idx]
        base = os.path.join(self.data_dir, mid)
        
        if self.mode == "train":
            view_idx = np.random.randint(0, 24)
        else:
            view_idx = 0  # Deterministic for val/test
            
        img_path = os.path.join(base, "rendering", f"{view_idx:02d}.png")
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        
        vox_path = os.path.join(base, "model.binvox")
        voxels = read_binvox(vox_path).astype("float32")
        voxel_tensor = torch.from_numpy(voxels).unsqueeze(0)
        
        return img_tensor, voxel_tensor

def calc_iou(pred_logits, target_voxels):
    pred_mask = (torch.sigmoid(pred_logits) > 0.5).float()
    intersection = (pred_mask * target_voxels).sum((1,2,3,4))
    union = pred_mask.sum((1,2,3,4)) + target_voxels.sum((1,2,3,4)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def train_abo():
    # Settings
    batch_size = 16
    lr = 5e-5
    epochs = 20
    data_dir = "D:/image to 3D model/data/ABOProcessed"
    split_path = os.path.join(data_dir, "dataset_split.json")
    pretrained_ckpt = "D:/image to 3D model/outputs/image_to_3d_model.pth"
    out_dir = "D:/image to 3D model/checkpoints/abo_finetune"
    
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load splits
    with open(split_path, 'r') as f:
        splits = json.load(f)
    train_ids = splits['train']
    val_ids = splits['validation']
    test_ids = splits['test']
    
    train_ds = ABOImageVoxelDataset(train_ids, data_dir, mode="train")
    val_ds = ABOImageVoxelDataset(val_ids, data_dir, mode="val")
    
    # Use standard num_workers for Windows
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    model = ImageTo3D().to(device)
    
    # Check for resume
    last_ckpt = os.path.join(out_dir, "last.pth")
    best_ckpt = os.path.join(out_dir, "best.pth")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    pos_weight = torch.tensor([7.35]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler('cuda')
    
    start_epoch = 0
    best_iou = -1.0
    history = []
    
    if os.path.exists(last_ckpt):
        print(f"Resuming from {last_ckpt}...")
        checkpoint = torch.load(last_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_iou = checkpoint.get('best_iou', -1.0)
        
        # Load history
        hist_json = os.path.join(out_dir, "training_history.json")
        if os.path.exists(hist_json):
            with open(hist_json, 'r') as f:
                history = json.load(f)
    else:
        print(f"Loading pretrained weights from {pretrained_ckpt}...")
        state_dict = torch.load(pretrained_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    print("Starting training...")
    
    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        epoch_start = time.time()
        
        # Training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for imgs, voxels in pbar:
            imgs, voxels = imgs.to(device), voxels.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                preds = model(imgs)
                loss = criterion(preds, voxels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        train_loss /= len(train_ds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for imgs, voxels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                imgs, voxels = imgs.to(device), voxels.to(device)
                
                with torch.amp.autocast('cuda'):
                    preds = model(imgs)
                    loss = criterion(preds, voxels)
                    
                val_loss += loss.item() * imgs.size(0)
                
                # Batch IoU
                v_iou = calc_iou(preds, voxels)
                val_iou += v_iou * imgs.size(0)
                
        val_loss /= len(val_ds)
        val_iou /= len(val_ds)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch}/{epochs} | Time: {epoch_time:.1f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
        
        # Record
        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "lr": lr,
            "time_sec": epoch_time
        }
        history.append(epoch_stats)
        
        # Save History
        with open(os.path.join(out_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=4)
            
        with open(os.path.join(out_dir, "training_history.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=epoch_stats.keys())
            writer.writeheader()
            writer.writerows(history)
            
        # Checkpointing
        ckpt_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_iou': best_iou
        }
        
        torch.save(ckpt_state, last_ckpt)
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(ckpt_state, best_ckpt)
            print(f" => New best model saved! (IoU: {best_iou:.4f})")
            
    # Final Report
    print("\n--- FINAL TRAINING REPORT ---")
    print(f"Completed Epochs: {epochs}")
    print(f"Best Validation IoU: {best_iou:.4f}")
    if history:
        print(f"Final Train Loss: {history[-1]['train_loss']:.4f}")
        print(f"Final Val Loss: {history[-1]['val_loss']:.4f}")
    print(f"Checkpoints saved to: {out_dir}")
    print("-----------------------------\n")

if __name__ == "__main__":
    train_abo()
