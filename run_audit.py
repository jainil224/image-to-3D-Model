import os
import sys
import json
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Adjust imports for local modules
sys.path.append("D:/image to 3D model")
from train import ImageTo3D
from data_utils import read_binvox

def backup_checkpoint():
    ckpt_path = "D:/image to 3D model/outputs/image_to_3d_model.pth"
    backup_dir = "D:/image to 3D model/checkpoints/backup_before_abo"
    os.makedirs(backup_dir, exist_ok=True)
    if os.path.exists(ckpt_path):
        backup_path = os.path.join(backup_dir, "image_to_3d_model_backup.pth")
        shutil.copy2(ckpt_path, backup_path)
        return ckpt_path, backup_path
    return None, None

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

def run():
    report = []
    
    # 1. Backup
    orig_ckpt, backup_ckpt = backup_checkpoint()
    report.append(f"1. Existing checkpoint found? {'YES' if orig_ckpt else 'NO'}")
    report.append(f"2. Checkpoint path: {orig_ckpt}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 3. Architecture compat
    model = ImageTo3D().to(device)
    if orig_ckpt:
        state_dict = torch.load(orig_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        report.append("3. Architecture compatibility: COMPATIBLE (state_dict loaded successfully)")
    else:
        report.append("3. Architecture compatibility: N/A")
        
    # Dataset splits
    split_path = "D:/image to 3D model/data/ABOProcessed/dataset_split.json"
    with open(split_path, 'r') as f:
        splits = json.load(f)
        
    train_ids = splits['train']
    val_ids = splits['validation']
    test_ids = splits['test']
    
    report.append(f"4. Train/val/test counts: {len(train_ids)} / {len(val_ids)} / {len(test_ids)}")
    
    s_tr, s_v, s_te = set(train_ids), set(val_ids), set(test_ids)
    no_overlap = len(s_tr & s_v) == 0 and len(s_tr & s_te) == 0 and len(s_v & s_te) == 0
    report.append(f"5. Zero split overlap confirmed? {no_overlap}")
    
    data_dir = "D:/image to 3D model/data/ABOProcessed"
    train_ds = ABOImageVoxelDataset(train_ids, data_dir, mode="train")
    val_ds = ABOImageVoxelDataset(val_ids, data_dir, mode="val")
    test_ds = ABOImageVoxelDataset(test_ids, data_dir, mode="test")
    
    img, vox = train_ds[0]
    report.append(f"6. Input tensor shape: {list(img.shape)}")
    report.append(f"7. Target voxel shape: {list(vox.shape)}")
    
    # Voxel occupancy
    occ = []
    for i in range(min(50, len(train_ids))):
        _, v = train_ds[i]
        occ.append(v.mean().item())
    mean_occ = np.mean(occ)
    report.append(f"8. Training voxel occupancy statistics: ~{mean_occ*100:.2f}% (measured on subset)")
    
    pos_weight = (1.0 - mean_occ) / mean_occ
    report.append(f"9. Recommended loss: BCEWithLogitsLoss with pos_weight={pos_weight:.2f}")
    
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    report.append(f"10. GPU/CUDA status: {gpu_name} (Available: {torch.cuda.is_available()})")
    
    batch_size = 16
    report.append(f"11. Recommended batch size: {batch_size}")
    lr = 5e-5
    report.append(f"12. Recommended learning rate: {lr}")
    epochs = 20
    report.append(f"13. Recommended epochs: {epochs}")
    report.append(f"14. Fine-tuning strategy: Option A (End-to-end fine-tuning with low LR {lr}) because we want adaptation without destroying the pre-trained ShapeNet features.")
    
    # Dry Run
    dry_train_ids = train_ids[:20]
    dry_val_ids = val_ids[:5]
    d_train_ds = ABOImageVoxelDataset(dry_train_ids, data_dir, mode="train")
    d_val_ds = ABOImageVoxelDataset(dry_val_ids, data_dir, mode="val")
    d_train_loader = DataLoader(d_train_ds, batch_size=4, shuffle=True)
    d_val_loader = DataLoader(d_val_ds, batch_size=4, shuffle=False)
    
    pos_weight_tensor = torch.tensor([pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Running dry run...")
    model.train()
    total_loss = 0
    start_t = time.time()
    for imgs, voxels in d_train_loader:
        imgs, voxels = imgs.to(device), voxels.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, voxels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_time = time.time() - start_t
    init_train_loss = total_loss / len(d_train_loader)
    
    model.eval()
    val_loss = 0
    val_iou = 0
    with torch.no_grad():
        for imgs, voxels in d_val_loader:
            imgs, voxels = imgs.to(device), voxels.to(device)
            preds = model(imgs)
            val_loss += criterion(preds, voxels).item()
            val_iou += calc_iou(preds, voxels)
    init_val_loss = val_loss / len(d_val_loader)
    init_val_iou = val_iou / len(d_val_loader)
    
    report.append(f"15. Dry-run result: SUCCESS (Forward, backward, validation completed)")
    report.append(f"16. Initial train loss: {init_train_loss:.4f}")
    report.append(f"17. Initial validation loss: {init_val_loss:.4f}")
    report.append(f"18. Initial validation IoU: {init_val_iou:.4f}")
    
    # Checkpoint save/reload
    test_ckpt = "D:/image to 3D model/checkpoints/abo_finetune/test.pth"
    os.makedirs(os.path.dirname(test_ckpt), exist_ok=True)
    torch.save(model.state_dict(), test_ckpt)
    
    model2 = ImageTo3D().to(device)
    model2.load_state_dict(torch.load(test_ckpt, weights_only=True))
    report.append(f"19. Checkpoint save/reload test result: SUCCESS")
    
    # Estimate time: train_time was for 20 samples. Epoch has 360 samples. 
    # Time per epoch = train_time * (360/20)
    epoch_time = train_time * (360 / len(dry_train_ids))
    est_total_time = epoch_time * epochs
    report.append(f"20. Estimated full training time: {est_total_time/60:.2f} minutes for {epochs} epochs")
    
    report_out = "\n".join(report)
    with open("D:/image to 3D model/training_audit_report.txt", "w") as f:
        f.write(report_out)
        
    print(report_out)

if __name__ == "__main__":
    run()
