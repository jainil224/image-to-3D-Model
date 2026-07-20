import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import sys

# Ensure data_utils and train are accessible
sys.path.append("D:/image to 3D model")
from data_utils import read_binvox
from train import ImageTo3D

def test_pipeline():
    print("Testing ML Pipeline compatibility with ABO object B07S74D9T7...")
    
    mid = "B07S74D9T7"
    obj_dir = f"D:/image to 3D model/data/ABOProcessed/{mid}"
    img_path = f"{obj_dir}/rendering/00.png"
    vox_path = f"{obj_dir}/model.binvox"
    
    if not os.path.exists(img_path) or not os.path.exists(vox_path):
        print("Required ABO files not found. Are you sure they were generated?")
        return
        
    print("\n--- STEP 4: Test DataLoader logic ---")
    img = Image.open(img_path).convert("RGB")
    
    # Same transforms as train.py
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
    ])
    
    img_tensor = transform(img)
    print(f"Image tensor shape: {list(img_tensor.shape)}")
    
    # Voxel load
    voxels = read_binvox(vox_path).astype("float32")
    voxel_tensor = torch.from_numpy(voxels).unsqueeze(0)
    print(f"Voxel tensor shape: {list(voxel_tensor.shape)}")
    
    # DataLoader Batch Size 1 simulation
    img_batch = img_tensor.unsqueeze(0)
    vox_batch = voxel_tensor.unsqueeze(0)
    
    print(f"\nBatch Shapes:")
    print(f"Image batch: {list(img_batch.shape)}, dtype: {img_batch.dtype}, min: {img_batch.min().item():.2f}, max: {img_batch.max().item():.2f}")
    print(f"Voxel batch: {list(vox_batch.shape)}, dtype: {vox_batch.dtype}, min: {vox_batch.min().item():.2f}, max: {vox_batch.max().item():.2f}")
    
    print("\n--- STEP 5: Forward / Loss / Backward ---")
    model = ImageTo3D()
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    try:
        optimizer.zero_grad()
        print("Running forward pass...")
        pred = model(img_batch)
        print(f"Prediction shape: {list(pred.shape)}")
        
        loss = criterion(pred, vox_batch)
        print(f"Loss is finite: {torch.isfinite(loss).item()}, Value: {loss.item():.4f}")
        
        print("Running backward pass...")
        loss.backward()
        
        # Check gradients
        has_grads = any(p.grad is not None for p in model.parameters())
        print(f"Gradients computed: {has_grads}")
        
        optimizer.step()
        print("Optimizer step successful. Pipeline is compatible!")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    test_pipeline()
