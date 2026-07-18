import os
import glob
import random
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from data_utils import read_binvox, voxels_to_mesh


CATEGORIES = {
    "02691156": "airplane",
    "02958343": "car",
}


class ImageVoxelDataset(Dataset):
    def __init__(self, samples, image_size=128):
        self.samples = samples
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        view_idx = random.randint(0, 23)
        img = Image.open(f"{s['render_dir']}/{view_idx:02d}.png").convert("RGB")
        img_tensor = self.transform(img)

        voxels = read_binvox(s["voxel_path"]).astype("float32")
        voxel_tensor = torch.from_numpy(voxels).unsqueeze(0)

        return img_tensor, voxel_tensor


class ImageEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = self.conv(x).flatten(1)
        return self.fc(x)


class VoxelDecoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 4, 2, 1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.ConvTranspose3d(128, 64, 4, 2, 1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.ConvTranspose3d(64, 1, 4, 2, 1),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 4, 4, 4)
        return self.deconv(x)


class ImageTo3D(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = ImageEncoder(latent_dim)
        self.decoder = VoxelDecoder(latent_dim)

    def forward(self, image):
        z = self.encoder(image)
        return self.decoder(z)


def build_samples(data_dir, categories=CATEGORIES):
    render_root = os.path.join(data_dir, "ShapeNetRendering")
    voxel_root = os.path.join(data_dir, "ShapeNetVox32")
    samples = []
    for cat_id in categories:
        model_dirs = glob.glob(os.path.join(render_root, cat_id, "*"))
        for model_dir in model_dirs:
            model_id = os.path.basename(model_dir)
            voxel_path = os.path.join(voxel_root, cat_id, model_id, "model.binvox")
            if os.path.exists(voxel_path):
                samples.append({
                    "render_dir": os.path.join(model_dir, "rendering"),
                    "voxel_path": voxel_path,
                    "category": categories[cat_id],
                })
    return samples


def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    samples = build_samples(args.data_dir)
    if len(samples) == 0:
        print("No samples found. Please download the ShapeNetRendering and ShapeNetVox32 folders into", args.data_dir)
        return

    random.seed(42)
    random.shuffle(samples)
    split = int(0.9 * len(samples))
    train_samples, val_samples = samples[:split], samples[split:]

    train_ds = ImageVoxelDataset(train_samples, image_size=args.image_size)
    val_ds = ImageVoxelDataset(val_samples, image_size=args.image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = ImageTo3D().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for imgs, voxels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs, voxels = imgs.to(device), voxels.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, voxels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, voxels in val_loader:
                imgs, voxels = imgs.to(device), voxels.to(device)
                pred = model(imgs)
                val_loss += criterion(pred, voxels).item() * imgs.size(0)
        val_loss /= len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "image_to_3d_model.pth")
    torch.save(model.state_dict(), ckpt_path)
    print("Saved checkpoint to", ckpt_path)

    # export a single prediction
    model.eval()
    sample = val_samples[0]
    img = Image.open(os.path.join(sample['render_dir'], "00.png")).convert("RGB")
    img_tensor = train_ds.transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_logits = model(img_tensor)
        pred_probs = torch.sigmoid(pred_logits)[0, 0].cpu().numpy()
        pred_voxels = pred_probs > 0.5

    mesh = voxels_to_mesh(pred_probs)
    mesh.export(os.path.join(args.output_dir, "prediction.obj"))
    mesh.export(os.path.join(args.output_dir, "prediction.glb"))
    print("Exported prediction.obj and prediction.glb to", args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train Image->3D voxel model")
    parser.add_argument('--data-dir', type=str, default='./data', help='Path to folder containing ShapeNetRendering and ShapeNetVox32')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--output-dir', type=str, default='./outputs')
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()

    train(args)
