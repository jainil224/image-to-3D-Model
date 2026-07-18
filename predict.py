"""
predict.py — Turn one photo into a downloadable 3D model (.glb / .obj)

Usage:
    python predict.py --image path/to/photo.jpg --checkpoint ./outputs/image_to_3d_model.pth --output ./outputs/result

This loads your trained model and runs it on a single new image (not from the
training dataset) to produce a real .glb file.
"""

import argparse
import os

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

import numpy as np
from data_utils import voxels_to_mesh


def get_dominant_colors(image_path, n_colors=5):
    """Extracts the most common colors from the input photo using simple k-means."""
    img = Image.open(image_path).convert("RGB").resize((100, 100))
    pixels = np.array(img).reshape(-1, 3).astype(np.float32)

    # simple k-means (no sklearn dependency needed)
    rng = np.random.default_rng(42)
    centers = pixels[rng.choice(len(pixels), n_colors, replace=False)]
    for _ in range(10):
        dists = np.linalg.norm(pixels[:, None] - centers[None], axis=2)
        labels = dists.argmin(axis=1)
        for k in range(n_colors):
            if (labels == k).any():
                centers[k] = pixels[labels == k].mean(axis=0)
    counts = np.bincount(labels, minlength=n_colors)
    order = np.argsort(-counts)
    return centers[order].astype(np.uint8)  # sorted most -> least common


def colorize_mesh_from_image(mesh, image_path):
    """Tints the mesh's vertices using the dominant colors of the input photo,
    varying by height (Y axis) so it isn't a single flat color."""
    colors = get_dominant_colors(image_path, n_colors=4)
    verts = mesh.vertices
    y = verts[:, 1]
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
    bucket = np.clip((y_norm * len(colors)).astype(int), 0, len(colors) - 1)
    vertex_colors = colors[bucket]
    mesh.visual.vertex_colors = vertex_colors
    return mesh


# ---- Same model architecture as train.py (must match exactly to load the checkpoint) ----

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


def load_model(checkpoint_path, device):
    model = ImageTo3D().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def image_to_glb(model, image_path, output_path_no_ext, device, image_size=128):
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_logits = model(img_tensor)
        pred_probs = torch.sigmoid(pred_logits)[0, 0].cpu().numpy()

    mesh = voxels_to_mesh(pred_probs)
    mesh = colorize_mesh_from_image(mesh, image_path)

    obj_path = output_path_no_ext + ".obj"
    glb_path = output_path_no_ext + ".glb"
    mesh.export(obj_path)
    mesh.export(glb_path)

    return obj_path, glb_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert one image to a 3D model")
    parser.add_argument("--image", type=str, required=True, help="Path to input photo")
    parser.add_argument("--checkpoint", type=str, default="./outputs/image_to_3d_model.pth")
    parser.add_argument("--output", type=str, default="./outputs/result", help="Output path WITHOUT extension")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device)
    obj_path, glb_path = image_to_glb(model, args.image, args.output, device)

    print(f"Done. Saved:\n  {obj_path}\n  {glb_path}")