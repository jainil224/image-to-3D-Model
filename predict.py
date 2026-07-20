import argparse
import os
import sys

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# We avoid torchvision and use PIL+numpy to match ToTensor() precisely
# ToTensor(): PIL (0-255) -> np.float32 (0.0-1.0), and transpose HWC -> CHW
def preprocess_image(image_path, image_size=128):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC to CHW
    tensor = torch.from_numpy(arr).unsqueeze(0)  # Add batch dim
    return tensor

from data_utils import voxels_to_mesh

# ---- Same model architecture as train.py ----
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def image_to_glb(model, image_path, output_path_no_ext, device, threshold=0.5, image_size=128):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_tensor = preprocess_image(image_path, image_size).to(device)
    
    print(f"Input tensor shape: {img_tensor.shape}")

    with torch.inference_mode():
        pred_logits = model(img_tensor)
        print(f"Output logits shape: {pred_logits.shape}")
        
        pred_probs = torch.sigmoid(pred_logits)[0, 0].cpu().numpy()

    print(f"Probabilities - Min: {pred_probs.min():.4f}, Max: {pred_probs.max():.4f}, Mean: {pred_probs.mean():.4f}")
    
    occupied_count = np.sum(pred_probs > threshold)
    total_voxels = pred_probs.size
    print(f"Occupied voxels (> {threshold}): {occupied_count} / {total_voxels} ({occupied_count/total_voxels*100:.2f}%)")

    if occupied_count == 0:
        raise ValueError("Model predicted an entirely empty voxel grid. Cannot generate mesh.")

    # Convert to binary voxel grid for voxels_to_mesh
    # Wait, voxels_to_mesh takes float values and applies marching cubes at `threshold` internally if we pass probs
    # It does: `verts, faces, normals, _ = measure.marching_cubes(padded, level=threshold)`
    # Passing probabilities directly allows smoother meshes, so we pass pred_probs.
    mesh = voxels_to_mesh(pred_probs, threshold=threshold)
    
    print(f"Generated Mesh - Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

    # We do NOT try to aggressively color the mesh since it often fails or causes unexpected bugs.
    # The network predicts geometry, not true textures.

    os.makedirs(os.path.dirname(output_path_no_ext) or '.', exist_ok=True)
    obj_path = output_path_no_ext + ".obj"
    glb_path = output_path_no_ext + ".glb"
    
    mesh.export(obj_path)
    mesh.export(glb_path)

    return obj_path, glb_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert one image to a 3D model")
    parser.add_argument("--image", type=str, required=True, help="Path to input photo")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output path WITHOUT extension")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for occupancy")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device)
    
    obj_path, glb_path = image_to_glb(model, args.image, args.output, device, threshold=args.threshold)
    
    print(f"Done. Saved:\n  {obj_path}\n  {glb_path}")