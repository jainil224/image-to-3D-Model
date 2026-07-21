"""
Image-to-3D Voxel Reconstruction Model & Utilities
--------------------------------------------------
This module defines the upgraded 3D Reconstruction architecture:
- ResNet50-based Image Encoder (2048-d feature output)
- Conv3D-based Voxel Decoder (32x32x32 voxel grid output)
- Combined Loss (BCEWithLogitsLoss + DiceLoss)
- Evaluation Metrics (IoU, Dice, Precision, Recall)
- Utility Functions (count_parameters, get_device)
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


# =====================================================================
# 1. ResNet50 Image Encoder
# =====================================================================

class ResNetEncoder(nn.Module):
    """
    Pretrained ResNet50 encoder extracting a 2048-dimensional feature vector.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        # Remove the final fully-connected classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, 3, H, W)
        x = self.backbone(x)  # Output shape: (batch_size, 2048, 1, 1)
        return torch.flatten(x, 1)  # Output shape: (batch_size, 2048)


# =====================================================================
# 2. 3D Voxel Decoder
# =====================================================================

class VoxelDecoder(nn.Module):
    """
    3D ConvTranspose decoder converting 2048-d feature vector to a 32x32x32 voxel grid.
    """
    def __init__(self, latent_dim: int = 2048):
        super().__init__()
        # Project 2048 latent vector to 512 * 4 * 4 * 4
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4 * 4)

        self.deconv = nn.Sequential(
            # (512, 4, 4, 4) -> (256, 8, 8, 8)
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            # (256, 8, 8, 8) -> (128, 16, 16, 16)
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            # (128, 16, 16, 16) -> (64, 32, 32, 32)
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            # Final output layer: (64, 32, 32, 32) -> (1, 32, 32, 32)
            nn.ConvTranspose3d(64, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, 2048)
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4, 4)
        return self.deconv(x)  # Output shape: (batch_size, 1, 32, 32, 32)


# =====================================================================
# 3. Main ImageTo3D Model API
# =====================================================================

class ImageTo3D(nn.Module):
    """
    End-to-End Image to 3D Voxel Model.
    Input: RGB Image Tensor (batch_size, 3, H, W)
    Output: 3D Voxel Logits (batch_size, 1, 32, 32, 32)
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained)
        self.decoder = VoxelDecoder(latent_dim=2048)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        z = self.encoder(image)
        return self.decoder(z)


# =====================================================================
# 4. Loss Functions
# =====================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for 3D voxel segmentation to handle severe class imbalance.
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(pred_logits)
        probs_flat = probs.view(-1)
        target_flat = target.view(-1)

        intersection = (probs_flat * target_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (probs_flat.sum() + target_flat.sum() + self.smooth)
        return 1.0 - dice_score


_bce_loss_fn = nn.BCEWithLogitsLoss()
_dice_loss_fn = DiceLoss()

def combined_loss(pred: torch.Tensor, target: torch.Tensor, pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Combines Binary Cross-Entropy with Logits and Dice Loss.
    Supports optional pos_weight for class imbalance handling.
    """
    if pos_weight is not None:
        bce = nn.functional.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
    else:
        bce = _bce_loss_fn(pred, target)
    dice = _dice_loss_fn(pred, target)
    return bce + dice


# =====================================================================
# 5. Evaluation Metrics
# =====================================================================

def calculate_iou(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Calculates Intersection over Union (IoU) / Jaccard Index."""
    with torch.no_grad():
        probs = torch.sigmoid(pred_logits)
        preds = (probs > threshold).float()
        
        intersection = (preds * target).sum().item()
        union = preds.sum().item() + target.sum().item() - intersection
        return float((intersection + smooth) / (union + smooth))


def calculate_dice(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Calculates Dice Similarity Coefficient (DSC)."""
    with torch.no_grad():
        probs = torch.sigmoid(pred_logits)
        preds = (probs > threshold).float()
        
        intersection = (preds * target).sum().item()
        total = preds.sum().item() + target.sum().item()
        return float((2.0 * intersection + smooth) / (total + smooth))


def calculate_precision(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Calculates Voxel Reconstruction Precision."""
    with torch.no_grad():
        probs = torch.sigmoid(pred_logits)
        preds = (probs > threshold).float()
        
        tp = (preds * target).sum().item()
        fp = (preds * (1.0 - target)).sum().item()
        return float((tp + smooth) / (tp + fp + smooth))


def calculate_recall(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Calculates Voxel Reconstruction Recall / Sensitivity."""
    with torch.no_grad():
        probs = torch.sigmoid(pred_logits)
        preds = (probs > threshold).float()
        
        tp = (preds * target).sum().item()
        fn = ((1.0 - preds) * target).sum().item()
        return float((tp + smooth) / (tp + fn + smooth))


# =====================================================================
# 6. Helper Utilities
# =====================================================================

def count_parameters(model: nn.Module) -> int:
    """Counts and prints the trainable parameters of a PyTorch model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {trainable_params:,} | Total Parameters: {total_params:,}")
    return trainable_params


def get_device() -> torch.device:
    """Returns CUDA device if available, otherwise CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
