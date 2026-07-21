"""
ImageTo3D Inference Script
--------------------------
Loads a trained ImageTo3D model and runs inference on a single input image.
Converts the output voxel probabilities to a 3D mesh (OBJ format) using Marching Cubes.
"""
import os
import argparse
import torch
import torchvision.transforms as T
from PIL import Image

from train import ImageTo3D, get_device
from data_utils import voxels_to_mesh


def run_inference(image_path: str, weights_path: str, output_path: str, threshold: float = 0.5):
    device = get_device()
    print(f"Using device: {device}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}. Please train the model first.")
        
    print("Loading model...")
    # Load model (pretrained=False as we are loading our own trained weights)
    model = ImageTo3D(pretrained=False).to(device)
    
    # Load state dict
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    
    # Handle both standalone weights and full checkpoint dictionaries
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
        
    model.load_state_dict(state_dict)
    model.eval()
    
    print("Processing image...")
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    print("Generating voxel grid...")
    with torch.no_grad():
        # Predict logits
        logits = model(input_tensor)
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        
    # Extract the voxel grid (shape: 32, 32, 32)
    voxel_grid = probs.squeeze().cpu().numpy()
    
    print("Converting voxel to mesh...")
    # Convert probability grid to a Trimesh object
    mesh = voxels_to_mesh(voxel_grid, threshold=threshold)
    
    print("Exporting OBJ...")
    mesh.export(output_path)
    
    glb_output_path = os.path.splitext(output_path)[0] + ".glb"
    print("Exporting GLB...")
    mesh.export(glb_output_path)
    
    print("Inference completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ImageTo3D Inference")
    parser.add_argument("-i", "--image", type=str, required=True, help="Path to input RGB image")
    parser.add_argument("-w", "--weights", type=str, default="checkpoints/abo_resnet50/image_to_3d_best_weights.pth", help="Path to trained model weights")
    parser.add_argument("-o", "--output", type=str, default="output.obj", help="Path to save output 3D mesh (e.g., .obj or .ply)")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Voxel probability threshold (default: 0.5)")
    
    args = parser.parse_args()
    run_inference(args.image, args.weights, args.output, args.threshold)
