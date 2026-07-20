import trimesh
import numpy as np
import sys

def test_solid_vox(obj_path):
    mesh = trimesh.load(obj_path, force='mesh')
    
    # Scale to [-0.5, 0.5]
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    extents = bounds[1] - bounds[0]
    max_dim = np.max(extents)
    if max_dim == 0: max_dim = 1
    
    mesh.apply_translation(-center)
    mesh.apply_scale(1.0 / max_dim)
    
    # Voxelize
    pitch = 1.0 / 32.0
    vox_obj = mesh.voxelized(pitch=pitch)
    
    # Fill interior
    vox_obj = vox_obj.fill()
    
    # Extract 32x32x32 grid
    grid = np.zeros((32, 32, 32), dtype=bool)
    
    # Map vox_obj points to grid
    points = vox_obj.points
    indices = np.floor((points + 0.5) * 32).astype(int)
    indices = np.clip(indices, 0, 31)
    
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    
    occ = np.sum(grid)
    ratio = occ / (32**3)
    print(f"Solid Occupancy: {occ} / 32768 ({ratio*100:.2f}%)")

if __name__ == "__main__":
    test_solid_vox("D:/image to 3D model/scratch/recon_1005ca47e516495512da0dbf3c68e847.obj")
    # Test on an ABO model we normalized earlier
    test_solid_vox("D:/image to 3D model/ABO/subset/B07S74D9T7/model.glb")
