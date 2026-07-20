import os
import numpy as np
import trimesh
import glob

# Ensure data_utils is in path
import sys
sys.path.append("D:/image to 3D model")
import data_utils

def analyze_shapenet(cat_id, model_id):
    base_dir = "D:/image to 3D model/data"
    vox_path = os.path.join(base_dir, f"ShapeNetVox32/{cat_id}/{model_id}/model.binvox")
    render_dir = os.path.join(base_dir, f"ShapeNetRendering/{cat_id}/{model_id}/rendering")
    
    if not os.path.exists(vox_path):
        print(f"File not found: {vox_path}")
        return
        
    print(f"--- Analyzing {cat_id} / {model_id} ---")
    voxels = data_utils.read_binvox(vox_path)
    
    print(f"Shape: {voxels.shape}")
    occ = np.sum(voxels)
    total = np.prod(voxels.shape)
    print(f"Occupancy: {occ} / {total} ({occ/total*100:.2f}%)")
    
    # Check if solid by looking at a center slice
    # For a solid object, the interior should be True. 
    # For a hollow object, the interior is False.
    # Let's take 3 center slices across all axes and count True vs False
    cx, cy, cz = voxels.shape[0]//2, voxels.shape[1]//2, voxels.shape[2]//2
    slice_x = voxels[cx, :, :]
    slice_y = voxels[:, cy, :]
    slice_z = voxels[:, :, cz]
    
    print(f"Center slice X occupancy: {np.sum(slice_x)}")
    print(f"Center slice Y occupancy: {np.sum(slice_y)}")
    print(f"Center slice Z occupancy: {np.sum(slice_z)}")
    
    # Let's check a ray through the center
    ray_x = voxels[:, cy, cz]
    print(f"Ray through X axis: {''.join(['1' if v else '0' for v in ray_x])}")
    ray_y = voxels[cx, :, cz]
    print(f"Ray through Y axis: {''.join(['1' if v else '0' for v in ray_y])}")
    ray_z = voxels[cx, cy, :]
    print(f"Ray through Z axis: {''.join(['1' if v else '0' for v in ray_z])}")
    
    # Check renders
    renders = glob.glob(os.path.join(render_dir, "*.png"))
    print(f"Found {len(renders)} renders.")

    # Reconstruct mesh
    mesh = data_utils.voxels_to_mesh(voxels)
    tmp_obj = f"D:/image to 3D model/scratch/recon_{model_id}.obj"
    mesh.export(tmp_obj)
    print(f"Exported reconstructed mesh to {tmp_obj}")
    
if __name__ == "__main__":
    # Categories: 02691156 (airplanes), 02958343 (cars), 03001627 (chairs), 02933112 (cabinets)
    analyze_shapenet("02958343", "1005ca47e516495512da0dbf3c68e847") # car
    analyze_shapenet("03001627", "1006be65e7bc937e9141f9b58470d646") # chair
    analyze_shapenet("02933112", "1055dc4f3f2079f7e6c5cd45aa112726") # cabinet
