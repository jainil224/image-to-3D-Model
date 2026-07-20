import os
import glob
import data_utils
import numpy as np

vox_dir = "D:/image to 3D model/data/ShapeNetVox32"
render_dir = "D:/image to 3D model/data/ShapeNetRendering"

categories = os.listdir(vox_dir)
count = 0

print("Checking ShapeNet Voxel Conventions...")

for cat in categories:
    cat_path = os.path.join(vox_dir, cat)
    if not os.path.isdir(cat_path): continue
    
    models = os.listdir(cat_path)
    for model in models[:2]:
        binvox_path = os.path.join(cat_path, model, "model.binvox")
        if not os.path.exists(binvox_path): continue
        
        try:
            voxels = data_utils.read_binvox(binvox_path)
            shape = voxels.shape
            occ_count = np.sum(voxels)
            ratio = occ_count / float(np.prod(shape))
            
            print(f"[{count+1}] Cat: {cat} Model: {model}")
            print(f"    Shape: {shape}")
            print(f"    Occupied: {occ_count} / {np.prod(shape)} ({ratio*100:.2f}%)")
            
            try:
                mesh = data_utils.voxels_to_mesh(voxels)
                print(f"    Marching Cubes: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
                if len(mesh.vertices) > 0:
                    mins = np.min(mesh.vertices, axis=0)
                    maxs = np.max(mesh.vertices, axis=0)
                    extents = maxs - mins
                    print(f"    Extents [X,Y,Z]: {extents}")
            except Exception as e:
                print(f"    MC Error: {e}")
                
            print("-" * 30)
            count += 1
            if count >= 10:
                break
        except Exception as e:
            pass
            
    if count >= 10:
        break
