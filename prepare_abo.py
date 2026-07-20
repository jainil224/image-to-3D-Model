import os
import json
import time
import argparse
import numpy as np
import trimesh
import subprocess
import cv2
from PIL import Image

def write_binvox(voxel_grid, file_path):
    with open(file_path, 'wb') as f:
        f.write(b'#binvox 1\n')
        f.write(f'dim {voxel_grid.shape[0]} {voxel_grid.shape[1]} {voxel_grid.shape[2]}\n'.encode())
        f.write(b'translate 0.0 0.0 0.0\n')
        f.write(b'scale 1.0\n')
        f.write(b'data\n')
        
        flat_data = voxel_grid.flatten()
        if len(flat_data) == 0: return
            
        state = flat_data[0]
        count = 0
        for val in flat_data:
            if val == state and count < 255:
                count += 1
            else:
                f.write(bytes([int(state), count]))
                state = val
                count = 1
        if count > 0:
            f.write(bytes([int(state), count]))

def generate_solid_voxels(obj_path, resolution=32):
    mesh = trimesh.load(obj_path, force='mesh')
    
    # Mesh is already in [-0.5, 0.5] from Blender
    pitch = 1.0 / resolution
    vox_obj = mesh.voxelized(pitch=pitch)
    vox_obj = vox_obj.fill()
    
    points = vox_obj.points
    grid = np.zeros((resolution, resolution, resolution), dtype=bool)
    
    # Map points in [-0.5, 0.5] to [0, 31]
    indices = np.floor((points + 0.5) * resolution).astype(int)
    indices = np.clip(indices, 0, resolution - 1)
    
    # Blender axes: X=Right, Y=Depth, Z=Up
    # ShapeNet axes: 0=Depth, 1=Width, 2=Height
    # We map Blender Y -> Axis 0, Blender X -> Axis 1, Blender Z -> Axis 2
    # Check if we need to flip anything. 
    grid[indices[:, 1], indices[:, 0], indices[:, 2]] = True
    
    return grid

blender_script = """
import bpy
import math
import os
import sys
import json
import mathutils

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

glb_path = sys.argv[-3]
out_dir = sys.argv[-2]
tmp_obj = sys.argv[-1]

clear_scene()

bpy.ops.import_scene.gltf(filepath=glb_path)

for obj in bpy.context.scene.objects:
    if obj.type in ['CAMERA', 'LIGHT']:
        bpy.data.objects.remove(obj, do_unlink=True)

meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
if not meshes:
    sys.exit(1)

min_coords = [float('inf')] * 3
max_coords = [float('-inf')] * 3

for obj in meshes:
    for corner in obj.bound_box:
        world_corner = obj.matrix_world @ mathutils.Vector(corner)
        for i in range(3):
            if world_corner[i] < min_coords[i]: min_coords[i] = world_corner[i]
            if world_corner[i] > max_coords[i]: max_coords[i] = world_corner[i]
            
center = [(min_coords[i] + max_coords[i]) / 2 for i in range(3)]
dims = [max_coords[i] - min_coords[i] for i in range(3)]
max_dim = max(dims)
if max_dim == 0: max_dim = 1

bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
parent = bpy.context.active_object

for obj in meshes:
    if obj.parent is None:
        obj.parent = parent
        
# Normalize to [-0.5, 0.5]
parent.location = (-center[0], -center[1], -center[2])
scale_factor = 1.0 / max_dim
parent.scale = (scale_factor, scale_factor, scale_factor)

bpy.context.view_layer.update()

# Save metadata
with open(os.path.join(out_dir, "metadata.json"), "w") as f:
    json.dump({"center": center, "scale_factor": scale_factor, "dims": dims}, f)

# Export OBJ
bpy.ops.wm.obj_export(filepath=tmp_obj, export_triangulated_mesh=True, forward_axis='Y', up_axis='Z')

# Calculate the new bounding box corners in world space after normalization
norm_corners = []
for x in [-0.5, 0.5]:
    for y in [-0.5, 0.5]:
        for z in [-0.5, 0.5]:
            norm_corners.append(mathutils.Vector((x * dims[0]/max_dim, y * dims[1]/max_dim, z * dims[2]/max_dim)))

cam_data = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

# Lighting
bpy.context.scene.world = bpy.data.worlds.new("World")
bpy.context.scene.world.use_nodes = True
bg_node = bpy.context.scene.world.node_tree.nodes.get("Background")
if bg_node:
    bg_node.inputs[0].default_value = (0.9, 0.9, 0.9, 1.0) # Light grey background
    bg_node.inputs[1].default_value = 1.0 # Strength

light_data = bpy.data.lights.new(name="Key", type='SUN')
light_data.energy = 3.0
light_obj = bpy.data.objects.new(name="Key", object_data=light_data)
bpy.context.collection.objects.link(light_obj)

fill_data = bpy.data.lights.new(name="Fill", type='SUN')
fill_data.energy = 1.0
fill_obj = bpy.data.objects.new(name="Fill", object_data=fill_data)
bpy.context.collection.objects.link(fill_obj)

bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT' if hasattr(bpy.types.SceneEEVEE, "use_raytracing") else 'BLENDER_EEVEE'
bpy.context.scene.render.resolution_x = 256
bpy.context.scene.render.resolution_y = 256

import bpy_extras

def check_framing(cam_obj, corners):
    scene = bpy.context.scene
    margin = 0.15 * 256
    for corner in corners:
        co2d = bpy_extras.object_utils.world_to_camera_view(scene, cam_obj, corner)
        x_pix = co2d.x * 256
        y_pix = co2d.y * 256
        if x_pix < margin or x_pix > 256 - margin or y_pix < margin or y_pix > 256 - margin:
            return False
    return True

render_dir = os.path.join(out_dir, "rendering")
os.makedirs(render_dir, exist_ok=True)

for i in range(24):
    azimuth = (i * 15) % 360
    elevation = 20 if i < 12 else 45
    
    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)
    
    # Dynamic camera positioning
    dist = 1.5
    while dist < 10.0:
        cam_x = dist * math.cos(el_rad) * math.sin(az_rad)
        cam_y = dist * math.cos(el_rad) * math.cos(az_rad)
        cam_z = dist * math.sin(el_rad)
        cam_obj.location = (cam_x, cam_y, cam_z)
        
        direction = -cam_obj.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam_obj.rotation_euler = rot_quat.to_euler()
        
        bpy.context.view_layer.update()
        
        if check_framing(cam_obj, norm_corners):
            break
        dist += 0.2
        
    light_obj.location = cam_obj.location
    light_obj.rotation_euler = cam_obj.rotation_euler
    
    fill_obj.location = (-cam_x, -cam_y, cam_z)
    fill_direction = -fill_obj.location
    fill_obj.rotation_euler = fill_direction.to_track_quat('-Z', 'Y').to_euler()
    
    out_img = os.path.join(render_dir, f"{i:02d}.png")
    bpy.context.scene.render.filepath = out_img
    bpy.ops.render.render(write_still=True)
"""

def is_valid_image(img_path):
    try:
        img = Image.open(img_path).convert('L')
        arr = np.array(img)
        
        # Check if nearly black or blank
        if np.mean(arr) < 10 or np.var(arr) < 5:
            return False
            
        # Check margins
        edges = np.concatenate([arr[0,:], arr[-1,:], arr[:,0], arr[:,-1]])
        if np.var(edges) > 50: # The object is touching the boundary
            return False
            
        return True
    except:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-objects', type=int, default=5)
    args = parser.parse_args()

    base_dir = "D:/image to 3D model/ABO"
    out_dir = "D:/image to 3D model/data/ABOProcessed"
    
    blender_script_path = "D:/image to 3D model/blender_render.py"
    with open(blender_script_path, "w") as f:
        f.write(blender_script)
        
    report_path = os.path.join(base_dir, "validation_report.json")
    with open(report_path, 'r') as f:
        report = json.load(f)
        
    valid_ids = report.get("validated_ids", [])
    corrupt_id = list(report.get("corrupt_model_details", {}).keys())
    if corrupt_id:
        valid_ids = [vid for vid in valid_ids if vid != corrupt_id[0]]
        
    manifest_path = os.path.join(base_dir, "subset", "manifest.jsonl")
    records = {}
    with open(manifest_path, 'r') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r['3dmodel_id']] = r
                
    valid_records = [records[mid] for mid in valid_ids if mid in records]
    
    successful = 0
    
    # User requested to re-run only the SAME 5 objects.
    # We will pick the first 5 records
    test_records = valid_records[:args.max_objects]
    
    for idx, r in enumerate(test_records):
        mid = r['3dmodel_id']
        obj_out_dir = os.path.join(out_dir, mid)
        render_dir = os.path.join(obj_out_dir, "rendering")
        binvox_path = os.path.join(obj_out_dir, "model.binvox")
        
        print(f"[{idx+1}/{len(test_records)}] Processing {mid}...")
        
        os.makedirs(render_dir, exist_ok=True)
        glb_path = os.path.join(base_dir, r['model_path'])
        tmp_obj = os.path.join(obj_out_dir, "temp_normalized.obj")
        
        cmd = ["python", blender_script_path, glb_path, obj_out_dir, tmp_obj]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if not os.path.exists(tmp_obj):
            print(f"  -> Blender rendering failed.")
            continue
            
        # Voxelize
        voxels = generate_solid_voxels(tmp_obj)
        write_binvox(voxels, binvox_path)
        
        # Verify images
        valid_imgs = 0
        for i in range(24):
            img_path = os.path.join(render_dir, f"{i:02d}.png")
            if os.path.exists(img_path) and is_valid_image(img_path):
                valid_imgs += 1
                
        occ = np.sum(voxels)
        ratio = occ / 32768
        
        print(f"  -> Renders valid: {valid_imgs}/24, Occupancy: {ratio*100:.2f}%")
        
        if os.path.exists(tmp_obj): os.remove(tmp_obj)
        if os.path.exists(tmp_obj.replace(".obj", ".mtl")): os.remove(tmp_obj.replace(".obj", ".mtl"))
        successful += 1

if __name__ == "__main__":
    main()
