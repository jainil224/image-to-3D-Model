import os
import sys
import json
import time
import shutil
import subprocess
import traceback
import numpy as np
import random
import trimesh
from PIL import Image

sys.path.append("D:/image to 3D model")
from data_utils import read_binvox

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
    pitch = 1.0 / resolution
    vox_obj = mesh.voxelized(pitch=pitch)
    vox_obj = vox_obj.fill()
    points = vox_obj.points
    grid = np.zeros((resolution, resolution, resolution), dtype=bool)
    indices = np.floor((points + 0.5) * resolution).astype(int)
    indices = np.clip(indices, 0, resolution - 1)
    grid[indices[:, 1], indices[:, 0], indices[:, 2]] = True
    return grid

blender_script = """
import bpy
import math
import os
import sys
import json
import mathutils
import bpy_extras

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

glb_path = sys.argv[-4]
out_dir = sys.argv[-3]
tmp_obj = sys.argv[-2]
margin_pct = float(sys.argv[-1])

clear_scene()

try:
    bpy.ops.import_scene.gltf(filepath=glb_path)
except Exception as e:
    print(f"GLTF IMPORT ERROR: {e}")
    sys.exit(1)

for obj in bpy.context.scene.objects:
    if obj.type in ['CAMERA', 'LIGHT']:
        bpy.data.objects.remove(obj, do_unlink=True)

meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
if not meshes:
    print("NO MESHES FOUND")
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
        
parent.location = (-center[0], -center[1], -center[2])
scale_factor = 1.0 / max_dim
parent.scale = (scale_factor, scale_factor, scale_factor)

bpy.context.view_layer.update()

os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "metadata.json"), "w") as f:
    json.dump({"center": center, "scale_factor": scale_factor, "dims": dims}, f)

try:
    bpy.ops.wm.obj_export(filepath=tmp_obj, export_triangulated_mesh=True, forward_axis='Y', up_axis='Z')
except Exception as e:
    print(f"OBJ EXPORT ERROR: {e}")
    sys.exit(1)

norm_corners = []
for x in [-0.5, 0.5]:
    for y in [-0.5, 0.5]:
        for z in [-0.5, 0.5]:
            norm_corners.append(mathutils.Vector((x * dims[0]/max_dim, y * dims[1]/max_dim, z * dims[2]/max_dim)))

cam_data = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

bpy.context.scene.world = bpy.data.worlds.new("World")
bpy.context.scene.world.use_nodes = True
bg_node = bpy.context.scene.world.node_tree.nodes.get("Background")
if bg_node:
    bg_node.inputs[0].default_value = (0.9, 0.9, 0.9, 1.0)
    bg_node.inputs[1].default_value = 1.0

light_data = bpy.data.lights.new(name="Key", type='SUN')
light_data.energy = 3.0
light_obj = bpy.data.objects.new(name="Key", object_data=light_data)
bpy.context.collection.objects.link(light_obj)

fill_data = bpy.data.lights.new(name="Fill", type='SUN')
fill_data.energy = 1.0
fill_obj = bpy.data.objects.new(name="Fill", object_data=fill_data)
bpy.context.collection.objects.link(fill_obj)

bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT' if hasattr(bpy.types.SceneEEVEE, "use_raytracing") else 'BLENDER_EEVEE'
bpy.context.scene.render.film_transparent = True
bpy.context.scene.render.resolution_x = 256
bpy.context.scene.render.resolution_y = 256

def check_framing(cam_obj, corners):
    scene = bpy.context.scene
    margin = margin_pct * 256
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
    
    dist = 1.2
    while dist < 8.0:
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
    try:
        bpy.ops.render.render(write_still=True)
    except Exception as e:
        print(f"RENDER ERROR FRAME {i}: {e}")
        sys.exit(1)
"""

def is_valid_image(img_path):
    try:
        img_rgba = Image.open(img_path)
        if img_rgba.mode != 'RGBA':
            return False, "Not RGBA"
        fg = np.array(img_rgba)[:, :, 3] > 10
        rows = np.any(fg, axis=1)
        cols = np.any(fg, axis=0)
        if not np.any(rows):
            return False, "No foreground found"
        
        # Check object is not too small
        occupancy = np.sum(fg) / (256 * 256)
        if occupancy < 0.02:
            return False, "Object too small"
            
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        if rmin == 0 or rmax == 255 or cmin == 0 or cmax == 255:
            return False, "Object touches bounds"
        return True, ""
    except Exception as e:
        return False, str(e)

def main():
    base_dir = "D:/image to 3D model/ABO"
    out_dir = "D:/image to 3D model/data/ABOProcessed"
    
    blender_script_path = "D:/image to 3D model/blender_render_recovery.py"
    with open(blender_script_path, "w") as f:
        f.write(blender_script)
        
    backup_failures_file = os.path.join(out_dir, "preprocessing_failures_bulk_pass.json")
    recovery_failures_file = os.path.join(out_dir, "recovery_failures.json")
    
    if not os.path.exists(backup_failures_file):
        print(f"Error: Could not find {backup_failures_file}")
        sys.exit(1)
        
    with open(backup_failures_file, "r") as f:
        bulk_failures = json.load(f)
        
    failed_mids = [f["model_id"] for f in bulk_failures]
    print(f"Loaded {len(failed_mids)} failed models to recover.")
    
    manifest_path = os.path.join(base_dir, "subset", "manifest.jsonl")
    records = {}
    with open(manifest_path, 'r') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r['3dmodel_id']] = r
                
    valid_records_to_recover = [records[mid] for mid in failed_mids if mid in records]
    
    if os.path.exists(recovery_failures_file):
        with open(recovery_failures_file, "r") as f:
            recovery_failures = json.load(f)
    else:
        recovery_failures = []
        
    recovered_mids = []
    permanently_failed = []
    
    # We will track which margins were attempted for the recovery log
    margins_to_try = [0.20, 0.25, 0.30, 0.35, 0.40]
    
    start_time_real = time.time()
    
    for idx, r in enumerate(valid_records_to_recover):
        mid = r['3dmodel_id']
        obj_out_dir = os.path.join(out_dir, mid)
        render_dir = os.path.join(obj_out_dir, "rendering")
        binvox_path = os.path.join(obj_out_dir, "model.binvox")
        
        # Check if already recovered in a previous interrupted run
        if os.path.exists(binvox_path) and os.path.exists(render_dir) and len(os.listdir(render_dir)) == 24:
            recovered = True
            for i in range(24):
                if not is_valid_image(os.path.join(render_dir, f"{i:02d}.png"))[0]:
                    recovered = False
                    break
            if recovered:
                try:
                    vox = read_binvox(binvox_path)
                    if vox.shape == (32, 32, 32) and np.sum(vox) > 0:
                        recovered_mids.append(mid)
                        continue
                except:
                    pass
                    
        print(f"Recovery [{idx+1}/{len(valid_records_to_recover)}] Processing {mid}...")
        
        glb_path = os.path.join(base_dir, r['model_path'])
        if not os.path.exists(glb_path):
            print(f"  -> Failed: GLB missing at {glb_path}")
            permanently_failed.append(mid)
            continue
            
        if os.path.exists(obj_out_dir):
            shutil.rmtree(obj_out_dir)
        os.makedirs(render_dir, exist_ok=True)
            
        tmp_obj = os.path.join(obj_out_dir, "temp_normalized.obj")
        
        t0 = time.time()
        success = False
        margins_attempted = []
        final_error = None
        failed_frame = None
        
        # Adaptive margin loop
        for margin in margins_to_try:
            print(f"  -> Attempting margin {margin*100:.0f}%")
            margins_attempted.append(margin)
            
            try:
                cmd = ["python", blender_script_path, glb_path, obj_out_dir, tmp_obj, str(margin)]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                
                if proc.returncode != 0 or not os.path.exists(tmp_obj):
                    raise Exception(f"Blender failed: {proc.stderr[-500:]}")
                    
                # Verify images
                images_valid = True
                for i in range(24):
                    img_path = os.path.join(render_dir, f"{i:02d}.png")
                    if not os.path.exists(img_path):
                        raise Exception(f"Missing frame {i}")
                    valid, msg = is_valid_image(img_path)
                    if not valid:
                        images_valid = False
                        final_error = msg
                        failed_frame = i
                        break
                        
                if images_valid:
                    success = True
                    break
            except Exception as e:
                final_error = str(e)
                break
                
        if success:
            try:
                # Reuse binvox if possible, otherwise generate
                voxels = generate_solid_voxels(tmp_obj)
                occ = np.sum(voxels)
                if occ == 0:
                    raise Exception("Voxel occupancy is 0")
                write_binvox(voxels, binvox_path)
                
                if os.path.exists(tmp_obj): os.remove(tmp_obj)
                mtl = tmp_obj.replace(".obj", ".mtl")
                if os.path.exists(mtl): os.remove(mtl)
                
                recovered_mids.append(mid)
                print("  -> Recovered!")
            except Exception as e:
                print(f"  -> Failed voxelization: {e}")
                permanently_failed.append(mid)
                final_error = str(e)
                success = False
                
        if not success:
            print(f"  -> Still failed at max margin: {final_error}")
            permanently_failed.append(mid)
            
            # Find original error
            orig_error = "Unknown"
            for f in bulk_failures:
                if f["model_id"] == mid:
                    orig_error = f["error"]
                    break
                    
            recovery_failures.append({
                "model_id": mid,
                "original_error": orig_error,
                "margins_attempted": margins_attempted,
                "stage": "recovery",
                "failed_frame": failed_frame,
                "final_error": final_error,
                "timestamp": time.time()
            })
            
            with open(recovery_failures_file, "w") as f:
                json.dump(recovery_failures, f, indent=4)
                
        # Progress update
        elapsed = time.time() - start_time_real
        total_processed = len(recovered_mids) + len(permanently_failed)
        avg_time = elapsed / max(1, total_processed)
        remaining = len(valid_records_to_recover) - total_processed
        est_rem = remaining * avg_time
        
        print(f"  Status: {len(recovered_mids)} recovered, {len(permanently_failed)} still failed, {remaining} remaining.")
        print(f"  Avg time: {avg_time:.1f}s. Est rem: {est_rem/60:.1f}m\n")

    print("\n--- RECOVERY COMPLETE ---")
    print(f"Total targeted: {len(valid_records_to_recover)}")
    print(f"Recovered: {len(recovered_mids)}")
    print(f"Permanently Failed: {len(permanently_failed)}")

if __name__ == "__main__":
    main()
