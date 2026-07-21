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

glb_path = sys.argv[-3]
out_dir = sys.argv[-2]
tmp_obj = sys.argv[-1]

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
    if not os.path.exists(img_path):
        return False, "Image file does not exist"
    try:
        with Image.open(img_path) as img_rgba:
            img_rgba.load()
            if img_rgba.mode != 'RGBA':
                return False, f"Not RGBA mode (got {img_rgba.mode})"
            alpha = np.array(img_rgba)[:, :, 3]
            if not np.any(alpha > 10):
                return False, "No foreground found"
            return True, ""
    except Exception as e:
        return False, f"Corrupted or invalid image: {str(e)}"

def check_disk_space():
    total, used, free = shutil.disk_usage("D:/")
    free_gb = free // (2**30)
    if free_gb < 10:
        return False, free_gb
    return True, free_gb

def is_completed(obj_out_dir):
    render_dir = os.path.join(obj_out_dir, "rendering")
    binvox_path = os.path.join(obj_out_dir, "model.binvox")
    metadata_path = os.path.join(obj_out_dir, "metadata.json")
    
    if not os.path.exists(metadata_path): return False
    if not os.path.exists(binvox_path): return False
    if not os.path.exists(render_dir): return False
    
    for i in range(24):
        if not os.path.exists(os.path.join(render_dir, f"{i:02d}.png")): return False
        
    try:
        vox = read_binvox(binvox_path)
        if vox.shape != (32, 32, 32): return False
        if np.sum(vox) == 0: return False
    except Exception:
        return False
        
    return True

def main():
    base_dir = "D:/image to 3D model/ABO"
    out_dir = "D:/image to 3D model/data/ABOProcessed5000"
    
    blender_script_path = "D:/image to 3D model/blender_render_full.py"
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
    
    os.makedirs(out_dir, exist_ok=True)
    failures_file = os.path.join(out_dir, "preprocessing_failures.json")
    progress_file = os.path.join(out_dir, "preprocessing_progress.json")
    
    if os.path.exists(failures_file):
        try:
            with open(failures_file, "r") as f:
                failures = json.load(f)
        except Exception:
            failures = []
    else:
        failures = []
        
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                progress = json.load(f)
        except Exception:
            progress = {
                "total_target": len(valid_records),
                "completed": 0,
                "failed": 0,
                "elapsed_time": 0.0
            }
    else:
        progress = {
            "total_target": len(valid_records),
            "completed": 0,
            "failed": 0,
            "elapsed_time": 0.0
        }
    
    start_time_real = time.time()

    # ============================================================
    # RESUME SUPPORT - rebuild state from actual processed files.
    # Safe across Ctrl+C, terminal close, reboot, and shutdown.
    # ============================================================
    completed_mids = []
    failed_mids = []
    occ_ratios = []

    print("\nScanning existing processed objects for resume...")

    for r in valid_records:
        mid = r["3dmodel_id"]
        obj_out_dir = os.path.join(out_dir, mid)
        binvox_path = os.path.join(obj_out_dir, "model.binvox")

        try:
            if is_completed(obj_out_dir):
                completed_mids.append(mid)
                try:
                    vox = read_binvox(binvox_path)
                    if vox.shape == (32, 32, 32):
                        occ_ratios.append(float(np.sum(vox)) / 32768.0)
                except Exception:
                    pass
        except Exception:
            pass

    completed_mids = list(dict.fromkeys(completed_mids))
    existing_completed = len(completed_mids)

    print(f"Found {existing_completed} already completed valid objects.")
    print(f"Remaining objects to process: {len(valid_records) - existing_completed}")

    # Current-run failures only. Failed objects will be retried next run.
    failed_mids = []

    # Never trust stale completion counters after an interrupted run.
    progress["total_target"] = len(valid_records)
    progress["completed"] = existing_completed
    progress["failed"] = 0
    progress["remaining"] = len(valid_records) - existing_completed
    progress["elapsed_time"] = 0.0
    progress["average_time_per_object"] = 0.0

    temp_progress_file = progress_file + ".tmp"
    with open(temp_progress_file, "w") as f:
        json.dump(progress, f, indent=4)
    os.replace(temp_progress_file, progress_file)

    print(f"Resume initialized: {existing_completed}/{len(valid_records)} completed.")

    for idx, r in enumerate(valid_records):
        mid = r['3dmodel_id']
        obj_out_dir = os.path.join(out_dir, mid)
        render_dir = os.path.join(obj_out_dir, "rendering")
        binvox_path = os.path.join(obj_out_dir, "model.binvox")
        
        if is_completed(obj_out_dir):
            if mid not in completed_mids:
                completed_mids.append(mid)
            try:
                vox = read_binvox(binvox_path)
                occ_ratios.append(np.sum(vox) / 32768)
            except Exception: pass
            continue
            
        print(f"[{len(completed_mids)+len(failed_mids)}/{len(valid_records)}] Processing {mid}...")
        
        disk_ok, free_gb = check_disk_space()
        if not disk_ok:
            print(f"DISK SPACE LOW ({free_gb} GB free). Stopping.")
            break
            
        if os.path.exists(obj_out_dir):
            shutil.rmtree(obj_out_dir)
        os.makedirs(render_dir, exist_ok=True)
            
        glb_path = os.path.join(base_dir, r['model_path'])
        tmp_obj = os.path.join(obj_out_dir, "temp_normalized.obj")
        
        t0 = time.time()
        
        try:
            # Render and output obj with timeout protection
            cmd = ["python", blender_script_path, glb_path, obj_out_dir, tmp_obj]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired as te:
                blender_log = (
                    te.stderr[-1000:] if te.stderr and te.stderr.strip()
                    else (te.stdout[-1000:] if te.stdout else "Process timed out after 300s")
                )
                raise Exception(f"Blender timed out after 300s:\n{blender_log}")
            
            if proc.returncode != 0 or not os.path.exists(tmp_obj):
                blender_log = (
                    proc.stderr[-1000:] if proc.stderr and proc.stderr.strip()
                    else (proc.stdout[-1000:] if proc.stdout else "No output")
                )
                raise Exception(f"Blender failed:\n{blender_log}")
                
            # Verify images
            for i in range(24):
                img_path = os.path.join(render_dir, f"{i:02d}.png")
                if not os.path.exists(img_path):
                    raise Exception(f"Missing frame {i}")
                valid, msg = is_valid_image(img_path)
                if not valid:
                    raise Exception(f"Invalid frame {i}: {msg}")
                    
            # Voxelize
            voxels = generate_solid_voxels(tmp_obj)
            occ = np.sum(voxels)
            if occ == 0:
                raise Exception("Voxel occupancy is 0")
                
            write_binvox(voxels, binvox_path)
            
            # Read verification
            vox_test = read_binvox(binvox_path)
            if vox_test.shape != (32, 32, 32):
                raise Exception(f"Voxel shape incorrect: {vox_test.shape}")
                
            if os.path.exists(tmp_obj): os.remove(tmp_obj)
            mtl = tmp_obj.replace(".obj", ".mtl")
            if os.path.exists(mtl): os.remove(mtl)
            
            completed_mids.append(mid)
            occ_ratios.append(occ / 32768)
            
        except Exception as e:
            err = traceback.format_exc()
            print(f"  -> Failed: {e}")
            failed_mids.append(mid)
            failures.append({
                "model_id": mid,
                "stage": "pipeline",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": time.time()
            })
            temp_failures_file = failures_file + ".tmp"
            with open(temp_failures_file, "w") as f:
                json.dump(failures, f, indent=4)
            os.replace(temp_failures_file, failures_file)
                
        # ============================================================
        # PROGRESS UPDATE
        # ============================================================
        elapsed_loop = time.time() - t0
        progress["elapsed_time"] += elapsed_loop

        completed_mids = list(dict.fromkeys(completed_mids))
        failed_mids = list(dict.fromkeys(failed_mids))

        completed_count = len(completed_mids)
        failed_count = len(failed_mids)

        # Failed objects remain unfinished so they can be retried later.
        remaining_count = len(valid_records) - completed_count

        progress["total_target"] = len(valid_records)
        progress["completed"] = completed_count
        progress["failed"] = failed_count
        progress["remaining"] = remaining_count
        progress["last_completed_model"] = mid

        # Estimate timing only from work attempted in this run.
        session_attempted = completed_count - existing_completed + failed_count
        avg_time = (
            progress["elapsed_time"] / session_attempted
            if session_attempted > 0
            else 0.0
        )
        progress["average_time_per_object"] = avg_time

        # Atomic progress save: protects JSON if execution is interrupted.
        temp_progress_file = progress_file + ".tmp"
        with open(temp_progress_file, "w") as f:
            json.dump(progress, f, indent=4)
        os.replace(temp_progress_file, progress_file)

        if session_attempted > 0 and session_attempted % 10 == 0:
            est_rem = remaining_count * avg_time
            print(
                f"  Checkpoint: {completed_count} completed, "
                f"{failed_count} failed this session, "
                f"{remaining_count} remaining. "
                f"Avg: {avg_time:.1f}s. "
                f"Est rem: {est_rem/60:.1f}m"
            )

    # Final splitting and report
    print("\n--- FINAL INTEGRITY AND REPORTING ---")
    
    # Split
    if len(completed_mids) > 0:
        completed_mids.sort()
        random.seed(42)
        shutil_mids = list(completed_mids)
        random.shuffle(shutil_mids)
        
        total_c = len(shutil_mids)
        train_c = int(total_c * 0.8)
        val_c = int(total_c * 0.1)
        test_c = total_c - train_c - val_c
        
        split = {
            "seed": 42,
            "train": shutil_mids[:train_c],
            "validation": shutil_mids[train_c:train_c+val_c],
            "test": shutil_mids[train_c+val_c:]
        }
        split_path = os.path.join(out_dir, "dataset_split.json")
        with open(split_path, "w") as f:
            json.dump(split, f, indent=4)
    else:
        split_path = "None"
        train_c = val_c = test_c = 0

    _, free_gb = check_disk_space()
    
    report_lines = [
        "1. Target valid objects: " + str(len(valid_records)),
        "2. Successfully processed objects: " + str(len(completed_mids)),
        "3. Failed objects: " + str(len(failed_mids)),
        "4. Failed model IDs + reasons: " + json.dumps([{f['model_id']: f['error']} for f in failures]),
        "5. Total PNG renders generated: " + str(len(completed_mids) * 24),
        "6. Expected renders vs actual renders: " + f"{len(valid_records)*24} vs {len(completed_mids)*24}",
        "7. Valid model.binvox count: " + str(len(completed_mids)),
        "8. Invalid voxel count: " + str(len(failed_mids)),
        "9. Average occupancy ratio: " + (f"{np.mean(occ_ratios)*100:.2f}%" if occ_ratios else "0%"),
        "10. Minimum/maximum occupancy ratio: " + (f"{np.min(occ_ratios)*100:.2f}% / {np.max(occ_ratios)*100:.2f}%" if occ_ratios else "0% / 0%"),
        "11. Total preprocessing duration: " + f"{progress['elapsed_time'] / 60:.2f} mins",
        "12. Average time per object: " + f"{progress['average_time_per_object']:.2f} s",
        "13. Final output disk usage (rough estimate): ~10GB",
        "14. Remaining D: free space: " + str(free_gb) + " GB",
        "15. Train object count: " + str(train_c),
        "16. Validation object count: " + str(val_c),
        "17. Test object count: " + str(test_c),
        "18. Path to dataset_split.json: " + split_path,
        "19. Final status: " + ("DATASET READY FOR TRAINING" if len(completed_mids) > 400 else "DATASET NOT READY - Too many failures")
    ]
    
    with open(os.path.join(out_dir, "final_preprocessing_report.txt"), "w") as f:
        f.write("\n".join(report_lines))
        
    print("\nFINAL REPORT GENERATED: " + os.path.join(out_dir, "final_preprocessing_report.txt"))
    for line in report_lines:
        print(line)

if __name__ == "__main__":
    main()
