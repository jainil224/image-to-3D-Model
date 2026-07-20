import os
import json
import random
import bpy
import math
import mathutils
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def setup_camera_and_lights(bbox_max_dim):
    # Add camera
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    
    # Distance based on bounding box
    dist = bbox_max_dim * 1.5
    cam_obj.location = (dist, -dist, dist * 0.8)
    
    # Point camera at origin
    from mathutils import Vector
    direction = -cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()
    
    # Add lights
    light_data = bpy.data.lights.new(name="Light1", type='SUN')
    light_data.energy = 2.0
    light_obj = bpy.data.objects.new(name="Light1", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.rotation_euler = (math.radians(45), 0, math.radians(45))
    
    light_data2 = bpy.data.lights.new(name="Light2", type='SUN')
    light_data2.energy = 1.0
    light_data2.color = (0.8, 0.9, 1.0)
    light_obj2 = bpy.data.objects.new(name="Light2", object_data=light_data2)
    bpy.context.collection.objects.link(light_obj2)
    light_obj2.rotation_euler = (math.radians(45), 0, math.radians(-135))

def render_glb(glb_path, output_path):
    clear_scene()
    
    # Import GLB
    bpy.ops.import_scene.gltf(filepath=glb_path)
    
    # Calculate bounding box
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not meshes:
        return False
        
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
    
    # Create empty to parent everything and transform
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    parent = bpy.context.active_object
    
    for obj in meshes:
        if obj.parent is None:
            obj.parent = parent
            
    # Normalize and center
    parent.location = (-center[0], -center[1], -center[2])
    scale_factor = 1.0 / max_dim
    parent.scale = (scale_factor, scale_factor, scale_factor)
    
    setup_camera_and_lights(1.0)
    
    # Render settings
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT' if hasattr(bpy.types.SceneEEVEE, "use_raytracing") else 'BLENDER_EEVEE'
    bpy.context.scene.render.resolution_x = 256
    bpy.context.scene.render.resolution_y = 256
    bpy.context.scene.render.filepath = output_path
    
    # Disable world background for neutral gray
    bpy.context.scene.world = bpy.data.worlds.new("World")
    bpy.context.scene.world.use_nodes = True
    bg_node = bpy.context.scene.world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs[0].default_value = (0.2, 0.2, 0.2, 1.0)
    
    bpy.ops.render.render(write_still=True)
    return True

def create_contact_sheet(images, output_path, cols=5):
    rows = math.ceil(len(images) / cols)
    if rows == 0: return
    
    w, h = images[0].size
    sheet = Image.new('RGB', (cols * w, rows * h), (50, 50, 50))
    
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        sheet.paste(img, (c * w, r * h))
        
    sheet.save(output_path)

def generate_visual_validation():
    base_dir = "D:/image to 3D model/ABO"
    out_dir = "D:/image to 3D model/outputs/abo_validation"
    os.makedirs(out_dir, exist_ok=True)
    
    report_path = os.path.join(base_dir, "validation_report.json")
    with open(report_path, 'r') as f:
        report = json.load(f)
        
    valid_ids = report.get("validated_ids", [])
    corrupt_id = list(report.get("corrupt_model_details", {}).keys())
    if corrupt_id:
        valid_ids = [vid for vid in valid_ids if vid != corrupt_id[0]]
        
    # Get metadata to find image paths
    manifest_path = os.path.join(base_dir, "subset", "manifest.jsonl")
    records = {}
    with open(manifest_path, 'r') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r['3dmodel_id']] = r
                
    selected = random.sample(valid_ids, min(20, len(valid_ids)))
    
    results = []
    
    for i, mid in enumerate(selected):
        r = records[mid]
        print(f"[{i+1}/20] Rendering {mid}...")
        
        glb_path = os.path.join(base_dir, r['model_path'])
        img_path = os.path.join(base_dir, r['image_path'])
        render_out = os.path.join(out_dir, f"render_{mid}.png")
        
        success = render_glb(glb_path, render_out)
        
        if success and os.path.exists(render_out):
            try:
                # Combine original and render
                orig_img = Image.open(img_path).convert('RGB').resize((256, 256))
                rend_img = Image.open(render_out).convert('RGB')
                
                combo = Image.new('RGB', (512, 256))
                combo.paste(orig_img, (0, 0))
                combo.paste(rend_img, (256, 0))
                
                # Draw label
                draw = ImageDraw.Draw(combo)
                draw.text((10, 10), f"ID: {mid}", fill=(255, 0, 0))
                
                combo_path = os.path.join(out_dir, f"compare_{mid}.png")
                combo.save(combo_path)
                results.append(combo)
            except Exception as e:
                print(f"Error combining images for {mid}: {e}")
        else:
            print(f"Failed to render {mid}")
            
    if results:
        sheet_path = os.path.join(out_dir, "contact_sheet.png")
        create_contact_sheet(results, sheet_path)
        print(f"Contact sheet saved to {sheet_path}")

if __name__ == "__main__":
    generate_visual_validation()
