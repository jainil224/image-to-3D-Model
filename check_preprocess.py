import os
import json
import trimesh
import subprocess
import glob
from PIL import Image, ImageDraw, ImageFont

def create_contact_sheet(images, output_path, cols=1):
    rows = (len(images) + cols - 1) // cols
    if rows == 0: return
    
    w, h = images[0].size
    sheet = Image.new('RGB', (cols * w, rows * h), (50, 50, 50))
    
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        sheet.paste(img, (c * w, r * h))
        
    sheet.save(output_path)

blender_render_script = """
import bpy
import math
import sys
import mathutils

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

obj_path = sys.argv[-2]
out_img = sys.argv[-1]

clear_scene()
bpy.ops.wm.obj_import(filepath=obj_path)

meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
if not meshes: sys.exit(1)

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
max_dim = max(dims) if max(dims) > 0 else 1

bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
parent = bpy.context.active_object
for obj in meshes:
    if obj.parent is None:
        obj.parent = parent
        
parent.location = (-center[0], -center[1], -center[2])
scale_factor = 1.0 / max_dim
parent.scale = (scale_factor, scale_factor, scale_factor)

cam_data = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

# Use a dynamic distance for framing
dist = 1.6
cam_x = dist * math.cos(math.radians(30)) * math.sin(math.radians(45))
cam_y = dist * math.cos(math.radians(30)) * math.cos(math.radians(45))
cam_z = dist * math.sin(math.radians(30))

cam_obj.location = (cam_x, cam_y, cam_z)
direction = -cam_obj.location
cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

light_data = bpy.data.lights.new(name="Light1", type='SUN')
light_data.energy = 4.0
light_obj = bpy.data.objects.new(name="Light1", object_data=light_data)
bpy.context.collection.objects.link(light_obj)
light_obj.rotation_euler = cam_obj.rotation_euler

fill_data = bpy.data.lights.new(name="Light2", type='SUN')
fill_data.energy = 2.0
fill_obj = bpy.data.objects.new(name="Light2", object_data=fill_data)
bpy.context.collection.objects.link(fill_obj)
fill_obj.location = (-cam_x, -cam_y, cam_z)
fill_obj.rotation_euler = (-fill_obj.location).to_track_quat('-Z', 'Y').to_euler()

bpy.context.scene.world = bpy.data.worlds.new("World")
bpy.context.scene.world.use_nodes = True
bg_node = bpy.context.scene.world.node_tree.nodes.get("Background")
if bg_node:
    bg_node.inputs[0].default_value = (0.3, 0.3, 0.3, 1.0)
    
bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT' if hasattr(bpy.types.SceneEEVEE, "use_raytracing") else 'BLENDER_EEVEE'
bpy.context.scene.render.resolution_x = 256
bpy.context.scene.render.resolution_y = 256
bpy.context.scene.render.filepath = out_img

bpy.ops.render.render(write_still=True)
"""

def main():
    import sys
    if "D:/image to 3D model" not in sys.path:
        sys.path.append("D:/image to 3D model")
    import data_utils
    
    out_dir = "D:/image to 3D model/outputs/abo_preprocessing_validation"
    os.makedirs(out_dir, exist_ok=True)
    
    processed_dir = "D:/image to 3D model/data/ABOProcessed"
    mids = os.listdir(processed_dir)
    
    # Write blender script
    blender_script_path = "D:/image to 3D model/blender_recon.py"
    with open(blender_script_path, "w") as f:
        f.write(blender_render_script)
        
    manifest_path = "D:/image to 3D model/ABO/subset/manifest.jsonl"
    records = {}
    with open(manifest_path, 'r') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r['3dmodel_id']] = r
                
    combo_images = []
    
    for mid in mids:
        obj_dir = os.path.join(processed_dir, mid)
        if not os.path.isdir(obj_dir): continue
        
        print(f"Validating {mid}...")
        
        # Original Image
        orig_img_path = os.path.join("D:/image to 3D model/ABO", records[mid]['image_path'])
        orig_img = Image.open(orig_img_path).convert('RGB').resize((256, 256))
        
        # Views
        views = []
        for v in ["00.png", "06.png", "12.png", "18.png"]:
            vpath = os.path.join(obj_dir, "rendering", v)
            if os.path.exists(vpath):
                img = Image.open(vpath).convert('RGB').resize((256, 256))
                views.append(img)
            else:
                views.append(Image.new('RGB', (256, 256)))
                
        # Binvox reconstruction
        binvox_path = os.path.join(obj_dir, "model.binvox")
        voxels = data_utils.read_binvox(binvox_path)
        mesh = data_utils.voxels_to_mesh(voxels)
        
        tmp_obj = os.path.join(out_dir, f"tmp_{mid}.obj")
        mesh.export(tmp_obj)
        
        recon_img_path = os.path.join(out_dir, f"recon_{mid}.png")
        cmd = ["python", blender_script_path, tmp_obj, recon_img_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        recon_img = Image.open(recon_img_path).convert('RGB').resize((256, 256))
        
        # Combine horizontally: Orig | V00 | V06 | V12 | V18 | Recon
        total_w = 256 * 6
        combo = Image.new('RGB', (total_w, 256))
        
        combo.paste(orig_img, (0, 0))
        for idx, view_img in enumerate(views):
            combo.paste(view_img, (256 * (idx + 1), 0))
        combo.paste(recon_img, (256 * 5, 0))
        
        draw = ImageDraw.Draw(combo)
        draw.text((10, 10), f"ID: {mid}", fill=(255, 0, 0))
        draw.text((256*5 + 10, 10), "Reconstructed Mesh", fill=(0, 255, 0))
        
        combo_path = os.path.join(out_dir, f"validate_{mid}.png")
        combo.save(combo_path)
        combo_images.append(combo)
        
        if os.path.exists(tmp_obj): os.remove(tmp_obj)
        if os.path.exists(tmp_obj.replace(".obj", ".mtl")): os.remove(tmp_obj.replace(".obj", ".mtl"))
        
    if combo_images:
        sheet_path = os.path.join(out_dir, "contact_sheet.png")
        create_contact_sheet(combo_images, sheet_path)
        print(f"Contact sheet saved to {sheet_path}")

if __name__ == "__main__":
    main()
