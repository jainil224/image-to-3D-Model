
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
