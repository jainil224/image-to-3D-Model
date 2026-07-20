
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

# Import GLB
bpy.ops.import_scene.gltf(filepath=glb_path)

# Delete existing cameras and lights
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

# Update scene
bpy.context.view_layer.update()

# Save metadata
with open(os.path.join(out_dir, "metadata.json"), "w") as f:
    json.dump({"center": center, "scale_factor": scale_factor, "dims": dims}, f)

# Export normalized OBJ for perfect voxelization alignment
bpy.ops.wm.obj_export(filepath=tmp_obj, export_triangulated_mesh=True, forward_axis='Y', up_axis='Z')

# Setup Camera and lights
cam_data = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

# Setup lights
light_data = bpy.data.lights.new(name="Light1", type='SUN')
light_data.energy = 2.0
light_obj = bpy.data.objects.new(name="Light1", object_data=light_data)
bpy.context.collection.objects.link(light_obj)

light_data2 = bpy.data.lights.new(name="Light2", type='SUN')
light_data2.energy = 1.0
light_data2.color = (0.8, 0.9, 1.0)
light_obj2 = bpy.data.objects.new(name="Light2", object_data=light_data2)
bpy.context.collection.objects.link(light_obj2)
light_obj2.rotation_euler = (math.radians(45), 0, math.radians(-135))

# Disable world background
bpy.context.scene.world = bpy.data.worlds.new("World")
bpy.context.scene.world.use_nodes = True
bg_node = bpy.context.scene.world.node_tree.nodes.get("Background")
if bg_node:
    bg_node.inputs[0].default_value = (0.0, 0.0, 0.0, 0.0) # Transparent
    
bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT' if hasattr(bpy.types.SceneEEVEE, "use_raytracing") else 'BLENDER_EEVEE'
bpy.context.scene.render.resolution_x = 256
bpy.context.scene.render.resolution_y = 256
bpy.context.scene.render.film_transparent = True

# 24 camera views
render_dir = os.path.join(out_dir, "rendering")
dist = 1.2

for i in range(24):
    azimuth = (i * 15) % 360
    elevation = 20 if i < 12 else 45
    
    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)
    
    cam_x = dist * math.cos(el_rad) * math.sin(az_rad)
    cam_y = dist * math.cos(el_rad) * math.cos(az_rad)
    cam_z = dist * math.sin(el_rad)
    
    cam_obj.location = (cam_x, cam_y, cam_z)
    
    direction = -cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()
    
    light_obj.location = cam_obj.location
    light_obj.rotation_euler = cam_obj.rotation_euler
    
    out_img = os.path.join(render_dir, f"{i:02d}.png")
    bpy.context.scene.render.filepath = out_img
    bpy.ops.render.render(write_still=True)
