
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
