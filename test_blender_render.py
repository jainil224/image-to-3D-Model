import bpy
import mathutils

bpy.ops.wm.read_factory_settings(use_empty=True)

# Create a cube
bpy.ops.mesh.primitive_cube_add(size=2)
bpy.context.active_object.location = (0, 0, 0)

# Camera
cam_data = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj
cam_obj.location = (3, -3, 3)
cam_obj.rotation_euler = (-cam_obj.location).to_track_quat('-Z', 'Y').to_euler()

# Light
light_data = bpy.data.lights.new(name="Light", type='SUN')
light_data.energy = 5.0
light_obj = bpy.data.objects.new("Light", object_data=light_data)
bpy.context.collection.objects.link(light_obj)
light_obj.rotation_euler = cam_obj.rotation_euler

# Render
bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT' if hasattr(bpy.types.SceneEEVEE, "use_raytracing") else 'BLENDER_EEVEE'
bpy.context.scene.render.resolution_x = 256
bpy.context.scene.render.resolution_y = 256
bpy.context.scene.render.filepath = 'D:/image to 3D model/scratch/test_blender_render.png'

bpy.ops.render.render(write_still=True)
