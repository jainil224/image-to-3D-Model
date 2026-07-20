import bpy
import sys
import math

def setup_scene(obj_path, out_path, az, el, dist):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Import OBJ
    bpy.ops.wm.obj_import(filepath=obj_path)
    
    # Camera
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    
    az_rad = math.radians(az)
    el_rad = math.radians(el)
    
    # Assuming standard spherical to cartesian where Z is up
    cam_x = dist * math.cos(el_rad) * math.sin(az_rad)
    cam_y = dist * math.cos(el_rad) * math.cos(az_rad)
    cam_z = dist * math.sin(el_rad)
    
    cam_obj.location = (cam_x, cam_y, cam_z)
    direction = -cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()
    
    # Light
    light_data = bpy.data.lights.new(name="Light", type='SUN')
    light_obj = bpy.data.objects.new(name="Light", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.rotation_euler = cam_obj.rotation_euler
    
    # Render
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT' if hasattr(bpy.types.SceneEEVEE, "use_raytracing") else 'BLENDER_EEVEE'
    bpy.context.scene.render.resolution_x = 256
    bpy.context.scene.render.resolution_y = 256
    bpy.context.scene.render.filepath = out_path
    
    bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    obj_path = sys.argv[-5]
    out_path = sys.argv[-4]
    az = float(sys.argv[-3])
    el = float(sys.argv[-2])
    dist = float(sys.argv[-1])
    setup_scene(obj_path, out_path, az, el, dist)
