import argparse
import sys
import os
import bpy 
import numpy as np
import random
from math import radians
from mathutils import Vector
import shutil
import cv2

# Set resolution and samples for rendering
RESOLUTION = 512
SAMPLES = 256

# Argument parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--object_folder_path", type=str, required=True, help="Path to object folder")
parser.add_argument("--material_folder_path", type=str, required=True, help="Path to material folder")
parser.add_argument("--output_folder_path", type=str, required=True, help="Path to output folder")
parser.add_argument("--num_samples", type=int, default=10, help="Number of samples")
parser.add_argument("--light_type", type=int, default=1, help="1: point, 2: sun")
parser.add_argument("--camera_pos", type=float, default=1.5, help="Camera position")
parser.add_argument("--max_obj_num", type=int, default=5, help="Maximum number of objects in the scene")
parser.add_argument("--min_obj_num", type=int, default=1, help="Minimum number of objects in the scene")
parser.add_argument("--projection_size", type=float, default=0.50, help="Projection size")

argv = sys.argv[sys.argv.index("--") + 1 :]

################### Reset Scene ###################

# Deletes unused data (meshes, materials, textures, and images) to free up memory
def delete_unused_data():
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)

# Clear all objects in the Blender scene
delete_unused_data()         
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Clear compositing nodes if enabled
if bpy.context.scene.use_nodes:
    bpy.context.scene.node_tree.nodes.clear()

################### Renderer settings ###################

# Configure rendering settings
bpy.ops.file.pack_all()
scene = bpy.context.scene
scene.use_nodes = True # Enable node-based compositing
scene.world.use_nodes = True  # Enable world shading nodes
scene.render.engine = 'CYCLES' # Use Cycles renderer

# Enable CUDA and select all CUDA devices
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    if device.type == 'CUDA':
        device.use = True
        
scene.render.film_transparent = True # Transparent background
scene.cycles.device = 'GPU'
scene.cycles.samples = SAMPLES # Set number of samples for rendering
scene.cycles.max_bounces = 10 # Maximum bounce settings
scene.cycles.diffuse_bounces = 10
scene.cycles.glossy_bounces = 10
scene.cycles.transmission_bounces = 10
scene.cycles.volume_bounces = 10
scene.cycles.transparent_max_bounces = 10
scene.cycles.use_denoising = True
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100

# Enable specific render passes (e.g., Z-depth, normal, object index)
render_layer = scene.view_layers["ViewLayer"]
render_layer.use_pass_z = True
render_layer.use_pass_normal = True
render_layer.use_pass_object_index = True

# Configure compositing nodes
render_layers_node = scene.node_tree.nodes.new(type='CompositorNodeRLayers')
composite_node = scene.node_tree.nodes.new(type='CompositorNodeComposite')
scene.node_tree.links.new(render_layers_node.outputs['Image'], composite_node.inputs['Image'])

################### Image output ###################

# Configure color management and output format
scene.display_settings.display_device = 'None'
scene.view_settings.view_transform = 'Standard'
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
scene.render.image_settings.file_format = 'TIFF'  # set output format to .png
scene.render.image_settings.color_depth = '16'
scene.render.image_settings.color_mode = 'RGB'
scene.render.image_settings.tiff_codec = 'NONE'

################### Apply Materials ###################

# Function to load textures (color, roughness, and metalness maps)
def load_texture(MATERIAL_PATH, MATERIAL_BASENAME, texture_type, mat):
    texture_filename = f"{MATERIAL_BASENAME}_{texture_type}.jpg"
    texture_filepath = os.path.abspath(os.path.join(MATERIAL_PATH, MATERIAL_BASENAME, texture_filename))
    if os.path.exists(texture_filepath):
        tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex_image.image = bpy.data.images.load(texture_filepath)
        return tex_image
    else:
        return None

# Function to apply material to an object
def apply_materials(obj, MATERIAL_PATH, MATERIAL_BASENAME):
    
    mat = bpy.data.materials.new(name="CustomMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    
    tex_coord = mat.node_tree.nodes.new('ShaderNodeTexCoord')
    mapping = mat.node_tree.nodes.new('ShaderNodeMapping')
    mat.node_tree.links.new(mapping.inputs["Vector"], tex_coord.outputs["UV"])

    color_map = load_texture(MATERIAL_PATH, MATERIAL_BASENAME, "Color", mat)
    roughness_map = load_texture(MATERIAL_PATH, MATERIAL_BASENAME, "Roughness", mat)
    metaric_map = load_texture(MATERIAL_PATH, MATERIAL_BASENAME, "Metalness", mat)

    if color_map:
        mat.node_tree.links.new(color_map.inputs["Vector"], mapping.outputs["Vector"])
        mat.node_tree.links.new(bsdf.inputs["Base Color"], color_map.outputs["Color"])
    if roughness_map:
        roughness_map.image.colorspace_settings.name = 'Non-Color'
        mat.node_tree.links.new(roughness_map.inputs["Vector"], mapping.outputs["Vector"])
        mat.node_tree.links.new(bsdf.inputs["Roughness"], roughness_map.outputs["Color"])
    if metaric_map:
        metaric_map.image.colorspace_settings.name = 'Non-Color'
        mat.node_tree.links.new(metaric_map.inputs["Vector"], mapping.outputs["Vector"])
        mat.node_tree.links.new(bsdf.inputs["Metallic"], metaric_map.outputs["Color"])
    
    if obj.type == 'MESH':
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
            
################### Camera Setting ###################

# Function to set up the camera in the scene
def camera_setting(camera_pos=3, projection_size=1.0):
    camera = bpy.data.cameras.new("Camera")
    camera_object = bpy.data.objects.new("Camera", camera)
    bpy.context.collection.objects.link(camera_object)
    
    if camera_object:
        camera_object.location = (0, 0, camera_pos)
        camera_object.rotation_euler = (0, 0, 0)
        camera.type = 'ORTHO' # Orthographic camera
        camera.ortho_scale = projection_size
    else:
        print("Camera object not found.")

    bpy.context.scene.camera = camera_object
    
################### Object setting ###################

# Function to check if a location is too close to other objects
def is_too_close(new_location, other_objects, min_distance):
    for obj in other_objects:
        if (new_location - obj.location).length < min_distance:
            return True
    return False

# Function to apply random transformations to objects (rotation and location)
def random_transform(obj, location_range, rotation_range, other_objects, min_distance=1.0):
    # Generate random rotation
    obj.rotation_euler.x += radians(random.uniform(-rotation_range, rotation_range))
    obj.rotation_euler.y += radians(random.uniform(-rotation_range, rotation_range))
    obj.rotation_euler.z += radians(random.uniform(-rotation_range, rotation_range))
    
    # Generate random location
    obj.location.x = 0.0
    obj.location.y = 0.0
    obj.location.z = 0.0
    
    for _ in range(50):  # Limit the number of attempts to avoid infinite loop
        new_location = obj.location + Vector((
            random.uniform(-location_range, location_range),
            random.uniform(-location_range, location_range),
            random.uniform(-0.1, 0.1)
        ))
        if not is_too_close(new_location, other_objects, min_distance):
            obj.location = new_location
            break

################### Lights setting ###################

# Sample point in spherical space for sun light
def sample_spherical1(radius=3.0, maxz=3.0, minz=0.):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        vec[2] = np.abs(vec[2])
        vec = vec / np.linalg.norm(vec, axis=0) * radius
        if maxz > vec[2] > minz:
            correct = True
    return vec

# Sample point in spherical space for point light
def sample_spherical2(radius_min=1.5, radius_max=2.0, maxz=1.6, minz=-0.75):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec

def set_point_light(name, radius_min=1.2, radius_max=1.8, maxz=1.8, minz=1.0, light_strength=100):
    x, y, z = sample_spherical2(radius_min=radius_min, radius_max=radius_max, maxz=maxz, minz=minz)
    light_data = bpy.data.lights.new(name=name, type='POINT')
    light_data.energy = light_strength
    light_object = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = x, y, z
    
    return light_object

def add_noise_to_direction(direction, noise_strength=0.5):
    noise = np.random.uniform(-noise_strength, noise_strength, 3)
    new_direction = direction + noise
    return new_direction / np.linalg.norm(new_direction)

def set_sun_light(name, radius=1.5, maxz=1.5, minz=1.0, light_strength=8.0):
    x, y, z = sample_spherical1(radius=radius, maxz=maxz, minz=minz)
    light_data = bpy.data.lights.new(name=name, type='SUN')
    light_data.energy = light_strength
    angle_degrees = random.uniform(0, 90)
    light_data.angle = np.radians(angle_degrees)
    
    light_object = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = x, y, z
    
    location = np.array(light_object.location)
    light_direction = add_noise_to_direction(location, 0.05)
    light_direction_vector = Vector(light_direction)
    light_object.rotation_euler = light_direction_vector.to_track_quat('Z', 'Y').to_euler()
    
    return light_object

# Function to delete objects (e.g., lights) from the scene
def delete_object(obj):
    bpy.data.objects.remove(obj, do_unlink=True)

################### Compositing Normal ###################

# Modify normal image compositing process
def edit_normal_compositing():
    # Get compositor nodes
    nodes = scene.node_tree.nodes
    links = scene.node_tree.links

    # Clear default nodes
    nodes.clear()
    render_layers_node = nodes.new(type='CompositorNodeRLayers')
    composite_node = nodes.new(type='CompositorNodeComposite')
    separate_rgb_node = nodes.new('CompositorNodeSepRGBA')
    
    links.new(render_layers_node.outputs['Normal'], separate_rgb_node.inputs['Image'])
    
    combined_channels = []
    for i, channel in enumerate(['R', 'G', 'B']):
        math_node = nodes.new('CompositorNodeMath')
        math_node.operation = 'MULTIPLY_ADD'
        math_node.inputs[1].default_value = 0.5
        math_node.inputs[2].default_value = 0.5
        
        links.new(separate_rgb_node.outputs[channel], math_node.inputs[0])
        combined_channels.append(math_node.outputs[0])
    
    combine_rgba_node = nodes.new('CompositorNodeCombRGBA')
    for i, channel in enumerate(['R', 'G', 'B']):
        links.new(combined_channels[i], combine_rgba_node.inputs[channel])
    
    links.new(combine_rgba_node.outputs['Image'], composite_node.inputs['Image'])

################### Main ###################
    
def main(args):
    MESH_PATH = args.object_folder_path
    MATERIAL_PATH = args.material_folder_path
    OUTPUT_PATH = args.output_folder_path
    
    OBJ_FILE_NAMEs = [dir for dir in os.listdir(MESH_PATH) if os.path.isdir(os.path.join(MESH_PATH, dir))]
    MATERIAL_BASENAMEs = [dir for dir in os.listdir(MATERIAL_PATH) if os.path.isdir(os.path.join(MATERIAL_PATH, dir))]
    
    num_obj = random.randint(args.min_obj_num, args.max_obj_num)
    
    OBJ_FILE_NAMES = random.sample(OBJ_FILE_NAMEs, num_obj)
    MATERIAL_BASENAMES = random.sample(MATERIAL_BASENAMEs, num_obj)
    
    output_env_file_path = os.path.join(OUTPUT_PATH, "selected_envs.txt")
    with open(output_env_file_path, 'w') as file:
        for OBJ_FILE_NAME in OBJ_FILE_NAMES:
            file.write(f"OBJ File: {OBJ_FILE_NAME}\n")
        for MATERIAL_BASENAME in MATERIAL_BASENAMES:
            file.write(f"Material: {MATERIAL_BASENAME}\n")
        file.write(f"Lights: {args.light_type}\n")

    color_management = bpy.context.scene.view_settings
    other_obj = []
    
    for OBJ_FILE_NAME, MATERIAL_BASENAME in zip(OBJ_FILE_NAMES, MATERIAL_BASENAMES):
        bpy.ops.import_scene.obj(filepath=os.path.join(MESH_PATH, OBJ_FILE_NAME, OBJ_FILE_NAME + ".obj"), axis_forward = '-Z', axis_up = 'Y')
        bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
        
        for obj in bpy.context.selected_objects:
            apply_materials(obj, MATERIAL_PATH, MATERIAL_BASENAME)
            obj.select_set(True)

        bpy.ops.object.join() # Join all selected objects
        bpy.ops.object.shade_smooth() # Apply smooth shading
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        
        obj = bpy.context.active_object

        # Calculate the dimensions of the bounding box
        dimensions = obj.dimensions
        longest_side = max(dimensions.x, dimensions.y)

        # Calculate the scale so that the longest side is around 20cm
        # Diligent Dataset Object Size: 20cm
        scale_factor = random.uniform(0.18, 0.22) / longest_side
        obj.scale *= scale_factor
        obj.location.x = 0
        obj.location.y = 0
        obj.location.z = 0
        random_transform(obj, 0.1, 20, other_obj, 0.05)
        
        other_obj.append(obj)
        
        bpy.ops.object.select_all(action='DESELECT')
        
    else:
        print("Imported mesh object not found.")
    
    for obj in other_obj:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
    bpy.ops.object.join()
    
    obj = bpy.context.active_object
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    current_dimensions = obj.dimensions.copy()
    longest_side = max(current_dimensions.x, current_dimensions.y)
    scale_factor = args.projection_size / longest_side
    obj.dimensions = current_dimensions * scale_factor
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    bbox_center = sum(bbox_corners, Vector()) / 8
    bpy.context.scene.cursor.location = bbox_center
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    
    obj.location.x = 0
    obj.location.y = 0
    obj.location.z = 0

    bpy.ops.object.select_all(action='DESELECT')
    
    camera_setting(camera_pos=args.camera_pos, projection_size=args.projection_size)
        
    if int(args.light_type) == 1:
        min_light_strength = 50
        max_light_strength = 100
        light_strength = random.uniform(min_light_strength, max_light_strength)
    elif int(args.light_type) == 2:
        min_light_strength = 3.0
        max_light_strength = 5.0
        light_strength = random.uniform(min_light_strength, max_light_strength)
    
    for i in range(args.num_samples):
        light = None
        if int(args.light_type) == 1:
            light = set_point_light(f"PointLight_{i}", light_strength=light_strength)
        elif int(args.light_type) == 2:
            light = set_sun_light(f"SunLight_{i}", light_strength=light_strength)
        else:
            raise ValueError("Invalid light type")
        
        # Save the rendered image
        bpy.context.scene.render.filepath = os.path.abspath(os.path.join(OUTPUT_PATH, "{:03}".format(i)))
        bpy.context.scene.render.image_settings.file_format = 'TIFF'  
        bpy.ops.render.render(write_still=True)

        # Remove the light
        if light is not None:
            delete_object(light)
        
    edit_normal_compositing()
    bpy.context.scene.render.filepath = os.path.abspath(os.path.join(OUTPUT_PATH, f"normal"))
    bpy.context.scene.render.image_settings.file_format = 'TIFF'  
    bpy.ops.render.render(write_still=True)
    
    # Check if the normal image is generated
    if not os.path.exists(os.path.join(OUTPUT_PATH, f"normal.tif")):
        print("Normal image is not found.")
        # Remove the output folder if the normal image is not generated
        shutil.rmtree(OUTPUT_PATH)
        return
    
    normal_path = os.path.abspath(os.path.join(OUTPUT_PATH, f"normal.tif"))
    normal_image = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
    target_color = np.array([65536/2, 65536/2, 65536/2], dtype=np.uint16)
    convert_color = np.array([0, 0, 0], dtype=np.uint16)
    
    mask = np.all(normal_image == target_color, axis=-1)
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    normal_image[dilated_mask == 1] = convert_color
    
    # If the number of non-zero elements in the mask is less than 2048, the target area is too small.
    nonzero_num = np.count_nonzero(dilated_mask)
    if nonzero_num < 2048:
        print("Target Area is too small.")
        # Remove the output folder if the normal image is not generated
        shutil.rmtree(OUTPUT_PATH)
        return
    
    cv2.imwrite(normal_path, normal_image, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    
if __name__ == '__main__':
    args = parser.parse_args(argv)
    main(args)