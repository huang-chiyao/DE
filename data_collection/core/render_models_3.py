#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path
import random
import bpy
import bmesh
import os
import mathutils
from mathutils import Vector
import bpy
import bmesh
import os
import mathutils
from mathutils import Vector
import math
import time

# Remove all bpy imports and BlenderOBJRenderer class

class BlenderOBJRenderer:
    def __init__(self):
        """Initialize the renderer and clear default scene"""
        print("Initializing BlenderOBJRenderer")
        self.clear_scene()
        self.setup_render_settings()
        self.original_materials = {}  # Store original materials
        print("BlenderOBJRenderer initialized")
    
    def clear_scene(self):
        """Clear all default objects from the scene"""
        print("Clearing scene")
        try:
            # Select all objects
            bpy.ops.object.select_all(action='SELECT')
            # Delete all selected objects
            bpy.ops.object.delete(use_global=False)
            
            # Clear all materials
            for material in bpy.data.materials:
                bpy.data.materials.remove(material)
            print("Scene cleared successfully")
        except Exception as e:
            print(f"Error clearing scene: {e}")
            raise
    
    def setup_render_settings(self):
        """Configure render settings for realistic output - CPU ONLY"""
        print("Setting up render settings")
        try:
            scene = bpy.context.scene
            
            # Set render engine to Cycles for realistic rendering
            scene.render.engine = 'CYCLES'
            
            # FORCE CPU RENDERING (no GPU/CUDA)
            bpy.context.scene.cycles.device = 'CPU'
            
            # Render settings
            scene.render.resolution_x = 512  # Smaller for faster rendering
            scene.render.resolution_y = 512
            scene.render.resolution_percentage = 100
            
            # Cycles specific settings - reduced for CPU
            scene.cycles.samples = 64  # Lower samples for CPU
            scene.cycles.use_denoising = True
            scene.cycles.denoiser = 'OPENIMAGEDENOISE'  # CPU-compatible denoiser
            
            # Color management for realistic output
            scene.view_settings.view_transform = 'Standard'
            scene.view_settings.look = 'None'
            print("Render settings configured successfully")
        except Exception as e:
            print(f"Error setting up render settings: {e}")
            raise
    def load_obj_file(self, obj_path):
        """Load OBJ file into the scene"""
        print(f"Loading OBJ file: {obj_path}")
        
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"OBJ file not found: {obj_path}")
        
        try:
            # Ensure we're in the correct context
            # Switch to object mode if not already
            if bpy.context.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            
            # Deselect all objects first
            bpy.ops.object.select_all(action='DESELECT')
            
            # Get list of objects before import
            objects_before = list(bpy.context.scene.objects)
            
            # Import OBJ file - updated for Blender 4.0+
            try:
                # Try new import method (Blender 4.0+)
                # Ensure proper context by overriding if needed
                with bpy.context.temp_override():
                    bpy.ops.wm.obj_import(filepath=obj_path)
                print("Used new obj_import method")
            except (AttributeError, RuntimeError):
                try:
                    # Fallback to older method (Blender 3.x)
                    bpy.ops.import_scene.obj(filepath=obj_path)
                    print("Used legacy import_scene.obj method")
                except (AttributeError, RuntimeError):
                    try:
                        # If both fail, try enabling the addon first
                        bpy.ops.preferences.addon_enable(module="io_scene_obj")
                        bpy.ops.import_scene.obj(filepath=obj_path)
                        print("Enabled addon and used import_scene.obj method")
                    except Exception as e:
                        # Last resort: try to manually parse and create object
                        print(f"All import methods failed: {e}")
                        print("Attempting manual OBJ loading...")
                        return self.manual_obj_load(obj_path)
            
            # Get the imported object
            # Find newly added objects
            objects_after = list(bpy.context.scene.objects)
            new_objects = [obj for obj in objects_after if obj not in objects_before]
            
            if new_objects:
                obj = new_objects[-1]  # Get the last imported object
            elif bpy.context.selected_objects:
                obj = bpy.context.selected_objects[-1]
            else:
                # If no new objects found, get the last object in the scene
                if len(bpy.context.scene.objects) > 0:
                    obj = bpy.context.scene.objects[-1]
                else:
                    raise RuntimeError("No object was imported or found in the scene")
            
            print(f"Loaded object: {obj.name}")
            
            # Ensure the object is selected and active
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            
            # Store original materials before modification
            self.store_original_materials(obj)
            
            # Center the object
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
            obj.location = (0, 0, 0)
            
            # Fix normal orientation issues
            self.fix_object_normals(obj)
            # Disable backface culling to show both sides of faces
            self.ensure_backface_culling_disabled(obj)
            
            print("Object loaded and processed successfully")
            return obj
            
        except Exception as e:
            print(f"Error loading OBJ file: {e}")
            raise
    
    def manual_obj_load(self, obj_path):
        """Manual OBJ loading as fallback method"""
        import bmesh
        
        print("Attempting manual OBJ parsing...")
        
        vertices = []
        faces = []
        
        try:
            with open(obj_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith('v '):
                        # Vertex
                        parts = line.split()
                        if len(parts) >= 4:
                            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    elif line.startswith('f '):
                        # Face
                        parts = line.split()
                        face_indices = []
                        for part in parts[1:]:
                            # Handle vertex/texture/normal format (v/vt/vn)
                            vertex_idx = int(part.split('/')[0]) - 1  # OBJ indices start at 1
                            face_indices.append(vertex_idx)
                        if len(face_indices) >= 3:
                            faces.append(face_indices)
            
            if not vertices:
                raise RuntimeError("No vertices found in OBJ file")
            
            # Create mesh
            mesh = bpy.data.meshes.new(name="imported_obj")
            mesh.from_pydata(vertices, [], faces)
            mesh.update()
            
            # Create object
            obj = bpy.data.objects.new("imported_obj", mesh)
            bpy.context.collection.objects.link(obj)
            
            print(f"Manually loaded OBJ with {len(vertices)} vertices and {len(faces)} faces")
            return obj
            
        except Exception as e:
            print(f"Manual OBJ loading failed: {e}")
            raise RuntimeError(f"Failed to load OBJ file: {obj_path}")
    
    def fix_object_normals(self, obj):
        """Fix normal orientation issues that cause black triangles"""
        print(f"Fixing normals for object: {obj.name}")
        try:
            # Make sure the object is selected and active
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            
            # Enter edit mode
            bpy.ops.object.mode_set(mode='EDIT')
            
            # Select all faces
            bpy.ops.mesh.select_all(action='SELECT')
            
            # Recalculate normals to point outward
            bpy.ops.mesh.normals_make_consistent(inside=False)
            
            # Exit edit mode
            bpy.ops.object.mode_set(mode='OBJECT')
            
            print("Normals fixed successfully")
        except Exception as e:
            print(f"Error fixing normals: {e}")
            # Don't raise here, just continue

    def ensure_backface_culling_disabled(self, obj):
        """Disable backface culling in materials to show both sides of faces"""
        print("Disabling backface culling")
        try:
            for slot in obj.material_slots:
                if slot.material:
                    mat = slot.material
                    if mat.use_nodes:
                        # Set material to show both sides
                        mat.use_backface_culling = False
                        
                        # Find Principled BSDF and ensure it's set up correctly
                        for node in mat.node_tree.nodes:
                            if node.type == 'BSDF_PRINCIPLED':
                                # Ensure alpha is set to 1.0 (fully opaque)
                                if 'Alpha' in node.inputs:
                                    node.inputs['Alpha'].default_value = 1.0
                                break
            print("Backface culling disabled successfully")
        except Exception as e:
            print(f"Error disabling backface culling: {e}")
    
    def store_original_materials(self, obj):
        """Store original material properties to preserve colors"""
        print("Storing original materials")
        try:
            for slot in obj.material_slots:
                if slot.material:
                    mat = slot.material
                    # Store original diffuse color if it exists
                    if mat.use_nodes:
                        for node in mat.node_tree.nodes:
                            if node.type == 'BSDF_PRINCIPLED':
                                self.original_materials[mat.name] = {
                                    'base_color': node.inputs['Base Color'].default_value[:],
                                    'material': mat
                                }
                                break
                    else:
                        # For non-node materials, store diffuse color
                        self.original_materials[mat.name] = {
                            'base_color': mat.diffuse_color[:],
                            'material': mat
                        }
            print(f"Stored {len(self.original_materials)} original materials")
        except Exception as e:
            print(f"Error storing original materials: {e}")
    
    def preserve_original_colors(self, obj):
        """Preserve original material colors instead of creating new materials"""
        print("Preserving original colors")
        try:
            for slot in obj.material_slots:
                if slot.material and slot.material.name in self.original_materials:
                    mat = slot.material
                    original_data = self.original_materials[slot.material.name]
                    
                    # Ensure material uses nodes
                    if not mat.use_nodes:
                        mat.use_nodes = True
                    
                    # Find or create Principled BSDF node
                    principled = None
                    for node in mat.node_tree.nodes:
                        if node.type == 'BSDF_PRINCIPLED':
                            principled = node
                            break
                    
                    if principled:
                        # Restore original color
                        principled.inputs['Base Color'].default_value = original_data['base_color']
                        # Keep some realistic properties but preserve color
                        principled.inputs['Metallic'].default_value = 0.0
                        principled.inputs['Roughness'].default_value = 0.5
            print("Original colors preserved successfully")
        except Exception as e:
            print(f"Error preserving original colors: {e}")
    
    def rotate_object(self, obj, azimuth=0, elevation=0, roll=0):
        """
        Rotate object with spherical coordinates
        azimuth: rotation around Z-axis (degrees)
        elevation: rotation around Y-axis (degrees) 
        roll: rotation around X-axis (degrees)
        """
        try:
            # Convert degrees to radians
            az_rad = math.radians(azimuth) 
            el_rad = math.radians(elevation)
            roll_rad = math.radians(roll) 
            
            # Apply rotations in order: Z (azimuth), Y (elevation), X (roll)
            obj.rotation_euler = (el_rad, az_rad, roll_rad)
            print(f"Rotated object: az={azimuth}, el={elevation}, roll={roll}")
        except Exception as e:
            print(f"Error rotating object: {e}")
    
    def setup_camera_with_depth(self, depth=1, target=(0, 0, 0)):
        """Setup camera at specified depth from target - positioned on Z-axis"""
        print(f"Setting up camera at depth {depth}")
        try:
            # Position camera on positive Z-axis (looking down)
            location = (0, 0, -depth)
            
            bpy.ops.object.camera_add(location=location)
            camera = bpy.context.object
            camera.name = "RenderCamera"
            
            # Point camera at target - fix the flip by using correct track axis
            direction = - (Vector(target) - Vector(location))
            # Use 'Z' as forward and 'Y' as up to avoid flip
            camera.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
            
            # Set as active camera
            bpy.context.scene.camera = camera
            
            # Camera settings
            camera.data.lens = 50
            camera.data.sensor_width = 36
            
            print("Camera setup successful")
            return camera
        except Exception as e:
            print(f"Error setting up camera: {e}")
            raise
    
    def add_light_setup(self, num_lights=1, light_energy_multiplier=1.0):
        """Add lighting setup with specified number of lights"""
        print(f"Adding {num_lights} lights with energy multiplier {light_energy_multiplier}")
        try:
            # Clear existing lights
            for obj in bpy.context.scene.objects:
                if obj.type == 'LIGHT':
                    bpy.data.objects.remove(obj, do_unlink=True)
            
            lights = []
            
            if num_lights >= 1:
                # Key light - positioned in positive Y direction
                bpy.ops.object.light_add(type='SUN', location=(0, 4, -2))
                key_light = bpy.context.object
                key_light.name = "KeyLight"
                key_light.data.energy = 3.0 * light_energy_multiplier
                key_light.data.color = (1, 1, 1)
                lights.append(key_light)
            
            if num_lights >= 2:
                # Fill light - also in positive Y but offset
                bpy.ops.object.light_add(type='AREA', location=(2, 3, -1))
                fill_light = bpy.context.object
                fill_light.name = "FillLight"
                fill_light.data.energy = 15.0 * light_energy_multiplier
                fill_light.data.color = (1, 0.9, 0.8)
                fill_light.data.size = 2
                lights.append(fill_light)
            
            if num_lights >= 3:
                # Rim light - opposite side for edge lighting
                bpy.ops.object.light_add(type='SPOT', location=(-2, 3, -2))
                rim_light = bpy.context.object
                rim_light.name = "RimLight"
                rim_light.data.energy = 25.0 * light_energy_multiplier
                rim_light.data.color = (0.8, 0.9, 1)
                rim_light.data.spot_size = 1.2
                rim_light.data.spot_blend = 0.3
                lights.append(rim_light)
            
            # Point all lights towards origin
            for light in lights:
                direction = Vector((0, 0, 0)) - Vector(light.location)
                light.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
            
            print(f"Successfully added {len(lights)} lights")
            return lights
        except Exception as e:
            print(f"Error adding lights: {e}")
            raise
    
    def render_image(self, output_path, samples=None):
        """Render the scene to an image file"""
        print(f"Rendering to: {output_path}")
        try:
            if samples:
                bpy.context.scene.cycles.samples = samples
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Set output path
            bpy.context.scene.render.filepath = output_path
            
            # Render
            bpy.ops.render.render(write_still=True)
            print("Render completed successfully")
        except Exception as e:
            print(f"Error during rendering: {e}")

def render_single_image(image_info):
    """Render a single image - NEW FUNCTION for image-level parallelization"""
    try:
        (model_info, output_base_path, az_range, el_range, roll_range, 
         image_idx, azimuth, elevation, roll) = image_info
        
        category, model_id, obj_path = model_info
        
        print(f"Rendering image {image_idx} for model {category}/{model_id}")
        
        # Create output directory
        output_dir = Path(output_base_path) / category / model_id / "rendering"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize renderer for this process
        renderer = BlenderOBJRenderer()
        
        # Load the 3D model
        obj = renderer.load_obj_file(str(obj_path))
        
        # Preserve original colors
        renderer.preserve_original_colors(obj)
        
        # Setup camera and lighting
        camera = renderer.setup_camera_with_depth(depth=1.5)
        lights = renderer.add_light_setup(num_lights=3, light_energy_multiplier=1.0)
        
        # Fixed values
        depth = 2.5
        fov = 60
        
        # Apply rotation to object
        renderer.rotate_object(obj, azimuth, elevation, roll)
        
        # Render image
        output_path = output_dir / f"{image_idx:03d}.png"
        renderer.render_image(str(output_path), samples=64)
        
        # Return metadata for this image
        metadata_line = f"{azimuth:.6f} {elevation:.6f} {roll:.6f} {depth:.6f} {fov:.6f}"
        
        print(f"Completed image {image_idx} for {category}/{model_id}")
        return (image_idx, metadata_line)
        
    except Exception as e:
        print(f"Error rendering image {image_idx}: {e}")
        return None
            
def render_single_model(model_info, output_base_path, az_range, el_range, roll_range, num_images):
    """Render a single model with multiple viewpoints using BlenderOBJRenderer"""
    category, model_id, obj_path = model_info
    
    print(f"=== RENDERING MODEL ===")
    print(f"Category: {category}")
    print(f"Model ID: {model_id}")
    print(f"OBJ path: {obj_path}")
    print(f"Output base path: {output_base_path}")
    print(f"Ranges: az={az_range}, el={el_range}, roll={roll_range}")
    print(f"Number of images: {num_images}")
    
    # Create output directory
    output_dir = Path(output_base_path) / category / model_id / "rendering"
    print(f"Output directory: {output_dir}")
    
    # Check if model is partially or fully rendered
    metadata_path = output_dir / "rendered_images_metadata.txt"
    start_idx = 0
    existing_metadata = []
    
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                existing_metadata = f.read().strip().split('\n')
                existing_metadata = [line for line in existing_metadata if line.strip()]  # Remove empty lines
            start_idx = len(existing_metadata)
            
            if start_idx >= num_images:
                print(f"✓ Model {category}/{model_id} already fully rendered ({start_idx}/{num_images}), skipping...")
                return True
            else:
                print(f"→ Model {category}/{model_id} partially rendered ({start_idx}/{num_images}), continuing from image {start_idx + 1}...")
        except Exception as e:
            print(f"Warning: Error reading metadata file, starting from beginning: {e}")
            start_idx = 0
            existing_metadata = []
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print("Output directory created")
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return False

    try:
        # Initialize renderer
        print("Initializing renderer...")
        renderer = BlenderOBJRenderer()
        
        # Load the 3D model
        print("Loading 3D model...")
        obj = renderer.load_obj_file(str(obj_path))
        
        # Preserve original colors
        print("Preserving original colors...")
        renderer.preserve_original_colors(obj)
        
        # Setup camera and lighting
        print("Setting up camera and lighting...")
        # Setup camera at a fixed depth, 1.5 for chair models
        camera = renderer.setup_camera_with_depth(depth=1.5)
        lights = renderer.add_light_setup(num_lights=3, light_energy_multiplier=1.0)
        
        # Generate random viewpoints and render
        print("Starting rendering loop...")
        
        # Prepare metadata file
        metadata_lines = existing_metadata.copy()  # Start with existing metadata
        
        for i in range(start_idx, num_images):
            print(f"Rendering image {i+1}/{num_images}")

            # Generate random rotations within specified ranges
            azimuth = random.uniform(az_range[0], az_range[1])
            elevation = random.uniform(el_range[0], el_range[1])
            roll = random.uniform(roll_range[0], roll_range[1])
            
            # Fixed values
            depth = 2.5
            fov = 60
            
            # Apply rotation to object
            renderer.rotate_object(obj, azimuth, elevation, roll)
            
            # Render image
            output_path = output_dir / f"{i:03d}.png"
            renderer.render_image(str(output_path), samples=64)
            
            # Add metadata line
            metadata_lines.append(f"{azimuth:.6f} {elevation:.6f} {roll:.6f} {depth:.6f} {fov:.6f}")
            
            print(f"Completed image {i+1}/{num_images}: {output_path}")
        
        # Write metadata file
        try:
            with open(metadata_path, 'w') as f:
                f.write('\n'.join(metadata_lines))
            print(f"Metadata written to: {metadata_path}")
        except Exception as e:
            print(f"Error writing metadata: {e}")
        
        print(f"✓ Successfully rendered {num_images - start_idx} new images for {category}/{model_id}")
        return True
        
    except Exception as e:
        print(f"✗ Error rendering {category}/{model_id}: {str(e)}")
        return False

def render_model_with_parallel_images(model_info, output_base_path, az_range, el_range, roll_range, num_images, num_processes):
    """NEW FUNCTION: Render a single model with parallel image processing"""
    category, model_id, obj_path = model_info
    
    print(f"=== RENDERING MODEL WITH PARALLEL IMAGES ===")
    print(f"Category: {category}")
    print(f"Model ID: {model_id}")
    print(f"Number of images: {num_images}")
    print(f"Image processes: {num_processes}")
    
    # Create output directory
    output_dir = Path(output_base_path) / category / model_id / "rendering"
    metadata_path = output_dir / "rendered_images_metadata.txt"
    
    # Check if already rendered
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                existing_lines = f.read().strip().split('\n')
                existing_lines = [line for line in existing_lines if line.strip()]
            if len(existing_lines) >= num_images:
                print(f"✓ Model {category}/{model_id} already fully rendered, skipping...")
                return True
        except Exception as e:
            print(f"Warning: Error reading metadata file: {e}")
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all image parameters first
        image_tasks = []
        for i in range(num_images):
            # Generate random rotations
            azimuth = random.uniform(az_range[0], az_range[1])
            elevation = random.uniform(el_range[0], el_range[1])
            roll = random.uniform(roll_range[0], roll_range[1])
            
            image_info = (model_info, output_base_path, az_range, el_range, roll_range, 
                         i, azimuth, elevation, roll)
            image_tasks.append(image_info)
        
        # Render images in parallel
        print(f"Starting parallel rendering of {num_images} images...")
        with Pool(processes=num_processes) as pool:
            results = pool.map(render_single_image, image_tasks)
        
        # Collect successful results and sort by image index
        successful_results = [r for r in results if r is not None]
        successful_results.sort(key=lambda x: x[0])  # Sort by image_idx
        
        # Write metadata file
        metadata_lines = [result[1] for result in successful_results]
        with open(metadata_path, 'w') as f:
            f.write('\n'.join(metadata_lines))
        
        success_count = len(successful_results)
        print(f"✓ Successfully rendered {success_count}/{num_images} images for {category}/{model_id}")
        
        return success_count == num_images
        
    except Exception as e:
        print(f"✗ Error rendering {category}/{model_id}: {str(e)}")
        return False

def render_single_model_wrapper(args):
    """Wrapper function for multiprocessing - safely unpacks arguments and calls render_single_model"""
    try:
        model_info, output_base_path, az_range, el_range, roll_range, num_images = args
        return render_single_model(model_info, output_base_path, az_range, el_range, roll_range, num_images)
    except Exception as e:
        model_id = args[0][1] if isinstance(args[0], tuple) and len(args[0]) > 1 else "UNKNOWN"
        print(f"[ERROR] Failed to render model {model_id}: {e}")
        return 0  # Indicate failure

def render_model_with_parallel_images_wrapper(args):
    """NEW WRAPPER: Wrapper for parallel image rendering"""
    try:
        model_info, output_base_path, az_range, el_range, roll_range, num_images, num_processes = args
        return render_model_with_parallel_images(model_info, output_base_path, az_range, el_range, roll_range, num_images, num_processes)
    except Exception as e:
        model_id = args[0][1] if isinstance(args[0], tuple) and len(args[0]) > 1 else "UNKNOWN"
        print(f"[ERROR] Failed to render model {model_id}: {e}")
        return False

def find_models(shapenet_path, category, begin_idx=None, end_idx=None):
    """Find models in category with optional begin and end indices"""
    category_path = Path(shapenet_path) / category
    all_models = []
    
    # Collect all valid models first
    for model_dir in category_path.iterdir():
        if model_dir.is_dir():
            obj_files = list((model_dir / "models").glob("*.obj"))
            if obj_files:
                all_models.append((category, model_dir.name, obj_files[0]))
    
    # Sort models alphabetically by model name
    all_models.sort(key=lambda x: x[1])  # Sort by model_id (x[1])
    
    # Apply begin/end indexing if specified
    if begin_idx is not None and end_idx is not None:
        models = all_models[begin_idx:end_idx]
    elif begin_idx is not None:
        models = all_models[begin_idx:]
    elif end_idx is not None:
        models = all_models[:end_idx]
    else:
        models = all_models
    
    return models


def main(
    CATEGORY,
    BEGIN_IDX,
    END_IDX,
    NUM_IMAGES,
    NUM_PROCESSES,
    OUTPUT_BASE_PATH=None,
    SHAPENET_PATH=None,
):
    # Configuration
    # Resolve paths relative to the repository root so this script
    # keeps working even when called from a different working directory.
    project_root = Path(__file__).resolve().parent.parent
    SHAPENET_PATH = Path(SHAPENET_PATH) if SHAPENET_PATH else (
        project_root / "ShapeNet" / "ShapeNetVox32"
    )  # ShapeNet dataset path
    OUTPUT_BASE_PATH = Path(OUTPUT_BASE_PATH) if OUTPUT_BASE_PATH else (
        project_root
        / "ShapeNet"
        / "ShapeNetRendering_az_90_el_ro_45_cls_id_30k"
    )  # Base output path
    BEGIN_IDX = BEGIN_IDX                           # Starting index (inclusive)
    END_IDX = END_IDX                             # Ending index (exclusive)
    NUM_IMAGES = NUM_IMAGES                         # Number of images per model
    NUM_PROCESSES = NUM_PROCESSES                   # Parallel processes
    
    # Orientation ranges (degrees)
    AZ_RANGE = (-90, 90)
    EL_RANGE = (-45, 45)
    ROLL_RANGE = (-45, 45)  # No roll variation
    
    # Record start time
    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    
    print(f"=== RENDERING STARTED ===")
    print(f"Start time: {start_time_str}")
    print(f"Rendering models from category {CATEGORY}")
    print(f"Model range: {BEGIN_IDX} to {END_IDX-1} (indices)")
    print(f"Generating {NUM_IMAGES} images per model")
    print(f"ShapeNet path: {SHAPENET_PATH}")
    print(f"Output base path: {OUTPUT_BASE_PATH}")
    print(f"Orientation ranges: az={AZ_RANGE}, el={EL_RANGE}, roll={ROLL_RANGE}")
    print(f"Using {NUM_PROCESSES} parallel processes")
    
    # Find models
    models = find_models(SHAPENET_PATH, CATEGORY, BEGIN_IDX, END_IDX)
    print(f"Found {len(models)} models (sorted alphabetically)")
    if models:
        print(f"First model: {models[0][1]}")
        print(f"Last model: {models[-1][1]}")
    
    # ADAPTIVE PARALLELIZATION STRATEGY
    num_models = len(models)
    
    # Determine optimal parallelization strategy
    if num_models <= 4 and NUM_IMAGES >= 100:
        # Few models, many images per model -> parallelize images within each model
        print(f"\n=== USING IMAGE-LEVEL PARALLELIZATION ===")
        print(f"Strategy: Few models ({num_models}), many images ({NUM_IMAGES})")
        print(f"Will process models sequentially, but parallelize images within each model")
        
        # Calculate processes per model (reserve some processes for model-level if needed)
        processes_per_model = min(NUM_PROCESSES, NUM_IMAGES)
        print(f"Using {processes_per_model} processes per model for image rendering")
        
        results = []
        for model in models:
            print(f"\nProcessing model {model[1]}...")
            result = render_model_with_parallel_images(
                model, OUTPUT_BASE_PATH, AZ_RANGE, EL_RANGE, ROLL_RANGE, 
                NUM_IMAGES, processes_per_model
            )
            results.append(result)
            
    elif num_models >= NUM_PROCESSES and NUM_IMAGES <= 50:
        # Many models, few images per model -> parallelize models
        print(f"\n=== USING MODEL-LEVEL PARALLELIZATION ===")
        print(f"Strategy: Many models ({num_models}), few images ({NUM_IMAGES})")
        print(f"Will process models in parallel, images sequentially within each model")
        
        # Prepare arguments for multiprocessing
        args = [(model, OUTPUT_BASE_PATH, AZ_RANGE, EL_RANGE, ROLL_RANGE, NUM_IMAGES) for model in models]
        
        # Render with multiprocessing
        with Pool(processes=NUM_PROCESSES) as pool:
            results = pool.map(render_single_model_wrapper, args)
            
    else:
        # Hybrid approach or balanced case
        print(f"\n=== USING HYBRID PARALLELIZATION ===")
        print(f"Strategy: Balanced case - models ({num_models}), images ({NUM_IMAGES})")
        
        if num_models <= NUM_PROCESSES // 2:
            # Use image-level parallelization with remaining processes
            processes_per_model = NUM_PROCESSES // num_models
            processes_per_model = min(processes_per_model, NUM_IMAGES)
            print(f"Using {processes_per_model} processes per model for image rendering")
            
            results = []
            for model in models:
                print(f"\nProcessing model {model[1]}...")
                result = render_model_with_parallel_images(
                    model, OUTPUT_BASE_PATH, AZ_RANGE, EL_RANGE, ROLL_RANGE, 
                    NUM_IMAGES, processes_per_model
                )
                results.append(result)
        else:
            # Fall back to model-level parallelization
            print(f"Falling back to model-level parallelization")
            args = [(model, OUTPUT_BASE_PATH, AZ_RANGE, EL_RANGE, ROLL_RANGE, NUM_IMAGES) for model in models]
            
            with Pool(processes=NUM_PROCESSES) as pool:
                results = pool.map(render_single_model_wrapper, args)
    
    # Record end time and calculate duration
    end_time = time.time()
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    duration = end_time - start_time
    duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))
    
    # Count successful results (handle both boolean and integer returns)
    success_count = 0
    for result in results:
        if isinstance(result, bool):
            success_count += 1 if result else 0
        elif isinstance(result, int):
            success_count += result
        else:
            success_count += 1  # Assume success for other types
    
    print(f"\n=== RENDERING COMPLETED ===")
    print(f"End time: {end_time_str}")
    print(f"Total duration: {duration_str}")
    print(f"Completed: {success_count}/{len(models)} successful")
    print(f"Total images generated: {success_count * NUM_IMAGES}")
    if len(models) > 0:
        print(f"Average time per model: {duration/len(models):.2f} seconds")


def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Render 3D models with specified parameters"
    )
    parser.add_argument("--category", type=str, required=True, help="category")
    parser.add_argument(
        "--begin_idx", type=int, required=True, help="Starting index (inclusive)"
    )
    parser.add_argument(
        "--end_idx", type=int, required=True, help="Ending index (exclusive)"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Number of images per model (default: 10)",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1)",
    )
    parser.add_argument(
        "--render_root",
        type=str,
        default=None,
        help="Optional output directory for rendered images.",
    )
    parser.add_argument(
        "--shapenet_root",
        type=str,
        default=None,
        help="Optional ShapeNetVox32 root directory.",
    )

    # When run through Blender, filter out Blender-specific args after '--'
    if "blender" in sys.argv[0].lower():
        try:
            script_args_start = sys.argv.index("--") + 1
            args_to_parse = sys.argv[script_args_start:]
        except ValueError:
            args_to_parse = []
        return parser.parse_args(args_to_parse)
    return parser.parse_args()


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    cli = _parse_cli_args()
    main(
        cli.category,
        cli.begin_idx,
        cli.end_idx,
        cli.num_images,
        cli.num_processes,
        cli.render_root,
        cli.shapenet_root,
    )
