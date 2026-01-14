import bpy
import os
import sys
import random
import json
import struct
import math
import numpy as np
import mathutils

def init_scene(output_dir):
    # Ensure Eevee
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.quality = 95
    
    # Setup Compositor for Depth Output
    scene.use_nodes = True
    tree = scene.node_tree
    links = tree.links
    
    # Clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)
        
    # Create Render Layers
    rl = tree.nodes.new('CompositorNodeRLayers')
    
    # Create File Output for Depth (Radiance HDR for compatibility)
    depth_out = tree.nodes.new('CompositorNodeOutputFile')
    depth_out.base_path = os.path.join(output_dir, "temp_depth")
    # RADIANCE supports floating point data (RGBE)
    depth_out.format.file_format = 'HDR' 
    # depth_out.format.color_depth is not relevant for RADIANCE, it's always float-ish
    
    # Link Z (Depth) to Input
    links.new(rl.outputs['Depth'], depth_out.inputs[0])
    
    return depth_out

def looks_like_human(obj):
    # Heuristic: Valid mesh, visible, not a floor/plane
    if obj.type != 'MESH': return False
    if obj.hide_render: return False
    name = obj.name.lower()
    if 'floor' in name or 'plane' in name or 'ground' in name: return False
    return True

def get_target_center():
    # Find the main object to look at
    targets = [o for o in bpy.context.scene.objects if looks_like_human(o)]
    print(f"DEBUG: Found {len(targets)} target objects: {[o.name for o in targets]}")
    
    if not targets:
        return np.array([0, 0, 0]), 1.0

    
    # Calculate geometric center of bounds from EVALUATED mesh (deformed)
    depsgraph = bpy.context.view_layer.depsgraph
    
    min_pt = np.array([float('inf')] * 3)
    max_pt = np.array([float('-inf')] * 3)
    
    has_valid_mesh = False
    
    for obj in targets:
        # Get evaluated object (with modifiers like Armature applied)
        obj_eval = obj.evaluated_get(depsgraph)
        
        # Create temp mesh to get deformed verts
        mesh = obj_eval.to_mesh()
        if mesh:
            has_valid_mesh = True
            # Transform verts to world space
            verts = [obj.matrix_world @ v.co for v in mesh.vertices]
            # Convert to numpy for range finding (could be large, but usually fine for simple characters)
            if len(verts) > 0:
                vs = np.array(verts)
                # Find min/max of this objects verts
                obj_min = np.min(vs, axis=0)
                obj_max = np.max(vs, axis=0)
                
                print(f"DEBUG: Object {obj.name} bounds: Min={obj_min}, Max={obj_max}")

                min_pt = np.minimum(min_pt, obj_min)
                max_pt = np.maximum(max_pt, obj_max)
            else:
                print(f"DEBUG: Object {obj.name} has 0 vertices.")
                
            obj_eval.to_mesh_clear()
    
    if not has_valid_mesh:
         print("DEBUG: No valid mesh data found in targets.")
         return np.array([0, 0, 0]), 1.0

    center = (min_pt + max_pt) / 2.0
    size = np.linalg.norm(max_pt - min_pt)
    print(f"DEBUG: Calculated Sequence Bounds - Center: {center}, Size: {size}")
    
    # Padding size slightly so we don't crop toes/hair
    size = size * 1.1 # Reduced padding
    
    return center, size

def setup_lighting():
    # Remove existing lights
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)
            
    # 1. Strong Key Light (Sun)
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    sun = bpy.context.object
    sun.data.energy = 1.5  # Reduced from 30.0 for less contrast
    # Angle it to look down at the center roughly
    sun.rotation_euler = (math.radians(45), math.radians(15), math.radians(30))
    
    # 2. Fill Light (Area)
    bpy.ops.object.light_add(type='AREA', location=(5, -5, 5))
    fill = bpy.context.object
    fill.data.energy = 2000.0 # High fill to reduce contrast
    fill.data.size = 10.0
    fill.rotation_euler = (math.radians(60), 0, math.radians(45))

    # 3. Camera / Headlight (Point)
    # Useful to fill forward-facing shadows
    bpy.ops.object.light_add(type='POINT', location=(0, 0, 0))
    headlight = bpy.context.object
    headlight.name = "Headlight"
    headlight.data.energy = 800.0 # Moderate headlight
    headlight.data.shadow_soft_size = 2.0
    
    # Parent to Camera
    cam = bpy.data.objects.get("Camera")
    if cam:
        headlight.parent = cam
        headlight.location = (0, 0, 0) # At camera origin

    # 4. Global Ambient
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs[1].default_value = 6.0 # High ambient to lift shadows

def get_camera_pose_matrix(cam_obj):
    """
    Get 4x4 OpenGL Camera-to-World matrix.
    Blender: +Y Up, -Z Forward for Camera Local?
    Blender Camera: -Z view direction, +Y up, +X right.
    This MATCHES OpenGL convention.
    
    HOWEVER, Blender World is Z-up. PyRender/CO3D usually expect Y-up World.
    We need to rotate the MATRIX so that it represents the camera in a Y-up world.
    
    Rotation: Rotate -90 degrees around X axis.
    (x, y, z)_blender -> (x, z, -y)_gl_world
    
    M_new = R_x(-90) @ M_old
    """
    m = cam_obj.matrix_world
    
    # 4x4 Blender Matrix
    # Columns: R_col0, R_col1, R_col2, T
    m_np = np.array([
        [m[0][0], m[0][1], m[0][2], m[0][3]],
        [m[1][0], m[1][1], m[1][2], m[1][3]],
        [m[2][0], m[2][1], m[2][2], m[2][3]],
        [m[3][0], m[3][1], m[3][2], m[3][3]]
    ])
    
    # Rotate World Coordinate System from Z-up (Blender) to Y-up (OpenGL/PyRender)
    # Rotation -90 deg around X
    # [1  0  0  0]
    # [0  0  1  0]
    # [0 -1  0  0]
    # [0  0  0  1]
    
    rot_mat = np.array([
        [1.0, 0.0,  0.0, 0.0],
        [0.0, 0.0,  1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0,  0.0, 1.0]
    ])
    
    # We want to express the camera pose in the new coordinate system.
    # P_new = Rot @ P_old
    # Camera Center: C_new = Rot @ C_old
    # Rotation Basis: R_new = Rot @ R_old
    # So M_new = Rot @ M_old
    
    m_new = rot_mat @ m_np
    
    return m_new

def main():
    # Set seed for reproducibility (avoids creating new folders every run if settings change)
    # random.seed(42)
    # np.random.seed(42)
    pass

    # Parse Args
    # sys.argv includes blender args. Everything after "--" is ours.
    try:
        args_idx = sys.argv.index("--") + 1
    except ValueError:
        print("Error: Arguments must be separated by '--'")
        return

    our_args = sys.argv[args_idx:]
    
    output_dir = our_args[0]
    num_views = int(our_args[1])
    image_size = int(our_args[2])
    scale_adjustment = float(our_args[3])
    num_sequences = int(our_args[4])
    
    # 1. Setup Scene
    scene = bpy.context.scene
    scene.render.resolution_x = image_size
    scene.render.resolution_y = image_size
    
    # Ensure we use Eevee for speed
    scene.render.engine = 'BLENDER_EEVEE'
    
    # Setup folders
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    # 2. Setup Camera
    # Create Camera FIRST so Headlight can parent to it in setup_lighting
    cam = bpy.data.objects.get("Camera")
    if not cam:
        cam_data = bpy.data.cameras.new("Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        scene.collection.objects.link(cam)
    scene.camera = cam
    cam.data.clip_start = 0.1
    cam.data.clip_end = 1000.0 # Ensure far plane is enough

    # Setup Compositor for Depth
    depth_node = init_scene(output_dir)
    
    # 1.5 Setup Lighting
    setup_lighting()

    # Frame Range
    frame_start = scene.frame_start
    frame_end = scene.frame_end
    
    # Sample random frames for sequences
    # If not enough frames, we might duplicate, but usually 4D assets have many frames.
    available_frames = list(range(frame_start, frame_end + 1))
    if num_sequences > len(available_frames):
        # Repetition needed
        sampled_frames = [random.choice(available_frames) for _ in range(num_sequences)]
    else:
        # Strided Stratified Sampling
        # Divide timeline into N buckets and pick 1 from each to ensure time coverage
        # This prevents picking two very close frames (e.g. 10 and 11) randomly.
        sampled_frames = []
        # Use floating point step to handle uneven divisions fairly
        step = len(available_frames) / float(num_sequences)
        
        for i in range(num_sequences):
             idx_start = int(i * step)
             idx_end = int((i + 1) * step)
             
             # Handle edge case where step is small or rounding makes start=end
             if idx_end <= idx_start:
                 idx_end = idx_start + 1
             
             # Clamp
             idx_start = max(0, min(idx_start, len(available_frames) - 1))
             idx_end = max(0, min(idx_end, len(available_frames)))
             
             if idx_start == idx_end:
                 # Single frame available
                 frame = available_frames[idx_start]
             else:
                 frame = random.choice(available_frames[idx_start:idx_end])
                 
             sampled_frames.append(frame)
             
        # Shuffle to randomize processing order
        random.shuffle(sampled_frames)

    # Apply Scale Adjustment (Relative)
    # The original script overwrote scale with (1.0, 1.0, 1.0), breaking imported assets that use 0.01 scale.
    # We now MULTIPLY existing scale.
    if scale_adjustment != 1.0:
        targets = [o for o in bpy.context.scene.objects if looks_like_human(o)]
        for obj in targets:
            s = obj.scale
            obj.scale = (s[0] * scale_adjustment, s[1] * scale_adjustment, s[2] * scale_adjustment)

    # 3. Determine Object Center & Size
    # We do this at frame 0 (or start)
    center, size = get_target_center()
    print(f"Target Center: {center}, Size: {size}")

    # Camera Distance Logic
    # FOV 60 deg (matches renderer_utils.py default)
    fov = 60 * (math.pi / 180.0)
    cam.data.angle = fov 
    
    dist = (size / 2.0) / math.tan(fov / 2.0)
    radius = dist * 1.5  # Standard margin
    
    # Metadata list
    meta_data = []

    global_index = 0

    for seq_idx, frame_num in enumerate(sampled_frames):
        # Set the time frame once for this sequence
        scene.frame_set(frame_num)
        # Update scene to ensure objects move
        bpy.context.view_layer.update()
        
        # Recalculate Center & Size for this specific frame (Animation Support)
        center, size = get_target_center()
        # Recalculate camera distance based on new size
        dist = (size / 2.0) / math.tan(fov / 2.0)
        radius = dist * 1.1 # Reduced padding
        
        # Determine Sequence Name
        # e.g. "frame_000123"
        seq_name = f"frame_{frame_num:06d}"
        print(f"Generating Sequence: {seq_name} (Frame {frame_num}) - Center: {center}")

        # Define Turntable Views (consistent with generate_dataset.py)
        elevations = [math.radians(e) for e in [-10, 0, 10, 30, 50]]
        views_per_elevation = num_views // len(elevations)

        for i in range(num_views):
            # 2. Pick Camera Angle (Structured Turntable)
            # Determine which elevation block we are in
            elev_idx = i // views_per_elevation
            # Clamp to last elevation if we have remainder frames
            if elev_idx >= len(elevations):
                elev_idx = len(elevations) - 1
                
            elevation = elevations[elev_idx]
            
            # Azimuth should complete a full circle (or part of it) for EACH elevation
            # Frame index within this elevation block
            frame_in_block = i % views_per_elevation
            if views_per_elevation > 1:
                azimuth = (2 * math.pi * frame_in_block) / views_per_elevation
            else:
                azimuth = 0.0

            # Spherical coords
            # Z-up Spherical Conversion:
            # x = r * cos(elev) * cos(azi)  <-- Correction: Azi 0 is +X axis
            # y = r * cos(elev) * sin(azi)
            # z = r * sin(elev)
            x = radius * math.cos(elevation) * math.cos(azimuth)
            y = radius * math.cos(elevation) * math.sin(azimuth)
            z = radius * math.sin(elevation)
            
            # Map to Blender Coords (Assuming Z up).
            # Blender: +X Right, +Y Forward, +Z Up.
            cam_pos = center + np.array([x, y, z])
            
            cam.location = cam_pos
            
            # Look At (track constraint or manual rotation)
            # direction = Target - Camera
            direction = mathutils.Vector(center - cam_pos)
            
            # NOTE: track_quat('-Z', 'Y') might introduce roll or inconsistencies if direct Up alignment isn't perfect.
            # Using look_at helper equivalent.
            rot_quat = direction.to_track_quat('-Z', 'Y') # Camera looks down -Z, Up is Y
            cam.rotation_euler = rot_quat.to_euler()
            
            # Update dependency graph
            bpy.context.view_layer.update()
            
            # 3. Render
            # Filepath pattern - Use global index to avoid collision
            filename = f"render_{global_index:08d}"
            scene.render.filepath = os.path.join(img_dir, filename + ".jpg")
            
            # Output Node filename format
            depth_node.file_slots[0].path = filename + "_" 
            
            # Render
            bpy.ops.render.render(write_still=True)
            
            # 4. Save Metadata (Pose, timestamp)
            # Get Matrix
            c2w = get_camera_pose_matrix(cam)
            
            meta = {
                "index": i, # Index within sequence
                "global_index": global_index,
                "filename_base": filename,
                "sequence_name": seq_name,
                "frame_number": frame_num, # The time frame
                "timestamp": frame_num / 30.0,
                "c2w_matrix": c2w.tolist(),
                "focal_length": cam.data.lens, # mm
                "sensor_width": cam.data.sensor_width, # mm
                "params": {"radius": radius, "azimuth": azimuth, "elevation": elevation}
            }
            meta_data.append(meta)
            
            global_index += 1

    # Save all metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta_data, f, indent=2)

if __name__ == "__main__":
    main()
