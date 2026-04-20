import numpy as np
import pyrender
import trimesh
import os

def normalize_scene_geometry(scene_or_mesh, target_scale=1.0):
    """
    Centers the mesh/scene at origin and scales it to fit within a unit sphere (or box).
    
    Args:
        scene_or_mesh: trimesh.Scene or trimesh.Trimesh object
        target_scale: The max dimension of the normalized object (default: 1.0)
    
    Returns:
        The normalized scene/mesh object
        norm_metrics: dict containing 'original_scale', 'scale_factor', 'offset'
    """
    bounds = scene_or_mesh.bounds
    # Calculate extents (size)
    extents = bounds[1] - bounds[0]
    max_extent = np.max(extents)
    
    # Calculate scale factor to make max_extent == target_scale
    if max_extent <= 1e-6:
        scale_factor = 1.0
    else:
        scale_factor = target_scale / max_extent
        
    # Calculate translation to center
    centroid = scene_or_mesh.centroid
    translation = -centroid
    
    # Create transform matrices
    # 1. Translate to origin
    T_center = np.eye(4)
    T_center[:3, 3] = translation
    
    # 2. Scale
    S = np.eye(4)
    S[:3, :3] *= scale_factor
    
    # Combined Transform: Scale * Translate (Translate first, then Scale)
    T_final = S @ T_center
    
    # Apply transform
    scene_or_mesh.apply_transform(T_final)
    
    print(f"[Normalization] Original Scale (Max Dim): {max_extent:.4f}")
    print(f"[Normalization] Applied Scale Factor: {scale_factor:.4f}")
    print(f"[Normalization] New Scale (Max Dim): {target_scale:.4f}")
    
    return scene_or_mesh, {
        "scale_factor": scale_factor,
        "offset": translation,
        "original_max_dim": max_extent
    }

def load_model(model_path, normalize=False, scale_adjustment=1.0):
    """
    Load a 3D model (GLB/GLTF/OBJ) into a pyrender Scene.
    
    Args:
        model_path: Path to the 3D model file
        normalize: If True, centers the model and scales it to fit in a unit cube.
        scale_adjustment: Manual scaling factor to apply to the model (default: 1.0)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # trimesh handles loading and texture resolution
    trimesh_scene = trimesh.load(model_path, process=False)
    
    # Custom Scaling (Manual)
    if scale_adjustment != 1.0:
        print(f"[Custom Scale] Applying manual scale factor: {scale_adjustment}")
        matrix = np.eye(4)
        matrix[:3, :3] *= scale_adjustment
        trimesh_scene.apply_transform(matrix)
    
    # Check if texture is loaded correctly. 
    # For OBJs without MTL or failing to resolve, we might need to manually load texture.
    # Heuristic: look for a texture file with similar name in the same directory.
    if isinstance(trimesh_scene, trimesh.Trimesh):
        # It's a single mesh
        meshes = [trimesh_scene]
    elif isinstance(trimesh_scene, trimesh.Scene):
        # Use geometry dictionary
        meshes = list(trimesh_scene.geometry.values())
    else:
        meshes = []

    # Try to find a texture file in the same directory if missing
    # ONLY for OBJs. GLB/GLTF should be self-contained.
    texture_path = None
    if model_path.lower().endswith('.obj'):
        model_dir = os.path.dirname(model_path)
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        # Common texture extensions
        tex_exts = ['.jpg', '.png', '.jpeg']
        
        # 0. Check 'tex' subfolder for diffuse textures (common in some assets)
        tex_subdir = os.path.join(model_dir, 'tex')
        if os.path.exists(tex_subdir) and os.path.isdir(tex_subdir):
            # Look for files containing 'dif' (diffuse)
            potential_textures = []
            for f in os.listdir(tex_subdir):
                if 'dif' in f.lower() and os.path.splitext(f)[1].lower() in tex_exts:
                    potential_textures.append(os.path.join(tex_subdir, f))
            
            # Prefer higher resolution if possible (e.g. 8k over 2k)
            if potential_textures:
                # Sort to maybe pick 8k or just pick first
                # "8k" > "2k"
                potential_textures.sort(reverse=True) 
                texture_path = potential_textures[0]

        # 1. Try name match: model_name + .jpg
        if texture_path is None:
            for ext in tex_exts:
                p = os.path.join(model_dir, model_name + ext)
                if os.path.exists(p):
                    texture_path = p
                    break
                
        # 2. If not found, try searching for any image file that looks related (e.g. contains 'diffuse', or just share prefix)
        # Specifically for this case: rp_dennis_posed_004_100k.obj -> rp_dennis_posed_004_A.jpg
        # They share strict prefix "rp_dennis_posed_004_"
        if texture_path is None:
            # Try to match the "base" name (e.g. remove resolution suffix like _100k)
            # Split by underscore and try to match the prefix
            parts = model_name.split('_')
            # Try matching progressively shorter prefixes
            for i in range(len(parts), 0, -1):
                prefix = "_".join(parts[:i])
                # List files in dir
                for f in os.listdir(model_dir):
                    if f.startswith(prefix) and os.path.splitext(f)[1].lower() in tex_exts:
                        texture_path = os.path.join(model_dir, f)
                        break
                if texture_path:
                    break

    if texture_path:
        from PIL import Image
        print(f"Loading texture from: {texture_path}")
        try:
            image = Image.open(texture_path)
            # Apply to all meshes that don't have a valid image or have a dummy one
            for m in meshes:
                has_valid_texture = False
                if hasattr(m.visual, 'material') and hasattr(m.visual.material, 'image'):
                     # Check if it's a dummy 2x2 image (often white/grey placeholder)
                     if m.visual.material.image is not None and m.visual.material.image.size != (2, 2):
                         has_valid_texture = True
                
                if not has_valid_texture:
                     # Create a simple material with this texture
                     m.visual.material = trimesh.visual.material.SimpleMaterial(image=image)
        except Exception as e:
            print(f"Failed to load texture: {e}")

    # Normalize if requested
    if normalize:
        trimesh_scene, _ = normalize_scene_geometry(trimesh_scene)

    if isinstance(trimesh_scene, trimesh.Scene):

        scene = pyrender.Scene.from_trimesh_scene(trimesh_scene)
        bounds = trimesh_scene.bounds
        centroid = trimesh_scene.centroid
    else:
        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_trimesh(trimesh_scene))
        bounds = trimesh_scene.bounds
        centroid = trimesh_scene.centroid
        
    # Add some ambient light
    scene.ambient_light = [0.3, 0.3, 0.3]
    
    return scene, bounds, centroid

def get_camera_pose(radius, azimuth, elevation, center=None):
    """
    Calculate camera pose matrix (world-to-camera) looking at the center.
    
    Args:
        radius: Distance from center
        azimuth: Azimuth angle in radians
        elevation: Elevation angle in radians
        center: Target point to look at (default: [0, 0, 0])
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
        
    # Convert spherical to cartesian coordinates for camera position
    # Using Y-up convention for the world (Standard for GLB/Human models)
    # Elevation is angle from the XZ plane (Ground)
    # Azimuth is angle in the XZ plane
    
    y = radius * np.sin(elevation)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    
    camera_position = np.array([x, y, z]) + center
    
    # Construct Look-At Matrix
    # Forward vector (camera looks at center)
    forward = center - camera_position
    forward = forward / np.linalg.norm(forward)
    
    # Up vector (global Y-up)
    up = np.array([0.0, 1.0, 0.0])
    
    # Right vector
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        # Handle case where looking straight down/up
        # If looking down Y, right can be X
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    
    # Recompute Up vector to be orthogonal
    new_up = np.cross(right, forward)
    new_up = new_up / np.linalg.norm(new_up)
    
    # PyRender/OpenGL Camera Convention:
    # -Z is forward, +Y is up, +X is right
    # We calculated forward pointing to target, so camera forward is -forward
    # But the matrix we need for pyrender is the Camera-to-World transform (Pose)
    
    # Rotation matrix columns: [Right, Up, -Forward]
    R = np.column_stack([right, new_up, -forward])
    
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = camera_position
    
    return pose

def render_view(scene, camera_pose, width=800, height=800, fov_y_deg=60.0):
    """
    Render a single view of the scene.
    
    Returns:
        color: (H, W, 3) uint8 array
        depth: (H, W) float32 array
        mask: (H, W) uint8 array (0 or 255)
    """
    camera = pyrender.PerspectiveCamera(yfov=np.radians(fov_y_deg), aspectRatio=width/height)
    
    # Create a temporary node for the camera
    camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
    scene.add_node(camera_node)
    
    # Add a directional light attached to the camera
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    light_node = pyrender.Node(light=light, matrix=camera_pose)
    scene.add_node(light_node)
    
    # Render
    r = pyrender.OffscreenRenderer(width, height)
    # flags=pyrender.RenderFlags.RGBA ensures we get alpha channel if available, 
    # but for mask we usually check depth or specific object IDs. 
    # Simple approach: check depth > 0 for mask.
    color, depth = r.render(scene)
    
    # Generate mask from depth
    mask = (depth > 0).astype(np.uint8) * 255
    
    # Cleanup
    scene.remove_node(camera_node)
    scene.remove_node(light_node)
    r.delete()
    
    return color, depth, mask

def get_co3d_viewpoint(camera_pose_gl):
    """
    Convert an OpenGL camera pose (Camera-to-World) to CO3D viewpoint format.
    
    CO3D/PyTorch3D uses: X_cam = X_world @ R + T (row vectors)
    Where R is 3x3 with det(R) = +1 (proper rotation)
    
    Args:
        camera_pose_gl: 4x4 OpenGL Camera-to-World matrix
        
    Returns:
        R: 3x3 Rotation matrix (Row-Major list)
        T: 3-element Translation vector (list)
    """
    # 1. Invert to get World-to-Camera (OpenGL)
    c2w_gl = camera_pose_gl
    w2c_gl = np.linalg.inv(c2w_gl)
    
    # Extract rotation and translation
    R_w2c_gl = w2c_gl[:3, :3]  # World to camera rotation (OpenGL)
    T_w2c_gl = w2c_gl[:3, 3]    # Translation
    
    # 2. Convert coordinate systems
    # OpenGL camera: +X right, +Y up, -Z forward
    # PyTorch3D camera: +X LEFT, +Y up, +Z forward
    # Change of basis for camera coordinates: flip X and Z
    S_cam = np.diag([-1.0, 1.0, -1.0])
    
    # OpenGL world: +Y up
    # PyTorch3D world: +Y up (same)
    # No world coordinate change needed
    
    # Transform: X_cam_pt3d = S_cam @ X_cam_gl
    # For world-to-camera: X_cam_pt3d = S_cam @ R_w2c_gl @ X_world + S_cam @ T_w2c_gl
    # Since world coords are same: X_cam_pt3d = (S_cam @ R_w2c_gl) @ X_world + (S_cam @ T_w2c_gl)
    R_w2c_pt3d_colmajor = S_cam @ R_w2c_gl
    T_w2c_pt3d = S_cam @ T_w2c_gl
    
    # 3. Convert to row-major format
    # Column-major: X_cam = R @ X_world + T (column vectors)
    # Row-major: X_cam = X_world @ R + T (row vectors)
    # These are related by transpose: R_row = R_col.T
    R_w2c_pt3d_rowmajor = R_w2c_pt3d_colmajor.T
    
    return R_w2c_pt3d_rowmajor.tolist(), T_w2c_pt3d.tolist()


def get_opencv_w2c(camera_pose_gl):
    """
    Convert an OpenGL camera pose (Camera-to-World) to OpenCV world-to-camera extrinsics.

    OpenGL camera coordinates: +X right, +Y up, -Z forward
    OpenCV camera coordinates: +X right, +Y down, +Z forward

    Args:
        camera_pose_gl: 4x4 OpenGL Camera-to-World matrix

    Returns:
        R: 3x3 world-to-camera rotation matrix (column-vector convention)
        t: 3-element world-to-camera translation vector
    """
    c2w_gl = camera_pose_gl
    w2c_gl = np.linalg.inv(c2w_gl)

    R_w2c_gl = w2c_gl[:3, :3]
    t_w2c_gl = w2c_gl[:3, 3]

    # Convert camera coordinates from OpenGL to OpenCV by flipping Y and Z.
    S_cam = np.diag([1.0, -1.0, -1.0])
    R_w2c_cv = S_cam @ R_w2c_gl
    t_w2c_cv = S_cam @ t_w2c_gl

    return R_w2c_cv.tolist(), t_w2c_cv.tolist()

