import numpy as np
import pyrender
import trimesh
import os

def load_model(model_path):
    """
    Load a 3D model (GLB/GLTF/OBJ) into a pyrender Scene.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # trimesh handles loading and texture resolution
    trimesh_scene = trimesh.load(model_path)
    
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
