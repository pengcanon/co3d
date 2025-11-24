"""
Debug script to visualize a single frame point cloud step by step.
"""
import numpy as np
import open3d as o3d
import gzip
import json
import os
from PIL import Image

def load_jgz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)

def load_16big_png_depth(depth_png):
    """Load CO3D depth maps stored as float16 in 16-bit PNG format."""
    with Image.open(depth_png) as pil_img:
        depth = np.frombuffer(
            np.array(pil_img, dtype=np.uint16), 
            dtype=np.float16
        ).reshape(pil_img.size[1], pil_img.size[0])
    return depth.astype(np.float32)

# Paths
dataset_root = r"d:\GitHub\co3d\custom_dataset_generator\output"
category = "human_body_0"
annotation_path = os.path.join(dataset_root, category, "frame_annotations.jgz")

# Load annotations
print("Loading annotations...")
annotations = load_jgz(annotation_path)
print(f"Loaded {len(annotations)} frames")

# Process four frames FROM THE SAME ELEVATION
# 100 frames / 4 elevations = 25 frames per elevation
# Frames 0-24: elevation -10°
# Frames 25-49: elevation 0°
# Frames 50-74: elevation 10°
# Frames 75-99: elevation 30°
frame_indices = [0, 5, 10, 15]  # All at -10° elevation, different azimuths
all_pcds = []
camera_markers = []

for frame_idx in frame_indices:
    frame = annotations[frame_idx]
    print(f"\n{'='*60}")
    print(f"Processing Frame {frame_idx}:")
    print(f"  Sequence: {frame['sequence_name']}")
    print(f"  Image: {frame['image']['path']}")

    # Load RGB
    rgb_path = os.path.join(dataset_root, frame['image']['path'])
    rgb = np.array(Image.open(rgb_path))
    print(f"  RGB shape: {rgb.shape}")

    # Load depth
    depth_path = os.path.join(dataset_root, frame['depth']['path'])
    depth_raw = load_16big_png_depth(depth_path)
    scale_adj = frame['depth']['scale_adjustment']
    depth_metric = depth_raw * scale_adj
    print(f"  Depth range: [{depth_metric.min():.3f}, {depth_metric.max():.3f}]")
    print(f"  Valid pixels: {(depth_metric > 0).sum()} / {depth_metric.size}")

    # Get camera parameters
    vp = frame['viewpoint']
    focal_length = vp['focal_length']
    principal_point = vp['principal_point']
    R = np.array(vp['R'])
    T = np.array(vp['T'])

    # Convert to pixel coordinates
    H, W = depth_metric.shape
    half_image_size = min(W, H) / 2.0
    fx_px = focal_length[0] * half_image_size
    fy_px = focal_length[1] * half_image_size
    px_px = principal_point[0] * half_image_size + W / 2.0
    py_px = principal_point[1] * half_image_size + H / 2.0

    # Create pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Valid mask
    valid = (depth_metric > 0) & np.isfinite(depth_metric)

    # Extract valid pixels
    u_valid = u[valid]
    v_valid = v[valid]
    depth_valid = depth_metric[valid]

    # TEST ALL 4 UNPROJECTION FORMULAS
    # PyTorch3D camera: +X=LEFT, +Y=UP, +Z=FORWARD
    # Image coords: u right (+x), v down (+y)
    formulas = {
        1: ("X = +(u-px)*Z/fx, Y = -(v-py)*Z/fy", 1, -1),
        2: ("X = -(u-px)*Z/fx, Y = -(v-py)*Z/fy", -1, -1),
        3: ("X = +(u-px)*Z/fx, Y = +(v-py)*Z/fy", 1, 1),
        4: ("X = -(u-px)*Z/fx, Y = +(v-py)*Z/fy", -1, 1),
    }
    
    # User: Change this number (1-4) to test different formulas
    TEST_FORMULA = 2
    
    desc, flip_x, flip_y = formulas[TEST_FORMULA]
    print(f"  Using formula {TEST_FORMULA}: {desc}")
    
    X_cam = flip_x * (u_valid - px_px) * depth_valid / fx_px
    Y_cam = flip_y * (v_valid - py_px) * depth_valid / fy_px
    Z_cam = depth_valid

    points_cam = np.stack([X_cam, Y_cam, Z_cam], axis=1)

    # Transform to world coordinates
    # PyTorch3D convention: X_cam = X_world @ R + T
    # Inverse: X_world = (X_cam - T) @ R^T
    points_world = (points_cam - T[None, :]) @ R.T
    
    # Camera center for visualization
    C = -T @ R.T
    
    print(f"  Camera center: [{C[0]:.3f}, {C[1]:.3f}, {C[2]:.3f}]")
    print(f"  Point cloud: {len(points_world)} points")

    # Get colors
    colors = rgb[valid].astype(np.float32) / 255.0

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    all_pcds.append(pcd)
    
    # Create camera marker
    camera_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    colors_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]  # Red, Green, Blue, Yellow
    camera_marker.paint_uniform_color(colors_list[frame_indices.index(frame_idx)])
    camera_marker.translate(C)
    camera_markers.append(camera_marker)

# Combine point clouds
print(f"\n{'='*60}")
print(f"Combining {len(all_pcds)} point clouds...")
combined_pcd = all_pcds[0]
for pcd in all_pcds[1:]:
    combined_pcd += pcd

print(f"Combined point cloud: {len(combined_pcd.points)} points")

# Add coordinate frame at origin
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

print("\nLaunching visualizer...")
print(f"Camera positions:")
for i, idx in enumerate(frame_indices):
    color_names = ["Red", "Green", "Blue", "Yellow"]
    print(f"  {color_names[i]} sphere = camera {idx}")
print("RGB axes = world origin (R=X, G=Y, B=Z)")
print("\nNOTE: Model is 17cm tall - use mouse wheel to zoom in!")

geometries = [combined_pcd, coord_frame] + camera_markers
o3d.visualization.draw_geometries(geometries, window_name="Debug Point Cloud - Four Frames")
