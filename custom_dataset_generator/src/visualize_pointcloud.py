"""
Visualize CO3D point clouds from depth maps and camera parameters.

Usage:
    python visualize_pointcloud.py --dataset_root path/to/dataset --category apple --frame_idx 0 --show_cameras
    python visualize_pointcloud.py --dataset_root path/to/dataset --category human_body_0 --sequence sequence_001 --stride 5 --show_cameras
"""

import argparse
import gzip
import json
import numpy as np
import open3d as o3d
import os
from PIL import Image


def load_jgz(path):
    """Load gzipped JSON file."""
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


def unproject_depth_to_pointcloud(
    depth_map,
    rgb_image,
    focal_length,
    principal_point,
    R,
    T,
    max_depth=None,
    intrinsics_format="ndc_isotropic",
    camera_convention="pytorch3d",
):
    """
    Unproject depth map to 3D point cloud using camera parameters.
    
    Args:
        depth_map: (H, W) depth in meters
        rgb_image: (H, W, 3) RGB image as numpy array
        focal_length: (fx, fy) focal lengths in NDC or pixel coordinates
        principal_point: (px, py) principal point in NDC or pixel coordinates
        R: (3, 3) rotation matrix (world to camera)
        T: (3,) translation vector (world to camera)
        max_depth: Maximum depth to include (filter far points)
        intrinsics_format: "ndc_isotropic" or "pixel"
    
    Returns:
        Open3D PointCloud object
    """
    H, W = depth_map.shape
    
    # Create pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Valid depth mask (non-zero, finite, and within range)
    valid = (depth_map > 0) & np.isfinite(depth_map)
    if max_depth is not None:
        valid = valid & (depth_map < max_depth)
    
    # Extract valid pixels
    u_valid = u[valid]
    v_valid = v[valid]
    depth_valid = depth_map[valid]
    
    # Convert focal length and principal point to pixel coordinates if needed
    fx, fy = focal_length
    px, py = principal_point
    
    if intrinsics_format == "ndc_isotropic":
        # NDC isotropic: normalized by min(W, H) / 2
        # For square images: f_px = f_ndc * (image_size / 2)
        half_image_size = min(W, H) / 2.0
        fx_px = fx * half_image_size
        fy_px = fy * half_image_size
        px_px = px * half_image_size + W / 2.0
        py_px = py * half_image_size + H / 2.0
    else:
        fx_px, fy_px = fx, fy
        px_px, py_px = px, py
    
    # Unproject to camera coordinates according to annotation convention.
    if camera_convention == "opencv":
        # OpenCV camera: +X right, +Y down, +Z forward
        X_cam = (u_valid - px_px) * depth_valid / fx_px
        Y_cam = (v_valid - py_px) * depth_valid / fy_px
    else:
        # PyTorch3D camera: +X left, +Y up, +Z forward
        X_cam = -(u_valid - px_px) * depth_valid / fx_px
        Y_cam = -(v_valid - py_px) * depth_valid / fy_px
    Z_cam = depth_valid
    
    points_cam = np.stack([X_cam, Y_cam, Z_cam], axis=1)  # (N, 3)
    
    # Transform to world coordinates.
    if camera_convention == "opencv":
        # OpenCV convention (column): X_cam = R @ X_world + T
        # Row form inverse: X_world = (X_cam - T) @ R
        points_world = (points_cam - T[None, :]) @ R
    else:
        # PyTorch3D convention (row): X_cam = X_world @ R + T
        points_world = (points_cam - T[None, :]) @ R.T

    # Extract colors
    colors = rgb_image[valid].astype(np.float32) / 255.0
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


def create_camera_frustum(
    R,
    T,
    focal_length,
    principal_point,
    image_size,
    scale=0.1,
    color=[1, 0, 0],
    intrinsics_format="ndc_isotropic",
    camera_convention="pytorch3d",
):
    """
    Create a wireframe frustum to visualize a camera in world coordinates.
    
    Args:
        R: (3, 3) rotation matrix (world to camera)
        T: (3,) translation vector (world to camera)
        focal_length: (fx, fy) focal lengths
        principal_point: (px, py) principal point
        image_size: (W, H) image dimensions
        scale: Scale factor for frustum size
        color: RGB color for frustum
        intrinsics_format: "ndc_isotropic" or "pixel"
    
    Returns:
        Open3D LineSet object
    """
    W, H = image_size
    fx, fy = focal_length
    px, py = principal_point
    
    # Convert to pixel coordinates if needed
    if intrinsics_format == "ndc_isotropic":
        half_image_size = min(W, H) / 2.0
        fx = fx * half_image_size
        fy = fy * half_image_size
        px = px * half_image_size + W / 2.0
        py = py * half_image_size + H / 2.0
    
    # Camera center in camera coordinates
    center_cam = np.array([0, 0, 0])
    
    # Image corners in camera coordinates at depth=scale
    if camera_convention == "opencv":
        corners_cam = np.array([
            [(0 - px) * scale / fx, (0 - py) * scale / fy, scale],      # Top-left
            [(W - px) * scale / fx, (0 - py) * scale / fy, scale],      # Top-right
            [(W - px) * scale / fx, (H - py) * scale / fy, scale],      # Bottom-right
            [(0 - px) * scale / fx, (H - py) * scale / fy, scale],      # Bottom-left
        ])
    else:
        corners_cam = np.array([
            [-(0 - px) * scale / fx, -(0 - py) * scale / fy, scale],      # Top-left
            [-(W - px) * scale / fx, -(0 - py) * scale / fy, scale],      # Top-right
            [-(W - px) * scale / fx, -(H - py) * scale / fy, scale],      # Bottom-right
            [-(0 - px) * scale / fx, -(H - py) * scale / fy, scale],      # Bottom-left
        ])

    # Transform to world coordinates.
    if camera_convention == "opencv":
        center_world = -T @ R
        corners_world = (corners_cam - T[None, :]) @ R
    else:
        center_world = -T @ R.T
        corners_world = (corners_cam - T[None, :]) @ R.T

    # Create frustum geometry
    points = np.vstack([center_world[None, :], corners_world])
    
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Lines from center to corners
        [1, 2], [2, 3], [3, 4], [4, 1]   # Image plane rectangle
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    return line_set


def visualize_frame(annotation, dataset_root, max_depth=10.0, show_camera=True, viewpoint_format="auto"):
    """
    Visualize a single frame as a point cloud.
    
    Args:
        annotation: Frame annotation dictionary
        dataset_root: Root directory of the dataset
        max_depth: Maximum depth to include
        show_camera: Whether to show camera frustum
    
    Returns:
        List of Open3D geometries to visualize
    """
    # Load RGB image
    rgb_path = os.path.join(dataset_root, annotation['image']['path'])
    if not os.path.exists(rgb_path):
        print(f"Warning: RGB image not found: {rgb_path}")
        return []
    
    rgb_img = np.array(Image.open(rgb_path))
    
    # Load depth map
    depth_path = os.path.join(dataset_root, annotation['depth']['path'])
    if not os.path.exists(depth_path):
        print(f"Warning: Depth map not found: {depth_path}")
        return []
    
    depth_raw = load_16big_png_depth(depth_path)
    depth_metric = depth_raw * annotation['depth']['scale_adjustment']
    
    # Check if depth is valid
    if (depth_metric > 0).sum() == 0:
        print(f"Warning: No valid depth in frame {annotation['image']['path']}")
        return []
    
    # Get camera parameters
    viewpoint = annotation['viewpoint']
    focal_length = viewpoint['focal_length']
    principal_point = viewpoint['principal_point']
    R = np.array(viewpoint['R'])
    T = np.array(viewpoint['T'])
    intrinsics_format = viewpoint.get('intrinsics_format', 'ndc_isotropic')
    if viewpoint_format == "opencv":
        camera_convention = "opencv"
    elif viewpoint_format == "pytorch3d":
        camera_convention = "pytorch3d"
    else:
        conv_tag = str(viewpoint.get('camera_convention', '')).lower()
        if 'opencv' in conv_tag or intrinsics_format == 'opencv_pixels':
            camera_convention = "opencv"
        else:
            camera_convention = "pytorch3d"

    print(f"Frame: {annotation['image']['path']}")
    print(f"  RGB shape: {rgb_img.shape}")
    print(f"  Depth range: [{depth_metric.min():.3f}, {depth_metric.max():.3f}] meters")
    print(f"  Valid depth pixels: {(depth_metric > 0).sum()} / {depth_metric.size}")
    print(f"  Camera convention: {camera_convention}")

    # Create point cloud
    pcd = unproject_depth_to_pointcloud(
        depth_metric, 
        rgb_img, 
        focal_length, 
        principal_point, 
        R, T,
        max_depth=max_depth,
        intrinsics_format=intrinsics_format,
        camera_convention=camera_convention,
    )
    
    print(f"  Point cloud: {len(pcd.points)} points")
    
    geometries = [pcd]
    
    # Add camera frustum
    if show_camera:
        H, W = rgb_img.shape[:2]
        # Use small fixed scale appropriate for the object size
        frustum_scale = 0.02  # 2cm frustum for visualization
        frustum = create_camera_frustum(R, T, focal_length, principal_point, (W, H), 
                                       scale=frustum_scale, color=[1, 0, 0], 
                                       intrinsics_format=intrinsics_format,
                                       camera_convention=camera_convention)
        geometries.append(frustum)
    
    return geometries


def visualize_sequence(
    annotations,
    dataset_root,
    sequence_name,
    stride=1,
    max_depth=10.0,
    show_cameras=True,
    viewpoint_format="auto",
):
    """
    Visualize multiple frames from a sequence as a combined point cloud.
    
    Args:
        annotations: List of all frame annotations
        dataset_root: Root directory of the dataset
        sequence_name: Name of the sequence to visualize
        stride: Only process every Nth frame
        max_depth: Maximum depth to include
        show_cameras: Whether to show camera frustums
    
    Returns:
        List of Open3D geometries to visualize
    """
    # Filter frames for this sequence
    sequence_frames = [anno for anno in annotations if anno['sequence_name'] == sequence_name]
    
    if len(sequence_frames) == 0:
        print(f"Error: No frames found for sequence '{sequence_name}'")
        return []
    
    print(f"Sequence '{sequence_name}': {len(sequence_frames)} frames")
    print(f"Processing every {stride} frame(s)...")
    
    all_geometries = []
    
    for i, anno in enumerate(sequence_frames[::stride]):
        print(f"\nProcessing frame {i * stride + 1}/{len(sequence_frames)}...")
        geometries = visualize_frame(
            anno,
            dataset_root,
            max_depth=max_depth,
            show_camera=show_cameras,
            viewpoint_format=viewpoint_format,
        )
        all_geometries.extend(geometries)
    
    return all_geometries


def main():
    parser = argparse.ArgumentParser(description="Visualize CO3D point clouds from depth maps")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root directory of the dataset")
    parser.add_argument("--category", type=str, required=True,
                        help="Category name (e.g., 'apple', 'human_body_0')")
    parser.add_argument("--frame_idx", type=int, default=None,
                        help="Index of single frame to visualize (0-based)")
    parser.add_argument("--sequence", type=str, default=None,
                        help="Sequence name to visualize (e.g., '110_13051_23361')")
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every Nth frame when visualizing sequence")
    parser.add_argument("--max_depth", type=float, default=10.0,
                        help="Maximum depth to include (meters)")
    parser.add_argument("--show_cameras", action="store_true",
                        help="Show camera frustums")
    parser.add_argument("--output", type=str, default=None,
                        help="Save point cloud to file (e.g., output.ply)")
    parser.add_argument(
        "--viewpoint_format",
        type=str,
        default="auto",
        choices=["auto", "pytorch3d", "opencv"],
        help="Camera convention in annotations; 'auto' uses metadata/intrinsics format",
    )

    args = parser.parse_args()
    
    # Construct annotation path from dataset_root and category
    args.annotation_path = os.path.join(args.dataset_root, args.category, "frame_annotations.jgz")
    
    # Load annotations
    print(f"Loading annotations from {args.annotation_path}...")
    annotations = load_jgz(args.annotation_path)
    print(f"Loaded {len(annotations)} frames.")
    
    # Generate geometries
    geometries = []
    
    if args.frame_idx is not None:
        # Visualize single frame
        if args.frame_idx < 0 or args.frame_idx >= len(annotations):
            print(f"Error: frame_idx {args.frame_idx} out of range [0, {len(annotations)-1}]")
            return
        
        print(f"\nVisualizing frame {args.frame_idx}...")
        geometries = visualize_frame(annotations[args.frame_idx], args.dataset_root, 
                                      max_depth=args.max_depth, show_camera=args.show_cameras,
                                      viewpoint_format=args.viewpoint_format)

    elif args.sequence is not None:
        # Visualize sequence
        geometries = visualize_sequence(annotations, args.dataset_root, args.sequence,
                                        stride=args.stride, max_depth=args.max_depth,
                                        show_cameras=args.show_cameras,
                                        viewpoint_format=args.viewpoint_format)

    else:
        print("Error: Must specify either --frame_idx or --sequence")
        return
    
    if len(geometries) == 0:
        print("No geometries to visualize.")
        return
    
    # Save if requested
    if args.output:
        # Combine all point clouds
        combined_pcd = o3d.geometry.PointCloud()
        for geom in geometries:
            if isinstance(geom, o3d.geometry.PointCloud):
                combined_pcd += geom
        
        print(f"\nSaving point cloud to {args.output}...")
        o3d.io.write_point_cloud(args.output, combined_pcd)
        print(f"Saved {len(combined_pcd.points)} points.")
    
    # Visualize
    print("\nLaunching Open3D visualizer...")
    print("TIP: Use mouse wheel to zoom, especially for small objects")
    
    # Add coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    geometries.append(coord_frame)
    
    o3d.visualization.draw_geometries(geometries, window_name="CO3D Point Cloud Visualization")


if __name__ == "__main__":
    main()
