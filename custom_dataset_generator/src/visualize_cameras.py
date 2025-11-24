import argparse
import gzip
import json
import numpy as np
import open3d as o3d
import os

def load_jgz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)

def create_camera_frustum(scale=0.1, color=[1, 0, 0]):
    """
    Create a wireframe frustum to visualize a camera.
    The frustum points along the +Z axis.
    Includes local axes: Red (+X), Green (+Y), Blue (+Z).
    """
    # Vertices of the frustum in Camera coordinates
    # PyTorch3D: +X Left, +Y Up, +Z Forward
    # We draw a pyramid with base at Z=scale
    
    # Base corners
    # Top-Left: (+X, +Y) -> [scale, scale, scale]
    # Top-Right: (-X, +Y) -> [-scale, scale, scale]
    # Bottom-Right: (-X, -Y) -> [-scale, -scale, scale]
    # Bottom-Left: (+X, -Y) -> [scale, -scale, scale]
    
    points = [
        [0, 0, 0],                  # 0: Center
        [scale, scale, scale],      # 1: Top-Left
        [-scale, scale, scale],     # 2: Top-Right
        [-scale, -scale, scale],    # 3: Bottom-Right
        [scale, -scale, scale]      # 4: Bottom-Left
    ]
    
    # Lines connecting the vertices
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4], # Lines from center to corners
        [1, 2], [2, 3], [3, 4], [4, 1]  # Image plane rectangle
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    # Add Local Axes
    # X-axis (Red) - Points Left (+X)
    # Y-axis (Green) - Points Up (+Y)
    # Z-axis (Blue) - Points Forward (+Z)
    axis_len = scale * 1.5
    axis_points = [
        [0, 0, 0],
        [axis_len, 0, 0],  # +X
        [0, axis_len, 0],  # +Y
        [0, 0, axis_len]   # +Z
    ]
    axis_lines = [[0, 1], [0, 2], [0, 3]]
    axis_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # R, G, B
    
    axis_set = o3d.geometry.LineSet()
    axis_set.points = o3d.utility.Vector3dVector(axis_points)
    axis_set.lines = o3d.utility.Vector2iVector(axis_lines)
    axis_set.colors = o3d.utility.Vector3dVector(axis_colors)
    
    # Combine
    return line_set + axis_set

def main():
    parser = argparse.ArgumentParser(description="Visualize CO3D camera poses in Open3D")
    parser.add_argument("--annotation_path", type=str, required=True, help="Path to frame_annotations.jgz")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale of camera frustums")
    parser.add_argument("--stride", type=int, default=1, help="Show every Nth camera")
    args = parser.parse_args()

    print(f"Loading annotations from {args.annotation_path}...")
    annotations = load_jgz(args.annotation_path)
    print(f"Loaded {len(annotations)} frames.")

    # Auto-calculate scale based on camera distances
    distances = []
    for frame in annotations:
        vp = frame.get('viewpoint')
        if vp:
            R = np.array(vp['R'])
            T = np.array(vp['T'])
            # C = -T @ R.T
            C = -np.dot(T, R.T)
            distances.append(np.linalg.norm(C))
            
    if distances:
        avg_dist = np.mean(distances)
        # A good frustum size is usually ~5% of the camera distance
        auto_scale = avg_dist * 0.05
        
        # If user kept the default (0.5), override it with auto_scale
        if args.scale == 0.5:
            print(f"Auto-adjusting frustum scale from {args.scale} to {auto_scale:.4f} (based on avg distance {avg_dist:.4f})")
            args.scale = auto_scale

    geometries = []
    
    # Add a coordinate frame at the origin
    # Scale coordinate frame to match frustum scale roughly
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.scale * 2.0, origin=[0, 0, 0])
    geometries.append(coord_frame)

    centers = []

    # Filter annotations based on stride
    annotations_to_show = annotations[::args.stride]
    print(f"Visualizing {len(annotations_to_show)} frames (stride={args.stride}).")

    for i, frame in enumerate(annotations_to_show):
        vp = frame.get('viewpoint')
        if not vp:
            continue
            
        # R and T from annotation
        # R is Row-Major World-to-Camera Rotation (M[:3, :3])
        # T is Row-Major World-to-Camera Translation (M[3, :3])
        R_row = np.array(vp['R'])
        T_row = np.array(vp['T'])
        
        # Construct the Row-Major World-to-Camera Matrix M_row
        # M_row = [[R00, R01, R02, 0],
        #          [R10, R11, R12, 0],
        #          [R20, R21, R22, 0],
        #          [Tx,  Ty,  Tz,  1]]
        # But standard linear algebra usually works with Column-Major matrices for transformation logic:
        # M_col = M_row.T
        
        # Let's work with Column-Major matrices for Open3D
        # M_col = [[R00, R10, R20, Tx],
        #          [R01, R11, R21, Ty],
        #          [R02, R12, R22, Tz],
        #          [0,   0,   0,   1]]
        
        R_col = R_row.T
        T_col = T_row.T # T_row is 1D array, so T_col is just the vector
        
        # World-to-Camera Matrix (Column-Major)
        w2c = np.eye(4)
        w2c[:3, :3] = R_col
        w2c[:3, 3] = T_col
        
        # Camera-to-World Matrix (Inverse of w2c)
        # This gives us the Camera Pose (Position and Orientation in World)
        c2w = np.linalg.inv(w2c)
        
        # Create a camera frustum
        # We use a color gradient to show sequence order (Red -> Blue)
        # Normalize i by the number of shown frames
        progress = i / max(1, len(annotations_to_show) - 1)
        color = [1.0 - progress, 0.0, progress]
        frustum = create_camera_frustum(scale=args.scale, color=color)
        
        # Apply transformation
        # Open3D uses Column-Major matrices for transformation
        frustum.transform(c2w)
        geometries.append(frustum)
        
        # Add camera axis (RGB = XYZ)
        # Size is relative to the frustum scale
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.scale * 0.6, origin=[0, 0, 0])
        axis.transform(c2w)
        geometries.append(axis)

        # Draw a line from camera center to world origin (to see where it is relative to 0,0,0)
        # This helps verify if the camera is "looking at" the origin
        center = c2w[:3, 3]
        line_to_origin = o3d.geometry.LineSet()
        line_to_origin.points = o3d.utility.Vector3dVector([center, [0, 0, 0]])
        line_to_origin.lines = o3d.utility.Vector2iVector([[0, 1]])
        line_to_origin.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8]]) # Grey line
        geometries.append(line_to_origin)

        centers.append(center)

    # Visualize
    print("Visualizing... (Close the window to exit)")
    o3d.visualization.draw_geometries(geometries, window_name="CO3D Camera Poses")

if __name__ == "__main__":
    main()
