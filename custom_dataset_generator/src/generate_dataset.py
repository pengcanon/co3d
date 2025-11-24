import argparse
import os
import sys
import json
import gzip
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add the parent directory to path to import co3d modules if needed
# Assuming we are running from the root of the workspace or similar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import renderer utils
from renderer_utils import load_model, get_camera_pose, render_view, get_co3d_viewpoint

def save_jgz(data, path):
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        json.dump(data, f)

def main():
    parser = argparse.ArgumentParser(description="Generate CO3D dataset from 3D model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .glb model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the dataset")
    parser.add_argument("--category", type=str, default="human_body", help="Category name for the dataset")
    parser.add_argument("--sequence_name", type=str, default="sequence_001", help="Sequence name")
    parser.add_argument("--num_views", type=int, default=100, help="Number of views to generate")
    parser.add_argument("--image_size", type=int, default=800, help="Image size (square)")
    
    args = parser.parse_args()
    
    print(f"Generating dataset for {args.model_path}...")
    print(f"Output directory: {args.output_dir}")
    
    # Setup directories
    seq_dir = os.path.join(args.output_dir, args.category, args.sequence_name)
    dirs = {
        "images": os.path.join(seq_dir, "images"),
        "depths": os.path.join(seq_dir, "depths"),
        "depth_masks": os.path.join(seq_dir, "depth_masks"),
        "masks": os.path.join(seq_dir, "masks"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        
    # Load model
    scene, bounds, centroid = load_model(args.model_path)
    
    frame_annotations = []
    
    # Calculate optimal camera distance
    # bounds is 2x3 [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    extents = bounds[1] - bounds[0]
    max_extent = np.max(extents)
    
    # FOV is 60 degrees (vertical).
    # We want the object to fit within the frame with some margin.
    # tan(fov/2) = (size/2) / distance
    # distance = (size/2) / tan(fov/2)
    fov_rad = np.radians(60)
    min_distance = (max_extent / 2.0) / np.tan(fov_rad / 2.0)
    
    # Add a margin (e.g., 1.2x to 1.5x)
    base_radius = min_distance * 1.5
    
    print(f"Object Bounds: {bounds}")
    print(f"Object Centroid: {centroid}")
    print(f"Max Extent: {max_extent:.4f}")
    print(f"Calculated Base Radius: {base_radius:.4f}")
    
    # Generate views (Turntable)
    # Radius can be adjusted or made an argument. 
    # For now, we assume the object is roughly unit size and centered.
    # radius = 3.0 
    elevations = [np.radians(e) for e in [-10, 0, 10, 30]]
    
    # We want to loop all views for one elevation first, then move to the next.
    # Total frames = num_views.
    # We should split num_views among the elevations.
    views_per_elevation = args.num_views // len(elevations)
    
    print("Rendering views...")
    for i in tqdm(range(args.num_views)):
        # Determine which elevation block we are in
        elev_idx = i // views_per_elevation
        # Clamp to last elevation if we have remainder frames
        if elev_idx >= len(elevations):
            elev_idx = len(elevations) - 1
            
        elevation = elevations[elev_idx]
        
        # Azimuth should complete a full circle (or part of it) for EACH elevation
        # Frame index within this elevation block
        frame_in_block = i % views_per_elevation
        azimuth = (2 * np.pi * frame_in_block) / views_per_elevation
        
        # Vary radius slightly for diversity if desired, or keep fixed
        radius = base_radius
        
        # Get Pose (Camera-to-World)
        # Look at the object centroid
        c2w = get_camera_pose(radius, azimuth, elevation, center=centroid)
        
        # Render
        color, depth, mask = render_view(scene, c2w, width=args.image_size, height=args.image_size)
        
        # Save files
        frame_num = i
        filename_base = f"frame{frame_num:06d}"
        
        # Image
        img_path_rel = os.path.join(args.category, args.sequence_name, "images", f"{filename_base}.jpg")
        img_path_abs = os.path.join(args.output_dir, img_path_rel)
        Image.fromarray(color).save(img_path_abs, quality=95)
        
        # Depth (16-bit png usually, or saved as raw float if needed, but CO3D uses png)
        # CO3D depth is stored as png with a scale adjustment.
        # We'll store depth in millimeters as uint16 for precision, or just use the float directly if we were using .npy
        # But CO3D spec says: "path to png file... storing depth / scale_adjustment"
        # Let's define scale_adjustment such that max depth fits in uint16.
        max_depth = np.max(depth)
        if max_depth == 0: max_depth = 1.0
        scale_adjustment = max_depth / 65535.0
        depth_uint16 = (depth / scale_adjustment).astype(np.uint16)
        
        depth_path_rel = os.path.join(args.category, args.sequence_name, "depths", f"{filename_base}.png")
        depth_path_abs = os.path.join(args.output_dir, depth_path_rel)
        Image.fromarray(depth_uint16).save(depth_path_abs)
        
        # Mask
        mask_path_rel = os.path.join(args.category, args.sequence_name, "masks", f"{filename_base}.png")
        mask_path_abs = os.path.join(args.output_dir, mask_path_rel)
        Image.fromarray(mask).save(mask_path_abs)
        
        # Depth Mask (same as mask for synthetic data usually)
        depth_mask_path_rel = os.path.join(args.category, args.sequence_name, "depth_masks", f"{filename_base}.png")
        depth_mask_path_abs = os.path.join(args.output_dir, depth_mask_path_rel)
        Image.fromarray(mask).save(depth_mask_path_abs)
        
        # Calculate Viewpoint Annotation
        # Use the helper from renderer_utils to convert OpenGL pose to CO3D format
        R, T = get_co3d_viewpoint(c2w)
        
        # Focal length and principal point
        # PyRender PerspectiveCamera: yfov is vertical FOV.
        # f_y = (H / 2) / tan(yfov / 2)
        # Assuming square pixels, f_x = f_y
        yfov = np.radians(60.0) # Matches renderer_utils default
        focal_length_px = (args.image_size / 2.0) / np.tan(yfov / 2.0)
        
        # NDC conversion for CO3D
        # "ndc_norm_image_bounds": [-1, 1] x [-1, 1]
        # focal_length_ndc = focal_length_px / (image_size / 2)
        focal_length_ndc = focal_length_px / (args.image_size / 2.0)
        
        # Principal point is usually center
        principal_point_ndc = [0.0, 0.0]
        
        frame_ann = {
            "sequence_name": args.sequence_name,
            "frame_number": frame_num,
            "frame_timestamp": float(frame_num) / 30.0, # Assuming 30fps
            "image": {
                "path": img_path_rel,
                "size": [args.image_size, args.image_size]
            },
            "depth": {
                "path": depth_path_rel,
                "scale_adjustment": float(scale_adjustment),
                "mask_path": depth_mask_path_rel
            },
            "mask": {
                "path": mask_path_rel,
                "mass": float(np.sum(mask) / 255.0)
            },
            "viewpoint": {
                "R": R,
                "T": T,
                "focal_length": [focal_length_ndc, focal_length_ndc],
                "principal_point": principal_point_ndc,
                "intrinsics_format": "ndc_norm_image_bounds"
            }
        }
        frame_annotations.append(frame_ann)

    # Save Frame Annotations
    # CO3D stores all frame annotations for a category in one file usually, 
    # but here we are generating a single sequence.
    # We will save it in the category folder.
    cat_dir = os.path.join(args.output_dir, args.category)
    frame_ann_path = os.path.join(cat_dir, "frame_annotations.jgz")
    save_jgz(frame_annotations, frame_ann_path)
    
    # Save Sequence Annotations
    seq_ann = {
        "sequence_name": args.sequence_name,
        "category": args.category,
        "viewpoint_quality_score": 1.0
    }
    seq_ann_path = os.path.join(cat_dir, "sequence_annotations.jgz")
    save_jgz([seq_ann], seq_ann_path)
    
    # Create Set Lists (train/val split)
    # We'll put all frames in 'manyview_dev_0' for simplicity
    set_lists_dir = os.path.join(cat_dir, "set_lists")
    os.makedirs(set_lists_dir, exist_ok=True)
    
    set_list_data = [
        (args.sequence_name, i, ann["image"]["path"]) 
        for i, ann in enumerate(frame_annotations)
    ]
    
    set_list_path = os.path.join(set_lists_dir, "set_lists_manyview_dev_0.json")
    with open(set_list_path, 'w') as f:
        json.dump(set_list_data, f)
        
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()
