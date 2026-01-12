import argparse
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import sys
import json
import gzip
import numpy as np
import shutil
import glob
import subprocess
try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image
from tqdm import tqdm

# Add the parent directory to path to import co3d modules if needed
# Assuming we are running from the root of the workspace or similar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import renderer utils
from renderer_utils import load_model, get_camera_pose, render_view, get_co3d_viewpoint

def load_jgz(path):
    if not os.path.exists(path):
        return []
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {path}: {e}")
        return []

def save_jgz(data, path):
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        json.dump(data, f)

def append_and_save_jgz(new_data, path, key_field=None):
    """
    Appends new_data (list) to existing data in path.
    If key_field is provided, it avoids duplicates by checking that field.
    """
    existing = load_jgz(path)
    if not isinstance(existing, list):
        print(f"Warning: Existing file {path} is not a list. Overwriting.")
        existing = []
        
    if key_field:
        existing_keys = {item.get(key_field) for item in existing}
        for item in new_data:
            if item.get(key_field) not in existing_keys:
                existing.append(item)
            else:
                # Optionally update? For now, skip to preserve existing?
                # Or replace? CO3D dataset generation usually implies adding new frames.
                # If we re-run, we might want to update.
                # Let's simple append new ones. 
                pass
    else:
        existing.extend(new_data)
        
    save_jgz(existing, path)

def append_and_save_json(new_data, path):
    """
    Appends new_data (list) to existing JSON file.
    """
    if os.path.exists(path):
        with open(path, 'r') as f:
            existing = json.load(f)
    else:
        existing = []
        
    if not isinstance(existing, list):
         existing = []
         
    existing.extend(new_data)
    
    with open(path, 'w') as f:
        json.dump(existing, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Generate CO3D dataset from 3D model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .glb or .obj model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the dataset")
    parser.add_argument("--category", type=str, default="human_body", help="Category name for the dataset")
    parser.add_argument("--sequence_name", type=str, default="sequence_001", help="Sequence name")
    parser.add_argument("--num_views", type=int, default=100, help="Number of views to generate")
    parser.add_argument("--image_size", type=int, default=800, help="Image size (square)")
    parser.add_argument("--scale_adjustment", type=float, default=1.0, help="Manual scale adjustment factor (default 1.0)")
    parser.add_argument("--blender_path", type=str, default="blender", help="Path to blender executable")
    parser.add_argument("--num_sequences", type=int, default=1, help="Number of random time sequences to generate (Blender only)")
    
    args = parser.parse_args()
    
    print(f"Generating dataset for {args.model_path}...")
    print(f"Output directory: {args.output_dir}")
    print(f"Scale Adjustment: {args.scale_adjustment}")
    
    # Setup directories
    seq_dir = os.path.join(args.output_dir, args.category, args.sequence_name)
    dirs = {
        "images": os.path.join(seq_dir, "images"),
        "depths": os.path.join(seq_dir, "depths"),
        "depth_masks": os.path.join(seq_dir, "depth_masks"),
        "masks": os.path.join(seq_dir, "masks"),
    }
    # Only create if NOT blender logic, or use as temp for blender (blender script creates its own)
    # The new blender logic uses seq_dir as a Root for all sequences? 
    # Or output_dir/category is the root, and we create sequence folders inside.
    # User said: "current argument sequence_name should not be used ... each sample time frame will generate dataset under a different sequence name"
    
    # So if Blender, we use args.output_dir/args.category as base, and subfolders will be create dynamically.
    if args.model_path.lower().endswith('.blend') or args.model_path.lower().endswith('.fbx'):
        # Use tempfile to create a truly temporary directory that we control
        import tempfile
        
        # We will use this content manager to ensure cleanup
        with tempfile.TemporaryDirectory() as temp_build_dir:
            temp_build_dir = os.path.abspath(temp_build_dir) # Use absolute path for Blender
            
            is_fbx = args.model_path.lower().endswith('.fbx')
            
            if is_fbx:
                script_name = "blender_fbx_script.py"
                script_path = os.path.join(os.path.dirname(__file__), script_name)
                # For FBX, we run empty blender and import via script
                cmd = [
                    args.blender_path,
                    "-b", 
                    "-P", script_path,
                    "--",
                    args.model_path,   # Arg 0 for script
                    temp_build_dir,    # Arg 1
                    str(args.num_views),
                    str(args.image_size),
                    str(args.scale_adjustment),
                    str(args.num_sequences)
                ]
            else:
                # .blend execution
                script_name = "blender_script.py"
                script_path = os.path.join(os.path.dirname(__file__), script_name)
                cmd = [
                    args.blender_path,
                    "-b", args.model_path,
                    "-P", script_path,
                    "--",
                    temp_build_dir,
                    str(args.num_views),
                    str(args.image_size),
                    str(args.scale_adjustment),
                    str(args.num_sequences)
                ]
            
            print(f"Running Blender ({script_name}): {' '.join(cmd)}")
            try:
                import cv2
                subprocess.check_call(cmd)
            except Exception as e:
                print(f"Blender Execution Error: {e}")
                return

            # Post-Processing
            meta_path = os.path.join(temp_build_dir, "metadata.json")
            if not os.path.exists(meta_path):
                 print(f"Error: Metadata file not found at {meta_path}. Blender likely failed.")
                 return

            with open(meta_path, "r") as f:
                meta_data = json.load(f)
                
            all_frame_annotations = []
            unique_sequences = set()
            
            print("Processing Blender outputs...")
            for meta in tqdm(meta_data):
                filename_base = meta['filename_base']
                seq_name = meta['sequence_name'] # e.g. frame_000123
                
                # Check if we should override sequence name if num_sequences=1?
                # For consistency with OBJ pipeline which uses 'sequence_name' arg...
                # But Blender renders actual animation frames. 
                # Keeping 'seq_name' from Blender is safer for multi-frame support.
                
                unique_sequences.add(seq_name)
                
                # Destination folders for this specific sequence
                # output/category/seq_name/...
                dest_seq_dir = os.path.join(args.output_dir, args.category, seq_name)
                dest_dirs = {
                    "images": os.path.join(dest_seq_dir, "images"),
                    "depths": os.path.join(dest_seq_dir, "depths"),
                    "depth_masks": os.path.join(dest_seq_dir, "depth_masks"),
                    "masks": os.path.join(dest_seq_dir, "masks"),
                }
                for d in dest_dirs.values():
                    os.makedirs(d, exist_ok=True)
                    
                # Move/Process Image
                src_img = os.path.join(temp_build_dir, "images", f"{filename_base}.jpg")
                dst_img_rel = os.path.join(args.category, seq_name, "images", f"{filename_base}.jpg")
                dst_img_abs = os.path.join(args.output_dir, dst_img_rel)
                
                if os.path.exists(src_img):
                    shutil.copy(src_img, dst_img_abs)
                    print(f"  [Image] Copied to: {dst_img_abs}")
                else:
                    print(f"Missing image: {src_img}")
                    continue
                
                # Depth Processing
                temp_depth_dir = os.path.join(temp_build_dir, "temp_depth")
                
                # Look for NPY first (New Blender Script)
                search_pattern_npy = os.path.join(temp_depth_dir, f"{filename_base}*.npy")
                files = glob.glob(search_pattern_npy)
                
                depth_img = None
                
                if files:
                    file_path = files[0]
                    try:
                        # Load NPY
                        depth_img = np.load(file_path)
                        # NPY is already (H, W) float32
                        
                        # Ensure it is 2D
                        if len(depth_img.shape) == 3:
                            depth_img = depth_img[:, :, 0]
                            
                    except Exception as e:
                         print(f"NPY read failed: {e}")
                
                else:
                    # Fallback to TIFF/EXR/HDR
                    possible_exts = [".hdr", ".exr", ".tif", ".tiff"]
                    files = []
                    for ext in possible_exts:
                        search_pattern = os.path.join(temp_depth_dir, f"{filename_base}*{ext}")
                        files = glob.glob(search_pattern)
                        if files: break
                    
                    if files:
                        file_path = files[0]
                        # Attempt to read with OpenCV first
                        depth_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                        
                        if depth_img is None:
                             print(f"DEBUG: cv2.imread returned None for {file_path}. Trying imageio.")
                             try:
                                import imageio.v2 as imageio
                                depth_img = imageio.imread(file_path)
                             except Exception as e:
                                print(f"ImageIO read failed: {e}")

                if depth_img is None:
                    print(f"Warning: No valid depth file (npy/exr/tif) found for {filename_base}")
                    continue
                    
                # Ensure it is 2D (take Red channel)
                if len(depth_img.shape) == 3:
                    depth_img = depth_img[:, :, 0]
                    
                depth_map = depth_img

                    
                # Convert
                depth_float16 = depth_map.astype(np.float16)
                depth_uint16 = np.frombuffer(depth_float16.tobytes(), dtype=np.uint16).reshape(depth_map.shape)
                
                dst_depth_rel = os.path.join(args.category, seq_name, "depths", f"{filename_base}.png")
                dst_depth_abs = os.path.join(args.output_dir, dst_depth_rel)
                Image.fromarray(depth_uint16).save(dst_depth_abs)
                
                # Mask
                mask = ((depth_map > 0) & (depth_map < 1000.0)).astype(np.uint8) * 255
                
                dst_mask_rel = os.path.join(args.category, seq_name, "masks", f"{filename_base}.png")
                dst_mask_abs = os.path.join(args.output_dir, dst_mask_rel)
                Image.fromarray(mask).save(dst_mask_abs)
                
                dst_dmask_rel = os.path.join(args.category, seq_name, "depth_masks", f"{filename_base}.png")
                dst_dmask_abs = os.path.join(args.output_dir, dst_dmask_rel)
                Image.fromarray(mask).save(dst_dmask_abs)
                
                # Annotations
                c2w = np.array(meta['c2w_matrix'])
                R, T = get_co3d_viewpoint(c2w)
                
                # Calculate Focal Length
                # Prefer metadata values if available (from Blender)
                if 'focal_length' in meta and 'sensor_width' in meta:
                    fl_mm = meta['focal_length']
                    sw_mm = meta['sensor_width']
                    # Focal length in NDC: fx_ndc = 2 * fl_mm / sw_mm
                    if sw_mm > 0:
                        focal_length_ndc = 2.0 * fl_mm / sw_mm
                    else:
                        yfov = np.radians(60.0)
                        focal_length_px = (args.image_size / 2.0) / np.tan(yfov / 2.0)
                        focal_length_ndc = focal_length_px / (args.image_size / 2.0)
                else:
                    yfov = np.radians(60.0)
                    focal_length_px = (args.image_size / 2.0) / np.tan(yfov / 2.0)
                    focal_length_ndc = focal_length_px / (args.image_size / 2.0)
                
                frame_ann = {
                    "sequence_name": seq_name,
                    "frame_number": meta['index'], # Index within SEQUENCE
                    "frame_timestamp": meta['timestamp'],
                    "image": {"path": dst_img_rel, "size": [args.image_size, args.image_size]},
                    "depth": {"path": dst_depth_rel, "scale_adjustment": 1.0, "mask_path": dst_dmask_rel},
                    "mask": {"path": dst_mask_rel, "mass": float(np.sum(mask) / 255.0)},
                    "viewpoint": {
                        "R": R, "T": T,
                        "focal_length": [focal_length_ndc, focal_length_ndc],
                        "principal_point": [0.0, 0.0],
                        "intrinsics_format": "ndc_isotropic"
                    },
                    "meta": {"original_frame": meta['frame_number']}
                }
                all_frame_annotations.append(frame_ann)
                
            # Cleanup - HANDLED BY CONTEXT MANAGER
            print(f"Debug: cleaned up temp build dir")
        
        # Save Annotations
        print("Saving Blender annotations...")
        
        # Frame Annotations (Global for category)
        ann_path = os.path.join(args.output_dir, args.category, "frame_annotations.jgz")
        append_and_save_jgz(all_frame_annotations, ann_path)
        
        # Sequence Annotations
        seq_ann_path = os.path.join(args.output_dir, args.category, "sequence_annotations.jgz")
        seq_ann = []
        for seq in unique_sequences:
            seq_ann.append({
                "sequence_name": seq,
                "category": args.category,
                "viewpoint_quality_score": 1.0
            })
        append_and_save_jgz(seq_ann, seq_ann_path, key_field="sequence_name")
        
        # Update Set Lists
        set_lists_dir = os.path.join(args.output_dir, args.category, "set_lists")
        os.makedirs(set_lists_dir, exist_ok=True)
        
        set_list_data = []
        for ann in all_frame_annotations:
             set_list_data.append((ann["sequence_name"], ann["frame_number"], ann["image"]["path"]))
             
        set_list_path = os.path.join(set_lists_dir, "set_lists_manyview_dev_0.json")
        append_and_save_json(set_list_data, set_list_path)
        
        print(f"Dataset generation complete (Blender Pipeline). Generated {len(unique_sequences)} sequences.")
        return

    # Create directories for PyRender pipeline
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Load model (PyRender path) for obj and glb files
    scene, bounds, centroid = load_model(args.model_path, normalize=False, scale_adjustment=args.scale_adjustment)
    
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
        
        # Depth - CO3D uses float16 values stored in 16-bit PNG format
        # The depth is stored as: depth_png = depth_metric / scale_adjustment
        # We use scale_adjustment = 1.0 and store depth directly as float16
        scale_adjustment = 1.0
        
        # Convert depth to float16, then reinterpret as uint16 for PNG storage
        depth_float16 = depth.astype(np.float16)
        depth_uint16 = np.frombuffer(depth_float16.tobytes(), dtype=np.uint16).reshape(depth.shape)
        
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
        # For square images, ndc_isotropic is the same as normalizing by half image size
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
                "intrinsics_format": "ndc_isotropic"
            }
        }
        frame_annotations.append(frame_ann)

    # Save Frame Annotations
    # CO3D stores all frame annotations for a category in one file usually.
    cat_dir = os.path.join(args.output_dir, args.category)
    frame_ann_path = os.path.join(cat_dir, "frame_annotations.jgz")
    append_and_save_jgz(frame_annotations, frame_ann_path)
    
    # Save Sequence Annotations
    seq_ann = {
        "sequence_name": args.sequence_name,
        "category": args.category,
        "viewpoint_quality_score": 1.0
    }
    seq_ann_path = os.path.join(cat_dir, "sequence_annotations.jgz")
    append_and_save_jgz([seq_ann], seq_ann_path, key_field="sequence_name")
    
    # Create Set Lists (train/val split)
    # We'll put all frames in 'manyview_dev_0' for simplicity
    set_lists_dir = os.path.join(cat_dir, "set_lists")
    os.makedirs(set_lists_dir, exist_ok=True)
    
    set_list_data = [
        (args.sequence_name, i, ann["image"]["path"]) 
        for i, ann in enumerate(frame_annotations)
    ]
    
    set_list_path = os.path.join(set_lists_dir, "set_lists_manyview_dev_0.json")
    append_and_save_json(set_list_data, set_list_path)
        
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()
