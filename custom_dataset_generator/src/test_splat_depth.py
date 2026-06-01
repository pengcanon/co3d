import argparse
import os
import sys
import math
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from splat_script import (
    load_ply_splat,
    render_view_gsplat,
    estimate_scene_params,
    make_opencv_viewpoint,
)
from renderer_utils import get_camera_pose

def main():
    parser = argparse.ArgumentParser(description="Test depth rendering from Splat file")
    parser.add_argument("--ply_path", type=str, required=True, help="Path to .ply file")
    parser.add_argument("--output_dir", type=str, default="output/splat_depth_test", help="Output directory")
    parser.add_argument("--num_views", type=int, default=5, help="Number of test views")
    parser.add_argument("--image_size", type=int, default=800, help="Image size")
    parser.add_argument("--radius", type=float, default=0.0, help="Orbit radius (0 = auto-estimate)")
    
    args = parser.parse_args()
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load Data
    splat_data = load_ply_splat(args.ply_path, device=device)
    
    # Analyze geometry
    center, auto_radius = estimate_scene_params(splat_data)
    radius = args.radius if args.radius > 0 else auto_radius
    
    print(f"Center: {center}")
    print(f"Radius: {radius:.2f}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    frame_annotations = []
    
    # Render loop
    elevations = [0.0] * args.num_views
    azimuths = np.linspace(0, 2*np.pi, args.num_views, endpoint=False)
    
    print(f"Rendering to {args.output_dir}...")
    for i in tqdm(range(args.num_views)):
        azimuth = azimuths[i]
        elevation = elevations[i]
        
        c2w = get_camera_pose(radius, azimuth, elevation, center=center)
        
        try:
            rgb_map, depth_map, alpha_map = render_view_gsplat(
                splat_data, c2w, args.image_size, args.image_size, device=device
            )
        except Exception as e:
            print(f"\nRender failed: {e}")
            import traceback
            traceback.print_exc()
            break
            
        # 1. Save Raw Depth
        # Use CO3D float16 -> uint16 bitcast convention to perfectly match generate_dataset.py
        depth_float16 = depth_map.astype(np.float16)
        depth_uint16 = np.frombuffer(depth_float16.tobytes(), dtype=np.uint16).reshape(depth_map.shape)
        Image.fromarray(depth_uint16).save(os.path.join(args.output_dir, f"depth_{i:04d}.png"))

        # 2. Save RGB image
        rgb_uint8 = (rgb_map * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb_uint8).save(os.path.join(args.output_dir, f"rgb_{i:04d}.jpg"), quality=95)
        
        # 3. Save Visualization (Color mapped depth JPG)
        valid_depth = depth_map[depth_map > 0]
        if len(valid_depth) > 0:
            d_min, d_max = valid_depth.min(), valid_depth.max()
            depth_norm = (depth_map - d_min) / (d_max - d_min + 1e-5)
            depth_norm[depth_map == 0] = 0
            try:
                plt.imsave(os.path.join(args.output_dir, f"viz_depth_{i:04d}.jpg"), depth_norm, cmap='turbo')
            except ValueError:
                pass
            
        # 4. Save Mask/Alpha
        alpha_img = (alpha_map * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(alpha_img).save(os.path.join(args.output_dir, f"alpha_{i:04d}.png"))
        
        # 5. Save annotation
        yfov = np.radians(60.0)
        focal_length_px = (args.image_size / 2.0) / np.tan(yfov / 2.0)
        
        viewpoint = make_opencv_viewpoint(c2w, args.image_size, focal_length_px)
        
        frame_ann = {
            "sequence_name": "splat_test",
            "frame_number": i,
            "image": {
                "path": f"viz_depth_{i:04d}.jpg",
                "size": [args.image_size, args.image_size]
            },
            "depth": {
                "path": f"depth_{i:04d}.png",
                "scale_adjustment": 1.0,
                "mask_path": f"alpha_{i:04d}.png"
            },
            "viewpoint": viewpoint
        }
        frame_annotations.append(frame_ann)
        
    # Save the annotations JSON file following OpenCV convention
    with open(os.path.join(args.output_dir, "frame_annotations_opencv.json"), "w") as f:
        json.dump(frame_annotations, f, indent=2)
        
    print(f"Done. Outputs in {args.output_dir}")

if __name__ == "__main__":
    main()
