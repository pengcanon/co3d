import argparse
import os
import sys
import torch
import numpy as np
import math
from plyfile import PlyData
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent dir to path to import renderer_utils
# This file is in custom_dataset_generator/src/, and renderer_utils.py is in the same dir
sys.path.append(os.path.dirname(__file__))
try:
    from renderer_utils import get_camera_pose
except ImportError:
    # Fallback if running from root
    sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
    from renderer_utils import get_camera_pose

def load_ply_splat(path, device="cuda"):
    """
    Loads a Nerfstudio/Splatfacto .ply file into torch tensors for gsplat.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PLY file not found: {path}")

    print(f"Loading PLY: {path}")
    plydata = PlyData.read(path)
    v = plydata['vertex']
    
    # 1. Means (Positions)
    means3d = np.stack((v['x'], v['y'], v['z']), axis=-1)
    
    # 2. Opacities
    # Nerfstudio/Splatfacto usually stores opacities as raw logits.
    # We will assume they are logits and apply sigmoid during rendering if needed.
    opacities = v['opacity']
    
    # 3. Scales
    # Splatfacto writes log scales. gsplat expects scales (exp).
    scales = np.stack((v['scale_0'], v['scale_1'], v['scale_2']), axis=-1)
    scales = np.exp(scales) 
    
    # 4. Rotations (Quaternions)
    # Splatfacto keys: rot_0, rot_1, rot_2, rot_3
    if 'rot_0' in v:
        rots = np.stack((v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']), axis=-1)
    else:
        # Fallback if names differ
        rots = np.zeros((means3d.shape[0], 4))
        rots[:, 0] = 1.0 # Identity
        
    print(f"Loaded {means3d.shape[0]} splats")
    
    return {
        'means': torch.from_numpy(means3d).float().to(device),
        'scales': torch.from_numpy(scales).float().to(device),
        'quats': torch.from_numpy(rots).float().to(device),
        'opacities': torch.from_numpy(opacities).float().to(device),
    }

def render_view_gsplat(splat_data, c2w, width, height, fov_rad=np.deg2rad(60), device="cuda"):
    """
    Renders depth using gsplat.
    """
    try:
        from gsplat.project_gaussians import project_gaussians
        from gsplat.rasterize import rasterize_gaussians
    except ImportError:
        print("Error: gsplat not installed. Please install with `pip install gsplat`")
        sys.exit(1)
    
    # World-to-Camera
    w2c = np.linalg.inv(c2w)
    R = torch.from_numpy(w2c[:3, :3]).float().to(device)
    T = torch.from_numpy(w2c[:3, 3]).float().to(device)
    
    # Intrinsics
    fx = (width / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    
    # 1. Project
    means3d = splat_data['means']
    scales = splat_data['scales']
    glob_scale = 1.0
    quats = splat_data['quats']
    
    viewmat = torch.eye(4, device=device)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = T
    
    # Attempt to call project_gaussians (args vary by version)
    # Trying v1.0+ signature: (means3d, scales, glob_scale, quats, viewmat, fx, fy, cx, cy, img_height, img_width, tile_bounds)
    try:
        xys, depths, radii, conics, conic_opacity, num_tiles_hit, cov3d = project_gaussians(
            means3d, scales, glob_scale, quats, viewmat, 
            fx, fy, cx, cy, height, width, 
            None # tile_bounds
        )
    except Exception as e:
        # Fallback for older versions needing block_width
        xys, depths, radii, conics, conic_opacity, num_tiles_hit, cov3d = project_gaussians(
            means3d, scales, glob_scale, quats, viewmat, 
            fx, fy, cx, cy, height, width, 
            16 # block width 
        )

    # 2. Rasterize Depth
    # Treat depth as the "color" feature (N, 1)
    depth_features = depths.unsqueeze(-1)
    
    # Opacities
    # Standard Nerfstudio PLY opacities are logits -> Apply Sigmoid
    opacities = torch.sigmoid(splat_data['opacities']).unsqueeze(-1)
    
    render_depth, render_alpha = rasterize_gaussians(
        xys, depth_features, opacities, radii, conics, num_tiles_hit, 
        height, width
    )
    
    # 3. Normalize (Expected Depth)
    # render_depth is weighted sum. Divide by accumulated alpha to get average depth.
    render_depth = render_depth / (render_alpha + 1e-6)
    
    # Mask out background
    render_depth[render_alpha < 0.5] = 0.0
    
    return render_depth.squeeze().cpu().numpy(), render_alpha.squeeze().cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Test depth rendering from Splat file")
    parser.add_argument("--ply_path", type=str, required=True, help="Path to .ply file")
    parser.add_argument("--output_dir", type=str, default="output/splat_depth_test", help="Output directory")
    parser.add_argument("--num_views", type=int, default=5, help="Number of test views")
    parser.add_argument("--image_size", type=int, default=800, help="Image size")
    parser.add_argument("--radius", type=float, default=0.0, help="Orbit radius (0 = auto-estimate)")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load Data
    splat_data = load_ply_splat(args.ply_path, device=device)
    
    # Analyze geometry
    means = splat_data['means'].cpu().numpy()
    center = np.median(means, axis=0) # Median is more robust to outliers
    
    # Radius estimation
    if args.radius > 0:
        radius = args.radius
    else:
        # Percentile distance
        dists = np.linalg.norm(means - center, axis=1)
        radius = np.percentile(dists, 95) * 2.5
        if radius < 0.1: radius = 3.0
    
    print(f"Center: {center}")
    print(f"Radius: {radius:.2f}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Render loop
    elevations = [0.0] * args.num_views
    azimuths = np.linspace(0, 2*np.pi, args.num_views, endpoint=False)
    
    print(f"Rendering to {args.output_dir}...")
    for i in tqdm(range(args.num_views)):
        azimuth = azimuths[i]
        elevation = elevations[i]
        
        c2w = get_camera_pose(radius, azimuth, elevation, center=center)
        
        try:
            depth_map, alpha_map = render_view_gsplat(
                splat_data, c2w, args.image_size, args.image_size, device=device
            )
        except Exception as e:
            print(f"\nRender failed: {e}")
            import traceback
            traceback.print_exc()
            break
            
        # 1. Save Raw Depth (16-bit PNG)
        # Scale by 1000 (mm)
        depth_uint16 = (depth_map * 1000).astype(np.uint16)
        Image.fromarray(depth_uint16).save(os.path.join(args.output_dir, f"depth_{i:04d}.png"))
        
        # 2. Save Visualization (Color mapped JPG)
        # Normalize min-max
        valid_depth = depth_map[depth_map > 0]
        if len(valid_depth) > 0:
            d_min, d_max = valid_depth.min(), valid_depth.max()
            depth_norm = (depth_map - d_min) / (d_max - d_min + 1e-5)
            # Mask background
            depth_norm[depth_map == 0] = 0
            
            # Apply colormap
            try:
                plt.imsave(os.path.join(args.output_dir, f"viz_depth_{i:04d}.jpg"), depth_norm, cmap='turbo')
            except ValueError:
                 # Handle all-zero or NaN
                 pass
            
        # 3. Save Mask/Alpha
        alpha_img = (alpha_map * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(alpha_img).save(os.path.join(args.output_dir, f"alpha_{i:04d}.png"))
        
    print(f"Done. Outputs in {args.output_dir}")

if __name__ == "__main__":
    main()
