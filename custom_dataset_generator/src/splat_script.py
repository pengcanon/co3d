"""
splat_script.py

Core rendering utilities for generating CO3D-format datasets from
Gaussian Splat .ply files trained with gsplat/Nerfstudio.

Designed to be called from generate_dataset.py via the .ply pipeline.
"""
import os
import sys
import math
import json
import gzip
import numpy as np
import torch
from plyfile import PlyData
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# renderer_utils is in the same directory
sys.path.append(os.path.dirname(__file__))
from renderer_utils import get_camera_pose, get_opencv_w2c

FOV_RAD = np.deg2rad(60.0)


# ---------------------------------------------------------------------------
# PLY Loading
# ---------------------------------------------------------------------------

def load_ply_splat(path, device="cuda"):
    """
    Loads a Nerfstudio/Splatfacto .ply file into torch tensors for gsplat.

    Expected vertex properties:
        x, y, z          — Gaussian centers
        opacity           — raw logits (sigmoid applied here)
        scale_0/1/2       — log scales (exp applied here)
        rot_0/1/2/3       — quaternion [w, x, y, z]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PLY file not found: {path}")

    print(f"  Loading PLY: {path}")
    plydata = PlyData.read(path)
    v = plydata['vertex']

    means3d = np.stack((v['x'], v['y'], v['z']), axis=-1)
    opacities = v['opacity']  # raw logits — sigmoid applied in render
    scales = np.exp(np.stack((v['scale_0'], v['scale_1'], v['scale_2']), axis=-1))

    if 'rot_0' in v.data.dtype.names:
        rots = np.stack((v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']), axis=-1)
    else:
        rots = np.zeros((means3d.shape[0], 4), dtype=np.float32)
        rots[:, 0] = 1.0  # identity quaternion

    # Spherical Harmonics coefficients.
    # Standard 3DGS PLY layout:
    #   f_dc_0/1/2      — degree-0 coefficient (1 band, 3 channels)
    #   f_rest_0 ..     — bands 1-3, stored as ALL coefficients for channel R,
    #                     then all for G, then all for B.
    # gsplat expects sh_coeffs: (N, (sh_degree+1)^2, 3)
    names = v.data.dtype.names
    N = means3d.shape[0]
    if 'f_dc_0' in names:
        f_dc = np.stack((v['f_dc_0'], v['f_dc_1'], v['f_dc_2']), axis=-1)  # (N, 3)

        # Count available f_rest fields
        rest_keys = sorted([k for k in names if k.startswith('f_rest_')],
                           key=lambda k: int(k.split('_')[-1]))
        num_rest_total = len(rest_keys)  # = num_rest_per_channel * 3

        if num_rest_total > 0 and num_rest_total % 3 == 0:
            num_rest_per_ch = num_rest_total // 3
            # Determine SH degree from number of coefficients beyond DC:
            # total bands = 1 + num_rest_per_ch  →  sh_degree = sqrt(1+num_rest_per_ch) - 1
            sh_degree = int(round(np.sqrt(1 + num_rest_per_ch) - 1))

            f_rest_arr = np.stack([v[k] for k in rest_keys], axis=-1)  # (N, num_rest_total)
            # Stored as [R0..R_n, G0..G_n, B0..B_n] → reshape to (N, 3, num_rest_per_ch)
            f_rest_ch = f_rest_arr.reshape(N, 3, num_rest_per_ch)
            # Reorder to (N, num_rest_per_ch, 3)
            f_rest_reordered = f_rest_ch.transpose(0, 2, 1)

            # Full SH: prepend DC band → (N, 1+num_rest_per_ch, 3) = (N, (sh_degree+1)^2, 3)
            sh_coeffs = np.concatenate(
                [f_dc[:, np.newaxis, :], f_rest_reordered], axis=1
            ).astype(np.float32)
        else:
            # Only DC available
            sh_degree = 0
            sh_coeffs = f_dc[:, np.newaxis, :].astype(np.float32)  # (N, 1, 3)
    else:
        # No SH at all — fall back to white
        sh_degree = 0
        sh_coeffs = np.ones((N, 1, 3), dtype=np.float32) * (0.5 / 0.28209479177387814)

    print(f"  Loaded {N} splats  (SH degree {sh_degree}, {sh_coeffs.shape[1]} bands)")

    return {
        'means':     torch.from_numpy(means3d).float().to(device),
        'scales':    torch.from_numpy(scales).float().to(device),
        'quats':     torch.from_numpy(rots).float().to(device),
        'opacities': torch.from_numpy(opacities).float().to(device),
        'sh_coeffs': torch.from_numpy(sh_coeffs).float().to(device),  # (N, K, 3)
        'sh_degree': sh_degree,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_view_gsplat(splat_data, c2w_gl, width, height, fov_rad=FOV_RAD, device="cuda"):
    """
    Renders a depth map using gsplat v1.0+ rasterization API.

    Args:
        splat_data: dict of torch tensors from load_ply_splat()
        c2w_gl:     (4,4) OpenGL Camera-to-World matrix (Y-up world)
        width, height: output resolution in pixels
        fov_rad:    vertical field of view in radians (default 60°)
        device:     torch device string

    Returns:
        depth_map:  (H, W) float32 numpy array — metric depth, 0 = background
        alpha_map:  (H, W) float32 numpy array — accumulated alpha [0, 1]
    """
    try:
        from gsplat.rendering import rasterization
    except ImportError as e:
        raise ImportError(
            f"gsplat not available: {e}\n"
            "Install with: pip install gsplat"
        )

    # Convert OpenGL c2w -> OpenCV w2c (matches generate_dataset.py convention)
    R_cv, T_cv = get_opencv_w2c(c2w_gl)
    R = torch.tensor(R_cv, dtype=torch.float32, device=device)
    T = torch.tensor(T_cv, dtype=torch.float32, device=device)

    viewmat = torch.eye(4, device=device)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = T

    fx = (width / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx
    K = torch.tensor(
        [[fx, 0, width / 2.0],
         [0, fy, height / 2.0],
         [0,  0,          1.0]],
        dtype=torch.float32, device=device
    )

    opacities = torch.sigmoid(splat_data['opacities'])
    sh_coeffs = splat_data['sh_coeffs']   # (N, K, 3)
    sh_degree = splat_data['sh_degree']   # int

    render_colors, render_alphas, _ = rasterization(
        means=splat_data['means'],
        quats=splat_data['quats'],
        scales=splat_data['scales'],
        opacities=opacities,
        colors=sh_coeffs,        # (N, K, 3) SH coefficients
        viewmats=viewmat.unsqueeze(0),   # (1, 4, 4)
        Ks=K.unsqueeze(0),               # (1, 3, 3)
        width=width,
        height=height,
        sh_degree=sh_degree,     # gsplat evaluates SH per view direction
        render_mode="RGB+ED",            # last channel = Expected Depth
    )

    # render_colors: (1, H, W, 4) = [R, G, B, depth]; render_alphas: (1, H, W, 1)
    rgb_map   = render_colors[0, ..., :3]              # (H, W, 3)
    depth_map = render_colors[0, ...,  3].squeeze()    # (H, W)
    alpha_map = render_alphas[0].squeeze()             # (H, W)

    # Zero out background
    depth_map = depth_map.clone()
    depth_map[alpha_map < 0.5] = 0.0

    return rgb_map.cpu().numpy(), depth_map.cpu().numpy(), alpha_map.cpu().numpy()


# ---------------------------------------------------------------------------
# Camera placement helpers
# ---------------------------------------------------------------------------

def estimate_scene_params(splat_data):
    """
    Estimates scene center and a suitable orbit radius from splat positions.

    Returns:
        center: (3,) numpy array
        radius: float
    """
    means = splat_data['means'].cpu().numpy()
    center = np.median(means, axis=0)
    dists = np.linalg.norm(means - center, axis=1)
    radius = max(np.percentile(dists, 95) * 2.5, 0.1)
    return center, radius


def build_turntable_cameras(num_views, elevations_deg=None):
    """
    Returns a list of (azimuth_rad, elevation_rad) tuples for a turntable.

    Args:
        num_views:       total number of views
        elevations_deg:  list of elevation angles in degrees. Views are evenly
                         distributed across elevations. Default matches the
                         generate_dataset.py PyRender pipeline: [-10, 0, 10, 30, 45].
    """
    if elevations_deg is None:
        elevations_deg = [-10, 0, 10, 30, 45]

    views_per_elev = max(1, num_views // len(elevations_deg))
    cameras = []
    for i in range(num_views):
        elev_idx = min(i // views_per_elev, len(elevations_deg) - 1)
        elev_rad = np.radians(elevations_deg[elev_idx])
        frame_in_block = i % views_per_elev
        azim_rad = (2 * np.pi * frame_in_block) / views_per_elev
        cameras.append((azim_rad, elev_rad))
    return cameras


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def make_opencv_viewpoint(c2w_gl, image_size, focal_length_px):
    """Builds a viewpoint annotation dict in OpenCV convention."""
    R_cv, T_cv = get_opencv_w2c(c2w_gl)
    principal = (image_size - 1) / 2.0
    return {
        "R": R_cv,
        "T": T_cv,
        "focal_length": [focal_length_px, focal_length_px],
        "principal_point": [principal, principal],
        "intrinsics_format": "opencv_pixels",
        "camera_convention": "opencv_w2c",
    }


# ---------------------------------------------------------------------------
# Main dataset generation pipeline
# ---------------------------------------------------------------------------

def generate_splat_dataset(
    ply_path,
    output_dir,
    category,
    sequence_name,
    num_views=100,
    image_size=800,
    radius=0.0,
    fov_deg=60.0,
    elevations_deg=None,
    device=None,
):
    """
    Generates a CO3D-format dataset (images/depths/masks + annotations) from a
    Gaussian Splat .ply file.

    Mirrors the structure produced by the PyRender pipeline in generate_dataset.py:
        output_dir/
          category/
            sequence_name/
              images/          ← RGB renders  (JPG)
              depths/          ← float16-bitcast uint16 PNG  (CO3D format)
              masks/           ← uint8 PNG
              depth_masks/     ← uint8 PNG
            frame_annotations.jgz
            sequence_annotations.jgz
            set_lists/
              set_lists_manyview_dev_0.json

    Returns:
        frame_annotations: list of annotation dicts
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    fov_rad = np.radians(fov_deg)

    # Directories -----------------------------------------------------------------
    seq_dir = os.path.join(output_dir, category, sequence_name)
    dirs = {
        "images":      os.path.join(seq_dir, "images"),
        "depths":      os.path.join(seq_dir, "depths"),
        "masks":       os.path.join(seq_dir, "masks"),
        "depth_masks": os.path.join(seq_dir, "depth_masks"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Load splats -----------------------------------------------------------------
    splat_data = load_ply_splat(ply_path, device=device)

    # Scene geometry --------------------------------------------------------------
    center, auto_radius = estimate_scene_params(splat_data)
    if radius <= 0:
        radius = auto_radius
    print(f"  Center: {center}  Radius: {radius:.3f}")

    # Camera schedule -------------------------------------------------------------
    cameras = build_turntable_cameras(num_views, elevations_deg)

    focal_length_px = (image_size / 2.0) / math.tan(fov_rad / 2.0)

    frame_annotations = []

    print("  Rendering views...")
    for frame_num, (azim_rad, elev_rad) in enumerate(tqdm(cameras)):
        filename_base = f"frame{frame_num:06d}"

        c2w = get_camera_pose(radius, azim_rad, elev_rad, center=center)

        rgb_map, depth_map, alpha_map = render_view_gsplat(
            splat_data, c2w, image_size, image_size, fov_rad=fov_rad, device=device
        )

        # ── Image (RGB from SH DC component) ────────────────────────────────────────
        rgb_uint8 = (rgb_map * 255).clip(0, 255).astype(np.uint8)
        img_path_rel = os.path.join(category, sequence_name, "images", f"{filename_base}.jpg")
        img_path_abs = os.path.join(output_dir, img_path_rel)
        Image.fromarray(rgb_uint8).save(img_path_abs, quality=95)

        # ── Depth (float16 bitcast → uint16 PNG — CO3D convention) ──────────────────
        depth_float16 = depth_map.astype(np.float16)
        depth_uint16 = np.frombuffer(depth_float16.tobytes(), dtype=np.uint16).reshape(depth_map.shape)
        depth_path_rel = os.path.join(category, sequence_name, "depths", f"{filename_base}.png")
        depth_path_abs = os.path.join(output_dir, depth_path_rel)
        Image.fromarray(depth_uint16).save(depth_path_abs)

        # ── Mask ──────────────────────────────────────────────────────────────────────
        mask = (alpha_map > 0.7).astype(np.uint8) * 255
        mask_path_rel = os.path.join(category, sequence_name, "masks", f"{filename_base}.png")
        mask_path_abs = os.path.join(output_dir, mask_path_rel)
        Image.fromarray(mask).save(mask_path_abs)

        dmask_path_rel = os.path.join(category, sequence_name, "depth_masks", f"{filename_base}.png")
        dmask_path_abs = os.path.join(output_dir, dmask_path_rel)
        Image.fromarray(mask).save(dmask_path_abs)

        # ── Annotation ───────────────────────────────────────────────────────────────
        viewpoint = make_opencv_viewpoint(c2w, image_size, focal_length_px)
        frame_ann = {
            "sequence_name": sequence_name,
            "frame_number":  frame_num,
            "frame_timestamp": float(frame_num) / 30.0,
            "image":  {"path": img_path_rel,   "size": [image_size, image_size]},
            "depth":  {"path": depth_path_rel, "scale_adjustment": 1.0, "mask_path": dmask_path_rel},
            "mask":   {"path": mask_path_rel,  "mass": float(np.sum(mask) / 255.0)},
            "viewpoint": viewpoint,
        }
        frame_annotations.append(frame_ann)

    return frame_annotations
