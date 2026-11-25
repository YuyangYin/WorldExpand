"""Pipeline utility functions shared across rendering scripts.

Provides common I/O helpers (EXR loading, video saving, depth colorization, etc.)
so that pipeline scripts can import them instead of duplicating code.
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime


# ============================================================================
# Data loading
# ============================================================================

def load_exr_depth(exr_path):
    """Load EXR depth map.

    Args:
        exr_path: path to .exr file

    Returns:
        depth: [H, W] float32 numpy array
    """
    if not os.path.exists(exr_path):
        raise FileNotFoundError(f"EXR file not found: {exr_path}")
    depth = cv2.imread(exr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
    if depth is None:
        raise ValueError(f"Failed to load EXR file: {exr_path}")
    if len(depth.shape) == 3:
        depth = depth[:, :, 1]
    return depth


def load_position_rotation_from_json(json_path):
    """Load camera position and rotation from a JSON file.

    Args:
        json_path: path to JSON file with 'Position' and 'Rotation' keys

    Returns:
        position: dict with 'X', 'Y', 'Z'
        rotation: dict with 'Yaw', 'Pitch', 'Roll'
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['Position'], data['Rotation']


# ============================================================================
# Depth visualization
# ============================================================================

def colorize_depth(depth, mask=None, normalize=True, cmap='Spectral'):
    """Colorize depth map for visualization.

    Args:
        depth: [H, W] float32 numpy array
        mask: optional [H, W] bool mask (True = valid)
        normalize: whether to normalize disparity
        cmap: matplotlib colormap name

    Returns:
        colored: [H, W, 3] uint8 numpy array
    """
    import matplotlib
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        depth = np.where((depth > 0) & mask, depth, np.nan)
    disp = 1 / depth
    if normalize:
        min_disp, max_disp = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.99)
        disp = (disp - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


# ============================================================================
# Video I/O
# ============================================================================

def reencoder_video(video_path):
    """Re-encode video with ffmpeg for browser compatibility."""
    import subprocess
    temp_video = video_path.replace('.mp4', '_temp.mp4')
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        temp_video
    ]
    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    os.remove(video_path)
    os.rename(temp_video, video_path)


def save_video_from_frames(frames_tensor, output_path, fps=24):
    """Save a sequence of RGB frames (torch tensor) as an mp4 video.

    Args:
        frames_tensor: tensor of shape [N, H, W, 3], values in [0, 1]
        output_path: output .mp4 file path
        fps: frames per second
    """
    N, H, W, C = frames_tensor.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for i in range(N):
        frame_np = (frames_tensor[i].cpu().numpy() * 255.).astype(np.uint8)
        frame_bgr = frame_np[..., ::-1]
        writer.write(frame_bgr)
    writer.release()
    reencoder_video(output_path)


def save_depth_video_from_frames(depth_frames, mask_frames, output_path, fps=24):
    """Save a sequence of depth frames as a colorized mp4 video.

    Args:
        depth_frames: tensor of shape [N, H, W], depth values
        mask_frames: tensor of shape [N, H, W], boolean mask
        output_path: output .mp4 file path
        fps: frames per second
    """
    N, H, W = depth_frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for i in range(N):
        depth_np = depth_frames[i].cpu().numpy()
        mask_np = mask_frames[i].cpu().numpy().astype(bool)
        colored = colorize_depth(depth_np.copy(), mask=mask_np)
        colored_bgr = colored[..., ::-1]
        writer.write(colored_bgr)
    writer.release()
    reencoder_video(output_path)


def save_mask_video_from_frames(mask_frames, output_path, fps=24):
    """Save a sequence of mask frames as an mp4 video.

    Args:
        mask_frames: tensor of shape [N, H, W], values in [0, 1]
        output_path: output .mp4 file path
        fps: frames per second
    """
    N, H, W = mask_frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for i in range(N):
        mask_np = (mask_frames[i].cpu().numpy() * 255).astype(np.uint8)
        mask_bgr = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
        writer.write(mask_bgr)
    writer.release()
    reencoder_video(output_path)


# ============================================================================
# Performance logging
# ============================================================================

def save_perf_stats_to_file(perf_stats, log_path, case_info=None, extra_info=None):
    """Append performance stats for one case to a text file.

    Args:
        perf_stats: dict with keys like 'render_mode', 'num_frames', 'total_time', etc.
        log_path: path to the output txt file (append mode)
        case_info: optional case metadata dict
        extra_info: optional dict of additional key-value pairs to log
    """
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, 'a') as f:
        f.write("=" * 60 + "\n")
        f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if case_info:
            f.write(f"  Case: {case_info.get('directory', 'N/A')}\n")
        if extra_info:
            for k, v in extra_info.items():
                f.write(f"  {k}: {v}\n")
        if perf_stats:
            for k, v in perf_stats.items():
                f.write(f"  {k:<20s}: {v}\n")
        f.write("=" * 60 + "\n\n")


# ============================================================================
# Case naming / output directory helpers
# ============================================================================

def get_case_basename(case_info):
    """Construct case basename from directory + start/end frame.

    e.g. 'Scene/Scene/239/0113_0172'

    Args:
        case_info: dict with 'directory', 'start', 'end' keys

    Returns:
        basename: str
    """
    directory = case_info['directory']
    start_frame = os.path.basename(os.path.dirname(case_info['start']['jpg']))
    end_frame = os.path.basename(os.path.dirname(case_info['end']['jpg']))
    return f"{directory}/{start_frame}_{end_frame}"


def check_output_exists(output_dir, image_basename, required_files=None):
    """Check if required output files already exist.

    Args:
        output_dir: root output directory
        image_basename: case subdirectory name
        required_files: list of filenames to check. If None, uses the default
            set for the full pipeline (start.png, end.png, warp.png, etc.)

    Returns:
        True if all required files exist
    """
    case_output_dir = os.path.join(output_dir, image_basename)
    if required_files is None:
        required_files = [
            "start.png",
            "end.png",
            "warp.png",
            "concat.png",
            "mask.png",
            "global_mesh.ply",
            "global_pcd.ply",
        ]
    for filename in required_files:
        file_path = os.path.join(case_output_dir, filename)
        if not os.path.exists(file_path):
            return False
    return True


def check_pcd_output_exists(output_dir, required_files=None):
    """Check if PCD render outputs already exist.

    Args:
        output_dir: case output directory
        required_files: list of filenames to check

    Returns:
        True if all required files exist
    """
    if required_files is None:
        required_files = [
            "warp_pcd.png",
            "depth_pcd.npy",
            "depth_pcd_colored.png",
        ]
    for filename in required_files:
        if not os.path.exists(os.path.join(output_dir, filename)):
            return False
    return True


# ============================================================================
# PCD path resolution (for pipeline_pcd_render_multi_gpu)
# ============================================================================

def resolve_pcd_path(case_info, pcd_root=None):
    """Determine PCD file path for a case.

    - If case has warp_output.global_pcd_ply: use it directly
    - If pcd_root specified: pcd_root / case_basename / global_pcd.ply

    Args:
        case_info: case dict
        pcd_root: optional root directory containing pre-generated PCD files

    Returns:
        pcd_path: str
    """
    warp_output = case_info.get('warp_output')
    if warp_output and 'global_pcd_ply' in warp_output and pcd_root is None:
        return warp_output['global_pcd_ply']
    basename = get_case_basename(case_info)
    if pcd_root:
        return os.path.join(pcd_root, basename, 'global_pcd.ply')
    if warp_output and 'global_pcd_ply' in warp_output:
        return warp_output['global_pcd_ply']
    raise ValueError("Cannot determine PCD path: no warp_output and --pcd_root not specified")


def resolve_output_dir(case_info, output_dir_root):
    """Determine output directory for a case.

    Args:
        case_info: case dict
        output_dir_root: custom output root (or None)

    Returns:
        output_dir: str
    """
    warp_output = case_info.get('warp_output')
    if warp_output and output_dir_root is None:
        return warp_output['output_dir']
    basename = get_case_basename(case_info)
    if output_dir_root:
        return os.path.join(output_dir_root, basename)
    raise ValueError("No warp_output in JSON and --output_dir not specified")


# ============================================================================
# Multi-machine task splitting
# ============================================================================

def split_tasks_for_machine(cases_with_indices, total_machines, machine_id, seed=42):
    """Split tasks deterministically across multiple machines.

    Uses a seeded shuffle to ensure all machines agree on the partition,
    then assigns a contiguous slice to each machine.

    Args:
        cases_with_indices: list of (case_index, case_info) tuples
        total_machines: total number of machines
        machine_id: 0-based ID of this machine
        seed: random seed for deterministic shuffling

    Returns:
        my_cases: list of (case_index, case_info) for this machine
    """
    import random as _random
    rng = _random.Random(seed)
    shuffled = list(cases_with_indices)
    rng.shuffle(shuffled)

    n = len(shuffled)
    # Even split: each machine gets ceil(n / total_machines) tasks at most
    chunk_size = (n + total_machines - 1) // total_machines
    start = machine_id * chunk_size
    end = min(start + chunk_size, n)
    return shuffled[start:end]
