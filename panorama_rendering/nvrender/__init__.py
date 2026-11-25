"""nvrender — Panoramic rendering package for HY-World.

Submodules:
    utils          — General-purpose utilities (image_uv, depth tools, perf stats)
    cubemap        — Cubemap constants and ERP stitching
    geometry       — Point cloud spatial transforms and mesh construction
    mesh_renderer  — Mesh-based rendering via nvdiffrast
    pcd_renderer   — Point-cloud rendering via pytorch3d / TorchPointCloudRenderer
    camera_motion  — Camera movement orchestration and public entry points

Optional renderers load when their exported functions are used.
"""

from importlib import import_module


_EXPORTS = {
    'perform_camera_movement_with_json_final_only': 'camera_motion',
    'perform_camera_movement_with_json': 'camera_motion',
    'perform_camera_movement': 'camera_motion',
    'generate_pc_render': 'camera_motion',
    'generate_mesh_render': 'camera_motion',
    'euler_to_rotation_matrix': 'camera_motion',
    'merge_panorama_color_torch': 'cubemap',
    'depth_repair': 'utils',
    'depth_edge_torch': 'utils',
    'collect_perf_stats': 'utils',
    'get_mesh_from_pano_Rt': 'geometry',
    'get_world_pcs_pano_torch_Rt': 'geometry',
    'get_pano_pcs_torch': 'geometry',
    'get_mesh_render_color': 'mesh_renderer',
    'mesh_pano_render_color': 'mesh_renderer',
    'mesh_pano_render_depth': 'mesh_renderer',
    'mesh_pano_render_rgb_and_depth': 'mesh_renderer',
    'point_rendering_distance': 'pcd_renderer',
    'pcd_pano_render_color': 'pcd_renderer',
    'pcd_pano_render_color_torch_renderer': 'pcd_renderer',
    'load_exr_depth': 'pipeline_utils',
    'load_position_rotation_from_json': 'pipeline_utils',
    'colorize_depth': 'pipeline_utils',
    'save_video_from_frames': 'pipeline_utils',
    'save_depth_video_from_frames': 'pipeline_utils',
    'save_mask_video_from_frames': 'pipeline_utils',
    'save_perf_stats_to_file': 'pipeline_utils',
    'get_case_basename': 'pipeline_utils',
    'check_output_exists': 'pipeline_utils',
    'check_pcd_output_exists': 'pipeline_utils',
    'resolve_pcd_path': 'pipeline_utils',
    'resolve_output_dir': 'pipeline_utils',
    'split_tasks_for_machine': 'pipeline_utils',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f'{__name__!r} has no attribute {name!r}')
    module = import_module(f'.{_EXPORTS[name]}', __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
