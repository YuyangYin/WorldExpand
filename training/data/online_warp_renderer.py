"""On-the-fly warp rendering for the online training path.

Builds an isotropic spherical-Gaussian point cloud from a single panoramic
RGBD frame and renders the end-view via :func:`gs_pano_render_color`. The
ZIM + GroundingDINO sky-mask models are lazy-loaded **on the training
rank's GPU** (no subprocess, no extra CUDA context). All heavy work stays
in the training process so the cost is just one synchronous render call
per sample, sitting between the dataloader fetch and the existing VAE
encode in ``training/train.py``.

This is a deliberate thin layer over the existing ``panorama_rendering.nvrender``
gsplat path; the only behavioural difference from the offline step4 ->
step5 pipeline is that we never write ``global_pcd.ply`` / ``warp.png`` /
``*_latent.pt`` to disk.
"""

from __future__ import annotations

# Resolve resource defaults from the external release configuration.
import sys as _we_sys
from pathlib import Path as _WePath
_we_root = next((_p for _p in _WePath(__file__).resolve().parents
                 if (_p / "worldexpand" / "paths.py").is_file()), None)
if _we_root is not None and str(_we_root) not in _we_sys.path:
    _we_sys.path.insert(0, str(_we_root))
from worldexpand.paths import resolve_resource as _release_path


import json
import math
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import utils3d

from panorama_rendering.nvrender.gs_renderer import (
    build_isotropic_gs_from_pcd,
    gs_pano_render_color,
)
# ``pcd_pano_render_color`` is only needed when render_engine='pytorch3d';
# we lazy-import it inside __init__ to keep pytorch3d out of the import
# graph for users who only run the default gsplat path.
from panorama_rendering.render_utils.panorama_utils import spherical_uv_to_directions


# ---------------------------------------------------------------------------
# Profiling helpers (used by render_one + by training/train.py online branch).
# CudaTimer accumulates one (start, end) torch.cuda.Event pair per section and
# defers timing materialisation to ``finalize()`` — a single synchronize at
# the end, instead of one sync per section. NoOpTimer is a zero-cost stand-in
# so the instrumented call sites stay branchless in the hot path.
# ---------------------------------------------------------------------------


class NoOpTimer:
    """Drop-in stand-in for :class:`CudaTimer` when profiling is off."""

    enabled = False

    @contextmanager
    def section(self, name: str):
        yield

    def finalize(self) -> Dict[str, float]:
        return {}


NOOP_TIMER = NoOpTimer()


class CudaTimer:
    """CUDA-event-based stage timer.

    Records ``(start, end)`` events per section, defers ``elapsed_time`` and
    the single ``torch.cuda.synchronize`` to :meth:`finalize`. This avoids
    inserting one sync per section (which would skew the very thing we are
    trying to measure).
    """

    enabled = True

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        self._events: List[Tuple[str, "torch.cuda.Event", "torch.cuda.Event"]] = []

    @contextmanager
    def section(self, name: str):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._events.append((name, start, end))

    def finalize(self) -> Dict[str, float]:
        """Block once, then convert all event pairs to seconds. Idempotent."""
        if not self._events:
            return {}
        torch.cuda.synchronize(self.device)
        out: Dict[str, float] = {}
        for name, s, e in self._events:
            ms = s.elapsed_time(e)
            out[name] = out.get(name, 0.0) + ms / 1000.0
        self._events.clear()
        return out


# ---------------------------------------------------------------------------
# Coordinate conversions copied from panorama_rendering/nvrender/camera_motion.py
# so we do not need to import private helpers (and so the rotation matrices
# live next to the renderer that actually consumes them).
# ---------------------------------------------------------------------------

# UE5: X=forward, Y=right, Z=up. Render (OpenCV): X=right, Y=down, Z=forward.
_ROT_MATRIX_3x3 = np.array(
    [
        [0, 1, 0],
        [0, 0, -1],
        [1, 0, 0],
    ],
    dtype=np.float32,
)
_ROT_MATRIX_4x4 = np.eye(4, dtype=np.float32)
_ROT_MATRIX_4x4[:3, :3] = _ROT_MATRIX_3x3

# Sphere-coords -> render-OpenCV (used inside the PCD branch of
# generate_pc_render to rotate the ply produced by convert_rgbd2pcd_panorama).
_PLY_ROT = np.array(
    [
        [0, 1, 0],
        [0, 0, -1],
        [-1, 0, 0.0],
    ],
    dtype=np.float32,
)


def compute_erp_plucker_maps(
    d_render: torch.Tensor,
    start_w2c: np.ndarray,
    end_w2c: np.ndarray,
    trans_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Analytic ERP Plücker ray maps for the START / END views.

    Shared by the online renderer (training) and the inference demo so the
    geometry can never drift between the two. Everything is expressed in the
    START camera frame, so the START origin is 0 (moment channels = 0) and the
    END origin is the relative camera translation.

    Args
    ----
    d_render : (H, W, 3) unit ERP ray directions already rotated into the
        render-OpenCV frame (i.e. ``sphere_dirs @ _PLY_ROT.T``).
    start_w2c, end_w2c : (4, 4) render-OpenCV world-to-camera matrices
        (e.g. from ``_build_w2c_render_from_ue5``).
    trans_scale : divides the END translation so the moment channels stay O(1).

    Returns
    -------
    (plucker_start, plucker_end) each (6, H, W); P = (o x d, d).
    """
    device, dtype = d_render.device, d_render.dtype
    end_c2w = np.linalg.inv(end_w2c)
    # Relative transform: END camera -> START camera (world cancels).
    T_rel = (start_w2c @ end_c2w).astype(np.float32)
    R_rel = torch.from_numpy(T_rel[:3, :3]).to(device=device, dtype=dtype)
    o_end = torch.from_numpy(T_rel[:3, 3]).to(device=device, dtype=dtype) / float(trans_scale)

    zeros3 = torch.zeros_like(d_render)
    plucker_start = torch.cat([zeros3, d_render], dim=-1)  # (H, W, 6)

    d_end = d_render @ R_rel.T  # (H, W, 3)
    moment_end = torch.cross(o_end.expand_as(d_end), d_end, dim=-1)
    plucker_end = torch.cat([moment_end, d_end], dim=-1)  # (H, W, 6)

    return (
        plucker_start.permute(2, 0, 1).contiguous(),
        plucker_end.permute(2, 0, 1).contiguous(),
    )


def compute_relative_translation(
    start_w2c: np.ndarray, end_w2c: np.ndarray, scale: float = 1.0
) -> np.ndarray:
    """END camera centre expressed in the START camera frame, divided by
    ``scale``. This is the global camera-position vector injected "like
    timestep" (Options A/B). Same ``T_rel = start_w2c @ inv(end_w2c)`` as the
    Plücker path, so the two conditioning signals share one coordinate frame.
    Returns a (3,) float32 array.
    """
    T_rel = (start_w2c @ np.linalg.inv(end_w2c)).astype(np.float32)
    return (T_rel[:3, 3] / float(scale)).astype(np.float32)


def _build_w2c_render_from_ue5(
    position: Dict[str, float], yaw_deg: float
) -> np.ndarray:
    yaw_rad = math.radians(yaw_deg)
    rotation_matrix = np.array(
        [
            [math.cos(yaw_rad), math.sin(yaw_rad), 0],
            [-math.sin(yaw_rad), math.cos(yaw_rad), 0],
            [0, 0, 1],
        ]
    )
    c2w_ue5 = np.eye(4)
    c2w_ue5[:3, -1] = np.array(
        [position["X"], position["Y"], position["Z"]]
    )
    c2w_ue5[:3, :3] = rotation_matrix.T
    w2c_ue5 = np.linalg.inv(c2w_ue5)
    return (_ROT_MATRIX_4x4 @ w2c_ue5 @ _ROT_MATRIX_4x4.T).astype(np.float32)


# ---------------------------------------------------------------------------
# OnlineWarpRenderer
# ---------------------------------------------------------------------------

# Optional sky segmentation reads the public checkpoint directories when enabled.
ZIM_CHECKPOINT = _release_path(os.environ.get("WE_ZIM_CHECKPOINT", "@repo/weights/zim_vit_l_2092"))
GD_CHECKPOINT = _release_path(os.environ.get("WE_GD_CHECKPOINT", "@repo/weights/grounding-dino-tiny"))


class OnlineWarpRenderer:
    """Per-rank online warp renderer.

    Parameters
    ----------
    device : torch.device
        Training rank's GPU. ZIM + GD will also live here once loaded.
    sky_mask_mode : {"lazy", "skip", "indoor_only", "cached"}
        ``lazy`` loads ZIM + GD on the first ``outdoor`` case.
        ``skip`` never loads them (sky_mask = all zeros, matches
        the original offline warp renderer's indoor branch).
        ``indoor_only`` is expected upstream (dataset filters), but we
        defensively treat unknown / outdoor as indoor in that mode.
        ``cached`` reads the dataset-supplied ``start_sky_mask_for_pcd``
        tensor (precomputed with
        the offline sky-mask exporter), and lazily
        falls back to ZIM/GD on cache miss with a one-shot WARN per
        rank. This is the recommended production mode once the cache is
        built — it sidesteps the ~22 s/step ZIM bottleneck without
        changing numerics (same ``get_zim_mask`` call as before, just
        run offline once).
    max_pcd_points : int
        Random downsample cap, mirroring step4's ``--max_pcd_points``.
    gs_scale_factor : float
        Per-splat sigma multiplier passed to
        :func:`build_isotropic_gs_from_pcd`.
    render_h, render_w : int
        Output ERP panorama size; default matches the trainer's
        ``img_size=512`` -> (512, 1024).
    rtol_edge : float
        Tolerance for ``utils3d.numpy.depth_edge``; same default (0.1) as
        the step4 full pipeline.
    """

    def __init__(
        self,
        device: torch.device,
        sky_mask_mode: str = "lazy",
        max_pcd_points: int = 5_000_000,
        gs_scale_factor: float = 1.0,
        render_h: int = 512,
        render_w: int = 1024,
        rtol_edge: float = 0.1,
        depth_contract: float = 0.0,
        prewarm_seg_models: bool = True,
        profile: Optional[bool] = None,
        render_engine: str = "gsplat",
        pcd_render_radius: float = 0.008,
        pcd_points_per_pixel: int = 8,
        pcd_use_fp16_cache: bool = True,
        zim_checkpoint: str = ZIM_CHECKPOINT,
        gd_checkpoint: str = GD_CHECKPOINT,
        # Debug-dump knobs. When ``debug_dump_dir`` is non-None, the first
        # ``debug_dump_first_n_steps`` calls to ``render_batch`` will dump
        # every sample's input / mask / warp / GT to disk for visual
        # inspection. Defaults disabled. The trainer is expected to pass
        # a non-None dir only on the main rank (so other ranks never write).
        debug_dump_dir: Optional[str] = None,
        debug_dump_first_n_steps: int = 1,
        # Cube layout: when set, the per-batch ERP renderer outputs
        # (wrap_image / wrap_mask) are re-sampled into a 2x4 cube grid
        # at the end of ``render_batch`` to match the cube-grid
        # ``start_rgb_train`` / ``end_rgb_train`` produced by the
        # dataset. See training/data/cube_layout.py and
        # CubeMap_Related_Work/2x4_padding_strategy.md.
        cube_layout: Optional[str] = None,
        cube_face_size: int = 480,
        cube_grid_pad: int = 32,
        # Plücker ray conditioning (ERP route). When ``use_plucker`` is True,
        # ``render_batch`` also returns per-sample (B, 6, H, W) ERP Plücker maps
        # for the START (condition) and END (target/warp) views, computed
        # analytically from the UE5 camera poses (no extra rendering). The END
        # ray origin (camera translation, START frame) is divided by
        # ``plucker_trans_scale`` so the moment channels stay well-conditioned.
        use_plucker: bool = False,
        plucker_trans_scale: float = 1.0,
        # Global camera-position conditioning ("like timestep", Options A/B).
        # When ``use_pose`` is True, ``render_batch`` also returns a (B, 3)
        # pose_vec = relative END translation in the START frame, divided by
        # ``pose_scale`` to keep it O(1).
        use_pose: bool = False,
        pose_scale: float = 1.0,
        # When False, skip the (heavy) per-sample warp render entirely:
        # render_batch returns wrap_rgb=None, wrap_mask=None (the trainer then
        # runs a 2-stream [noisy, condition] model). pose / plucker still
        # computed if their flags are on. Big speedup for warp-free ablations.
        use_warp: bool = True,
        # Surface-normal conditioning (ERP route, ablation 2). When True,
        # render_batch also returns per-sample (B, 3, H, W) normal maps in
        # [0, 1] for the START (condition) and the END-warp (wrap) views. The
        # START-frame per-pixel normal is computed from the start depth; the
        # END-warp normal is produced by splatting that same per-point normal
        # (as an RGB colour) through the SAME gsplat pass as the warp RGB, so
        # its holes match warp_rgb/warp_mask exactly. Sky / over-range depth /
        # large-gradient (anti-alias) pixels are culled to a neutral 0.5.
        use_normal: bool = False,
        normal_max_depth: float = 5000.0,
        normal_grad_rel: float = 1.0,
        normal_facing_camera: bool = True,
        warp_cache_dir: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device)
        self.sky_mask_mode = str(sky_mask_mode)
        if self.sky_mask_mode not in {"lazy", "skip", "indoor_only", "cached", "depth_v2"}:
            raise ValueError(
                f"sky_mask_mode must be lazy/skip/indoor_only/cached/depth_v2, "
                f"got {sky_mask_mode!r}"
            )
        # v2 depth-based mask: pre-load params from env at __init__ so we
        # log them once on this rank and don't hit os.environ on every
        # render. Algorithm + dataclass live in dataset_utils/sky_mask_v2.py;
        # see training/data/sky_mask_v2.py for the env-var contract.
        self._v2_params = None
        if self.sky_mask_mode == "depth_v2":
            from training.data.sky_mask_v2 import params_from_env
            self._v2_params = params_from_env(verbose=True)
            print(
                f"[OnlineWarpRenderer] sky_mask_mode='depth_v2' on "
                f"{self.device}: using inline depth-based mask "
                f"(no ZIM/GD load, no separate depth_edge pass). "
                f"Params={self._v2_params.as_dict()}"
            )
        # Warn-once state: in ``cached`` mode every cache miss falls back to
        # a synchronous ZIM/GD call. We log only the first miss per rank to
        # avoid log-flooding on a misconfigured cache_dir; the miss counter
        # itself is exposed via ``self.cache_miss_count`` for the trainer.
        self._cache_miss_warned: bool = False
        self.cache_miss_count: int = 0
        self.max_pcd_points = int(max_pcd_points)
        self.gs_scale_factor = float(gs_scale_factor)
        self.render_h = int(render_h)
        self.render_w = int(render_w)
        self.use_plucker = bool(use_plucker)
        self.plucker_trans_scale = float(plucker_trans_scale)
        if self.plucker_trans_scale == 0.0:
            raise ValueError("plucker_trans_scale must be non-zero")
        self.use_pose = bool(use_pose)
        self.pose_scale = float(pose_scale)
        if self.pose_scale == 0.0:
            raise ValueError("pose_scale must be non-zero")
        self.use_warp = bool(use_warp)
        self.use_normal = bool(use_normal)
        self.normal_max_depth = float(normal_max_depth)
        self.normal_grad_rel = float(normal_grad_rel)
        self.normal_facing_camera = bool(normal_facing_camera)
        if self.use_normal and not self.use_warp:
            raise ValueError(
                "use_normal=True requires use_warp=True (the START normal is "
                "computed inside render_one and the END-warp normal reuses the "
                "warp PCD/gsplat pass)."
            )
        self.rtol_edge = float(rtol_edge)
        self.depth_contract = float(depth_contract)
        self.zim_checkpoint = zim_checkpoint
        self.gd_checkpoint = gd_checkpoint

        # Render backend. ``gsplat`` keeps the current behaviour (isotropic
        # GS via build_isotropic_gs_from_pcd + gs_pano_render_color). The
        # ``pytorch3d`` path skips the GS dict build and rasterizes raw
        # points via panorama_rendering/nvrender/pcd_renderer.pcd_pano_render_color
        # — the same backend exposed by step4's --render_engine point_renderer.
        # ``pcd_render_radius`` / ``pcd_points_per_pixel`` default to the
        # values hard-coded inside ``pcd_pano_render_color`` itself (which is
        # also what the offline pipeline ends up using since step4 does not
        # expose CLI overrides), so swapping engine alone is equivalent to
        # the offline pytorch3d branch with zero further tuning.
        #
        # ``pytorch3d_torch_renderer`` is the **ablation** path: it calls
        # ``pcd_pano_render_color_torch_renderer``, which uses a pure-PyTorch
        # ``TorchPointCloudRenderer`` instead of pytorch3d's
        # ``PointsRasterizer``. Its distinguishing property is the
        # ``use_fp16_cache`` flag (default True): the point cloud lives in
        # fp16 inside the renderer to save ~60 MB / 5M-point scene, but on
        # UE5 absolute world coordinates (~10^2 m) fp16's ~0.25 m precision
        # visibly quantises near-ground geometry. The ``pcd_use_fp16_cache``
        # kwarg below (default True) toggles this behaviour and is the
        # exposed knob for the "crippled point renderer" ablation.
        # ``render_radius`` / ``points_per_pixel`` are NOT accepted by this
        # function (the renderer manages those internally), so those two
        # kwargs above are silently ignored on this path.
        self.render_engine = str(render_engine).lower()
        if self.render_engine not in {"gsplat", "pytorch3d", "pytorch3d_torch_renderer", "mesh", "mesh_cached"}:
            raise ValueError(
                f"render_engine must be 'gsplat', 'pytorch3d', "
                f"'pytorch3d_torch_renderer', 'mesh' or 'mesh_cached', got {render_engine!r}"
            )
        # mesh_cached: read pre-rendered mesh warps from disk (offline prerender);
        # no online mesh build. warp_cache_dir/<case_id>/warp_<dir>.png + mask_<dir>.png
        self.warp_cache_dir = warp_cache_dir
        if self.use_normal and self.render_engine != "gsplat":
            raise ValueError(
                "use_normal is currently only wired for render_engine='gsplat' "
                f"(got {self.render_engine!r}). The normal map is splatted via the "
                "gsplat colour pass; add the pytorch3d branch if needed."
            )
        self.pcd_render_radius = float(pcd_render_radius)
        self.pcd_points_per_pixel = int(pcd_points_per_pixel)
        self.pcd_use_fp16_cache = bool(pcd_use_fp16_cache)
        self._pcd_pano_render_color = None
        self._pcd_pano_render_color_torch_renderer = None
        if self.render_engine == "pytorch3d":
            # Lazy import: pulls pytorch3d into the import graph only when
            # actually needed. Users on the default gsplat path keep a
            # slimmer dependency surface.
            from panorama_rendering.nvrender.pcd_renderer import pcd_pano_render_color
            self._pcd_pano_render_color = pcd_pano_render_color
        elif self.render_engine == "pytorch3d_torch_renderer":
            # Same lazy-import hygiene; the _torch_renderer path additionally
            # constructs a TorchPointCloudRenderer (pure-PyTorch
            # rasterizer + AlphaCompositor) on every render_one call.
            from panorama_rendering.nvrender.pcd_renderer import (
                pcd_pano_render_color_torch_renderer,
            )
            self._pcd_pano_render_color_torch_renderer = (
                pcd_pano_render_color_torch_renderer
            )

        # Profiling. When None, fall back to the ONLINE_WARP_PROFILE env so
        # the same launcher works for production and debug runs without a
        # rebuild. Per-batch sub-timings land in ``self.last_timings`` for
        # the trainer to read after ``render_batch``.
        if profile is None:
            profile = os.environ.get("ONLINE_WARP_PROFILE", "0").lower() in {
                "1", "true", "yes", "on",
            }
        self.profile = bool(profile)
        self.last_timings: Dict[str, float] = {}

        # Lazy state.
        self._zim_predictor = None
        self._gd_processor = None
        self._gd_model = None

        # Rays are a function of (H, W) only; we cache one tensor per
        # resolution to avoid rebuilding on every sample.
        self._rays_cache: Dict[Tuple[int, int], torch.Tensor] = {}

        # Fix C — defensive set_device. ``accelerate.Accelerator`` already
        # binds the current CUDA device per rank before we get here, but if
        # any future code path inside gsplat / ZIM / GD uses
        # ``torch.cuda.current_device()`` instead of the tensor's device,
        # this guarantees it lands on the right GPU.
        if self.device.type == "cuda" and self.device.index is not None:
            torch.cuda.set_device(self.device)

        # Fix A — optional pre-warm of ZIM + GD. Loading these eagerly at
        # __init__ on every rank eliminates the "rank N pays the +5-15 s
        # ZIM/GD load on its first outdoor case, the other 7 ranks block at
        # accelerator.wait_for_everyone()" skew. Only meaningful for
        # sky_mask_mode='lazy'; ``skip`` / ``indoor_only`` never load these
        # models anyway so we keep them out of memory. ``cached`` should
        # cover ~all outdoor frames so we also skip pre-warm there — if the
        # cache happens to miss, we'll lazy-load on the first miss instead
        # (paying the 5-15 s once, on one rank, only when the cache is
        # actually incomplete).
        if prewarm_seg_models and self.sky_mask_mode == "lazy":
            print(
                f"[OnlineWarpRenderer] prewarm_seg_models=True: loading "
                f"ZIM + GD eagerly on {self.device} so all 8 ranks finish "
                f"the load before training step 0 (no first-step skew)."
            )
            self._ensure_seg_models()

        if self.render_engine == "pytorch3d":
            print(
                f"[OnlineWarpRenderer] render_engine=pytorch3d on {self.device}: "
                f"render_radius={self.pcd_render_radius} "
                f"points_per_pixel={self.pcd_points_per_pixel} "
                f"(no isotropic-GS construction; expect ~1.5-3x render time "
                f"vs the default gsplat backend at MAX_PCD_POINTS=5M)."
            )
        elif self.render_engine == "pytorch3d_torch_renderer":
            print(
                f"[OnlineWarpRenderer] render_engine=pytorch3d_torch_renderer "
                f"on {self.device}: use_fp16_cache={self.pcd_use_fp16_cache} "
                f"(ABLATION path — TorchPointCloudRenderer; with "
                f"use_fp16_cache=True on UE5 absolute world coords this "
                f"quantises near-ground geometry to ~0.25 m precision)."
            )

        if self.profile:
            print(
                f"[OnlineWarpRenderer] profile=True on {self.device}: "
                f"per-batch sub-timings recorded via CUDA events; "
                f"trainer reads them from ``renderer.last_timings``."
            )

        # Debug-dump state. The trainer is responsible for passing
        # ``debug_dump_dir=None`` on non-main ranks; we just trust the
        # caller. ``_dump_step_count`` tracks how many render_batch
        # calls we've already dumped — we stop dumping once we hit
        # ``debug_dump_first_n_steps``. ``_batch_dump_idx`` is reset
        # at the start of each dumped batch and used as a stable
        # per-sample index inside the dump dir.
        self.debug_dump_dir: Optional[str] = (
            str(debug_dump_dir) if debug_dump_dir else None
        )
        self.debug_dump_first_n_steps: int = int(debug_dump_first_n_steps)
        self._dump_step_count: int = 0
        self._batch_dump_idx: int = 0
        self._this_batch_should_dump: bool = False
        if self.debug_dump_dir is not None:
            os.makedirs(self.debug_dump_dir, exist_ok=True)
            print(
                f"[OnlineWarpRenderer] debug_dump_dir='{self.debug_dump_dir}' "
                f"on {self.device}: will dump start_rgb / depth / masks / "
                f"warp / GT for the first {self.debug_dump_first_n_steps} "
                f"render_batch call(s) to disk for visual inspection. "
                f"Dump path layout: <dir>/step_<NNNN>/<safe_case_id>/..."
            )

        # Cube layout: normalise to None | "2x4_padding". The ERP-shaped
        # ``render_one`` output is converted to a 2x4 cube grid at the
        # end of ``render_batch`` so the rest of the pipeline doesn't
        # need to know whether we're in ERP or cube mode (only the
        # transformer-input spatial dims change).
        from training.data import cube_layout as _cube_layout_mod
        if cube_layout in (None, "", "off", "null", "none"):
            self.cube_layout: Optional[str] = None
        else:
            cl = str(cube_layout).strip().lower()
            if cl not in _cube_layout_mod.VALID_LAYOUTS:
                raise ValueError(
                    f"OnlineWarpRenderer: cube_layout must be one of "
                    f"{_cube_layout_mod.VALID_LAYOUTS} or null, "
                    f"got {cube_layout!r}"
                )
            self.cube_layout = cl
        self.cube_face_size = int(cube_face_size)
        self.cube_grid_pad = int(cube_grid_pad)
        if self.cube_layout is not None:
            _cube_layout_mod.validate_dims(self.cube_face_size, self.cube_grid_pad)
            _h_grid, _w_grid = _cube_layout_mod.grid_shape(
                self.cube_face_size, self.cube_grid_pad
            )
            print(
                f"[OnlineWarpRenderer] cube_layout={self.cube_layout} "
                f"(face_size={self.cube_face_size}, "
                f"grid_pad={self.cube_grid_pad} -> grid={_h_grid}x{_w_grid}). "
                f"render_one still renders ERP at "
                f"{self.render_h}x{self.render_w}; the grid is sampled "
                f"out of that ERP at the end of render_batch."
            )

    # ------------------------------------------------------------------
    # Lazy ZIM + GD load.
    # ------------------------------------------------------------------
    def _ensure_seg_models(self) -> None:
        if self._zim_predictor is not None:
            return
        # Import here so a worker / process that never hits an outdoor case
        # never even touches the optional zim_anything + transformers
        # AutoModelForZeroShotObjectDetection dependencies.
        from panorama_rendering.render_utils.seg_utils import (
            build_gd_model,
            build_zim_model,
        )

        print(
            f"[OnlineWarpRenderer] lazy-loading ZIM(vit_l) + GroundingDINO "
            f"on device={self.device} (sky_mask_mode=lazy, first outdoor case)"
        )
        self._zim_predictor = build_zim_model(
            "vit_l", self.zim_checkpoint, device=self.device
        )
        self._gd_processor, self._gd_model = build_gd_model(
            self.gd_checkpoint, device=self.device
        )

    # ------------------------------------------------------------------
    # Sky mask resolution.
    # ------------------------------------------------------------------
    def _build_sky_mask(
        self,
        start_rgb_pcd: torch.Tensor,    # (H, W, 3) float32 in [0, 1] on self.device
        label: Optional[str],
        sample: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Return ``True`` for sky pixels (i.e. *excluded* from the PCD).

        Mode semantics:

        * ``skip`` / ``indoor_only`` / outdoor=indoor → all-zero mask.
        * ``cached`` → load from dataset's ``start_sky_mask_for_pcd``;
          on miss, fall back to lazy ZIM/GD with a one-shot warning.
        * ``lazy`` → run ZIM + GroundingDINO on the RGB.
        * ``depth_v2`` → run :func:`compute_sky_mask_torch` on the start
          depth tensor. This subsumes both the v1 sky mask AND the v1
          ``depth_edge`` step (the renderer's ``depth_edge`` section is
          a no-op in this mode).
        """
        H, W = start_rgb_pcd.shape[:2]
        if self.sky_mask_mode == "skip" or label == "indoor":
            return torch.zeros((H, W), dtype=torch.bool, device=self.device)
        if self.sky_mask_mode == "indoor_only":
            # Dataset should have filtered already; fall back to no mask.
            return torch.zeros((H, W), dtype=torch.bool, device=self.device)

        if self.sky_mask_mode == "depth_v2":
            # Depth-only mask: needs the raw depth tensor from the sample
            # (the dataset must NOT p99-clip in depth_v2 mode; see
            # OnlineWarpDataset._clean_depth — clip is auto-disabled when
            # sky_mask_mode='depth_v2'). The algorithm computes its own
            # combined sky/invalid/edge mask + dilation, so the caller
            # (render_one) skips the separate utils3d depth_edge pass.
            from training.data.sky_mask_v2 import compute_sky_mask_torch
            assert sample is not None, "depth_v2 needs sample[start_depth_for_pcd]"
            depth = sample.get("start_depth_for_pcd")
            if depth is None:
                raise RuntimeError(
                    "sky_mask_mode='depth_v2' requires "
                    "sample['start_depth_for_pcd'] to be present."
                )
            return compute_sky_mask_torch(
                depth, self._v2_params, device=self.device,
            )

        if self.sky_mask_mode == "cached":
            cached = sample.get("start_sky_mask_for_pcd") if sample else None
            if cached is not None and torch.is_tensor(cached):
                cached = cached.to(self.device, non_blocking=True)
                if cached.dtype != torch.bool:
                    cached = cached.bool()
                if cached.shape != (H, W):
                    # Edge case: dataset's pcd_build_h/w changed at runtime
                    # but the cache wasn't regenerated. Nearest-resize so
                    # we never linearly-interp a binary boundary.
                    cached = (
                        F.interpolate(
                            cached.float().unsqueeze(0).unsqueeze(0),
                            size=(H, W),
                            mode="nearest",
                        )
                        .squeeze(0)
                        .squeeze(0)
                        .bool()
                    )
                return cached
            # Cache miss → fall through to lazy ZIM. Bump the counter and
            # warn once.
            self.cache_miss_count += 1
            if not self._cache_miss_warned:
                self._cache_miss_warned = True
                print(
                    f"[OnlineWarpRenderer] sky_mask_mode='cached' but the "
                    f"dataset returned no start_sky_mask_for_pcd on this "
                    f"rank ({self.device}); falling back to a lazy ZIM/GD "
                    f"call. Subsequent misses on this rank will be silent. "
                    f"Total misses are tracked in renderer.cache_miss_count. "
                    f"Re-run the offline sky-mask exporter "
                    f"if the cache is incomplete."
                )

        # lazy / cached-miss / "outdoor with no label" → run ZIM + GD.
        self._ensure_seg_models()
        from panorama_rendering.render_utils.seg_utils import get_zim_mask
        from PIL import Image

        rgb_np = (start_rgb_pcd.detach().clamp(0, 1).cpu().numpy() * 255).astype(
            np.uint8
        )
        pil_img = Image.fromarray(rgb_np)
        # get_zim_mask returns True for the foreground (= non-sky) under the
        # "~" used inside step4. Match that semantics exactly: sky_mask is
        # True where pixels should be **excluded** (sky pixels).
        non_sky = get_zim_mask(
            pil_img,
            "sky.",
            0.3,
            0.3,
            self._zim_predictor,
            self._gd_processor,
            self._gd_model,
            DEVICE=self.device,
        )
        sky_mask_np = ~np.asarray(non_sky, dtype=bool)
        return torch.from_numpy(sky_mask_np).to(self.device)

    # ------------------------------------------------------------------
    # Rays cache.
    # ------------------------------------------------------------------
    def _rays(self, h: int, w: int) -> torch.Tensor:
        key = (int(h), int(w))
        if key not in self._rays_cache:
            rays = torch.from_numpy(
                spherical_uv_to_directions(
                    utils3d.numpy.image_uv(width=int(w), height=int(h))
                ).astype(np.float32)
            ).to(self.device)
            self._rays_cache[key] = rays
        return self._rays_cache[key]

    # ------------------------------------------------------------------
    # Surface normals (ERP route). Per-pixel START-camera-frame normal from a
    # panoramic distance map, computed on GPU (torch) to avoid a numpy CPU
    # round-trip. Mirrors the math in
    # panorama_rendering/render_utils/panorama_utils.erp_distance_ray_to_normal:
    #   P = d * ray ; Tu = dP/du ; Tv = dP/dv ; N = Tu x Tv ; normalise ; orient.
    # The u (width) axis is ERP-periodic so we use a wrap-around (roll) central
    # difference; the v (height) axis is not periodic (one-sided at the poles).
    # ------------------------------------------------------------------
    def _erp_normal_from_depth(
        self, distance: torch.Tensor, rays: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (normal (H,W,3) in [-1,1] start-cam frame, grad (H,W)).

        ``grad = max(|Tu|, |Tv|)`` is the local tangent magnitude used to cull
        depth-discontinuity / anti-aliasing pixels (large grad => unreliable
        normal direction).
        """
        d = torch.nan_to_num(distance.float(), nan=0.0, posinf=0.0, neginf=0.0)
        d = d.clamp(0.0, float(self.normal_max_depth))
        P = d.unsqueeze(-1) * rays                                  # (H, W, 3)
        # Horizontal (width) — ERP periodic central difference.
        Tu = 0.5 * (torch.roll(P, -1, dims=1) - torch.roll(P, 1, dims=1))
        # Vertical (height) — central interior, one-sided at top/bottom rows.
        Tv = torch.zeros_like(P)
        Tv[1:-1] = 0.5 * (P[2:] - P[:-2])
        Tv[0] = P[1] - P[0]
        Tv[-1] = P[-1] - P[-2]
        n = torch.cross(Tu, Tv, dim=-1)
        n = n / torch.linalg.norm(n, dim=-1, keepdim=True).clamp_min(1e-8)
        if self.normal_facing_camera:
            # Normal should face the camera (origin): dot(n, ray) < 0. Flip otherwise.
            dot = (n * rays).sum(dim=-1, keepdim=True)
            n = torch.where(dot > 0, -n, n)
        n = torch.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        grad = torch.maximum(
            torch.linalg.norm(Tu, dim=-1), torch.linalg.norm(Tv, dim=-1)
        )                                                           # (H, W)
        return n, grad

    # ------------------------------------------------------------------
    # Core: one sample -> (warp_rgb, warp_mask) at (render_h, render_w).
    #
    # ``timer`` is a CudaTimer / NoOpTimer instance owned by the caller
    # (typically ``render_batch``); the sub-sections below feed straight into
    # ``timer.finalize()`` so the trainer can attribute the per-step render
    # cost to sky_mask vs depth_edge vs pcd build vs gsplat.
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def render_one(
        self,
        sample: Dict[str, Any],
        timer: Optional["NoOpTimer"] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns (warp_rgb HWC[0,1], warp_mask HW bool, normal_start HWC[0,1] or
        None, warp_normal HWC[0,1] or None, normal_invalid HW bool or None).

        The last three are None unless ``self.use_normal``. ``normal_invalid`` is
        only used by the debug dump.
        """
        if timer is None:
            timer = NOOP_TIMER
        device = self.device

        def _neutral_normal():
            # 0.5 == (n=0+1)/2 -> neutral / "unknown normal" colour.
            return torch.full(
                (self.render_h, self.render_w, 3), 0.5, dtype=torch.float32, device=device
            )

        with timer.section("io_upload"):
            start_rgb = sample["start_rgb_for_pcd"].to(device, non_blocking=True)
            start_depth = sample["start_depth_for_pcd"].to(device, non_blocking=True)
        H, W = start_depth.shape[-2:]

        rays = self._rays(H, W)

        # ---- depth cleanup: sky mask | edge mask | p99 clip ----------
        with timer.section("sky_mask"):
            sky_mask = self._build_sky_mask(
                start_rgb, sample.get("label"), sample=sample
            )
        with timer.section("depth_edge"):
            if self.sky_mask_mode == "depth_v2":
                # v2 already runs a halo-gated gradient pass inside
                # compute_sky_mask and dilates it; an extra utils3d
                # depth_edge would be redundant and would re-introduce
                # the global gradient masking we explicitly tuned out.
                edge_mask = torch.zeros_like(sky_mask, dtype=torch.bool)
            else:
                edge_mask = torch.from_numpy(
                    utils3d.numpy.depth_edge(start_depth.detach().cpu().numpy(), rtol=self.rtol_edge)
                ).to(device).bool()
        full_mask = sky_mask | edge_mask

        # ---- Surface normal (START-cam frame) + invalid mask --------------
        # Computed from the (raw) start depth, then sky / over-range /
        # large-gradient (anti-alias) pixels are culled to a neutral 0 vector.
        normal_map_full = None      # (H, W, 3) in [-1, 1], invalid -> 0
        normal_invalid = None       # (H, W) bool
        normal_start_hwc = None     # (render_h, render_w, 3) in [0, 1]
        if self.use_normal:
            with timer.section("normal"):
                n_map, n_grad = self._erp_normal_from_depth(start_depth, rays)
                sd = start_depth.float()
                normal_invalid = (
                    sky_mask
                    | ~torch.isfinite(sd)
                    | (sd <= 0)
                    | (sd > float(self.normal_max_depth))
                    | (n_grad > float(self.normal_grad_rel) * sd.clamp_min(1e-6))
                )
                n_map = n_map.clone()
                n_map[normal_invalid] = 0.0
                normal_map_full = n_map
                ns01 = ((normal_map_full + 1.0) * 0.5).permute(2, 0, 1).unsqueeze(0)
                ns01 = F.interpolate(
                    ns01, size=(self.render_h, self.render_w),
                    mode="bilinear", align_corners=False,
                )
                normal_start_hwc = ns01.squeeze(0).permute(1, 2, 0).contiguous().clamp(0.0, 1.0)

        with timer.section("pcd_filter"):
            valid = ~full_mask & torch.isfinite(start_depth) & (start_depth > 0)
            if valid.sum().item() < 100:
                # Pathological case: depth is all bad. Bail out with a zero warp.
                zero_rgb = torch.zeros(
                    (self.render_h, self.render_w, 3), dtype=torch.float32, device=device
                )
                zero_mask = torch.zeros(
                    (self.render_h, self.render_w), dtype=torch.bool, device=device
                )
                if self.use_normal:
                    return zero_rgb, zero_mask, normal_start_hwc, _neutral_normal(), normal_invalid
                return zero_rgb, zero_mask, None, None, None

            if self.sky_mask_mode == "skip":
                # skip mode keeps the sky in the warp; a p99 clip would crush
                # the (intentionally kept) far/sky pixels toward the real-
                # geometry range. Ship raw depth so the sky stays at its
                # ~65504 m "infinity" distance (only NaN/Inf are dropped below).
                distance = start_depth.float()
            else:
                max_d = torch.quantile(start_depth[valid], q=0.99).item()
                distance = torch.clip(start_depth, 0.0, max_d).float()

            if self.depth_contract is not None and self.depth_contract > 0:
                median = torch.median(distance.reshape(-1)).item()
                cd = median * self.depth_contract
                cmask = distance > cd
                distance[cmask] = (2 * cd) - (cd ** 2 / (distance[cmask] + 1e-6))

            # vertices in sphere coords (origin = camera, unit ray * distance).
            verts_sphere = (distance.unsqueeze(-1) * rays).reshape(-1, 3)
            colors = start_rgb.reshape(-1, 3).clamp(0.0, 1.0)
            # Per-point START-frame normal (invalid -> 0). Threaded through the
            # SAME keep/finite/sane/idx index ops as ``colors`` so the normal GS
            # below uses an identical point set (=> identical holes as warp_rgb).
            normal_pp = (
                normal_map_full.reshape(-1, 3) if self.use_normal else None
            )
            keep = (~full_mask).reshape(-1) & torch.isfinite(verts_sphere).all(dim=-1)
            verts_sphere = verts_sphere[keep]
            colors = colors[keep]
            if normal_pp is not None:
                normal_pp = normal_pp[keep]

            # ---- sphere coords -> render OpenCV -----------------------
            rot_ply = torch.from_numpy(_PLY_ROT).to(device=device, dtype=verts_sphere.dtype)
            verts_render = verts_sphere @ rot_ply.T

            # ---- render-OpenCV (camera at origin) -> world ------------
            pos_start = {k: float(sample["start_position"][k]) / 100.0 for k in ("X", "Y", "Z")}
            start_yaw = float(sample["start_yaw"])
            start_w2c_np = _build_w2c_render_from_ue5(pos_start, start_yaw)
            start_c2w_np = np.linalg.inv(start_w2c_np).astype(np.float32)
            c2w = torch.from_numpy(start_c2w_np).to(device=device, dtype=verts_render.dtype)
            verts_world = verts_render @ c2w[:3, :3].T + c2w[:3, 3]

            # ---- NaN / Inf / extreme filter ---------------------------
            finite = torch.isfinite(verts_world).all(dim=-1) \
                & torch.isfinite(colors).all(dim=-1)
            verts_world = verts_world[finite]
            colors = colors[finite]
            if normal_pp is not None:
                normal_pp = normal_pp[finite]
            if self.sky_mask_mode != "skip":
                # The abs_max < 1e4 m extreme-distance cull drops far-away
                # outliers and, as a side effect, the ~65504 m sky points. In
                # skip mode we intentionally keep the sky, so we apply only the
                # NaN/Inf filter above and leave distant points in place.
                abs_max = verts_world.abs().max(dim=-1).values
                sane = abs_max < 1e4
                verts_world = verts_world[sane]
                colors = colors[sane]
                if normal_pp is not None:
                    normal_pp = normal_pp[sane]

        with timer.section("pcd_downsample"):
            if (
                self.max_pcd_points is not None
                and self.max_pcd_points > 0
                and verts_world.shape[0] > self.max_pcd_points
            ):
                idx = torch.randperm(
                    verts_world.shape[0], device=verts_world.device
                )[: self.max_pcd_points]
                verts_world = verts_world[idx]
                colors = colors[idx]
                if normal_pp is not None:
                    normal_pp = normal_pp[idx]

        if verts_world.shape[0] == 0:
            zero_rgb = torch.zeros(
                (self.render_h, self.render_w, 3), dtype=torch.float32, device=device
            )
            zero_mask = torch.zeros(
                (self.render_h, self.render_w), dtype=torch.bool, device=device
            )
            if self.use_normal:
                return zero_rgb, zero_mask, normal_start_hwc, _neutral_normal(), normal_invalid
            return zero_rgb, zero_mask, None, None, None

        # Both engines need colours in [-1, 1] (gsplat for its sigma+colour
        # convention, pytorch3d for compositor symmetry with the warp space).
        colors_signed = (colors * 2.0 - 1.0).contiguous()

        gs = None
        if self.render_engine == "gsplat":
            with timer.section("gs_build"):
                gs = build_isotropic_gs_from_pcd(
                    points=verts_world.contiguous(),
                    colors=colors_signed,
                    first_frame_Rt=torch.from_numpy(start_w2c_np),
                    scale_factor=self.gs_scale_factor,
                    device=device,
                )

        # ---- End-frame w2c -------------------------------------------
        pos_end = {k: float(sample["end_position"][k]) / 100.0 for k in ("X", "Y", "Z")}
        end_yaw = float(sample["end_yaw"])
        end_w2c_np = _build_w2c_render_from_ue5(pos_end, end_yaw)
        Rts_np = end_w2c_np[None]  # (1, 4, 4)

        if self.render_engine == "gsplat":
            with timer.section("gs_render"):
                rgb_list, mask_list = gs_pano_render_color(
                    gs,
                    Rts_np,
                    self.render_h,
                    self.render_w,
                    device,
                    render_depth=False,
                    erp_mask_aware=True,
                )
        elif self.render_engine == "pytorch3d":
            # pcd_pano_render_color takes raw world-space points + signed
            # colours and rasterizes via pytorch3d's PointsRasterizer per
            # cubemap face (no GS dict, no first_frame_Rt). Returns the
            # same (rgb_list, mask_list) interface as gs_pano_render_color
            # — RGB in [0, 1], mask in bool, length F.
            with timer.section("pcd_render"):
                rgb_list, mask_list = self._pcd_pano_render_color(
                    points=verts_world.contiguous(),
                    colors=colors_signed,
                    Rts=Rts_np,
                    height=self.render_h,
                    width=self.render_w,
                    device=device,
                    render_radius=self.pcd_render_radius,
                    points_per_pixel=self.pcd_points_per_pixel,
                    render_depth=False,
                    erp_mask_aware=True,
                )
        else:  # pytorch3d_torch_renderer (ablation path)
            # pcd_pano_render_color_torch_renderer uses a pure-PyTorch
            # TorchPointCloudRenderer instead of pytorch3d's
            # PointsRasterizer. Same (rgb_list, mask_list) interface.
            # ``use_fp16_cache=True`` (the default) is the "crippled"
            # behaviour we ablate against: it stores the point cloud in
            # fp16 inside the renderer; on UE5 absolute world coords
            # (~10^2 m) this quantises near-ground geometry (~0.25 m
            # precision). render_radius / points_per_pixel are NOT
            # accepted by this function — the renderer manages them
            # internally — so we omit them here.
            with timer.section("pcd_torch_render"):
                rgb_list, mask_list = self._pcd_pano_render_color_torch_renderer(
                    points=verts_world.contiguous(),
                    colors=colors_signed,
                    Rts=Rts_np,
                    height=self.render_h,
                    width=self.render_w,
                    device=device,
                    render_depth=False,
                    erp_mask_aware=True,
                    use_fp16_cache=self.pcd_use_fp16_cache,
                )
        warp_rgb_np = rgb_list[0].astype(np.float32, copy=False)   # (H, W, 3) in [0, 1]
        warp_mask_np = mask_list[0]                                # (H, W) bool

        # ---- Normal warp (second gsplat pass: normal-as-colour) ----------
        # Splat the per-point START-frame normal (in [-1, 1], invalid -> 0) as
        # an RGB colour through an identical gsplat at the SAME end pose. The
        # point set is identical to the RGB pass, so the holes match
        # warp_mask exactly. build_isotropic_gs_from_pcd maps colour (c+1)/2,
        # so the output is already in [0, 1].
        warp_normal_np = None
        if self.use_normal:
            with timer.section("gs_normal"):
                gs_n = build_isotropic_gs_from_pcd(
                    points=verts_world.contiguous(),
                    colors=normal_pp.contiguous(),
                    first_frame_Rt=torch.from_numpy(start_w2c_np),
                    scale_factor=self.gs_scale_factor,
                    device=device,
                )
                nrgb_list, _ = gs_pano_render_color(
                    gs_n,
                    Rts_np,
                    self.render_h,
                    self.render_w,
                    device,
                    render_depth=False,
                    erp_mask_aware=True,
                )
                warp_normal_np = nrgb_list[0].astype(np.float32, copy=False)  # (H,W,3) [0,1]
                del gs_n

        # Fix B — let PyTorch's caching allocator handle freed buffers; no
        # need to torch.cuda.empty_cache() per sample (that would serialize
        # all CUDA streams and add BS sync points per training step).
        del verts_world, colors, colors_signed, gs, verts_sphere, verts_render
        if self.use_normal:
            del normal_pp

        with timer.section("out_h2d"):
            warp_rgb = torch.from_numpy(warp_rgb_np).to(device)
            warp_mask = torch.from_numpy(np.ascontiguousarray(warp_mask_np)).to(device)
            warp_normal = None
            if self.use_normal:
                warp_normal = torch.from_numpy(warp_normal_np).to(device)
                # Make holes (no splatted point) neutral 0.5, matching the
                # START map's invalid encoding and warp_rgb's hole semantics.
                warp_normal[~warp_mask] = 0.5

        # Debug dump (off in production). Gated by `_this_batch_should_dump`
        # which is set/cleared inside render_batch — when this flag is
        # False the entire helper call is skipped, so the hot path is just
        # one bool check.
        if getattr(self, "_this_batch_should_dump", False):
            try:
                self._dump_sample(
                    sample=sample,
                    step_dir=self._current_step_dir,
                    sample_idx=self._batch_dump_idx,
                    start_rgb=start_rgb,
                    start_depth=start_depth,
                    sky_mask=sky_mask,
                    edge_mask=edge_mask,
                    full_mask=full_mask,
                    warp_rgb=warp_rgb,
                    warp_mask=warp_mask,
                    normal_start=normal_start_hwc,
                    warp_normal=warp_normal,
                    normal_invalid=normal_invalid,
                )
            except Exception as e:
                # Dump failures must NEVER break training.
                print(
                    f"[OnlineWarpRenderer] debug dump FAILED for "
                    f"case_id={sample.get('case_id')!r}: "
                    f"{type(e).__name__}: {e}"
                )
            self._batch_dump_idx += 1
        if self.use_normal:
            return warp_rgb, warp_mask, normal_start_hwc, warp_normal, normal_invalid
        return warp_rgb, warp_mask, None, None, None

    # ------------------------------------------------------------------
    # Debug dump helpers (only fire when ``debug_dump_dir`` is set and
    # ``_dump_step_count < debug_dump_first_n_steps``). Saves PNG / NPY /
    # JSON for one sample into:
    #     ``<debug_dump_dir>/step_<NNNN>/<safe_case_id>/``
    # The hot path stays untouched — gated by a cheap bool check.
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_case_id(case_id: str) -> str:
        """Make the case_id filesystem-safe (replace '/', strip whitespace)."""
        if not case_id:
            return "unknown"
        return (
            str(case_id)
            .replace("/", "__")
            .replace("\\", "__")
            .replace(" ", "_")
            .strip("._")
        )

    @staticmethod
    def _depth_to_uint8_vis(depth_np: np.ndarray) -> np.ndarray:
        """Map depth to a 0..255 grayscale via log + p2/p98 stretch.

        The raw depth has long sky tails (p99 ≈ 65504), a linear stretch
        would crush all real geometry into one bucket. log + percentile
        stretch keeps near and mid range readable in PNG.
        """
        d = depth_np.astype(np.float32)
        valid = np.isfinite(d) & (d > 0)
        if not np.any(valid):
            return np.zeros_like(d, dtype=np.uint8)
        d_log = np.log1p(np.clip(d, 0, None))
        v = d_log[valid]
        lo, hi = float(np.quantile(v, 0.02)), float(np.quantile(v, 0.98))
        if hi <= lo:
            hi = lo + 1e-3
        out = np.clip((d_log - lo) / (hi - lo), 0, 1)
        out[~valid] = 0
        return (out * 255).astype(np.uint8)

    def _dump_sample(
        self,
        sample: Dict[str, Any],
        step_dir: str,
        sample_idx: int,
        # tensors gathered from render_one's scope
        start_rgb: torch.Tensor,        # (H, W, 3) [0,1]
        start_depth: torch.Tensor,      # (H, W)
        sky_mask: torch.Tensor,         # (H, W) bool
        edge_mask: torch.Tensor,        # (H, W) bool
        full_mask: torch.Tensor,        # (H, W) bool
        warp_rgb: torch.Tensor,         # (h, w, 3) [0,1]
        warp_mask: torch.Tensor,        # (h, w) bool
        normal_start: Optional[torch.Tensor] = None,   # (h, w, 3) [0,1]
        warp_normal: Optional[torch.Tensor] = None,     # (h, w, 3) [0,1]
        normal_invalid: Optional[torch.Tensor] = None,  # (H, W) bool (PCD res)
    ) -> None:
        """Write one sample's debug bundle. Called from render_one if active."""
        case_id_raw = sample.get("case_id") or self._case_id_fallback(sample)
        case_id_safe = self._safe_case_id(case_id_raw)
        out_dir = os.path.join(
            step_dir, f"{sample_idx:02d}_{case_id_safe}",
        )
        os.makedirs(out_dir, exist_ok=True)

        # numpy versions for cv2 / np writes
        rgb_np = (
            start_rgb.detach().clamp(0, 1).cpu().numpy() * 255
        ).astype(np.uint8)
        depth_np = start_depth.detach().cpu().numpy().astype(np.float32)
        sky_np = sky_mask.detach().cpu().numpy().astype(np.uint8) * 255
        edge_np = edge_mask.detach().cpu().numpy().astype(np.uint8) * 255
        full_np = full_mask.detach().cpu().numpy().astype(np.uint8) * 255
        warp_rgb_np = (
            warp_rgb.detach().clamp(0, 1).cpu().numpy() * 255
        ).astype(np.uint8)
        warp_mask_np = warp_mask.detach().cpu().numpy().astype(np.uint8) * 255

        # End-frame GT (training resolution). Convert from [-1,1] back to [0,1].
        end_rgb_t = sample.get("end_rgb_train")
        if end_rgb_t is not None and torch.is_tensor(end_rgb_t):
            end_np = (
                (end_rgb_t.detach().cpu().numpy() + 1.0) * 0.5
            ).clip(0, 1)
            # CHW -> HWC
            if end_np.ndim == 3 and end_np.shape[0] == 3:
                end_np = end_np.transpose(1, 2, 0)
            end_np = (end_np * 255).astype(np.uint8)
        else:
            end_np = None

        # PNGs (cv2 expects BGR for color)
        cv2.imwrite(
            os.path.join(out_dir, "01_start_rgb.png"),
            cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(out_dir, "02_start_depth_vis.png"),
            self._depth_to_uint8_vis(depth_np),
        )
        cv2.imwrite(os.path.join(out_dir, "03_mask_sky.png"), sky_np)
        cv2.imwrite(os.path.join(out_dir, "04_mask_edge.png"), edge_np)
        cv2.imwrite(os.path.join(out_dir, "05_mask_full.png"), full_np)
        cv2.imwrite(
            os.path.join(out_dir, "06_warp_rgb.png"),
            cv2.cvtColor(warp_rgb_np, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(os.path.join(out_dir, "07_warp_mask.png"), warp_mask_np)
        if end_np is not None:
            cv2.imwrite(
                os.path.join(out_dir, "08_end_rgb_gt.png"),
                cv2.cvtColor(end_np, cv2.COLOR_RGB2BGR),
            )

        # Normal debug bundle (only when use_normal). Normals are RGB-encoded
        # ((n+1)/2 in [0,1]); 0.5 == neutral / invalid / hole.
        if normal_start is not None:
            ns_np = (
                normal_start.detach().clamp(0, 1).cpu().numpy() * 255
            ).astype(np.uint8)
            cv2.imwrite(
                os.path.join(out_dir, "16_normal_start_vis.png"),
                cv2.cvtColor(ns_np, cv2.COLOR_RGB2BGR),
            )
        if warp_normal is not None:
            wn_np = (
                warp_normal.detach().clamp(0, 1).cpu().numpy() * 255
            ).astype(np.uint8)
            cv2.imwrite(
                os.path.join(out_dir, "17_warp_normal_vis.png"),
                cv2.cvtColor(wn_np, cv2.COLOR_RGB2BGR),
            )
        if normal_invalid is not None:
            ni_np = normal_invalid.detach().cpu().numpy().astype(np.uint8) * 255
            cv2.imwrite(os.path.join(out_dir, "18_mask_normal_invalid.png"), ni_np)

        # Raw depth (lossless, for re-running compute_sky_mask offline /
        # tweaking thresholds without re-loading EXR).
        np.save(
            os.path.join(out_dir, "start_depth_raw.npy"),
            depth_np.astype(np.float32),
        )

        # Metadata JSON
        meta: Dict[str, Any] = {
            "case_id": str(case_id_raw),
            "label": sample.get("label"),
            "caption": sample.get("caption", ""),
            "sky_mask_mode": self.sky_mask_mode,
            "render_engine": self.render_engine,
            "shapes": {
                "start_rgb": list(rgb_np.shape),
                "start_depth": list(depth_np.shape),
                "warp_rgb": list(warp_rgb_np.shape),
            },
            "depth_stats": {
                "min": float(np.nanmin(depth_np)) if np.isfinite(depth_np).any() else None,
                "p50": float(np.nanmedian(depth_np[np.isfinite(depth_np) & (depth_np > 0)])) if np.any(np.isfinite(depth_np) & (depth_np > 0)) else None,
                "p99": float(np.nanquantile(depth_np[np.isfinite(depth_np) & (depth_np > 0)], 0.99)) if np.any(np.isfinite(depth_np) & (depth_np > 0)) else None,
                "max": float(np.nanmax(depth_np)) if np.isfinite(depth_np).any() else None,
            },
            "mask_ratios": {
                "sky":   float(sky_np.mean() / 255),
                "edge":  float(edge_np.mean() / 255),
                "full":  float(full_np.mean() / 255),
                "warp":  float(warp_mask_np.mean() / 255),
            },
            "pose": {
                "start_position": sample.get("start_position"),
                "start_yaw_deg": sample.get("start_yaw"),
                "end_position": sample.get("end_position"),
                "end_yaw_deg": sample.get("end_yaw"),
            },
        }
        # If depth_v2 is in use, snapshot the params so the dump is
        # self-describing (you can re-run compute_sky_mask on the .npy
        # and reproduce the .png masks exactly).
        if self.sky_mask_mode == "depth_v2" and self._v2_params is not None:
            meta["sky_mask_v2_params"] = self._v2_params.as_dict()

        # If the dataset applied an ERP yaw alignment, the per-sample
        # bookkeeping (before/after yaws + per-tensor pixel shifts) was
        # stashed in sample["_yaw_align_meta"] by training.data.yaw_align.
        # Surface it here so the dump is self-describing — you can read
        # this meta.json and immediately tell whether the dumped images
        # are post-roll or not, and reconstruct the shift used.
        if sample.get("_yaw_align_meta") is not None:
            meta["yaw_align"] = sample["_yaw_align_meta"]

        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    @staticmethod
    def _case_id_fallback(sample: Dict[str, Any]) -> str:
        """Best-effort case id when sample doesn't carry one."""
        for k in ("case_id", "id"):
            v = sample.get(k)
            if v:
                return str(v)
        return "no_case_id"

    # ------------------------------------------------------------------
    # Batched entry point. Returns the four tensors the trainer needs.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Plücker ray maps (ERP route). Analytic from UE5 camera poses; no
    # rendering needed. Returns two (6, H, W) tensors on ``self.device``:
    #   - plucker_start: START (condition) view. Expressed in the START
    #     camera frame, so its origin is 0 -> moment channels are 0 and the
    #     direction channels are exactly the ERP sphere ray directions.
    #   - plucker_end:   END view, expressed in the START camera frame. The
    #     END ray directions are rotated by the relative rotation; the moment
    #     is ``o_end x d_end`` where ``o_end`` is the END camera centre in the
    #     START frame (translation), divided by ``plucker_trans_scale``.
    # Convention: P = (o x d, d), the standard (moment, direction) 6-vector.
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _plucker_maps_one(self, sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        H, W = self.render_h, self.render_w
        device = self.device

        # ERP sphere-coords ray directions (H, W, 3), unit -> render OpenCV frame
        d_sphere = self._rays(H, W)  # (H, W, 3) float32 on device
        rot_ply = torch.from_numpy(_PLY_ROT).to(device=device, dtype=d_sphere.dtype)
        d_render = d_sphere @ rot_ply.T  # (H, W, 3), unit, render frame

        # Camera poses in render-OpenCV world (same builder as the warp path).
        pos_start = {k: float(sample["start_position"][k]) / 100.0 for k in ("X", "Y", "Z")}
        pos_end = {k: float(sample["end_position"][k]) / 100.0 for k in ("X", "Y", "Z")}
        start_w2c = _build_w2c_render_from_ue5(pos_start, float(sample["start_yaw"]))
        end_w2c = _build_w2c_render_from_ue5(pos_end, float(sample["end_yaw"]))

        return compute_erp_plucker_maps(
            d_render, start_w2c, end_w2c, trans_scale=self.plucker_trans_scale
        )

    @torch.inference_mode()
    def _pose_vec_one(self, sample: Dict[str, Any]) -> torch.Tensor:
        """Relative END translation (START frame), normalised. Returns (3,)."""
        pos_start = {k: float(sample["start_position"][k]) / 100.0 for k in ("X", "Y", "Z")}
        pos_end = {k: float(sample["end_position"][k]) / 100.0 for k in ("X", "Y", "Z")}
        start_w2c = _build_w2c_render_from_ue5(pos_start, float(sample["start_yaw"]))
        end_w2c = _build_w2c_render_from_ue5(pos_end, float(sample["end_yaw"]))
        t = compute_relative_translation(start_w2c, end_w2c, scale=self.pose_scale)
        return torch.from_numpy(t).to(self.device)

    def render_batch(
        self,
        samples: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Render a whole batch sequentially on this rank's GPU.

        Returns
        -------
        warp_rgb : (B, 3, render_h, render_w) float32 in [-1, 1]
        warp_mask : (B, 1, render_h, render_w) float32 in {0, 1}
        plucker_start : (B, 6, render_h, render_w) float32, or None when
            ``self.use_plucker`` is False (START / condition view).
        plucker_end : (B, 6, render_h, render_w) float32, or None when
            ``self.use_plucker`` is False (END / target+warp view).
        pose_vec : (B, 3) float32 relative END translation in the START frame,
            or None when ``self.use_pose`` is False.
        normal_start : (B, 3, render_h, render_w) float32 in [0, 1], or None
            when ``self.use_normal`` is False (START / condition view normal).
        normal_end_warp : (B, 3, render_h, render_w) float32 in [0, 1], or None
            when ``self.use_normal`` is False (END / warp view normal, holey).

        When ``self.profile`` is True, per-sub-section timings (summed across
        the ``len(samples)`` ``render_one`` calls) are stashed in
        ``self.last_timings`` for the trainer to print alongside its own
        per-step timings.
        """
        if self.profile and self.device.type == "cuda":
            timer: NoOpTimer = CudaTimer(self.device)
        else:
            timer = NOOP_TIMER

        # Decide whether THIS batch should be dumped. Done once per
        # render_batch (not per sample) so the dump grouping mirrors the
        # trainer's notion of "step".
        self._this_batch_should_dump = (
            self.debug_dump_dir is not None
            and self._dump_step_count < self.debug_dump_first_n_steps
        )
        if self._this_batch_should_dump:
            self._batch_dump_idx = 0
            self._current_step_dir = os.path.join(
                self.debug_dump_dir,
                f"step_{self._dump_step_count:04d}",
            )
            os.makedirs(self._current_step_dir, exist_ok=True)
            print(
                f"[OnlineWarpRenderer] debug dump: writing batch "
                f"#{self._dump_step_count} ({len(samples)} sample(s)) "
                f"to {self._current_step_dir}"
            )

        warps: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        plk_start: List[torch.Tensor] = []
        plk_end: List[torch.Tensor] = []
        poses: List[torch.Tensor] = []
        nrm_start: List[torch.Tensor] = []
        nrm_warp: List[torch.Tensor] = []
        for s in samples:
            if self.use_warp:
                rgb, mask, n_start, n_warp, _n_inv = self.render_one(s, timer=timer)
                # (H, W, 3) [0, 1] -> (3, H, W) [-1, 1]
                warps.append((rgb.permute(2, 0, 1) * 2.0 - 1.0).contiguous())
                # (H, W) bool -> (1, H, W) float
                masks.append(mask.float().unsqueeze(0).contiguous())
                if self.use_normal:
                    # Normals stay in [0, 1] (the normal_encoder convention);
                    # (H, W, 3) -> (3, H, W). Holes / invalid already at 0.5.
                    nrm_start.append(n_start.permute(2, 0, 1).contiguous())
                    nrm_warp.append(n_warp.permute(2, 0, 1).contiguous())
            if self.use_plucker:
                ps, pe = self._plucker_maps_one(s)
                plk_start.append(ps)
                plk_end.append(pe)
            if self.use_pose:
                poses.append(self._pose_vec_one(s))
        self.last_timings = timer.finalize()

        if self._this_batch_should_dump:
            # Mark the batch as done and increment the step counter.
            self._dump_step_count += 1
            self._this_batch_should_dump = False

        rgb_stacked = torch.stack(warps, dim=0) if self.use_warp else None
        mask_stacked = torch.stack(masks, dim=0) if self.use_warp else None
        plk_start_stacked = torch.stack(plk_start, dim=0) if self.use_plucker else None
        plk_end_stacked = torch.stack(plk_end, dim=0) if self.use_plucker else None
        pose_stacked = torch.stack(poses, dim=0) if self.use_pose else None  # (B, 3)
        nrm_start_stacked = torch.stack(nrm_start, dim=0) if self.use_normal else None
        nrm_warp_stacked = torch.stack(nrm_warp, dim=0) if self.use_normal else None

        # Cube layout: re-sample the ERP outputs into the 2x4 padding
        # grid so the trainer sees the same spatial layout for
        # warp_image / wrap_mask as for start_rgb_train / end_rgb_train.
        # The dataset side does the equivalent transform; see
        # training/data/cube_layout.py for the shared layout module.
        if self.cube_layout == "2x4_padding" and rgb_stacked is not None:
            from training.data.cube_layout import erp_to_2x4_padding_grid
            rgb_stacked = erp_to_2x4_padding_grid(
                rgb_stacked,
                self.cube_face_size,
                self.cube_grid_pad,
                interp_mode="bilinear",
            )
            # The mask is technically binary, but downstream
            # ``mask_encoder`` is a Conv2d(in=1, out=64, k=16, s=16)
            # operating on float input, so bilinear is fine; we keep
            # the float[0, 1] semantics (cube_map_from_equirectangular
            # uses bilinear grid_sample with padding_mode='border').
            mask_stacked = erp_to_2x4_padding_grid(
                mask_stacked,
                self.cube_face_size,
                self.cube_grid_pad,
                interp_mode="bilinear",
            )
            # Plücker maps follow the same ERP->cube resampling so they stay
            # spatially aligned with warp_image / wrap_mask. Direction channels
            # are bilinearly interpolated (not renormalised) — acceptable for a
            # conditioning signal; the "ERP route" (cube_layout off) is exact.
            if plk_start_stacked is not None:
                plk_start_stacked = erp_to_2x4_padding_grid(
                    plk_start_stacked, self.cube_face_size, self.cube_grid_pad,
                    interp_mode="bilinear",
                )
                plk_end_stacked = erp_to_2x4_padding_grid(
                    plk_end_stacked, self.cube_face_size, self.cube_grid_pad,
                    interp_mode="bilinear",
                )
            # Normal maps follow the same ERP->cube resampling (bilinear; the
            # [0,1] encoding is preserved, holes/invalid stay ~0.5).
            if nrm_start_stacked is not None:
                nrm_start_stacked = erp_to_2x4_padding_grid(
                    nrm_start_stacked, self.cube_face_size, self.cube_grid_pad,
                    interp_mode="bilinear",
                )
                nrm_warp_stacked = erp_to_2x4_padding_grid(
                    nrm_warp_stacked, self.cube_face_size, self.cube_grid_pad,
                    interp_mode="bilinear",
                )

        return (
            rgb_stacked, mask_stacked, plk_start_stacked, plk_end_stacked,
            pose_stacked, nrm_start_stacked, nrm_warp_stacked,
        )

    # ==================================================================
    # Four-view online warp.
    #
    # ``render_one_multi`` builds the PCD / gsplat ONCE from the shared
    # ``start`` ingredients, then renders to each of ``V`` end cameras —
    # the PCD-reuse that makes 4-view online warp affordable. Mirrors the
    # build段 of ``render_one`` verbatim (so single-view behaviour is
    # untouched: ``render_one`` itself is unchanged). gsplat-only fast
    # path; other engines fall back to per-view ``render_one`` (rebuilds
    # the PCD each view — correct, just slower).
    #
    # ``render_batch_multiview`` is the 4-view analogue of ``render_batch``;
    # it stacks per-view warp / mask / plücker / pose for a whole batch.
    # ==================================================================
    @torch.inference_mode()
    def render_one_multi(
        self,
        sample: Dict[str, Any],
        timer: Optional["NoOpTimer"] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Build PCD once from the shared start, render to each end camera.

        Returns ``(warps, masks)`` where each is a length-``V`` list of
        ``(H, W, 3)`` float[0,1] RGB and ``(H, W)`` bool mask tensors on
        ``self.device`` (same per-view layout as ``render_one``).
        """
        if timer is None:
            timer = NOOP_TIMER
        device = self.device

        end_positions = sample["end_positions"]
        end_yaws = sample["end_yaws"]
        V = int(sample.get("num_views", len(end_positions)))

        def _zero_outputs() -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
            zr = [
                torch.zeros((self.render_h, self.render_w, 3), dtype=torch.float32, device=device)
                for _ in range(V)
            ]
            zm = [
                torch.zeros((self.render_h, self.render_w), dtype=torch.bool, device=device)
                for _ in range(V)
            ]
            return zr, zm

        # --- mesh_cached engine: read pre-rendered mesh warps from disk.
        if self.render_engine == "mesh_cached":
            return self._read_cached_warps(sample, V)

        # --- Mesh engine: build one world-space triangle mesh, render per view.
        if self.render_engine == "mesh":
            return self._render_multi_mesh(sample, end_positions, end_yaws, V, timer)

        # --- Non-gsplat engines: correctness over speed (per-view rebuild).
        if self.render_engine != "gsplat":
            warps: List[torch.Tensor] = []
            masks: List[torch.Tensor] = []
            for i in range(V):
                sub = dict(sample)
                sub["end_position"] = end_positions[i]
                sub["end_yaw"] = end_yaws[i]
                out = self.render_one(sub, timer=timer)
                rgb, mask = out[0], out[1]  # render_one returns a 5-tuple (rgb, mask, +normals)
                warps.append(rgb)
                masks.append(mask)
            return warps, masks

        # ---------- BUILD (shared, once) — mirrors render_one ----------
        built = self.build_world_gs(sample, timer=timer)
        if built is None:
            return _zero_outputs()
        gs, start_w2c_np = built

        # ---------- RENDER to each end camera ----------
        warps = []
        masks = []
        for i in range(V):
            pos_end = {k: float(end_positions[i][k]) / 100.0 for k in ("X", "Y", "Z")}
            end_w2c_np = _build_w2c_render_from_ue5(pos_end, float(end_yaws[i]))
            with timer.section("gs_render"):
                rgb_list, mask_list = gs_pano_render_color(
                    gs,
                    end_w2c_np[None],
                    self.render_h,
                    self.render_w,
                    device,
                    render_depth=False,
                    erp_mask_aware=True,
                )
            warps.append(torch.from_numpy(rgb_list[0].astype(np.float32, copy=False)).to(device))
            masks.append(torch.from_numpy(np.ascontiguousarray(mask_list[0])).to(device))

        del gs
        return warps, masks

    # ------------------------------------------------------------------ #
    # Reusable building blocks (shared by render_one_multi and the        #
    # interpolated "pin"-render fly-through used for video debugging).    #
    # ------------------------------------------------------------------ #
    def build_world_gs(
        self,
        sample: Dict[str, Any],
        timer: Optional["NoOpTimer"] = None,
    ) -> Optional[Tuple[Dict[str, torch.Tensor], np.ndarray]]:
        """Build the world-space isotropic-GS point cloud from the shared start.

        Extracted verbatim from the gsplat BUILD block of
        :meth:`render_one_multi` so the same point cloud can be rendered to an
        arbitrary set of camera poses (e.g. an interpolated fly-through).

        Returns ``(gs, start_w2c_np)`` or ``None`` when the depth is too
        degenerate to build a usable cloud.
        """
        if timer is None:
            timer = NOOP_TIMER
        device = self.device

        with timer.section("io_upload"):
            start_rgb = sample["start_rgb_for_pcd"].to(device, non_blocking=True)
            start_depth = sample["start_depth_for_pcd"].to(device, non_blocking=True)

        rays = self._rays(*start_depth.shape[-2:])

        with timer.section("sky_mask"):
            sky_mask = self._build_sky_mask(start_rgb, sample.get("label"), sample=sample)
        with timer.section("depth_edge"):
            if self.sky_mask_mode == "depth_v2":
                edge_mask = torch.zeros_like(sky_mask, dtype=torch.bool)
            else:
                edge_mask = torch.from_numpy(
                    utils3d.numpy.depth_edge(start_depth.detach().cpu().numpy(), rtol=self.rtol_edge)
                ).to(device).bool()
        full_mask = sky_mask | edge_mask

        with timer.section("pcd_filter"):
            valid = ~full_mask & torch.isfinite(start_depth) & (start_depth > 0)
            if valid.sum().item() < 100:
                return None

            if self.sky_mask_mode == "skip":
                distance = start_depth.float()
            else:
                max_d = torch.quantile(start_depth[valid], q=0.99).item()
                distance = torch.clip(start_depth, 0.0, max_d).float()

            if self.depth_contract is not None and self.depth_contract > 0:
                median = torch.median(distance.reshape(-1)).item()
                cd = median * self.depth_contract
                cmask = distance > cd
                distance[cmask] = (2 * cd) - (cd ** 2 / (distance[cmask] + 1e-6))

            verts_sphere = (distance.unsqueeze(-1) * rays).reshape(-1, 3)
            colors = start_rgb.reshape(-1, 3).clamp(0.0, 1.0)
            keep = (~full_mask).reshape(-1) & torch.isfinite(verts_sphere).all(dim=-1)
            verts_sphere = verts_sphere[keep]
            colors = colors[keep]

            rot_ply = torch.from_numpy(_PLY_ROT).to(device=device, dtype=verts_sphere.dtype)
            verts_render = verts_sphere @ rot_ply.T

            pos_start = {k: float(sample["start_position"][k]) / 100.0 for k in ("X", "Y", "Z")}
            start_yaw = float(sample["start_yaw"])
            start_w2c_np = _build_w2c_render_from_ue5(pos_start, start_yaw)
            start_c2w_np = np.linalg.inv(start_w2c_np).astype(np.float32)
            c2w = torch.from_numpy(start_c2w_np).to(device=device, dtype=verts_render.dtype)
            verts_world = verts_render @ c2w[:3, :3].T + c2w[:3, 3]

            finite = torch.isfinite(verts_world).all(dim=-1) & torch.isfinite(colors).all(dim=-1)
            verts_world = verts_world[finite]
            colors = colors[finite]
            if self.sky_mask_mode != "skip":
                abs_max = verts_world.abs().max(dim=-1).values
                sane = abs_max < 1e4
                verts_world = verts_world[sane]
                colors = colors[sane]

        with timer.section("pcd_downsample"):
            if (
                self.max_pcd_points is not None
                and self.max_pcd_points > 0
                and verts_world.shape[0] > self.max_pcd_points
            ):
                idx = torch.randperm(verts_world.shape[0], device=verts_world.device)[: self.max_pcd_points]
                verts_world = verts_world[idx]
                colors = colors[idx]

        if verts_world.shape[0] == 0:
            return None

        colors_signed = (colors * 2.0 - 1.0).contiguous()

        with timer.section("gs_build"):
            gs = build_isotropic_gs_from_pcd(
                points=verts_world.contiguous(),
                colors=colors_signed,
                first_frame_Rt=torch.from_numpy(start_w2c_np),
                scale_factor=self.gs_scale_factor,
                device=device,
            )

        del verts_world, colors, colors_signed, verts_sphere, verts_render
        return gs, start_w2c_np

    def _read_cached_warps(
        self, sample: Dict[str, Any], V: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """mesh_cached: load pre-rendered warp+mask PNGs from ``warp_cache_dir``.

        Layout: ``warp_cache_dir/<case_id>/warp_<direction>.png`` (RGB uint8) and
        ``mask_<direction>.png`` ({0,255}). Returns per-view (H,W,3)float[0,1] and
        (H,W)bool on device, matching render_one_multi's output. Missing files ->
        zero outputs (training is expected to filter to cached cases upstream).
        """
        import cv2
        device = self.device
        cdir = os.path.join(str(self.warp_cache_dir), str(sample.get("case_id", "")))
        directions = list(sample.get("directions", []))[:V]
        warps: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        for i in range(V):
            d = directions[i] if i < len(directions) else f"v{i}"
            wp = os.path.join(cdir, f"warp_{d}.png")
            mp = os.path.join(cdir, f"mask_{d}.png")
            if not (os.path.isfile(wp) and os.path.isfile(mp)):
                warps.append(torch.zeros((self.render_h, self.render_w, 3), dtype=torch.float32, device=device))
                masks.append(torch.zeros((self.render_h, self.render_w), dtype=torch.bool, device=device))
                continue
            bgr = cv2.imread(wp, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if (rgb.shape[0], rgb.shape[1]) != (self.render_h, self.render_w):
                rgb = cv2.resize(rgb, (self.render_w, self.render_h), interpolation=cv2.INTER_LINEAR)
                m = cv2.resize(m, (self.render_w, self.render_h), interpolation=cv2.INTER_NEAREST)
            warps.append(torch.from_numpy(np.ascontiguousarray(rgb)).to(device))
            masks.append(torch.from_numpy(np.ascontiguousarray(m > 127)).to(device))
        return warps, masks

    def _render_multi_mesh(
        self,
        sample: Dict[str, Any],
        end_positions: List[Dict[str, float]],
        end_yaws: List[float],
        V: int,
        timer: Optional["NoOpTimer"] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Mesh geometry path: build one world-space triangle mesh from the
        shared start RGBD (same world frame as ``build_world_gs``), then render
        it to each end camera with the nvdiffrast cubemap->ERP renderer.

        Uses the standalone helpers in ``training/data/mesh_warp_ext.py``.
        """
        if timer is None:
            timer = NOOP_TIMER
        device = self.device
        from training.data.mesh_warp_ext import build_world_mesh, render_mesh_multiview

        def _zero():
            zr = [torch.zeros((self.render_h, self.render_w, 3), dtype=torch.float32, device=device) for _ in range(V)]
            zm = [torch.zeros((self.render_h, self.render_w), dtype=torch.bool, device=device) for _ in range(V)]
            return zr, zm

        with timer.section("io_upload"):
            start_rgb = sample["start_rgb_for_pcd"].to(device, non_blocking=True)
            start_depth = sample["start_depth_for_pcd"].to(device, non_blocking=True)
        rays = self._rays(*start_depth.shape[-2:])

        with timer.section("sky_mask"):
            sky_mask = self._build_sky_mask(start_rgb, sample.get("label"), sample=sample)
        with timer.section("depth_edge"):
            if self.sky_mask_mode == "depth_v2":
                edge_mask = torch.zeros_like(sky_mask, dtype=torch.bool)
            else:
                edge_mask = torch.from_numpy(
                    utils3d.numpy.depth_edge(start_depth.detach().cpu().numpy(), rtol=self.rtol_edge)
                ).to(device).bool()
        full_mask = sky_mask | edge_mask

        with timer.section("pcd_filter"):
            valid = torch.isfinite(start_depth) & (start_depth > 0)
            if (valid & ~full_mask).sum().item() < 100:
                return _zero()
            if self.sky_mask_mode == "skip":
                distance = start_depth.float()
            else:
                max_d = torch.quantile(start_depth[valid & ~full_mask], q=0.99).item()
                distance = torch.clip(start_depth, 0.0, max_d).float()
            if self.depth_contract is not None and self.depth_contract > 0:
                median = torch.median(distance.reshape(-1)).item()
                cd = median * self.depth_contract
                cmask = distance > cd
                distance[cmask] = (2 * cd) - (cd ** 2 / (distance[cmask] + 1e-6))

            pos_start = {k: float(sample["start_position"][k]) / 100.0 for k in ("X", "Y", "Z")}
            start_w2c_np = _build_w2c_render_from_ue5(pos_start, float(sample["start_yaw"]))
            start_c2w_np = np.linalg.inv(start_w2c_np).astype(np.float32)

        with timer.section("gs_build"):  # reuse "gs_build" label = mesh build time
            mesh = build_world_mesh(
                start_rgb, distance, rays, full_mask, start_c2w_np, device,
            )
        if mesh is None:
            return _zero()

        far = float(np.linalg.norm(np.asarray(mesh.vertices), axis=1).max()) * 1.1 + 1e-3
        end_w2c_list = [
            _build_w2c_render_from_ue5(
                {k: float(end_positions[i][k]) / 100.0 for k in ("X", "Y", "Z")},
                float(end_yaws[i]),
            )
            for i in range(V)
        ]
        with timer.section("gs_render"):  # reuse "gs_render" label = mesh render time
            warps, masks = render_mesh_multiview(
                mesh, end_w2c_list, self.render_h, self.render_w, device, near=1e-3, far=far
            )
        return warps, masks

    @torch.inference_mode()
    def render_w2c_stack(
        self,
        gs: Dict[str, torch.Tensor],
        Rts_np: np.ndarray,
        timer: Optional["NoOpTimer"] = None,
    ) -> List[np.ndarray]:
        """Render a pre-built GS cloud to a stack of ``(F, 4, 4)`` w2c poses.

        Returns a length-``F`` list of ``(H, W, 3)`` float32 [0,1] RGB arrays
        (CPU numpy). Used by the interpolated fly-through video renderer.
        """
        if timer is None:
            timer = NOOP_TIMER
        if Rts_np.ndim == 2:
            Rts_np = Rts_np[None]
        with timer.section("gs_render"):
            rgb_list, _ = gs_pano_render_color(
                gs,
                Rts_np.astype(np.float32),
                self.render_h,
                self.render_w,
                self.device,
                render_depth=False,
                erp_mask_aware=True,
            )
        return [r.astype(np.float32, copy=False) for r in rgb_list]

    @torch.inference_mode()
    def render_batch_multiview(
        self,
        samples: List[Dict[str, Any]],
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """4-view batch render.

        Returns
        -------
        warp_rgb   : (B, V, 3, H, W) float in [-1, 1], or None if use_warp=False
        warp_mask  : (B, V, 1, H, W) float in {0, 1}, or None
        plucker_start : (B, 6, H, W) float (SHARED start view), or None
        plucker_end   : (B, V, 6, H, W) float, or None
        pose_vec      : (B, V, 3) float relative END translations, or None
        """
        if self.profile and self.device.type == "cuda":
            timer: NoOpTimer = CudaTimer(self.device)
        else:
            timer = NOOP_TIMER

        warp_b: List[torch.Tensor] = []
        mask_b: List[torch.Tensor] = []
        plk_start_b: List[torch.Tensor] = []
        plk_end_b: List[torch.Tensor] = []
        pose_b: List[torch.Tensor] = []

        for s in samples:
            V = int(s.get("num_views", len(s["end_positions"])))

            if self.use_warp:
                warps, masks = self.render_one_multi(s, timer=timer)
                # (H, W, 3)[0,1] -> (3, H, W)[-1,1]; (H, W)bool -> (1, H, W)float
                wv = torch.stack(
                    [(r.permute(2, 0, 1) * 2.0 - 1.0).contiguous() for r in warps], dim=0
                )  # (V, 3, H, W)
                mv = torch.stack(
                    [m.float().unsqueeze(0).contiguous() for m in masks], dim=0
                )  # (V, 1, H, W)
                warp_b.append(wv)
                mask_b.append(mv)

            if self.use_plucker or self.use_pose:
                pe_list: List[torch.Tensor] = []
                pv_list: List[torch.Tensor] = []
                ps_shared: Optional[torch.Tensor] = None
                for i in range(V):
                    sub = {
                        "start_position": s["start_position"],
                        "start_yaw": s["start_yaw"],
                        "end_position": s["end_positions"][i],
                        "end_yaw": s["end_yaws"][i],
                    }
                    if self.use_plucker:
                        ps, pe = self._plucker_maps_one(sub)
                        if ps_shared is None:
                            ps_shared = ps  # start view identical across views
                        pe_list.append(pe)
                    if self.use_pose:
                        pv_list.append(self._pose_vec_one(sub))
                if self.use_plucker:
                    plk_start_b.append(ps_shared)
                    plk_end_b.append(torch.stack(pe_list, dim=0))  # (V, 6, H, W)
                if self.use_pose:
                    pose_b.append(torch.stack(pv_list, dim=0))  # (V, 3)

        self.last_timings = timer.finalize()

        warp_rgb = torch.stack(warp_b, dim=0) if self.use_warp else None  # (B,V,3,H,W)
        warp_mask = torch.stack(mask_b, dim=0) if self.use_warp else None  # (B,V,1,H,W)
        plucker_start = torch.stack(plk_start_b, dim=0) if self.use_plucker else None  # (B,6,H,W)
        plucker_end = torch.stack(plk_end_b, dim=0) if self.use_plucker else None  # (B,V,6,H,W)
        pose_vec = torch.stack(pose_b, dim=0) if self.use_pose else None  # (B,V,3)

        return warp_rgb, warp_mask, plucker_start, plucker_end, pose_vec
