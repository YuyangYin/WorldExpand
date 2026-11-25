"""Online warp variant of the Pano2Pano dataset (CPU only).

Reads a step3.2 JSON (no `warp_output`, no `*_pt` latents) and yields the
*raw* ingredients needed by :class:`training.data.online_warp_renderer.
OnlineWarpRenderer` to build the PCD + gsplat render + VAE encode on the
training rank's GPU. The dataloader workers themselves stay CPU-only so
that fork multiprocessing does not have to spin up CUDA contexts.

Per-sample dict (returned by ``__getitem__``):

    case_id              : str
    label                : "indoor" | "outdoor" | None
    caption              : str
    start_rgb_for_pcd    : torch.float32 (H_pcd, W_pcd, 3) in [0, 1]
    start_depth_for_pcd  : torch.float32 (H_pcd, W_pcd)  cleaned EXR distance
    start_sky_mask_for_pcd : Optional[torch.bool] (H_pcd, W_pcd)
                              # Pre-computed sky mask loaded from
                              # ``sky_cache_dir`` when ``sky_mask_mode='cached'``.
                              # ``None`` on cache miss; the renderer falls back
                              # to a lazy ZIM/GD call in that case.
    start_rgb_train      : torch.float32 (3, img_size, img_size*2)  in [-1, 1]
    end_rgb_train        : torch.float32 (3, img_size, img_size*2)  in [-1, 1]
    cond_img_prompt      : torch.float32 (3, h_cond, w_cond)        in [-1, 1]
    start_position       : dict X/Y/Z (float, metres, **post-recenter**)
    start_yaw            : float (degrees, **post-recenter**: always 0)
    end_position         : dict X/Y/Z (float, metres, relative to start)
    end_yaw              : float (degrees, relative to start)

In the recentered coordinate convention, start is placed at the origin with yaw=0,
end is expressed as a delta rotated into the start-frame local frame.

The 5-view intersection tuple expansion (``front1/back1/front2/back2``) is
shared with the offline step4 entry point so this loader can consume both
legacy flat-form JSONs and TUPLE-v1 JSONs.
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Must be set before cv2 is imported; enables OpenEXR decoding in OpenCV.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from PIL import Image

from panorama_rendering.nvrender.pipeline_utils import (
    load_exr_depth,
    load_position_rotation_from_json,
)

from training.data.yaw_align import VALID_MODES as YAW_ALIGN_VALID_MODES
from training.data.yaw_align import apply_yaw_align


# ---------------------------------------------------------------------------
# Shared TUPLE-v1 expansion (mirrors the original offline warp renderer).
# ---------------------------------------------------------------------------
_NEIGHBOR_ROLES = ("front1", "back1", "front2", "back2")


def _paths_for(scene: str, traj: str, frame: str) -> Dict[str, str]:
    rel = f"{scene}/{scene}/{traj}/{frame}"
    return {
        "jpg": f"{rel}/{frame}.jpg",
        "exr": f"{rel}/{frame}.exr",
        "json": f"{rel}/{frame}.json",
    }


def _expand_tuple_case(tup: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One intersection tuple -> 4 sub-cases (one per neighbor role)."""
    scene = tup["scene"]
    main_traj = str(tup["main_trajectory"])
    comp_traj = str(tup.get("companion_trajectory", ""))
    center = tup["center"]
    center_frame = str(center["frame"])
    center_traj = str(center.get("trajectory", main_traj))
    parent_id = tup.get("intersection_id") \
        or f"{scene}-{main_traj}-{center_frame}"

    subs = []
    for role in _NEIGHBOR_ROLES:
        pos = tup.get(role)
        if not isinstance(pos, dict):
            continue
        v_traj = str(pos.get(
            "trajectory",
            comp_traj if role in ("front2", "back2") else main_traj,
        ))
        v_frame = str(pos["frame"])
        case_id = f"{parent_id}__{role}"
        directory = f"{scene}/{scene}/{main_traj}/{case_id}"
        sub = {
            "id": case_id,
            "directory": directory,
            "start": _paths_for(scene, center_traj, center_frame),
            "end": _paths_for(scene, v_traj, v_frame),
            "parent_case_id": parent_id,
            "role": role,
            "scene": scene,
            "main_trajectory": main_traj,
            "companion_trajectory": comp_traj,
            "center_frame": center_frame,
            "view_frame": v_frame,
        }
        if "label" in tup:
            sub["label"] = tup["label"]
        if "caption" in tup:
            sub["caption"] = tup["caption"]
        subs.append(sub)
    return subs


def _is_tuple_case(c: Any) -> bool:
    return (
        isinstance(c, dict)
        and isinstance(c.get("center"), dict)
        and "start" not in c
        and any(isinstance(c.get(r), dict) for r in _NEIGHBOR_ROLES)
    )


# ---------------------------------------------------------------------------
# Expanded-tuple expansion (flat schema from stepe_compute_pose_metadata).
#
# Flat schema produced by data_pipeline_fourview/stepe_compute_pose_metadata.py:
#
#   {
#     "case_id":              "<scene>__<main>__<comp>__<center_frame>",
#     "directory":            "<scene>/<scene>/<main_traj>",
#     "scene":                "...",
#     "main_trajectory":      "...",
#     "companion_trajectory": "...",
#     "label":                "indoor" | "outdoor" | "unknown",
#     "caption":              "...",
#     "intersection_xy":      [x, y] | null,
#     "start":  {"jpg": "..."},           # center frame — jpg only
#     "front1": {"jpg": "..."},
#     "back1":  {"jpg": "..."},
#     "front2": {"jpg": "..."},
#     "back2":  {"jpg": "..."},
#     "metadata": {
#       "start_yaw_deg": float,           # shared center-frame absolute yaw
#       "front1": {
#         "end_position_recentered": {"X": float, "Y": float, "Z": float},
#         "end_yaw_delta_deg":  float,
#         "distance_cm":        float,
#         "psnr_vs_center":     float | null,
#         "frame_index":        int
#       },
#       "back1": {...}, "front2": {...}, "back2": {...}
#     }
#   }
#
# exr/json sidecars are NOT stored; callers derive them by suffix swap:
#   exr  = jpg.replace(".jpg", ".exr")
#   json = jpg.replace(".jpg", ".json")
#
# Each expanded sub-case carries "_pose_metadata" so _load_one can use the
# pre-computed pose values directly without re-reading the UE5 JSON sidecars.
# ---------------------------------------------------------------------------

def _is_expanded_tuple_case(c: Any) -> bool:
    """Detect the flat expanded-tuple schema from stepe_compute_pose_metadata.

    Distinguishing features of the new flat schema:
    * ``start`` dict whose value has a ``jpg`` key (center frame — jpg only)
    * ``front1`` dict directly on the case (not inside a ``views`` list)

    This is distinct from:
    * TUPLE-v1: has ``center_frame`` / ``frames`` keys, no ``start`` dict
    * old expanded schema: has ``views`` list with nested ``end`` dicts
    * step2.3 views-format: has ``jpg``/``exr``/``json_`` at top level
    """
    if not isinstance(c, dict):
        return False
    start = c.get("start")
    if not isinstance(start, dict) or "jpg" not in start:
        return False
    front1 = c.get("front1")
    return isinstance(front1, dict) and "jpg" in front1


def _expand_expanded_tuple_case(tup: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One flat expanded-tuple → 4 sub-cases (one per view role).

    Sub-cases carry ``_pose_metadata`` so ``_load_one`` can skip the
    ``load_position_rotation_from_json`` + ``_recenter`` calls at training time.
    The pre-computed values already reflect the yaw-only recentering done by
    ``data_pipeline_fourview/_pose_utils.py::recenter_end_pose`` (identical
    math to ``OnlineWarpDataset._recenter``).
    """
    case_id = str(tup.get("case_id") or tup.get("id", ""))
    start_paths = tup["start"]  # {"jpg": "..."}
    case_meta: Dict[str, Any] = tup.get("metadata") or {}
    start_yaw = float(case_meta.get("start_yaw_deg", 0.0))
    subs: List[Dict[str, Any]] = []
    for role in _NEIGHBOR_ROLES:
        view = tup.get(role)
        if not isinstance(view, dict):
            continue
        role_meta: Dict[str, Any] = case_meta.get(role) or {}
        sub: Dict[str, Any] = {
            "id": f"{case_id}__{role}",
            "directory": tup.get("directory", ""),
            "start": start_paths,
            "end": view,              # {"jpg": "..."}
            "parent_case_id": case_id,
            "role": role,
            "scene": tup.get("scene", ""),
        }
        if "label" in tup:
            sub["label"] = tup["label"]
        if "caption" in tup:
            sub["caption"] = tup["caption"]
        # Carry pre-computed pose so _load_one can skip runtime JSON reads.
        if role_meta:
            epr = role_meta.get("end_position_recentered") or {}
            sub["_pose_metadata"] = {
                "start_yaw": start_yaw,
                "end_position": {
                    "X": float(epr.get("X", 0.0)),
                    "Y": float(epr.get("Y", 0.0)),
                    "Z": float(epr.get("Z", 0.0)),
                },
                "end_yaw": float(role_meta.get("end_yaw_delta_deg", 0.0)),
            }
        subs.append(sub)
    return subs


# ---------------------------------------------------------------------------
# Dimensions helper (same convention as the offline dataset).
# ---------------------------------------------------------------------------
CONDITION_IMAGE_SIZE = 384 * 384


def calculate_dimensions(target_area: int, ratio: float) -> Tuple[int, int]:
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height


def erp_circular_pad_amount(width: int, pad_frac: int = 16) -> int:
    """Per-side circular-pad width for an ERP tensor of the given pixel width.

    The nominal amount is ``W // pad_frac`` (the long-standing rule). To keep
    the padded width VAE/pack-friendly we round it to the nearest multiple of
    8, which guarantees ``(W + 2*pad) % 16 == 0`` whenever ``W % 16 == 0`` (so
    the latent width stays even after the VAE 8x downsample + 2x pack). For the
    canonical sizes (ERP width 1024 -> 64, 1920 -> 120) ``W // 16`` is already a
    multiple of 8, so this is a no-op and existing runs are byte-for-byte
    unchanged. It only kicks in for "odd" widths such as the 720-cond ERP
    (width 1440 -> 90 -> 88) where the raw ``//16`` would break divisibility.
    """
    base = int(width) // int(pad_frac)
    if base <= 0:
        return 0
    return int(round(base / 8.0)) * 8


def _circular_pad_width_torch(t: torch.Tensor, pad_frac: int = 16) -> torch.Tensor:
    """Circular-pad an ERP tensor on the width axis (last dim).

    Mirrors qwen-pano2pano2-diffusers and the NVSPano offline dataset:
    pad by ``erp_circular_pad_amount(W)`` pixels on each side, cycling from the
    opposite edge so the seam at u=0 / u=W becomes continuous before VAE encode.
    Accepts any tensor whose last dim is the ERP width (CHW, (1,H,W),
    (B,C,H,W), etc).
    """
    pad_w = erp_circular_pad_amount(int(t.shape[-1]), pad_frac)
    if pad_w <= 0:
        return t
    return torch.cat([t[..., -pad_w:], t, t[..., :pad_w]], dim=-1)


# ---------------------------------------------------------------------------
# Online dataset.
# ---------------------------------------------------------------------------
class OnlineWarpDataset(Dataset):
    """Returns CPU tensors + pose deltas for an on-the-fly warp + VAE encode."""

    # Two-part system prompt:
    # 1) Colleague's full-FT prefix — verbatim string they used at FT time on
    #    Qwen-Image-Edit-2509 (see CubeMap_Related_Work or
    #    /path/to/datasets/
    #    datasets_2x4_padding.py). Including this prefix puts the
    #    Qwen2.5-VL prompt encoder in the same input distribution as the FT
    #    base, so the FT prior transfers cleanly.
    # 2) Our existing NVS-task prompt — kept after the colleague's prefix so
    #    the model still sees what we want it to do (forward-projection).
    SYSTEM_PROMPT = (
      "Generate a cube map (front, back, left, right, top, bottom) "
        "based on the condition image and the following description: "
        "Translate the camera to a new nearby viewpoint and synthesize "
        "the resulting view, keeping geometry, style, lighting, and fine "
        "details consistent with the input."
    )

    def __init__(
        self,
        train_files: Sequence[str],
        data_root: str,
        img_size: int = 512,
        pcd_build_h: int = 960,
        pcd_build_w: int = 1920,
        drop_text_prob: float = 0.0,
        # "lazy" | "skip" | "indoor_only" | "cached" | "depth_v2"
        sky_mask_mode: str = "lazy",
        # Required iff sky_mask_mode == "cached". Mirrors the source layout:
        #   <sky_cache_dir>/<scene>/<scene>/<traj>/<frame>/<frame>_sky.png
        # Generated by the offline sky-mask exporter.
        sky_cache_dir: Optional[str] = None,
        depth_p99_clip: bool = True,           # clip depth at p99 to drop sky pixels
        recenter_first_frame: bool = True,
        # Pixel-space ERP circular padding. Mirrors the offline dataset
        # (training/data/datasets_pano2pano.py) and qwen-pano2pano2-diffusers
        # reference: pad each ERP image by W // 16 pixels on both left/right,
        # cycling from the opposite edge so the seam is smoothed before VAE
        # encode. When True, the trainer's "if not rgb_circular_pad" branch
        # is skipped so latent-space +4 pad does NOT also fire (no double pad).
        # ``wrap_image`` / ``wrap_mask`` come out of the GPU renderer in
        # train.py and are padded there, not here.
        rgb_circular_pad: bool = False,
        # Cube layout: when set, ERP start/end RGBs (the VAE-bound tensors)
        # are re-sampled into a 2x4 cube grid before being returned. See
        # ``training/data/cube_layout.py`` and
        # ``CubeMap_Related_Work/2x4_padding_strategy.md`` for the layout.
        #
        # When this is non-None we force-disable ``rgb_circular_pad`` for
        # the dataset side: the cube grid already carries its own
        # ``grid_pad`` columns at the seam, so a second ERP-style pad
        # would just stack two unrelated padding schemes on top of each
        # other. The trainer's latent-space "+4" wrap pad is also gated
        # on ``cube_layout`` in ``training/train.py``.
        #
        # ``cond_img_prompt`` (fed to Qwen2.5-VL) is intentionally left
        # as an ERP thumbnail: the text tower was pre-trained on natural
        # images, not cube atlases, and cond_img_prompt is not in the
        # VAE / transformer-token loss path anyway.
        cube_layout: Optional[str] = None,
        cube_face_size: int = 480,
        cube_grid_pad: int = 32,
        # ERP yaw-alignment pre-processing. Applied AFTER ``_recenter`` and
        # BEFORE the circular pad below. Modes are:
        #   ``off``             : no-op (legacy / default).
        #   ``end_to_start``    : roll end so its ERP center faces the same
        #                          world direction as start (end_yaw -> 0).
        #   ``to_displacement`` : roll BOTH start AND end so every ERP center
        #                          faces atan2(end_pos.Y, end_pos.X) (the
        #                          motion direction in start's local frame).
        # See ``training/data/yaw_align.py`` for the math + per-tensor list.
        yaw_align_mode: str = "off",
        # Optional SEPARATE resolution for the condition (start) ERP image.
        # When set and != ``img_size``, ``start_rgb_train`` is resized to a
        # ``cond_img_size`` x ``2*cond_img_size`` ERP (and circular-padded at
        # that width) while ``end_rgb_train`` (the prediction target) stays at
        # ``img_size``. The warp / mask / plucker condition streams are
        # down-sampled to match in train.py. ``None`` (default) keeps the
        # legacy single-resolution behaviour. This decouples the condition
        # token grid from the prediction token grid.
        cond_img_size: Optional[int] = None,
        # Aliases used elsewhere; accepted but ignored here for API parity.
        **_kwargs: Any,
    ) -> None:
        if isinstance(train_files, str):
            train_files = [train_files]
        train_files = [str(p).strip() for p in train_files]

        self.data_root = data_root
        self.img_size = int(img_size)
        # Separate condition (start) resolution. None / == img_size means the
        # legacy single-resolution path (condition same size as target).
        self.cond_img_size = (
            int(cond_img_size)
            if (cond_img_size is not None and int(cond_img_size) != int(img_size))
            else None
        )
        self.pcd_build_h = int(pcd_build_h)
        self.pcd_build_w = int(pcd_build_w)
        self.drop_text_prob = float(drop_text_prob)
        self.sky_mask_mode = str(sky_mask_mode)
        self.depth_p99_clip = bool(depth_p99_clip)
        self.recenter_first_frame = bool(recenter_first_frame)
        self.rgb_circular_pad = bool(rgb_circular_pad)

        # ---- Cube layout (2x4 padding) -----------------------------------
        # Normalise ``cube_layout`` to None | "2x4_padding". Empty / "off"
        # / "null" all mean "no cube transform" so the YAML default can be
        # an unambiguous off-switch.
        from training.data import cube_layout as _cube_layout_mod
        if cube_layout in (None, "", "off", "null", "none"):
            self.cube_layout: Optional[str] = None
        else:
            cl = str(cube_layout).strip().lower()
            if cl not in _cube_layout_mod.VALID_LAYOUTS:
                raise ValueError(
                    f"cube_layout must be one of "
                    f"{_cube_layout_mod.VALID_LAYOUTS} or null, "
                    f"got {cube_layout!r}"
                )
            self.cube_layout = cl
        self.cube_face_size = int(cube_face_size)
        self.cube_grid_pad = int(cube_grid_pad)
        if self.cube_layout is not None:
            _cube_layout_mod.validate_dims(self.cube_face_size, self.cube_grid_pad)
            if self.rgb_circular_pad:
                # Hard-fail rather than silently double-pad: the cube grid
                # already includes a ``grid_pad`` wrap column, and we
                # don't want the ERP-style +W/16 pad to stack on top.
                raise ValueError(
                    "cube_layout is incompatible with rgb_circular_pad=True; "
                    "the cube grid carries its own grid_pad columns. "
                    "Set rgb_circular_pad=False (or disable cube_layout)."
                )

        # Validate yaw-align mode up-front so a typo in the yaml fails loud
        # at __init__ instead of silently no-op'ing for the whole run.
        self.yaw_align_mode = str(yaw_align_mode)
        if self.yaw_align_mode not in YAW_ALIGN_VALID_MODES:
            raise ValueError(
                f"yaw_align_mode must be one of {YAW_ALIGN_VALID_MODES}, "
                f"got {self.yaw_align_mode!r}"
            )

        # Validate that img_size is compatible with the circular pad.
        # Pad expands ERP width as W -> W + 2*(W//16). After VAE 8x downsample
        # + 2x2 transformer patch packing the result must satisfy
        # padded_W % 16 == 0 (so latent_W is even). img_size=1440 is a
        # well-known UE5-native value that fails this check; the user must
        # either disable circular pad or move to a compatible size.
        # NOTE: we only validate the VAE/transformer path (train_w =
        # img_size*2). cond_img_prompt has its own smaller width
        # (~544 px for CONDITION_IMAGE_SIZE=384*384) AND is intentionally
        # left unpadded (see _load_one) because it feeds Qwen2.5-VL,
        # not the VAE. So its width does not need to satisfy this check.
        if self.rgb_circular_pad:
            _w = int(self.img_size) * 2
            _padded_w = _w + 2 * (_w // 16)
            if _padded_w % 16 != 0:
                raise ValueError(
                    f"[OnlineWarpDataset] rgb_circular_pad=True is "
                    f"incompatible with img_size={self.img_size}: padded "
                    f"ERP width = {_padded_w} (= {_w} + 2*{_w//16}) is "
                    f"not a multiple of 16, so VAE latent W would be odd "
                    f"and the transformer's 2x2 patch packing breaks. "
                    f"Compatible img_size values are multiples of 64 "
                    f"(e.g. 512/768/960/1024/1280/1600/1920/2048). "
                    f"Alternatively, disable rgb_circular_pad and the "
                    f"trainer falls back to latent-space +4 wrap pad."
                )

        # Normalise sky_cache_dir: accept None / "" / "null" all as "off".
        if (
            sky_cache_dir is None
            or (isinstance(sky_cache_dir, str) and sky_cache_dir.strip().lower() in {"", "null", "none"})
        ):
            self.sky_cache_dir: Optional[str] = None
        else:
            self.sky_cache_dir = str(sky_cache_dir)

        if self.sky_mask_mode not in {"lazy", "skip", "indoor_only", "cached", "depth_v2"}:
            raise ValueError(
                f"sky_mask_mode must be one of "
                f"lazy/skip/indoor_only/cached/depth_v2, "
                f"got {self.sky_mask_mode!r}"
            )
        # depth_v2 needs RAW depth to detect sky (sky_thresh check on
        # > 7000 m values). The legacy p99 clip would crush 65504 → ~max
        # real depth and defeat the sky detector. Force-disable here so
        # an old YAML that still says ``depth_p99_clip: True`` does the
        # right thing automatically when the operator flips to depth_v2.
        if self.sky_mask_mode == "depth_v2" and self.depth_p99_clip:
            print(
                f"[OnlineWarpDataset] sky_mask_mode='depth_v2': "
                f"forcing depth_p99_clip=False (the v2 sky mask needs "
                f"raw depth to identify sky/out-of-range pixels via "
                f"sky_thresh; p99-clipping would crush them to ~max "
                f"real depth and break mask_sky)."
            )
            self.depth_p99_clip = False
        # skip mode intentionally KEEPS the sky in the warp (sky is not masked
        # out of the PCD). The legacy p99 clip would clamp the sky's ~65504 m
        # EXR depth down toward the real-geometry range, moving the sky to a
        # wrong finite distance; combined with the renderer's extreme-distance
        # cull this is what made the sky vanish from the warp. Disable it so
        # skip mode ships raw depth, mirroring the depth_v2 guard above.
        if self.sky_mask_mode == "skip" and self.depth_p99_clip:
            print(
                f"[OnlineWarpDataset] sky_mask_mode='skip': forcing "
                f"depth_p99_clip=False so the kept sky/far pixels retain "
                f"their raw EXR depth (no p99 clamp)."
            )
            self.depth_p99_clip = False
        if self.sky_mask_mode == "cached":
            if not self.sky_cache_dir:
                raise ValueError(
                    "sky_mask_mode='cached' requires sky_cache_dir to be set "
                    "(point it at the directory produced by "
                    "the offline sky-mask exporter)."
                )
            if not os.path.isdir(self.sky_cache_dir):
                raise FileNotFoundError(
                    f"sky_cache_dir not found: {self.sky_cache_dir}"
                )
        elif self.sky_cache_dir is not None:
            print(
                f"[OnlineWarpDataset] WARN: sky_cache_dir is set ({self.sky_cache_dir}) "
                f"but sky_mask_mode={self.sky_mask_mode!r}; cache will NOT be "
                f"used. Set sky_mask_mode='cached' to consume it."
            )

        self.cases: List[Dict[str, Any]] = []
        for path in train_files:
            with open(path, "r") as f:
                payload = json.load(f)
            raw_cases = (
                payload["cases"]
                if isinstance(payload, dict) and "cases" in payload
                else payload
            )

            if raw_cases and _is_expanded_tuple_case(raw_cases[0]):
                # New schema from data_pipeline_fourview/stepe_compute_pose_metadata.py.
                # Paths and pose metadata are already fully expanded; just unpack
                # the views list into one sub-case per role.
                expanded: List[Dict[str, Any]] = []
                for tup in raw_cases:
                    expanded.extend(_expand_expanded_tuple_case(tup))
                self.cases.extend(expanded)
            elif raw_cases and _is_tuple_case(raw_cases[0]):
                expanded: List[Dict[str, Any]] = []
                for tup in raw_cases:
                    expanded.extend(_expand_tuple_case(tup))
                self.cases.extend(expanded)
            else:
                self.cases.extend(raw_cases)

        if self.sky_mask_mode == "indoor_only":
            kept = [c for c in self.cases if c.get("label") == "indoor"]
            self.cases = kept

        if len(self.cases) == 0:
            raise ValueError(
                f"OnlineWarpDataset: no cases loaded from {train_files} "
                f"(sky_mask_mode={self.sky_mask_mode})."
            )

        # --- bad-data robustness knobs ----------------------------------
        # __getitem__ substitutes a different case when one fails to load
        # (missing jpg/exr/json, corrupt image, etc.). We jump to RANDOM
        # indices on failure rather than idx+1, because a missing
        # scene/trajectory occupies a contiguous block of cases -- so the
        # neighbours of a broken case are very likely broken too. Bad
        # indices are memoised per worker to skip them without re-hitting
        # disk and to avoid log spam. Override the budget via env.
        self.max_getitem_retries = int(
            os.environ.get("DATA_GETITEM_MAX_RETRIES", "50")
        )
        self._bad_indices: set = set()

        cache_label = self.sky_cache_dir if self.sky_mask_mode == "cached" else "off"
        if self.cube_layout is None:
            cube_label = "off"
        else:
            from training.data.cube_layout import grid_shape as _gs
            _h_grid, _w_grid = _gs(self.cube_face_size, self.cube_grid_pad)
            cube_label = (
                f"{self.cube_layout} (face_size={self.cube_face_size}, "
                f"grid_pad={self.cube_grid_pad} -> grid={_h_grid}x{_w_grid})"
            )
        print(
            f"[OnlineWarpDataset] loaded {len(self.cases)} cases from "
            f"{len(train_files)} json(s); img_size={self.img_size}, "
            f"pcd={self.pcd_build_h}x{self.pcd_build_w}, "
            f"sky_mask_mode={self.sky_mask_mode}, "
            f"sky_cache_dir={cache_label}, "
            f"recenter_first_frame={self.recenter_first_frame}, "
            f"rgb_circular_pad={self.rgb_circular_pad}, "
            f"yaw_align_mode={self.yaw_align_mode}, "
            f"cube_layout={cube_label}"
        )

    def __len__(self) -> int:
        return len(self.cases)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _case_id(self, case: Dict[str, Any]) -> str:
        if "id" in case:
            return str(case["id"])
        directory = case.get("directory", "")
        start_frame = os.path.basename(
            os.path.dirname(case.get("start", {}).get("jpg", ""))
        )
        end_frame = os.path.basename(
            os.path.dirname(case.get("end", {}).get("jpg", ""))
        )
        return f"{directory}/{start_frame}_{end_frame}".lstrip("/")

    def _resolve(self, rel_path: str) -> str:
        if not rel_path:
            return rel_path
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.join(self.data_root, rel_path)

    # ------------------------------------------------------------------
    # Sky-mask cache plumbing.
    # ``sky_cache_dir`` mirrors the source-JPG layout. The cached PNG is
    # produced by the offline sky-mask exporter at the
    # same numeric resolution that ``_load_one`` will downsample
    # ``start_rgb_for_pcd`` to (default 960x1920). When the cache is hit we
    # convert ``uint8 -> bool`` (``> 127``) and ship a CPU bool tensor in the
    # sample dict; the renderer side does ``.to(device, non_blocking=True)``.
    # On a miss we return ``None`` and the renderer falls back to a lazy ZIM
    # call (same numerics as before, just slow). See ``docs/online_warp_training.md``
    # §5c for the full data-flow comparison.
    # ------------------------------------------------------------------
    def _cache_path_for(self, start_jpg_rel: str) -> Optional[str]:
        if not self.sky_cache_dir:
            return None
        rel = start_jpg_rel
        # If the JSON happens to store an absolute path under data_root,
        # strip the data_root prefix so the cache layout is consistent.
        if os.path.isabs(rel) and self.data_root and rel.startswith(self.data_root):
            rel = os.path.relpath(rel, self.data_root)
        # Tolerate .jpg / .jpeg / .png suffixes; everything else gets the
        # ``_sky.png`` suffix appended verbatim.
        for ext in (".jpg", ".jpeg", ".png"):
            if rel.endswith(ext):
                rel = rel[: -len(ext)] + "_sky.png"
                break
        else:
            rel = rel + "_sky.png"
        return os.path.join(self.sky_cache_dir, rel)

    def _load_cached_sky(
        self, start_jpg_rel: str
    ) -> Optional[torch.Tensor]:
        """Return (H, W) bool tensor (CPU) or ``None`` on miss."""
        if self.sky_mask_mode != "cached":
            return None
        cache_path = self._cache_path_for(start_jpg_rel)
        if not cache_path or not os.path.exists(cache_path):
            return None
        img = cv2.imread(cache_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        # If the cache was generated at a different resolution (e.g. a 2K
        # cache being consumed by a 960x1920 trainer), nearest-resize so the
        # mask aligns with start_rgb_for_pcd. Nearest preserves the binary
        # boundary; binary masks shouldn't be linearly interpolated.
        if img.shape != (self.pcd_build_h, self.pcd_build_w):
            img = cv2.resize(
                img,
                (self.pcd_build_w, self.pcd_build_h),
                interpolation=cv2.INTER_NEAREST,
            )
        return torch.from_numpy((img > 127)).to(torch.bool)

    def _load_resized_rgb(
        self, jpg_path: str, target_h: int, target_w: int
    ) -> np.ndarray:
        img_bgr = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"cv2.imread failed: {jpg_path}")
        if img_bgr.shape[:2] != (target_h, target_w):
            img_bgr = cv2.resize(
                img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def _clean_depth(self, depth_full: np.ndarray) -> np.ndarray:
        """Apply step4-style cleanup: clamp sky / inf pixels via p99 clip."""
        if not self.depth_p99_clip:
            return depth_full
        valid = (depth_full > 0) & np.isfinite(depth_full)
        if not np.any(valid):
            return depth_full
        p99 = float(np.quantile(depth_full[valid], 0.99))
        cleaned = np.clip(depth_full, 0.0, p99)
        return cleaned

    def _recenter(
        self,
        start_position: Dict[str, float],
        start_rotation: Dict[str, float],
        end_position: Dict[str, float],
        end_rotation: Dict[str, float],
    ) -> Tuple[Dict[str, float], float, Dict[str, float], float]:
        """Mirror of step4_warp_render's --recenter_first_frame branch.

        Always uses Yaw only (Pitch/Roll are zero across the UE5 dataset; see
        the comment block around step4_warp_render.py:478).
        """
        if not self.recenter_first_frame:
            return (
                dict(start_position),
                float(start_rotation.get("Yaw", 0.0)),
                dict(end_position),
                float(end_rotation.get("Yaw", 0.0)),
            )

        y1 = math.radians(float(start_rotation.get("Yaw", 0.0)))
        cos_y1, sin_y1 = math.cos(y1), math.sin(y1)

        dx_world = float(end_position["X"]) - float(start_position["X"])
        dy_world = float(end_position["Y"]) - float(start_position["Y"])
        dz_world = float(end_position["Z"]) - float(start_position["Z"])

        dx_local = cos_y1 * dx_world + sin_y1 * dy_world
        dy_local = -sin_y1 * dx_world + cos_y1 * dy_world
        dz_local = dz_world

        start_pos_out = {"X": 0.0, "Y": 0.0, "Z": 0.0}
        end_pos_out = {"X": dx_local, "Y": dy_local, "Z": dz_local}
        # Wrap to [-180, 180] so the value is the shortest signed rotation
        # (sin/cos-equivalent; only matters for stats / debug readability).
        raw_delta = (
            float(end_rotation.get("Yaw", 0.0))
            - float(start_rotation.get("Yaw", 0.0))
        )
        end_yaw_out = ((raw_delta + 180.0) % 360.0) - 180.0
        return start_pos_out, 0.0, end_pos_out, end_yaw_out

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        n = len(self.cases)
        # Tolerate subclasses that didn't set these via super().__init__.
        max_retries = getattr(self, "max_getitem_retries", 50)
        bad = getattr(self, "_bad_indices", None)
        if bad is None:
            bad = set()
            self._bad_indices = bad

        last_err: Optional[Exception] = None
        cur = idx % n
        for attempt in range(max_retries):
            # Skip indices already known-bad on this worker (no disk hit).
            if cur in bad:
                if len(bad) >= n:
                    break  # everything we've seen is broken; bail out below
                cur = random.randrange(n)
                continue
            try:
                return self._load_one(cur)
            except Exception as e:
                last_err = e
                bad.add(cur)
                # Throttle logging: first failure of this call + every 20th
                # afterwards, so a broken scene doesn't flood the log.
                if attempt == 0 or attempt % 20 == 0:
                    print(
                        f"[OnlineWarpDataset] idx={idx} attempt={attempt} "
                        f"skip {type(e).__name__}: {e} "
                        f"(bad-so-far={len(bad)})",
                        flush=True,
                    )
                # Random jump escapes contiguous blocks of broken cases
                # (a missing scene/traj spans many neighbouring indices).
                cur = random.randrange(n)

        raise RuntimeError(
            f"[OnlineWarpDataset] gave up after {max_retries} attempts "
            f"starting from idx={idx} ({len(bad)} bad indices seen on this "
            f"worker); last error: {last_err}"
        )

    def _load_one(self, idx: int) -> Dict[str, Any]:
        case = self.cases[idx]
        case_id = self._case_id(case)

        start_jpg = self._resolve(case["start"]["jpg"])
        start_jpg_rel = case["start"]["jpg"]
        start_exr = self._resolve(
            case["start"].get("exr") or start_jpg_rel.replace(".jpg", ".exr")
        )
        start_json = self._resolve(
            case["start"].get("json") or start_jpg_rel.replace(".jpg", ".json")
        )
        end_jpg = self._resolve(case["end"]["jpg"])
        end_jpg_rel = case["end"]["jpg"]
        end_json = self._resolve(
            case["end"].get("json") or end_jpg_rel.replace(".jpg", ".json")
        )

        for p in (start_jpg, start_exr, start_json, end_jpg, end_json):
            if not os.path.exists(p):
                raise FileNotFoundError(f"missing input: {p}")

        # --- Geometry inputs (PCD-build resolution) ---------------------
        start_rgb_full_bgr = cv2.imread(start_jpg, cv2.IMREAD_COLOR)
        if start_rgb_full_bgr is None:
            raise FileNotFoundError(f"cv2.imread failed: {start_jpg}")
        h_full, w_full = start_rgb_full_bgr.shape[:2]

        start_depth_full = load_exr_depth(start_exr)
        if start_depth_full.shape[:2] != (h_full, w_full):
            start_depth_full = cv2.resize(
                start_depth_full, (w_full, h_full), interpolation=cv2.INTER_LINEAR
            )

        start_depth_clean = self._clean_depth(start_depth_full)

        # Downsample to (pcd_build_h, pcd_build_w) for cheap PCD build.
        start_rgb_pcd_bgr = cv2.resize(
            start_rgb_full_bgr, (self.pcd_build_w, self.pcd_build_h),
            interpolation=cv2.INTER_AREA,
        )
        start_rgb_pcd = cv2.cvtColor(start_rgb_pcd_bgr, cv2.COLOR_BGR2RGB)
        start_depth_pcd = cv2.resize(
            start_depth_clean, (self.pcd_build_w, self.pcd_build_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # --- Training-resolution start / end RGBs -----------------------
        # ``end`` (prediction target) is always at the full training res.
        # ``start`` (condition) may be at a separate, lower ``cond_img_size``
        # so its VAE/token grid is decoupled from the target's.
        train_h = self.img_size
        train_w = self.img_size * 2
        if self.cond_img_size is not None:
            cond_train_h = self.cond_img_size
            cond_train_w = self.cond_img_size * 2
        else:
            cond_train_h = train_h
            cond_train_w = train_w
        start_rgb_train_np = cv2.cvtColor(
            cv2.resize(start_rgb_full_bgr, (cond_train_w, cond_train_h), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        end_rgb_train_np = self._load_resized_rgb(end_jpg, train_h, train_w)

        # cond_img_prompt size (matches the offline dataset's CONDITION_IMAGE_SIZE)
        cond_w, cond_h = calculate_dimensions(
            CONDITION_IMAGE_SIZE, train_w / train_h
        )
        cond_img_prompt_np = cv2.cvtColor(
            cv2.resize(start_rgb_full_bgr, (cond_w, cond_h), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )

        # --- Sky mask (cached only) -------------------------------------
        # We pass the ORIGINAL JSON-relative path so the cache key is
        # stable across machines (don't use the resolved abs path which
        # may include local mount prefixes).
        sky_cached = self._load_cached_sky(case["start"]["jpg"])

        # --- Pose -------------------------------------------------------
        if "_pose_metadata" in case:
            # Pre-computed by data_pipeline_fourview/stepe_compute_pose_metadata.py.
            # Values already reflect the yaw-only recentering (start at origin,
            # end expressed as delta in start's local frame), so we can skip
            # the load_position_rotation_from_json + _recenter calls entirely.
            pm = case["_pose_metadata"]
            start_pos_out: Dict[str, float] = {"X": 0.0, "Y": 0.0, "Z": 0.0}
            start_yaw_out: float = 0.0
            end_pos_out: Dict[str, float] = {
                "X": float(pm["end_position"]["X"]),
                "Y": float(pm["end_position"]["Y"]),
                "Z": float(pm["end_position"]["Z"]),
            }
            end_yaw_out: float = float(pm["end_yaw"])
        else:
            start_position, start_rotation = load_position_rotation_from_json(start_json)
            end_position, end_rotation = load_position_rotation_from_json(end_json)
            start_pos_out, start_yaw_out, end_pos_out, end_yaw_out = self._recenter(
                start_position, start_rotation, end_position, end_rotation
            )

        # --- Prompt -----------------------------------------------------
        caption = case.get("caption", "") or ""
        if random.random() < self.drop_text_prob:
            caption = ""
        prompt = self.SYSTEM_PROMPT + caption

        # --- Convert to torch tensors (CPU, contiguous) -----------------
        start_rgb_pcd_t = torch.from_numpy(
            np.ascontiguousarray(start_rgb_pcd.astype(np.float32) / 255.0)
        )
        start_depth_pcd_t = torch.from_numpy(
            np.ascontiguousarray(start_depth_pcd.astype(np.float32))
        )
        start_rgb_train_t = torch.from_numpy(
            np.ascontiguousarray(start_rgb_train_np.astype(np.float32) / 127.5 - 1.0)
        ).permute(2, 0, 1).contiguous()
        end_rgb_train_t = torch.from_numpy(
            np.ascontiguousarray(end_rgb_train_np.astype(np.float32) / 127.5 - 1.0)
        ).permute(2, 0, 1).contiguous()
        cond_img_prompt_t = torch.from_numpy(
            np.ascontiguousarray(cond_img_prompt_np.astype(np.float32) / 127.5 - 1.0)
        ).permute(2, 0, 1).contiguous()

        sample: Dict[str, Any] = {
            "case_id": case_id,
            "label": case.get("label"),
            "caption": prompt,
            "start_rgb_for_pcd": start_rgb_pcd_t,
            "start_depth_for_pcd": start_depth_pcd_t,
            "start_sky_mask_for_pcd": sky_cached,  # None on cache miss
            "start_rgb_train": start_rgb_train_t,
            "end_rgb_train": end_rgb_train_t,
            "cond_img_prompt": cond_img_prompt_t,
            "start_position": start_pos_out,
            "start_yaw": float(start_yaw_out),
            "end_position": end_pos_out,
            "end_yaw": float(end_yaw_out),
        }

        # --- ERP yaw alignment ------------------------------------------
        # Runs after _recenter (so start_yaw == 0 going in) but BEFORE the
        # circular pad below — yaw_align operates on the raw ERP width
        # (img_size*2), whereas circular pad expands the width by W/16 on
        # each side. Doing them in the other order would force yaw_align
        # to know about the pad and roll it consistently, which is fragile.
        if self.yaw_align_mode != "off":
            sample = apply_yaw_align(sample, self.yaw_align_mode)

        # --- Pixel-space ERP circular padding ---------------------------
        # Mirrors qwen-pano2pano2-diffusers/image_datasets/datasets_pano2pano.py
        # exactly: pad ONLY the VAE-bound tensors (start_rgb_train,
        # end_rgb_train) here, plus wrap_image/wrap_mask in train.py
        # after the GPU renderer. ``cond_img_prompt`` is fed to
        # Qwen2.5-VL (text tower), which was pretrained on natural
        # non-panoramic images; circular-padding its input puts an
        # off-distribution duplicated band at the edges and just wastes
        # visual tokens without conveying any wrap-around semantics.
        # The reference dataset deliberately leaves cond_img_prompt
        # unpadded - we do the same.
        if self.rgb_circular_pad:
            sample["start_rgb_train"] = _circular_pad_width_torch(sample["start_rgb_train"])
            sample["end_rgb_train"] = _circular_pad_width_torch(sample["end_rgb_train"])

        # --- Cube layout (2x4 padding) re-projection --------------------
        # Applied AFTER yaw-align (so the cube faces are sampled from the
        # already-aligned ERP) and AFTER the circular-pad branch (which
        # is a no-op anyway when cube_layout is on; see __init__ guard).
        # Only the VAE-bound tensors are converted; cond_img_prompt is
        # left as an ERP thumbnail. wrap_image / wrap_mask are produced
        # by the GPU renderer and converted on its side
        # (see training/data/online_warp_renderer.py).
        if self.cube_layout == "2x4_padding":
            from training.data.cube_layout import erp_to_2x4_padding_grid
            sample["start_rgb_train"] = erp_to_2x4_padding_grid(
                sample["start_rgb_train"],
                self.cube_face_size,
                self.cube_grid_pad,
                interp_mode="bilinear",
            )
            sample["end_rgb_train"] = erp_to_2x4_padding_grid(
                sample["end_rgb_train"],
                self.cube_face_size,
                self.cube_grid_pad,
                interp_mode="bilinear",
            )

        return sample


# ---------------------------------------------------------------------------
# Collate: render is per-case, so we keep a list-of-dicts for renderer + stack
# only the train-resolution tensors that the transformer ultimately consumes.
# ---------------------------------------------------------------------------
def _online_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "samples": batch,
        "captions": [b["caption"] for b in batch],
        "start_rgb_train": torch.stack([b["start_rgb_train"] for b in batch], dim=0),
        "end_rgb_train": torch.stack([b["end_rgb_train"] for b in batch], dim=0),
        "cond_img_prompt": torch.stack([b["cond_img_prompt"] for b in batch], dim=0),
    }


# ---------------------------------------------------------------------------
# Loader factory. Signature mirrors training/data/datasets_pano2pano.loader so
# train.py's ``loader(**args.data_config)`` call works unchanged.
# ---------------------------------------------------------------------------
def loader(
    train_batch_size: int,
    num_workers: int,
    **dataset_kwargs: Any,
) -> DataLoader:
    dataset = OnlineWarpDataset(**dataset_kwargs)

    sampler = None
    if dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            seed=42,
        )

    return DataLoader(
        dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        sampler=sampler,
        shuffle=(sampler is None),
        pin_memory=True,
        persistent_workers=True,
        # Same note as the offline loader: 'fork' avoids the spawn re-import
        # numpy._core.multiarray hazard, and keeps each worker CPU-only.
        multiprocessing_context=("fork" if num_workers and num_workers > 0 else None),
        collate_fn=_online_collate,
    )
