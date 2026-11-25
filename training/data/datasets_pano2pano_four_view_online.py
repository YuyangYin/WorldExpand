"""Four-view online-warp variant of the Pano2Pano dataset (CPU only).

This is the 4-view counterpart of :mod:`training.data.datasets_pano2pano_online`.
One ``__getitem__`` returns one shared ``start`` ingredient set + 4 ``end``
poses + 4 training-resolution ``end`` RGBs. The GPU-side renderer
(:meth:`OnlineWarpRenderer.render_one_multi`) then builds the PCD once and
renders 4 warps — that PCD-reuse is the whole point of the 4-view online path.

Two JSON input formats are supported:

**four-view-v1** (produced by ``four_view_dataset/step4_build_four_view_json.py``)::

    {
      "case_id":    "<scene>__<main>__<comp>__<frame>",
      "directory":  "<scene>/<version>/<scene>/<main_traj>",
      "scene":      "...",   "version": "v3" | "v2" | "legacy",
      "caption":    "...",
      "start":      {"jpg": "...", "exr": "...", "json": "..."},
      "start_yaw":  float,
      "views": [
        { "role": "front1",
          "end": {"jpg": "...", "exr": "...", "json": "..."},
          "metadata": { "end_yaw_deg": float,
                        "end_position_recentered": {"X": f, "Y": f, "Z": f},
                        "distance_cm": float } },
        ...  (back1, front2, back2)
      ]
    }

**Legacy TUPLE-v1 / flat expanded-tuple** (produced by older pipeline stages): the
parent ``OnlineWarpDataset`` expands these into single-view sub-cases carrying
``parent_case_id``; this class re-groups them back into 4-view groups.

Per-sample dict (returned by ``__getitem__``):

    case_id                : str
    label                  : "indoor" | "outdoor" | None
    caption                : str (shared system prompt + per-case caption)
    # ----- SHARED start ingredients (one PCD build feeds all 4 views) -----
    start_rgb_for_pcd      : torch.float32 (H_pcd, W_pcd, 3) in [0, 1]
    start_depth_for_pcd    : torch.float32 (H_pcd, W_pcd)  cleaned EXR distance
    start_sky_mask_for_pcd : Optional[torch.bool] (H_pcd, W_pcd)
    start_rgb_train        : torch.float32 (3, img_size, img_size*2) in [-1, 1]
    cond_img_prompt        : torch.float32 (3, h_cond, w_cond)       in [-1, 1]
    start_position         : dict X/Y/Z (metres, post-recenter -> origin)
    start_yaw              : float (degrees, post-recenter -> 0)
    # ----- per-view (V = num_views) ---------------------------------------
    end_rgb_train          : torch.float32 (V, 3, img_size, img_size*2) in [-1, 1]
    end_positions          : List[dict] X/Y/Z (metres, relative to start)
    end_yaws               : List[float] (degrees, relative; 0 after end_to_start)
    directions             : List[str] (role names: front1/back1/front2/back2)
    num_views              : int

Coordinate / yaw conventions are inherited verbatim from
:class:`OnlineWarpDataset`. yaw alignment is **end_to_start only** (each view's
``end_rgb_train`` is ERP-rolled to the shared start heading, ``end_yaw -> 0``);
``to_displacement`` is rejected at __init__ because it would have to roll the
single shared start (and its PCD) toward 4 different displacement directions at
once — incompatible with the one-PCD-build design.
"""

from __future__ import annotations

import os
import random
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Must be set before cv2 is imported; enables OpenEXR decoding in OpenCV.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from panorama_rendering.nvrender.pipeline_utils import (
    load_exr_depth,
    load_position_rotation_from_json,
)

from training.data.datasets_pano2pano_online import (
    CONDITION_IMAGE_SIZE,
    OnlineWarpDataset,
    _NEIGHBOR_ROLES,
    _circular_pad_width_torch,
    calculate_dimensions,
)
from training.data.yaw_align import align_end_to_start


_DEPTH_AUG_NONE_MODES = {"", "none", "off", "false", "0", "no"}
_DEPTH_AUG_PRESETS: Dict[str, Tuple[float, int, float]] = {
    "mild": (0.125, 11, 0.0),
    "medium": (0.0625, 15, 0.0),
    "down_up_blur": (0.0625, 19, 0.0),
    "strong": (0.03125, 23, 0.0),
}


def _normalise_depth_aug_config(
    mode: Any,
    downscale: Any,
    blur_kernel: Any,
    blur_sigma: Any,
) -> Tuple[str, float, int, float]:
    """Return canonical depth augmentation settings; ``mode='none'`` disables it."""
    mode_s = str(mode or "none").strip().lower()
    if mode_s in _DEPTH_AUG_NONE_MODES:
        return "none", 1.0, 1, 0.0
    if mode_s == "custom":
        scale = float(0.25 if downscale is None else downscale)
        kernel = int(5 if blur_kernel is None else blur_kernel)
        sigma = float(0.0 if blur_sigma is None else blur_sigma)
    elif mode_s in _DEPTH_AUG_PRESETS:
        scale, kernel, sigma = _DEPTH_AUG_PRESETS[mode_s]
    else:
        valid = sorted(_DEPTH_AUG_PRESETS.keys() | {"custom", "none"})
        raise ValueError(f"depth_aug_mode must be one of {valid}, got {mode!r}")

    if not (0.0 < scale <= 1.0):
        raise ValueError(f"depth_aug_downscale must be in (0, 1], got {scale}")
    if kernel < 1:
        raise ValueError(f"depth_aug_blur_kernel must be >= 1, got {kernel}")
    if kernel % 2 == 0:
        kernel += 1
    if sigma < 0.0:
        raise ValueError(f"depth_aug_blur_sigma must be >= 0, got {sigma}")
    return mode_s, scale, kernel, sigma


def _normalise_depth_aug_prob(prob: Any) -> float:
    prob_f = float(1.0 if prob is None else prob)
    if not (0.0 <= prob_f <= 1.0):
        raise ValueError(f"depth_aug_prob must be in [0, 1], got {prob_f}")
    return prob_f


def _augment_depth_down_up_blur(
    depth: np.ndarray,
    downscale: float,
    blur_kernel: int,
    blur_sigma: float,
) -> np.ndarray:
    """Simulate softer depth by down/up resizing, then applying a Gaussian kernel."""
    h, w = depth.shape[:2]
    small_w = max(1, int(round(w * float(downscale))))
    small_h = max(1, int(round(h * float(downscale))))
    out = depth.astype(np.float32, copy=False)
    if (small_h, small_w) != (h, w):
        out = cv2.resize(out, (small_w, small_h), interpolation=cv2.INTER_AREA)
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    if int(blur_kernel) > 1:
        k = int(blur_kernel)
        out = cv2.GaussianBlur(out, (k, k), float(blur_sigma))
    return out.astype(np.float32, copy=False)


def _is_fourview_v1_case(c: Any) -> bool:
    """Detect the four-view-v1 schema from step4_build_four_view_json.py.

    Key distinguishing features:
    * ``views`` is a list of dicts (not top-level role keys like ``front1``)
    * first view has a ``metadata`` dict with ``end_yaw_deg``
    """
    if not isinstance(c, dict):
        return False
    views = c.get("views")
    if not isinstance(views, list) or not views:
        return False
    first = views[0]
    return isinstance(first, dict) and isinstance(first.get("metadata"), dict) and "end_yaw_deg" in first["metadata"]


class FourViewOnlineWarpDataset(OnlineWarpDataset):
    """1 shared start + V directional ends, warps rendered online on the GPU.

    Inherits all parameter validation + per-sample helpers from
    :class:`OnlineWarpDataset`; overrides only (a) how cases are grouped
    (TUPLE-v1 kept as 1 four-view case instead of expanded into 4) and
    (b) ``_load_one`` (loads 1 start + V ends).
    """

    # Four-view path generates ERP panoramas (not cube maps), so override the
    # inherited cube-map SYSTEM_PROMPT with a panorama-specific one. Final
    # prompt is still ``SYSTEM_PROMPT + caption`` (see _load_one).
    SYSTEM_PROMPT = (
        "根据起始 ERP（等距柱状）全景图，为四个新视角生成对应的等距柱状全景图。"
        "严格遵循扭曲条件图像所给出的几何结构与布局，同时合成照片级、高保真的"
        "细节，保持纹理清晰、光照一致，并使整个场景的风格协调统一："
    )

    def __init__(
        self,
        *,
        num_views: int = 4,
        direction_names: Optional[Sequence[str]] = None,
        depth_aug_mode: str = "none",
        depth_aug_downscale: Optional[float] = None,
        depth_aug_blur_kernel: Optional[int] = None,
        depth_aug_blur_sigma: Optional[float] = None,
        depth_aug_prob: float = 1.0,
        warp_cache_dir: Optional[str] = None,
        require_cached_warp: bool = False,
        **kwargs: Any,
    ) -> None:
        # --- 4-view-specific guards (before super so we fail fast) --------
        ya = str(kwargs.get("yaw_align_mode", "off"))
        if ya == "to_displacement":
            raise ValueError(
                "FourViewOnlineWarpDataset does not support "
                "yaw_align_mode='to_displacement': it would roll the single "
                "SHARED start (and its PCD) toward 4 different displacement "
                "directions at once, which is incompatible with the "
                "one-PCD-build design. Use 'end_to_start' (recommended) or "
                "'off'."
            )
        if kwargs.get("cube_layout") not in (None, "", "off", "null", "none"):
            raise ValueError(
                "FourViewOnlineWarpDataset: cube_layout is not supported in "
                "the first 4-view release; set cube_layout=null."
            )

        # super().__init__ loads + TUPLE-expands the JSON into single-view
        # subs (each carrying parent_case_id + role); we regroup them below.
        super().__init__(**kwargs)

        (
            self.depth_aug_mode,
            self.depth_aug_downscale,
            self.depth_aug_blur_kernel,
            self.depth_aug_blur_sigma,
        ) = _normalise_depth_aug_config(
            depth_aug_mode,
            depth_aug_downscale,
            depth_aug_blur_kernel,
            depth_aug_blur_sigma,
        )
        self.depth_aug_prob = _normalise_depth_aug_prob(depth_aug_prob)
        self.num_views = int(num_views)
        self.direction_names: List[str] = (
            list(direction_names) if direction_names else list(_NEIGHBOR_ROLES)
        )
        self._regroup_cases_four_view()
        # mesh_cached path: keep only cases that have a pre-rendered warp on disk
        # (``warp_cache_dir/<case_id>/.done``). Cases whose offline mesh render
        # crashed/were skipped are dropped so training never hits a missing warp.
        self.warp_cache_dir = warp_cache_dir
        # Auto-enable the cache filter when the run uses the mesh_cached engine
        # (render_engine flows in via data_config). This keeps the launcher simple:
        # RENDER_ENGINE=mesh_cached + WARP_CACHE_DIR=... is enough.
        _re = str(kwargs.get("render_engine", "")).lower() if isinstance(kwargs, dict) else ""
        if _re == "mesh_cached":
            require_cached_warp = True
        if require_cached_warp and warp_cache_dir:
            before = len(self.cases)
            kept = []
            for c in self.cases:
                cid = str(c.get("case_id") or c.get("id") or "")
                if cid and os.path.isfile(os.path.join(str(warp_cache_dir), cid, ".done")):
                    kept.append(c)
            self.cases = kept
            print(f"[FourViewOnlineWarpDataset] warp_cache filter: kept {len(kept)}/{before} "
                  f"cases with pre-rendered warp under {warp_cache_dir}")
        if self.depth_aug_mode != "none":
            print(
                "[FourViewOnlineWarpDataset] depth augmentation: "
                f"mode={self.depth_aug_mode}, "
                f"downscale={self.depth_aug_downscale}, "
                f"blur_kernel={self.depth_aug_blur_kernel}, "
                f"blur_sigma={self.depth_aug_blur_sigma}, "
                f"prob={self.depth_aug_prob}"
            )

    # ------------------------------------------------------------------
    # Regroup / load cases into the internal 4-view representation.
    # Supports two input formats detected automatically:
    #   • four-view-v1  (step4_build_four_view_json.py) — already grouped
    #   • legacy TUPLE-v1 / flat expanded-tuple — needs regrouping
    # ------------------------------------------------------------------
    def _regroup_cases_four_view(self) -> None:
        if self.cases and _is_fourview_v1_case(self.cases[0]):
            self._regroup_fourview_v1()
        else:
            self._regroup_cases_four_view_legacy()

    def _regroup_fourview_v1(self) -> None:
        """Load four-view-v1 cases already grouped by step4_build_four_view_json.py.

        Translates each view's ``metadata`` (``end_position_recentered`` /
        ``end_yaw_deg``) into the internal ``_pose_metadata`` dict so
        ``_load_one`` can use the fast path without reading UE5 JSON sidecars.
        """
        grouped: List[Dict[str, Any]] = []
        n_dropped = 0
        for case in self.cases:
            views = case.get("views", [])
            by_role = {v.get("role"): v for v in views}
            ordered = [by_role[r] for r in _NEIGHBOR_ROLES if r in by_role]
            if len(ordered) < self.num_views:
                n_dropped += 1
                continue
            ordered = ordered[: self.num_views]

            translated_views: List[Dict[str, Any]] = []
            for v in ordered:
                meta = v.get("metadata") or {}
                epr = meta.get("end_position_recentered") or {}
                translated_views.append(
                    {
                        "end": v["end"],
                        "role": v.get("role"),
                        # Translate to the internal _pose_metadata schema that
                        # _load_one expects (start is always at origin after recenter).
                        "_pose_metadata": {
                            "end_position": {
                                "X": float(epr.get("X", 0.0)),
                                "Y": float(epr.get("Y", 0.0)),
                                "Z": float(epr.get("Z", 0.0)),
                            },
                            "end_yaw": float(meta.get("end_yaw_deg", 0.0)),
                        },
                    }
                )

            grouped.append(
                {
                    "id": case.get("case_id", ""),
                    "directory": case.get("directory", ""),
                    "start": case["start"],
                    "label": case.get("label"),
                    "caption": case.get("caption", ""),
                    "views": translated_views,
                }
            )

        if len(grouped) == 0:
            raise ValueError(
                "FourViewOnlineWarpDataset: no four-view-v1 cases loaded "
                f"(num_views={self.num_views}). Check the step4 JSON."
            )
        self.cases = grouped
        print(
            f"[FourViewOnlineWarpDataset] loaded {len(self.cases)} four-view-v1 "
            f"cases (num_views={self.num_views}, "
            f"dropped {n_dropped} incomplete, "
            f"yaw_align_mode={self.yaw_align_mode}, "
            f"directions={self.direction_names})"
        )

    def _regroup_cases_four_view_legacy(self) -> None:
        """Regroup parent-expanded single-view sub-cases back into 4-view groups.

        Used for legacy TUPLE-v1 / flat expanded-tuple input where the parent
        ``__init__`` has already expanded each tuple into sub-cases carrying
        ``parent_case_id`` and optionally ``_pose_metadata``.
        """
        groups: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
        for sub in self.cases:
            pid = sub.get("parent_case_id")
            if pid is None:
                raise ValueError(
                    "FourViewOnlineWarpDataset: case has neither a 'views' list "
                    "(four-view-v1) nor a 'parent_case_id' (legacy expanded-tuple). "
                    "Check the JSON format passed to the dataset."
                )
            groups.setdefault(str(pid), []).append(sub)

        grouped: List[Dict[str, Any]] = []
        n_dropped = 0
        for pid, subs in groups.items():
            by_role = {s.get("role"): s for s in subs}
            ordered = [by_role[r] for r in _NEIGHBOR_ROLES if r in by_role]
            if len(ordered) < self.num_views:
                n_dropped += 1
                continue
            ordered = ordered[: self.num_views]
            first = ordered[0]
            grouped.append(
                {
                    "id": pid,
                    "directory": first.get("directory", ""),
                    "start": first["start"],
                    "label": first.get("label"),
                    "caption": first.get("caption", ""),
                    "views": [
                        {
                            "end": s["end"],
                            "role": s.get("role"),
                            "_pose_metadata": s.get("_pose_metadata"),
                        }
                        for s in ordered
                    ],
                }
            )

        if len(grouped) == 0:
            raise ValueError(
                "FourViewOnlineWarpDataset: no complete 4-view groups after "
                f"regrouping (num_views={self.num_views}). Check that the "
                "TUPLE JSON has all of front1/back1/front2/back2 per tuple."
            )
        self.cases = grouped
        print(
            f"[FourViewOnlineWarpDataset] regrouped into {len(self.cases)} "
            f"four-view cases (num_views={self.num_views}, "
            f"dropped {n_dropped} incomplete groups, "
            f"yaw_align_mode={self.yaw_align_mode}, "
            f"directions={self.direction_names})"
        )

    # ------------------------------------------------------------------
    # _load_one: 1 shared start + V ends.  Reuses parent helpers
    # (_resolve / _clean_depth / _recenter / _load_cached_sky /
    #  _load_resized_rgb) and the parent's __getitem__ retry wrapper.
    # ------------------------------------------------------------------
    def _apply_depth_augmentation(self, depth: np.ndarray, *, force: bool = False) -> np.ndarray:
        if self.depth_aug_mode == "none":
            return depth
        if not force and random.random() >= self.depth_aug_prob:
            return depth
        return _augment_depth_down_up_blur(
            depth,
            self.depth_aug_downscale,
            self.depth_aug_blur_kernel,
            self.depth_aug_blur_sigma,
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
        for p in (start_jpg, start_exr, start_json):
            if not os.path.exists(p):
                raise FileNotFoundError(f"missing start input: {p}")

        # --- SHARED start geometry inputs (PCD-build resolution) ----------
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
        start_depth_for_pcd = self._apply_depth_augmentation(start_depth_clean)

        start_rgb_pcd_bgr = cv2.resize(
            start_rgb_full_bgr, (self.pcd_build_w, self.pcd_build_h),
            interpolation=cv2.INTER_AREA,
        )
        start_rgb_pcd = cv2.cvtColor(start_rgb_pcd_bgr, cv2.COLOR_BGR2RGB)
        start_depth_pcd = cv2.resize(
            start_depth_for_pcd, (self.pcd_build_w, self.pcd_build_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # --- SHARED training-resolution start + cond_img_prompt -----------
        train_h = self.img_size
        train_w = self.img_size * 2
        start_rgb_train_np = cv2.cvtColor(
            cv2.resize(start_rgb_full_bgr, (train_w, train_h), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        cond_w, cond_h = calculate_dimensions(CONDITION_IMAGE_SIZE, train_w / train_h)
        cond_img_prompt_np = cv2.cvtColor(
            cv2.resize(start_rgb_full_bgr, (cond_w, cond_h), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )

        sky_cached = self._load_cached_sky(case["start"]["jpg"])

        # Lazy start pose: only load when a view lacks _pose_metadata.
        _start_pose_loaded = False
        _start_position: Optional[Dict[str, float]] = None
        _start_rotation: Optional[Dict[str, float]] = None

        def _get_start_pose():
            nonlocal _start_pose_loaded, _start_position, _start_rotation
            if not _start_pose_loaded:
                _start_position, _start_rotation = load_position_rotation_from_json(
                    start_json
                )
                _start_pose_loaded = True
            return _start_position, _start_rotation

        # --- Prompt (shared) ----------------------------------------------
        caption = case.get("caption", "") or ""
        if random.random() < self.drop_text_prob:
            caption = ""
        prompt = self.SYSTEM_PROMPT + caption

        # --- Per-view ends ------------------------------------------------
        end_rgb_list: List[torch.Tensor] = []
        end_positions: List[Dict[str, float]] = []
        end_yaws: List[float] = []
        directions: List[str] = []
        start_pos_out: Optional[Dict[str, float]] = None
        start_yaw_out: float = 0.0

        for v in case["views"]:
            end_jpg = self._resolve(v["end"]["jpg"])
            if not os.path.exists(end_jpg):
                raise FileNotFoundError(f"missing end jpg: {end_jpg}")

            end_rgb_train_np = self._load_resized_rgb(end_jpg, train_h, train_w)

            # --- pose: fast-path from pre-computed metadata, or live read ---
            pm = v.get("_pose_metadata")
            if pm is not None:
                sp_out: Dict[str, float] = {"X": 0.0, "Y": 0.0, "Z": 0.0}
                sy_out: float = 0.0
                ep_out: Dict[str, float] = dict(pm["end_position"])
                ey_out: float = float(pm["end_yaw"])
            else:
                end_jpg_rel = v["end"]["jpg"]
                end_json = self._resolve(
                    v["end"].get("json") or end_jpg_rel.replace(".jpg", ".json")
                )
                if not os.path.exists(end_json):
                    raise FileNotFoundError(f"missing end json: {end_json}")
                sp, sr = _get_start_pose()
                end_position, end_rotation = load_position_rotation_from_json(end_json)
                sp_out, sy_out, ep_out, ey_out = self._recenter(
                    sp, sr, end_position, end_rotation
                )

            # start pose is identical for every view; capture once.
            if start_pos_out is None:
                start_pos_out, start_yaw_out = sp_out, sy_out

            end_rgb_t = torch.from_numpy(
                np.ascontiguousarray(end_rgb_train_np.astype(np.float32) / 127.5 - 1.0)
            ).permute(2, 0, 1).contiguous()

            # --- yaw alignment (end_to_start, per view) -------------------
            # Roll THIS view's end_rgb_train to the shared start heading and
            # zero its relative yaw. start/PCD untouched (rendered once).
            if self.yaw_align_mode == "end_to_start":
                mini = align_end_to_start(
                    {"start_yaw": sy_out, "end_yaw": ey_out, "end_rgb_train": end_rgb_t}
                )
                end_rgb_t = mini["end_rgb_train"]
                ey_out = float(mini["end_yaw"])  # == sy_out (0 after recenter)

            # --- pixel-space circular pad (VAE-bound end tensor) ----------
            if self.rgb_circular_pad:
                end_rgb_t = _circular_pad_width_torch(end_rgb_t)

            end_rgb_list.append(end_rgb_t)
            end_positions.append(ep_out)
            end_yaws.append(float(ey_out))
            directions.append(v.get("role") or f"v{len(directions)}")

        # --- SHARED start tensors -----------------------------------------
        start_rgb_pcd_t = torch.from_numpy(
            np.ascontiguousarray(start_rgb_pcd.astype(np.float32) / 255.0)
        )
        start_depth_pcd_t = torch.from_numpy(
            np.ascontiguousarray(start_depth_pcd.astype(np.float32))
        )
        start_rgb_train_t = torch.from_numpy(
            np.ascontiguousarray(start_rgb_train_np.astype(np.float32) / 127.5 - 1.0)
        ).permute(2, 0, 1).contiguous()
        cond_img_prompt_t = torch.from_numpy(
            np.ascontiguousarray(cond_img_prompt_np.astype(np.float32) / 127.5 - 1.0)
        ).permute(2, 0, 1).contiguous()
        if self.rgb_circular_pad:
            start_rgb_train_t = _circular_pad_width_torch(start_rgb_train_t)

        end_rgb_views = torch.stack(end_rgb_list, dim=0)  # (V, 3, H, W)

        return {
            "case_id": case_id,
            "label": case.get("label"),
            "caption": prompt,
            "start_rgb_for_pcd": start_rgb_pcd_t,
            "start_depth_for_pcd": start_depth_pcd_t,
            "start_sky_mask_for_pcd": sky_cached,
            "start_rgb_train": start_rgb_train_t,
            "cond_img_prompt": cond_img_prompt_t,
            "start_position": start_pos_out if start_pos_out is not None else {"X": 0.0, "Y": 0.0, "Z": 0.0},
            "start_yaw": float(start_yaw_out),
            "end_rgb_train": end_rgb_views,
            "end_positions": end_positions,
            "end_yaws": end_yaws,
            "directions": directions,
            "num_views": self.num_views,
        }


# ---------------------------------------------------------------------------
# Collate: render is per-case (list of dicts kept for the renderer); stack the
# train-resolution tensors the transformer consumes. end_rgb_train is
# (V, 3, H, W) per sample -> (B, V, 3, H, W).
# ---------------------------------------------------------------------------
def _four_view_online_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "samples": batch,
        "captions": [b["caption"] for b in batch],
        "start_rgb_train": torch.stack([b["start_rgb_train"] for b in batch], dim=0),
        "end_rgb_train": torch.stack([b["end_rgb_train"] for b in batch], dim=0),
        "cond_img_prompt": torch.stack([b["cond_img_prompt"] for b in batch], dim=0),
    }


# ---------------------------------------------------------------------------
# Loader factory. Signature mirrors datasets_pano2pano_online.loader so the
# trainer's ``loader(**args.data_config)`` call works unchanged.
# ---------------------------------------------------------------------------
def loader(
    train_batch_size: int,
    num_workers: int,
    **dataset_kwargs: Any,
) -> DataLoader:
    dataset = FourViewOnlineWarpDataset(**dataset_kwargs)

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
        persistent_workers=bool(num_workers and num_workers > 0),
        multiprocessing_context=("fork" if num_workers and num_workers > 0 else None),
        collate_fn=_four_view_online_collate,
    )


# ---------------------------------------------------------------------------
# Debug / visualisation entry point
#   python -m training.data.datasets_pano2pano_four_view_online \
#       --json  /path/to/step4_total.json \
#       --data_root /path/to/data \
#       --output_dir ./debug_vis
# ---------------------------------------------------------------------------
def _to_uint8(t: "torch.Tensor") -> "np.ndarray":
    """Convert a (3,H,W) or (H,W,3) tensor in [-1,1] or [0,1] to (H,W,3) uint8."""
    import numpy as _np
    arr = t.detach().cpu().float()
    if arr.dim() == 3 and arr.shape[0] == 3:
        arr = arr.permute(1, 2, 0)
    if arr.min() < -0.1:          # [-1, 1] → [0, 1]
        arr = (arr + 1.0) / 2.0
    return (arr.clamp(0.0, 1.0) * 255).byte().numpy()


def _add_label(img_bgr: "np.ndarray", text: str) -> "np.ndarray":
    """Burn a text label onto a BGR image (in-place clone)."""
    out = img_bgr.copy()
    cv2.putText(out, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def _depth_to_bgr(
    depth: "np.ndarray",
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    sky_thresh: Optional[float] = 60000.0,
) -> "np.ndarray":
    """Visualise a depth map with robust shared normalisation."""
    valid = (depth > 0) & np.isfinite(depth)
    if sky_thresh is not None and sky_thresh > 0:
        valid &= depth < float(sky_thresh)
    if not np.any(valid):
        return np.zeros((*depth.shape[:2], 3), dtype=np.uint8)
    if vmin is None:
        vmin = float(np.quantile(depth[valid], 0.01))
    if vmax is None:
        vmax = float(np.quantile(depth[valid], 0.99))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = np.clip((depth.astype(np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)
    out = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out[~valid] = 0
    return out


def _load_depth_aug_debug_pair(
    dataset: FourViewOnlineWarpDataset,
    idx: int,
) -> Tuple["np.ndarray", "np.ndarray"]:
    """Load the same start EXR path as training and return PCD-res before/after depth."""
    case = dataset.cases[idx]
    start_jpg_rel = case["start"]["jpg"]
    start_jpg = dataset._resolve(start_jpg_rel)
    start_exr = dataset._resolve(
        case["start"].get("exr") or start_jpg_rel.replace(".jpg", ".exr")
    )
    start_rgb_full_bgr = cv2.imread(start_jpg, cv2.IMREAD_COLOR)
    if start_rgb_full_bgr is None:
        raise FileNotFoundError(f"cv2.imread failed: {start_jpg}")
    h_full, w_full = start_rgb_full_bgr.shape[:2]
    depth_full = load_exr_depth(start_exr)
    if depth_full.shape[:2] != (h_full, w_full):
        depth_full = cv2.resize(
            depth_full, (w_full, h_full), interpolation=cv2.INTER_LINEAR
        )
    clean = dataset._clean_depth(depth_full)
    augmented = dataset._apply_depth_augmentation(clean, force=True)
    clean_pcd = cv2.resize(
        clean, (dataset.pcd_build_w, dataset.pcd_build_h),
        interpolation=cv2.INTER_LINEAR,
    )
    augmented_pcd = cv2.resize(
        augmented, (dataset.pcd_build_w, dataset.pcd_build_h),
        interpolation=cv2.INTER_LINEAR,
    )
    return clean_pcd.astype(np.float32, copy=False), augmented_pcd.astype(np.float32, copy=False)


if __name__ == "__main__":
    import argparse
    import sys
    import time as _time

    parser = argparse.ArgumentParser(
        description="Debug visualiser for FourViewOnlineWarpDataset. "
                    "Saves a 4×2 (no warp) or 4×3 (with warp) grid per sample."
    )
    parser.add_argument("--json",         required=True,
                        help="Path to step4_total.json (four-view-v1 format).")
    parser.add_argument("--data_root",    required=True,
                        help="Root directory for resolving relative paths.")
    parser.add_argument("--output_dir",   default="./debug_vis",
                        help="Directory where output images are saved.")
    parser.add_argument("--sky_cache_dir", default=None,
                        help="Optional sky-mask cache dir (sky_mask_mode=cached).")
    parser.add_argument("--sample_idx",   type=int, default=0,
                        help="Start index (default: 0).")
    parser.add_argument("--num_samples",  type=int, default=1,
                        help="How many consecutive samples to visualise (default: 1).")
    parser.add_argument("--num_views",     type=int, default=4,
                        help="Views rendered per case (1 = only the first "
                             "neighbour view; 4 = full four-view).")
    parser.add_argument("--device",       default="cuda:0",
                        help="Torch device for warp rendering, e.g. 'cuda:0'. "
                             "Pass '' to skip warp.")
    parser.add_argument("--img_size",     type=int, default=512)
    parser.add_argument("--pcd_h",        type=int, default=512)
    parser.add_argument("--pcd_w",        type=int, default=1024)
    parser.add_argument("--render_h",     type=int, default=512)
    parser.add_argument("--render_w",     type=int, default=1024)
    parser.add_argument("--render_engine", default="gsplat",
                        choices=["gsplat", "mesh", "pytorch3d"],
                        help="Geometry representation for warp render: "
                             "gsplat (isotropic-GS from PCD), mesh (nvdiffrast "
                             "mesh from RGBD), pytorch3d (raw point cloud). "
                             "Output files are suffixed with the engine name so "
                             "you can render the same case with each and compare.")
    parser.add_argument("--max_pcd_points", type=int, default=16000000,
                        help="Max PCD points (gsplat/pytorch3d) / mesh vertices cap.")
    parser.add_argument("--gs_scale_factor", type=float, default=1.0)
    parser.add_argument("--time",         action="store_true",
                        help="Profile and print per-stage timing statistics.")
    parser.add_argument("--yaw_align",    default="end_to_start",
                        choices=["end_to_start", "off"],
                        help="Yaw alignment mode (default: end_to_start). "
                             "end_to_start: roll each end ERP to the shared start heading. "
                             "off: no yaw adjustment.")
    parser.add_argument("--depth_aug_mode", default="none",
                        choices=["none", "mild", "medium", "down_up_blur", "strong", "custom"],
                        help="Depth augmentation mode for start EXR before PCD build.")
    parser.add_argument("--depth_aug_downscale", type=float, default=None,
                        help="Custom depth downscale ratio, used when depth_aug_mode=custom.")
    parser.add_argument("--depth_aug_blur_kernel", type=int, default=None,
                        help="Custom odd Gaussian blur kernel, used when depth_aug_mode=custom.")
    parser.add_argument("--depth_aug_blur_sigma", type=float, default=None,
                        help="Custom Gaussian sigma, used when depth_aug_mode=custom.")
    parser.add_argument("--depth_vis_sky_thresh", type=float, default=60000.0,
                        help="Depth values >= this are treated as sky/far and drawn black.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Lightweight wall-clock timer (only used when --time is set)         #
    # ------------------------------------------------------------------ #
    class _StageTimer:
        """Accumulate per-stage durations; print a summary table at the end."""
        def __init__(self, enabled: bool):
            self.enabled = enabled
            self._totals: Dict[str, float] = {}
            self._counts: Dict[str, int]   = {}
            self._t0: float = 0.0
            self._stage: str = ""

        def start(self, stage: str) -> None:
            if not self.enabled:
                return
            self._stage = stage
            self._t0 = _time.perf_counter()

        def stop(self) -> float:
            if not self.enabled:
                return 0.0
            elapsed = _time.perf_counter() - self._t0
            s = self._stage
            self._totals[s] = self._totals.get(s, 0.0) + elapsed
            self._counts[s] = self._counts.get(s, 0) + 1
            return elapsed

        def summary(self) -> None:
            if not self.enabled or not self._totals:
                return
            n_samp = max(self._counts.values())
            print(f"\n{'─'*62}")
            print(f"  Timing summary  ({n_samp} sample(s))")
            print(f"{'─'*62}")
            print(f"  {'Stage':<26} {'Total(s)':>9} {'Mean(s)':>9} {'Count':>6}")
            print(f"{'─'*62}")
            for s, tot in self._totals.items():
                cnt = self._counts[s]
                print(f"  {s:<26} {tot:9.3f} {tot/cnt:9.3f} {cnt:6d}")
            print(f"{'─'*62}\n")

    timer = _StageTimer(args.time)

    # ------------------------------------------------------------------ #
    # 1. Build dataset                                                     #
    # ------------------------------------------------------------------ #
    timer.start("dataset_init")
    sky_mode = "cached" if args.sky_cache_dir else "skip"
    dataset = FourViewOnlineWarpDataset(
        train_files=[args.json],
        data_root=args.data_root,
        num_views=args.num_views,
        img_size=args.img_size,
        pcd_build_h=args.pcd_h,
        pcd_build_w=args.pcd_w,
        sky_mask_mode=sky_mode,
        sky_cache_dir=args.sky_cache_dir,
        yaw_align_mode=args.yaw_align,
        depth_aug_mode=args.depth_aug_mode,
        depth_aug_downscale=args.depth_aug_downscale,
        depth_aug_blur_kernel=args.depth_aug_blur_kernel,
        depth_aug_blur_sigma=args.depth_aug_blur_sigma,
        depth_aug_prob=1.0,
        drop_text_prob=0.0,
    )
    t_init = timer.stop()
    print(f"[main] Dataset size: {len(dataset)}"
          + (f"  (init: {t_init:.2f}s)" if args.time else ""))

    # ------------------------------------------------------------------ #
    # 2. Initialise warp renderer once (reused across all samples)        #
    # ------------------------------------------------------------------ #
    renderer = None
    device_str = args.device if args.device else None
    if device_str is None and torch.cuda.is_available():
        device_str = "cuda"

    if device_str:
        try:
            from training.data.online_warp_renderer import OnlineWarpRenderer
            _dev = torch.device(device_str)
            timer.start("renderer_init")
            renderer = OnlineWarpRenderer(
                device=_dev,
                sky_mask_mode="skip",
                render_h=args.render_h,
                render_w=args.render_w,
                render_engine=args.render_engine,
                max_pcd_points=int(args.max_pcd_points),
                gs_scale_factor=float(args.gs_scale_factor),
            )
            t_ri = timer.stop()
            print(f"[main] Renderer ready on {device_str} (engine={args.render_engine})"
                  + (f"  (init: {t_ri:.2f}s)" if args.time else ""))
        except Exception as exc:
            print(f"[main] Renderer unavailable ({type(exc).__name__}: {exc}); "
                  "warp skipped.")

    # ------------------------------------------------------------------ #
    # 3. Loop over requested samples                                       #
    # ------------------------------------------------------------------ #
    n_total = min(args.num_samples, len(dataset))
    for si in range(n_total):
        idx = (args.sample_idx + si) % len(dataset)
        print(f"\n{'='*62}")
        print(f"  Sample {si+1}/{n_total}  (dataset idx={idx})")
        print(f"{'='*62}")

        # ---- 3a. CPU data load ----------------------------------------
        timer.start("cpu_load")
        sample = dataset[idx]
        t_load = timer.stop()
        if args.time:
            print(f"  [time] cpu_load: {t_load:.3f}s")

        # Print shapes only for the first sample.
        if si == 0:
            print(f"\n  Tensor shapes:")
            for key, val in sample.items():
                if isinstance(val, torch.Tensor):
                    print(f"    {key:30s}: {tuple(val.shape)}  dtype={val.dtype}")
                elif isinstance(val, list) and val and isinstance(val[0], (int, float, dict)):
                    print(f"    {key:30s}: {val}")
                else:
                    print(f"    {key:30s}: {val!r}")

        before_depth: Optional[np.ndarray] = None
        after_depth: Optional[np.ndarray] = None
        if dataset.depth_aug_mode != "none":
            before_depth, after_depth = _load_depth_aug_debug_pair(dataset, idx)

        # ---- 3b. GPU warp rendering (optional) ------------------------
        warps_np: Optional[List[np.ndarray]] = None
        warps_before_np: Optional[List[np.ndarray]] = None
        masks_np: Optional[List[np.ndarray]] = None
        if renderer is not None:
            try:
                timer.start("warp_render")
                if before_depth is not None:
                    sample_before = dict(sample)
                    sample_before["start_depth_for_pcd"] = torch.from_numpy(
                        np.ascontiguousarray(before_depth)
                    )
                    sample_before_gpu = {
                        k: (v.to(_dev) if isinstance(v, torch.Tensor) else v)
                        for k, v in sample_before.items()
                    }
                    warps_before_t, _ = renderer.render_one_multi(sample_before_gpu)
                    warps_before_np = [
                        cv2.resize(
                            (w.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8),
                            (args.img_size * 2, args.img_size),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        for w in warps_before_t
                    ]
                sample_gpu = {
                    k: (v.to(_dev) if isinstance(v, torch.Tensor) else v)
                    for k, v in sample.items()
                }
                warps_t, masks_t = renderer.render_one_multi(sample_gpu)
                t_render = timer.stop()
                if args.time:
                    print(f"  [time] warp_render: {t_render:.3f}s")
                H_r, W_r = warps_t[0].shape[:2]
                warps_np = [
                    cv2.resize(
                        (w.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8),
                        (args.img_size * 2, args.img_size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    for w in warps_t
                ]
                # Mask: bool (H, W) -> uint8 {0,255}, NEAREST so it stays binary.
                masks_np = [
                    cv2.resize(
                        (m.cpu().numpy().astype(np.uint8) * 255),
                        (args.img_size * 2, args.img_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    for m in masks_t
                ]
                print(f"  Warp OK: {len(warps_np)} views, raw ({H_r}×{W_r})")
            except Exception as exc:
                timer.stop()
                print(f"  Warp failed ({type(exc).__name__}: {exc})")

        # ---- 3c. Build 2x3 depth/warp diff grids + save ----------------
        timer.start("save_images")

        V          = sample["num_views"]
        directions = sample["directions"]
        end_yaws   = sample.get("end_yaws", [])

        # Sanitize case_id for use in filenames
        raw_case_id = sample.get("case_id", "") or ""
        case_id_safe = re.sub(r"[^A-Za-z0-9_\-]", "_", raw_case_id)[:64]
        file_prefix  = f"{case_id_safe}" if case_id_safe else f"sample_{si:04d}"
        # Suffix with the geometry engine so mesh/gsplat/pytorch3d runs of the
        # same case land in distinct files and can be compared side by side.
        file_prefix  = f"{file_prefix}__{args.render_engine}"

        compare_paths: List[str] = []
        if before_depth is not None and after_depth is not None:
            sky_thresh = float(args.depth_vis_sky_thresh)
            valid = (before_depth > 0) & np.isfinite(before_depth)
            if sky_thresh > 0:
                valid &= before_depth < sky_thresh
            vmin = float(np.quantile(before_depth[valid], 0.01)) if np.any(valid) else 0.0
            vmax = float(np.quantile(before_depth[valid], 0.99)) if np.any(valid) else 1.0
            before_bgr = _depth_to_bgr(
                before_depth, vmin=vmin, vmax=vmax, sky_thresh=sky_thresh
            )
            after_bgr = _depth_to_bgr(
                after_depth, vmin=vmin, vmax=vmax, sky_thresh=sky_thresh
            )
            depth_size = (args.img_size * 2, args.img_size)
            before_bgr = cv2.resize(before_bgr, depth_size, interpolation=cv2.INTER_LINEAR)
            after_bgr = cv2.resize(after_bgr, depth_size, interpolation=cv2.INTER_LINEAR)
            depth_diff = np.abs(after_depth - before_depth)
            depth_diff_bgr = _depth_to_bgr(
                cv2.resize(depth_diff, depth_size, interpolation=cv2.INTER_LINEAR),
                sky_thresh=None,
            )

        if (
            before_depth is not None
            and after_depth is not None
            and warps_before_np is not None
            and warps_np is not None
        ):
            for i in range(V):
                role    = directions[i] if i < len(directions) else f"v{i}"
                yaw_str = f"{end_yaws[i]:.1f}deg" if i < len(end_yaws) else "?"
                warp_before_bgr = cv2.cvtColor(warps_before_np[i], cv2.COLOR_RGB2BGR)
                warp_bgr = cv2.cvtColor(warps_np[i], cv2.COLOR_RGB2BGR)
                warp_diff_bgr = cv2.absdiff(warp_bgr, warp_before_bgr)
                compare_grid = np.concatenate(
                    [
                        np.concatenate(
                            [
                                _add_label(before_bgr, "depth before (sky black)"),
                                _add_label(after_bgr, "depth after aug"),
                                _add_label(depth_diff_bgr, "depth abs diff"),
                            ],
                            axis=1,
                        ),
                        np.concatenate(
                            [
                                _add_label(warp_before_bgr, f"warp before ({role})"),
                                _add_label(warp_bgr, f"warp after ({role})"),
                                _add_label(warp_diff_bgr, f"warp abs diff ({yaw_str})"),
                            ],
                            axis=1,
                        ),
                    ],
                    axis=0,
                )
                compare_path = os.path.join(
                    args.output_dir, f"{file_prefix}_{role}_depth_warp_aug_2x3.jpg"
                )
                cv2.imwrite(compare_path, compare_grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
                compare_paths.append(compare_path)
        else:
            print("  Aug 2x3 compare skipped: need depth_aug_mode != none and renderer enabled.")

        # Always dump the plain warp (+mask) per view so the geometry engines
        # (gsplat / mesh / pytorch3d) can be rendered on the same case and
        # compared side by side, regardless of depth-aug.
        warp_paths: List[str] = []
        if warps_np is not None:
            start_rgb_vis = None
            try:
                # start ERP for reference (train tensor is (3,H,W) in [-1,1])
                s = sample.get("start_rgb_train")
                if isinstance(s, torch.Tensor):
                    s_np = ((s.permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                    start_rgb_vis = cv2.cvtColor(s_np, cv2.COLOR_RGB2BGR)
            except Exception:
                start_rgb_vis = None
            row_imgs = []
            for i in range(V):
                role    = directions[i] if i < len(directions) else f"v{i}"
                yaw_str = f"{end_yaws[i]:.1f}deg" if i < len(end_yaws) else "?"
                warp_bgr = cv2.cvtColor(warps_np[i], cv2.COLOR_RGB2BGR)
                # per-view warp png
                wp = os.path.join(args.output_dir, f"{file_prefix}_{role}_warp.png")
                cv2.imwrite(wp, warp_bgr)
                warp_paths.append(wp)
                # per-view mask png
                if masks_np is not None:
                    mp = os.path.join(args.output_dir, f"{file_prefix}_{role}_mask.png")
                    cv2.imwrite(mp, masks_np[i])
                row_imgs.append(_add_label(warp_bgr, f"{args.render_engine} {role} {yaw_str}"))
            # one combined strip of all V warps for a quick look
            if row_imgs:
                strip = np.concatenate(row_imgs, axis=1)
                if start_rgb_vis is not None:
                    strip = np.concatenate(
                        [_add_label(cv2.resize(start_rgb_vis, (row_imgs[0].shape[1], row_imgs[0].shape[0])),
                                    "START"), strip], axis=1)
                strip_path = os.path.join(args.output_dir, f"{file_prefix}_warp_strip.jpg")
                cv2.imwrite(strip_path, strip, [cv2.IMWRITE_JPEG_QUALITY, 95])
                warp_paths.append(strip_path)
            print(f"  Warp images → {args.output_dir}/{file_prefix}_*_warp.png (+strip)")

        t_save = timer.stop()
        if args.time:
            print(f"  [time] save_images: {t_save:.3f}s")
        if compare_paths:
            print(f"  Aug 2x3 compare  → {compare_paths[0]} (+{len(compare_paths)-1} more)")

    # ------------------------------------------------------------------ #
    # 4. Timing summary                                                    #
    # ------------------------------------------------------------------ #
    timer.summary()
    sys.exit(0)
