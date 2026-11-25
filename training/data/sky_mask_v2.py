"""Training-side wrapper for the v2 depth-based sky mask.

The pure algorithm lives in ``dataset_utils/sky_mask_v2.py``; this
module adds two pieces of glue that the trainer needs:

1. :func:`params_from_env` — reads the ``DEPTH_*`` / ``SKY_DEPTH_THRESH``
   environment variables that ``configs/pipeline.env`` exports, with
   safe fall-back to the tuned dataclass defaults when an env var is
   absent or unparsable.

2. :func:`compute_sky_mask_torch` — takes a CPU torch ``float32``
   depth tensor (``H, W``) and returns a torch ``bool`` tensor on the
   requested device. Internally it does ``cpu().numpy()`` →
   :func:`compute_sky_mask` → ``torch.from_numpy``. The algorithm runs
   on CPU because ``cv2.Sobel`` / ``cv2.dilate`` are CPU-only; the
   round-trip is acceptable because the trainer already does the same
   round-trip for ``utils3d.numpy.depth_edge``.

Both are pure side-effect-free; safe to call from any worker / rank.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import numpy as np
import torch

from dataset_utils.sky_mask_v2 import (
    SkyMaskBreakdown,
    SkyMaskParams,
    compute_sky_mask,
)


# Mapping of env var names → ``SkyMaskParams`` field + Python type.
# Kept tabular so the round-trip test in ``params_from_env`` can iterate.
_ENV_FIELDS: Tuple[Tuple[str, str, type], ...] = (
    ("SKY_DEPTH_THRESH",      "sky_thresh",       float),
    ("DEPTH_GRAD_ABS_M",      "grad_abs",         float),
    ("DEPTH_GRAD_REL",        "grad_rel",         float),
    ("DEPTH_HALO_RADIUS",     "halo_radius",      int),
    ("DEPTH_GRAD_REL_GLOBAL", "grad_rel_global",  float),
    ("DEPTH_EDGE_DILATE",     "dilate",           int),
)


def params_from_env(verbose: bool = True) -> SkyMaskParams:
    """Build a :class:`SkyMaskParams` from the ``DEPTH_*`` env vars.

    Missing or unparsable env vars fall back to the tuned dataclass
    defaults — i.e. running with no env at all gives you the production
    config from ``configs/pipeline.env``.

    Parameters
    ----------
    verbose : bool
        Print one line listing the resolved params (helpful for trainer
        boot-time sanity check). Defaults to True.
    """
    p = SkyMaskParams()
    for env_name, field, cast in _ENV_FIELDS:
        raw = os.environ.get(env_name)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            setattr(p, field, cast(raw))
        except (TypeError, ValueError):
            print(
                f"[sky_mask_v2.params_from_env] WARN: ${env_name}={raw!r} "
                f"could not be cast to {cast.__name__}; using default "
                f"{getattr(p, field)!r} for SkyMaskParams.{field}."
            )
    if verbose:
        print(
            f"[sky_mask_v2.params_from_env] {p.as_dict()}"
        )
    return p


def compute_sky_mask_torch(
    depth: torch.Tensor,
    params: SkyMaskParams,
    device: Optional[torch.device] = None,
    return_breakdown: bool = False,
) -> Any:
    """Run the v2 sky mask on a torch depth tensor.

    Parameters
    ----------
    depth : torch.Tensor (H, W) float32 in metres
        Can be on CPU or GPU; will be moved to CPU+numpy for the cv2
        ops, then result is uploaded to ``device`` (or ``depth.device``
        if ``device`` is None).
    params : SkyMaskParams
    device : torch.device or None
        Where to put the returned bool tensor. Defaults to
        ``depth.device``.
    return_breakdown : bool
        If True, return ``(mask_tensor, SkyMaskBreakdown)`` so the
        caller can read per-stage ratios / sub-masks for profiling.
        Otherwise just return the final dilated bool mask tensor.

    Returns
    -------
    torch.Tensor (H, W) bool on ``device``, where ``True`` = excluded
    pixel (sky / invalid / edge halo). Optionally also the full
    SkyMaskBreakdown.
    """
    if device is None:
        device = depth.device
    depth_np = depth.detach().to("cpu", non_blocking=False).numpy().astype(
        np.float32, copy=False
    )
    br = compute_sky_mask(depth_np, params)
    mask_t = torch.from_numpy(br.mask_dilated).to(device)
    if return_breakdown:
        return mask_t, br
    return mask_t


__all__ = [
    "SkyMaskParams",
    "SkyMaskBreakdown",
    "compute_sky_mask",          # re-export
    "compute_sky_mask_torch",
    "params_from_env",
]
