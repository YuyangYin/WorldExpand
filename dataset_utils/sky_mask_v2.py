"""Depth-based sky-mask algorithm used by the online warp renderer.

The NumPy and OpenCV implementation is shared with the PyTorch wrapper in
``training/data/sky_mask_v2.py``. Its defaults come from the May 2026 outdoor
frame sweep on the 0519 dataset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Any, Dict

# OpenCV's prebuilt wheels ship with OpenEXR support disabled by default.
# Other modules that import this one may load EXR via cv2 — set the env
# var here too so they don't have to remember.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402  -- after env-var
import numpy as np  # noqa: E402


# =============================================================================
# Public dataclasses
# =============================================================================


@dataclass
class SkyMaskParams:
    """All knobs the inline sky mask exposes. Defaults mirror ``pipeline.env``."""

    sky_thresh: float = 7000.0      # metres; > this is sky / out-of-render-range
                                    # decoration. UE fp16 max ≈ 65504 also
                                    # caught by the same rule.
    grad_abs: float = 10.0          # m/px; halo-gated; only sharpest sky-
                                    # boundary jumps fire.
    grad_rel: float = 0.20          # unitless; halo-gated relative grad.
    dilate: int = 2                 # px; final mask dilation kernel radius.
    halo_radius: int = 2            # px; how far around sky/invalid the
                                    # gradient masks may fire.
    grad_global: bool = False       # True ⇒ kill halo gate for both grad
                                    # masks (legacy A/B flag).
    grad_rel_global: float = 0.0    # > 0 ⇒ extra global grad_rel pass at
                                    # this stricter threshold (AA-halo
                                    # catcher for synthetic UE data with
                                    # antialiasing). 0 disables.

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SkyMaskBreakdown:
    """Per-stage masks (bool, H×W) plus final dilated mask, kept around for viz."""

    mask_sky: np.ndarray
    mask_invalid: np.ndarray
    mask_grad_abs: np.ndarray
    mask_grad_rel: np.ndarray
    mask_grad_rel_global: np.ndarray   # all-zero if grad_rel_global <= 0
    mask_combined: np.ndarray
    mask_dilated: np.ndarray            # final mask actually fed downstream

    def ratios(self) -> Dict[str, float]:
        n = self.mask_sky.size
        return {
            "sky":             float(self.mask_sky.sum()) / n,
            "invalid":         float(self.mask_invalid.sum()) / n,
            "grad_abs":        float(self.mask_grad_abs.sum()) / n,
            "grad_rel":        float(self.mask_grad_rel.sum()) / n,
            "grad_rel_global": float(self.mask_grad_rel_global.sum()) / n,
            "combined":        float(self.mask_combined.sum()) / n,
            "dilated_final":   float(self.mask_dilated.sum()) / n,
        }


# =============================================================================
# Algorithm
# =============================================================================


def compute_sky_mask(
    depth: np.ndarray, params: SkyMaskParams,
) -> SkyMaskBreakdown:
    """Compute the depth-only sky mask.

    Algorithm
    ---------
    1. ``mask_sky``     = depth > sky_thresh                 (sky / out-of-range
                                                              decorations; this
                                                              also catches UE
                                                              fp16 far-plane
                                                              fill ≈ 65504 m)
    2. ``mask_invalid`` = ~isfinite(depth) | (depth <= 0)    (rendering holes)
    3. Halo gate: dilate ``mask_sky | mask_invalid`` by ``halo_radius`` to get
       the "object → sky" boundary region. (Skipped if ``grad_global=True``.)
    4. ``mask_grad_abs`` = (|∇d| > grad_abs)              within the halo
    5. ``mask_grad_rel`` = (|∇d|/d > grad_rel)            within the halo
    6. ``mask_grad_rel_global`` = (|∇d|/d > grad_rel_global) globally, all px.
       Only active when ``grad_rel_global > 0``. Defensive pass intended to
       catch sub-pixel anti-aliasing halos at non-sky object silhouettes
       (TAA/MSAA leaks). Uses a stricter threshold so it ignores gentle
       real geometry changes (mountain slopes etc.).
    7. ``mask_combined`` = union of 1, 2, 4, 5, 6
    8. ``mask_dilated``  = dilate(combined, dilate)       (final, fed to PCD)

    Why the halo gate matters
    -------------------------
    Without it, ``mask_grad_abs`` fires whenever any two adjacent pixels have
    > grad_abs metre depth difference — that catches real near→far jumps
    BETWEEN OBJECTS (e.g. a 5 m statue in front of a 50 m wall) and would
    delete genuine geometry. With the halo gate, gradient masks only fire
    on pixels adjacent to actual sky/invalid regions — i.e. antialias halos
    along the silhouette of foreground against sky — which is what we want.

    Why grad_rel_global is useful
    -----------------------------
    With halo gating + tight ``halo_radius`` (e.g. 2 px), antialias halos at
    object-to-object silhouettes far from sky are NOT masked. Synthetic data
    rendered with antialiasing leaves sub-pixel "in-between depth" rings
    around every silhouette; left in, these become floating thin point sheets
    in the warped PCD. ``grad_rel_global`` is a scale-invariant detector
    that catches them: AA halo pixels have huge ``|∇d|/d`` (≈ 0.5–1.0)
    because the depth jump is ~the depth itself, while real geometry
    surfaces have ``|∇d|/d`` ≪ 0.5.
    """
    d = depth.astype(np.float32, copy=False)

    # (1) Sky: depth above the render-range cull.
    mask_sky = d > float(params.sky_thresh)

    # (2) Invalid: nan / inf / <= 0.
    mask_invalid = (~np.isfinite(d)) | (d <= 0)
    sky_or_invalid = mask_sky | mask_invalid

    # (3) Halo around sky-or-invalid. Gradient masks (below) will only be
    # consulted within this region. If grad_global=True, halo = everywhere.
    if params.grad_global:
        halo = np.ones_like(sky_or_invalid, dtype=bool)
    elif sky_or_invalid.any() and params.halo_radius > 0:
        kh = int(2 * params.halo_radius + 1)
        kernel_halo = np.ones((kh, kh), dtype=np.uint8)
        halo = cv2.dilate(
            sky_or_invalid.astype(np.uint8), kernel_halo, iterations=1,
        ).astype(bool)
    else:
        halo = sky_or_invalid.copy()  # no halo to widen

    # (4) + (5) Gradient masks, computed everywhere but gated by halo.
    d_safe = np.where(np.isfinite(d) & (d > 0), d, 0.0)
    gx = cv2.Sobel(d_safe, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d_safe, cv2.CV_32F, 0, 1, ksize=3)
    g_abs = np.abs(gx) + np.abs(gy)              # L1 magnitude

    mask_grad_abs = (g_abs > float(params.grad_abs)) & halo

    with np.errstate(divide="ignore", invalid="ignore"):
        g_rel = np.where(d_safe > 0, g_abs / np.maximum(d_safe, 1e-6), 0.0)
    mask_grad_rel = (g_rel > float(params.grad_rel)) & halo

    # (6) Optional defensive global grad_rel pass (AA halo catcher).
    if float(params.grad_rel_global) > 0.0:
        mask_grad_rel_global = g_rel > float(params.grad_rel_global)
    else:
        mask_grad_rel_global = np.zeros_like(mask_sky, dtype=bool)

    # (7) Union of all 5.
    mask_combined = (
        mask_sky | mask_invalid
        | mask_grad_abs | mask_grad_rel | mask_grad_rel_global
    )

    # (8) Final dilation by `dilate` px ((2k+1) square kernel).
    if params.dilate > 0:
        k = int(2 * params.dilate + 1)
        kernel = np.ones((k, k), dtype=np.uint8)
        mask_dilated = cv2.dilate(
            mask_combined.astype(np.uint8), kernel, iterations=1
        ).astype(bool)
    else:
        mask_dilated = mask_combined.copy()

    return SkyMaskBreakdown(
        mask_sky=mask_sky,
        mask_invalid=mask_invalid,
        mask_grad_abs=mask_grad_abs,
        mask_grad_rel=mask_grad_rel,
        mask_grad_rel_global=mask_grad_rel_global,
        mask_combined=mask_combined,
        mask_dilated=mask_dilated,
    )


__all__ = ["SkyMaskParams", "SkyMaskBreakdown", "compute_sky_mask"]
