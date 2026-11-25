"""Viewpoint selection adapted from Pano-NVS-Recon/point_select/selection_core.py."""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np


def horizontal_ring_directions(n_dir: int) -> np.ndarray:
    """`n_dir` unit dirs evenly spaced on the y=0 ring (OpenCV x-right, z-fwd)."""
    angles = np.linspace(0.0, 2.0 * math.pi, int(n_dir), endpoint=False)
    return np.stack([np.cos(angles), np.zeros_like(angles), np.sin(angles)],
                    axis=1)



def build_candidates_rings(anchor_xyz: np.ndarray,
                           radii: Sequence[float],
                           n_dir: int = 16,
                           n_dir_base: int = None,
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concentric horizontal rings of candidate positions.

    Args:
        anchor_xyz:  (3,) anchor position (camera height, y fixed)
        radii:       explicit list of ring radii (world units), ring index = i+1
        n_dir:       azimuth directions per ring (uniform, used when n_dir_base
                     is None — legacy mode)
        n_dir_base:  if given, ring i (1-indexed) gets ``n_dir_base * i`` points,
                     keeping arc-length spacing roughly uniform across rings.
                     Each ring is phase-shifted by π/n_points(i) to avoid all
                     rings sharing the same azimuth lines.
                     Total candidates = n_dir_base * sum(1..n_rings).

    Returns:
        positions (M,3), directions (M,3) unit, radii_per_cand (M,)
    """
    anchor = np.asarray(anchor_xyz, dtype=np.float64).reshape(3)
    positions, directions, radii_out = [], [], []
    for ring_idx, r in enumerate(radii):
        r = float(r)
        if n_dir_base is not None:
            # adaptive: ring 1 → n_dir_base, ring 2 → 2*n_dir_base, ...
            n_pts = max(4, int(n_dir_base) * (ring_idx + 1))
            phase = math.pi / n_pts  # half-step offset between adjacent rings
        else:
            n_pts = int(n_dir)
            phase = 0.0
        angles = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=False) + phase
        unit_dirs = np.stack(
            [np.cos(angles), np.zeros(n_pts), np.sin(angles)], axis=1)
        for d in unit_dirs:
            positions.append(anchor + r * d)
            directions.append(d.copy())
            radii_out.append(r)
    return (np.asarray(positions, dtype=np.float64),
            np.asarray(directions, dtype=np.float64),
            np.asarray(radii_out, dtype=np.float64))



def subsample_points(points: np.ndarray, max_n: int = 5000) -> np.ndarray:
    if len(points) <= max_n:
        return points
    idx = np.random.RandomState(0).choice(len(points), max_n, replace=False)
    return points[idx]



def geometric_prefilter(candidates: np.ndarray,
                        points_subsample: np.ndarray,
                        scene_scale: float,
                        anchor_xyz: np.ndarray = None,
                        near_point_ratio: float = 0.05,
                        path_n_steps: int = 12,
                        path_check_radius_ratio: float = 0.06,
                        path_min_neighbors: int = 1,
                        path_max_empty_segments: int = 2,
                        ) -> Tuple[np.ndarray, dict]:
    """Boolean keep-mask. Only Rule B (near-point penetration) + Rule E
    (path traversal "did we cross a wall"). Bbox/hull are handled elsewhere
    (mesh occupancy grid), so they are omitted here."""
    M = candidates.shape[0]
    keep = np.ones(M, dtype=bool)
    rejected_by = {"near_point": 0, "path": 0}

    psub = (np.asarray(points_subsample, dtype=np.float64)
            if points_subsample is not None else None)

    # Rule B: near-point penetration (camera buried in geometry).
    if psub is not None and len(psub) > 0 and near_point_ratio > 0:
        thresh = float(near_point_ratio) * float(scene_scale)
        if thresh > 0:
            for i in range(M):
                d = np.linalg.norm(psub - candidates[i:i + 1], axis=1)
                if d.min() < thresh:
                    keep[i] = False
                    rejected_by["near_point"] += 1

    # Rule E: path traversal (crossed a wall to reach candidate).
    if (anchor_xyz is not None and psub is not None and len(psub) > 0
            and path_n_steps >= 3 and path_check_radius_ratio > 0):
        anchor = np.asarray(anchor_xyz, dtype=np.float64).reshape(3)
        check_r2 = (float(path_check_radius_ratio) * float(scene_scale)) ** 2
        ts = np.linspace(0.0, 1.0, int(path_n_steps))
        ts_use = ts[2:-1] if len(ts) > 4 else ts[1:-1]
        for i in range(M):
            if not keep[i]:
                continue
            cand = candidates[i]
            empty = 0
            for t in ts_use:
                x = anchor + t * (cand - anchor)
                d2 = np.sum((psub - x[None, :]) ** 2, axis=1)
                if int(np.count_nonzero(d2 < check_r2)) < int(path_min_neighbors):
                    empty += 1
            if empty > int(path_max_empty_segments):
                keep[i] = False
                rejected_by["path"] += 1

    return keep, rejected_by



def to_tile_mask(erp_mask: np.ndarray, tile_h: int = 8,
                 tile_w: int = 8) -> np.ndarray:
    """ERP bool mask (H, W) -> tile bool mask (H/tile_h, W/tile_w).
    Tile active iff any pixel inside it is active."""
    erp = np.asarray(erp_mask).astype(bool)
    H, W = erp.shape
    Hc = (H // tile_h) * tile_h
    Wc = (W // tile_w) * tile_w
    if (Hc, Wc) != (H, W):
        erp = erp[:Hc, :Wc]
    Ht, Wt = Hc // tile_h, Wc // tile_w
    return erp.reshape(Ht, tile_h, Wt, tile_w).any(axis=(1, 3))



def to_hole_tile_mask(erp_valid_mask: np.ndarray, tile_h: int = 8,
                      tile_w: int = 8, hole_frac: float = 0.5) -> np.ndarray:
    """ERP *valid* mask (True=covered) -> hole tile mask.

    A tile is a "hole tile" iff the fraction of NON-covered (hole) pixels in it
    is >= ``hole_frac``. Holes are the disoccluded / parallax-revealed regions
    that the NVS generator must fill, so selecting viewpoints that maximize the
    union of hole tiles surfaces "where the scene has the most holes".
    """
    valid = np.asarray(erp_valid_mask).astype(bool)
    H, W = valid.shape
    Hc = (H // tile_h) * tile_h
    Wc = (W // tile_w) * tile_w
    if (Hc, Wc) != (H, W):
        valid = valid[:Hc, :Wc]
    Ht, Wt = Hc // tile_h, Wc // tile_w
    hole = (~valid).reshape(Ht, tile_h, Wt, tile_w).mean(axis=(1, 3))
    return hole >= float(hole_frac)



def gs_proxy_tile_masks(gs, positions, w2c_fn, gs_render_fn, device,
                        erp_h: int = 128, erp_w: int = 256,
                        tile_h: int = 8, tile_w: int = 8,
                        mode: str = "hole", hole_frac: float = 0.5):
    """Render low-res proxy ERP masks for a list of camera positions with the
    prebuilt GS dict, pool to tile masks.

    Args:
        gs:           dict from build_isotropic_gs_from_pcd (built once).
        positions:    (M, 3) candidate camera positions (world).
        w2c_fn:       position -> (4,4) w2c numpy (e.g. warp_gsplat_test.w2c_at).
        gs_render_fn: gs_pano_render_color (returns (rgb_list, mask_list)).
        device:       torch device.
        mode:         "hole"     -> tile=1 where the view has a hole
                                    (disocclusion / parallax-revealed region),
                                    so greedy maximizes union of holes.
                      "coverage" -> tile=1 where the view is covered (legacy).
        hole_frac:    a tile counts as a hole iff >= this fraction of its pixels
                      are NOT covered (only used for mode="hole").
    Returns:
        tiles:      (M, Ht, Wt) bool  (hole tiles or coverage tiles per `mode`)
        hole_ratio: (M,) float        fraction of hole tiles per view (for diag)
    """
    import numpy as _np
    pos = _np.asarray(positions, dtype=_np.float32).reshape(-1, 3)
    Rts = _np.stack([w2c_fn(p) for p in pos], axis=0).astype(_np.float32)
    _rgb_list, mask_list = gs_render_fn(
        gs, Rts, erp_h, erp_w, device,
        render_depth=False, erp_mask_aware=True,
    )
    tiles = []
    hole_ratio = []
    for m in mask_list:
        if mode == "coverage":
            t = to_tile_mask(m, tile_h=tile_h, tile_w=tile_w)
        else:
            t = to_hole_tile_mask(m, tile_h=tile_h, tile_w=tile_w,
                                  hole_frac=hole_frac)
        tiles.append(t)
        hole_ratio.append(float((~_np.asarray(m).astype(bool)).mean()))
    return _np.stack(tiles, axis=0).astype(bool), _np.asarray(hole_ratio)



def best_partition_angle(angles: np.ndarray,
                         r_boundary: np.ndarray) -> float:
    """Align the depth-0 cross so one arm points at the *nearest* boundary.

    The depth-0 quadrant split is a cross (4 arms 90 deg apart) through the
    anchor. We orient it so an arm sits on the globally nearest boundary
    direction ``argmin_theta r_boundary(theta)`` (reduced mod pi/2 by the
    cross's 90-deg symmetry). For a roughly rectangular room this aligns the
    cross with the wall normals / footprint principal axes, so the four
    quadrants become balanced corner sectors (each holds a coherent corner +
    equal shares of the two adjacent walls) rather than a big/small split.

    NOTE: this targets *balanced* quadrants (the stated goal). It is NOT the
    min-cut "cut through thinnest free space" variant (that would minimize the
    summed 4-arm distance and, for a rectangle, land the arms on the diagonals
    -> unbalanced). Swap the objective here if min-cut is preferred.

    Args:
        angles:     (N,) ray directions in [0, 2pi), uniformly spaced.
        r_boundary: (N,) anchor->first-mesh-hit distance per ray.

    Returns:
        phi (radians) in [0, pi/2), the cross rotation to apply.
    """
    angles = np.asarray(angles, dtype=np.float64).reshape(-1)
    r = np.asarray(r_boundary, dtype=np.float64).reshape(-1)
    if len(angles) == 0:
        return 0.0
    theta_min = float(angles[int(np.argmin(r))])
    return theta_min % (math.pi / 2.0)



def partition_select(
    T_seed: np.ndarray,
    T_v_list: np.ndarray,
    positions_xz: np.ndarray,
    K: int = 4,
    max_depth: int = 2,
    skip_covered_threshold: float = 0.9,
    rotate_angle: float = 0.0,
) -> Tuple[List[int], List[dict]]:
    """Quadrant-recursive selection for spatial diversity.

    Algorithm (recursive, up to ``max_depth`` levels):
      1. Divide current region into 4 orthogonal quadrants at the split point
         (anchor (0,0) at depth 0; bbox centre at deeper levels).
      2. In each quadrant find the candidate with the most new tiles
         (tiles not yet in ``covered``).
      3. If that candidate's visible tiles are >= ``skip_covered_threshold``
         already covered, skip the quadrant (nothing new to gain there).
      4. Otherwise select it, update covered, and recurse into the quadrant
         (up to ``max_depth``).
      5. Stop when K points are selected.

    Args:
        T_seed:                  (Ht, Wt) bool  initial covered set
        T_v_list:                (M, Ht, Wt) bool  tile masks for candidates
        positions_xz:            (M, 2) float  (x, z) world coords of candidates
        K:                       max points to select
        max_depth:               recursion depth (0 = just 4 top-level quadrants)
        skip_covered_threshold:  fraction of a candidate's tiles that are already
                                 covered to skip that quadrant (default 0.9)
        rotate_angle:            rotate the depth-0 cross by this angle (radians).
                                 0.0 = axis-aligned quadrants (legacy). Non-zero
                                 assigns candidates to quadrants in a frame
                                 rotated by ``rotate_angle`` (world coords /
                                 chosen indices are unchanged).

    Returns:
        chosen : list of indices into T_v_list (length <= K)
        diag   : list of per-quadrant diagnostic dicts
    """
    pos = np.asarray(positions_xz, dtype=np.float64).reshape(-1, 2)
    if rotate_angle != 0.0:
        # Express candidate (x, z) in the cross frame (rotate by -angle) so the
        # axis-aligned quadrant split below cuts along the rotated cross arms.
        c, s = math.cos(-rotate_angle), math.sin(-rotate_angle)
        pos = pos @ np.array([[c, -s], [s, c]], dtype=np.float64).T
    M = T_v_list.shape[0]
    covered = T_seed.astype(bool).copy()
    chosen: List[int] = []
    diag: List[dict] = []

    margin = 1e-4
    global_bbox = (pos[:, 0].min() - margin, pos[:, 0].max() + margin,
                   pos[:, 1].min() - margin, pos[:, 1].max() + margin)

    def _recurse(indices: List[int],
                 split_x: float, split_z: float,
                 bbox: Tuple[float, float, float, float],
                 depth: int) -> None:
        nonlocal covered
        if len(chosen) >= K or depth > max_depth or not indices:
            return

        xmin, xmax, zmin, zmax = bbox
        quadrants = [
            ("NE", split_x, xmax,   split_z, zmax),
            ("NW", xmin,   split_x, split_z, zmax),
            ("SE", split_x, xmax,   zmin,   split_z),
            ("SW", xmin,   split_x, zmin,   split_z),
        ]

        non_skipped: List[Tuple[List[int], Tuple]] = []
        for qname, qxmin, qxmax, qzmin, qzmax in quadrants:
            if len(chosen) >= K:
                break

            q_idx = [i for i in indices
                     if qxmin <= pos[i, 0] < qxmax
                     and qzmin <= pos[i, 1] < qzmax]
            if not q_idx:
                continue

            gains = np.array([(T_v_list[i] & ~covered).sum()
                               for i in q_idx], dtype=np.int64)
            best_local = int(np.argmax(gains))
            best_i = q_idx[best_local]
            best_gain = int(gains[best_local])

            T_best = T_v_list[best_i]
            total = int(T_best.sum())
            cov = int((T_best & covered).sum())
            cov_ratio = cov / max(total, 1)
            skip = cov_ratio >= float(skip_covered_threshold)

            diag.append({
                "depth": depth,
                "quadrant": qname,
                "bbox": (round(qxmin, 3), round(qxmax, 3),
                         round(qzmin, 3), round(qzmax, 3)),
                "n_candidates": len(q_idx),
                "best_idx": int(best_i),
                "best_gain": best_gain,
                "covered_ratio": round(cov_ratio, 3),
                "skipped": skip,
                "chosen_rank": len(chosen) if not skip else None,
            })

            if skip:
                continue

            chosen.append(best_i)
            covered |= T_best
            non_skipped.append((q_idx, (qxmin, qxmax, qzmin, qzmax)))

        # Recurse into each non-skipped quadrant.
        for q_idx, q_bbox in non_skipped:
            if len(chosen) >= K:
                break
            nx = (q_bbox[0] + q_bbox[1]) / 2.0
            nz = (q_bbox[2] + q_bbox[3]) / 2.0
            _recurse(q_idx, nx, nz, q_bbox, depth + 1)

    # Level 0: always split at anchor (0, 0).
    _recurse(list(range(M)), 0.0, 0.0, global_bbox, depth=0)

    return chosen, diag
