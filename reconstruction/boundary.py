#!/usr/bin/env python3
"""Constrain auxiliary cameras using horizontal mesh rays and an eroded XZ footprint.

Intersect both checks when the footprint is available. The mesh and candidate
positions use OpenCV world coordinates, with X right, Y down and Z forward.
Adapted from Pano-NVS-Recon/point_select/boundary.py.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple

import numpy as np


# ------------------------------------------------------------------
# Minimal binary-PLY vertex reader — kept for backward compatibility
# (used by viz / other callers).
# ------------------------------------------------------------------
def read_ply_vertices(path: str) -> np.ndarray:
    """Return (N, 3) float64 vertex positions from a binary_little_endian PLY."""
    with open(path, "rb") as f:
        magic = f.readline().decode("ascii", errors="replace").strip()
        if magic != "ply":
            raise ValueError(f"{path}: not a PLY file (magic={magic!r})")
        fmt_line = f.readline().decode("ascii", errors="replace").strip()
        if "binary_little_endian" not in fmt_line:
            raise ValueError(f"{path}: only binary_little_endian supported "
                             f"(got {fmt_line!r})")
        n_vert = 0
        props = []
        in_vertex = False
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"{path}: incomplete PLY header")
            ln = raw.decode("ascii", errors="replace").strip()
            if ln == "end_header":
                break
            parts = ln.split()
            if not parts:
                continue
            if parts[0] == "element":
                in_vertex = (parts[1] == "vertex")
                if in_vertex:
                    n_vert = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                tmap = {
                    "float": "<f4", "float32": "<f4",
                    "double": "<f8", "float64": "<f8",
                    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
                    "short": "<i2", "ushort": "<u2",
                    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
                }
                props.append((tmap[parts[1]], parts[2]))
        dt = np.dtype([(name, t) for t, name in props])
        arr = np.frombuffer(f.read(dt.itemsize * n_vert), dtype=dt, count=n_vert)
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=-1).astype(np.float64)


# ------------------------------------------------------------------
# HorizonBoundary — polar boundary model
# ------------------------------------------------------------------
class HorizonBoundary:
    """Radial boundary (ray scan) and an XZ projection guard.

    Attributes
    ----------
    angles           : (N,) float  ray directions in [0, 2π), uniformly spaced
    r_boundary       : (N,) float  distance to first mesh hit per ray
    r_safe           : (N,) float  r_boundary - wall_margin (safe interior limit)
    anchor_xz        : (2,) float  anchor world (x, z)
    wall_margin      : float
    max_dist         : float       used when a ray hits nothing
    xz_guard         : optional eroded XZ occupancy grid
    """

    def __init__(self, angles: np.ndarray, r_boundary: np.ndarray,
                 r_safe: np.ndarray, anchor_xz: np.ndarray,
                 wall_margin: float, max_dist: float,
                 xz_guard: Optional["XZProjectionGuard"] = None):
        self.angles = np.asarray(angles, dtype=np.float64)
        self.r_boundary = np.asarray(r_boundary, dtype=np.float64)
        self.r_safe = np.asarray(r_safe, dtype=np.float64)
        self.anchor_xz = np.asarray(anchor_xz, dtype=np.float64).reshape(2)
        self.wall_margin = float(wall_margin)
        self.max_dist = float(max_dist)
        self.xz_guard = xz_guard  # Optional[XZProjectionGuard]

    # --- world (x, z) → ray-index ---
    def _theta_to_idx(self, theta: np.ndarray) -> np.ndarray:
        """Map angles (radians, any range) → nearest index in self.angles."""
        n = len(self.angles)
        dtheta = 2.0 * np.pi / n
        idx = np.round(((theta % (2.0 * np.pi)) / dtheta)).astype(np.int64) % n
        return idx

    def contains(self, positions_xyz: np.ndarray) -> np.ndarray:
        """True iff (x, z) is valid: passes BOTH ray-scan AND XZ-projection.

        Intersection rule: both checks must agree.
          - Ray scan  provides precise per-direction collision boundary.
          - XZ guard  provides the overall room footprint as a coarse bound.
        A point outside either check is rejected.  This is more conservative
        than union and prevents candidates from leaking through mesh gaps.
        """
        pos = np.asarray(positions_xyz, dtype=np.float64).reshape(-1, 3)
        dx = pos[:, 0] - self.anchor_xz[0]
        dz = pos[:, 2] - self.anchor_xz[1]
        r = np.sqrt(dx ** 2 + dz ** 2)
        theta = np.arctan2(dz, dx) % (2.0 * np.pi)
        idx = self._theta_to_idx(theta)
        ray_valid = r < self.r_safe[idx]
        if self.xz_guard is not None:
            xz_valid = self.xz_guard.contains(pos)
            # Intersection: must pass both.
            return ray_valid & xz_valid
        return ray_valid

# ------------------------------------------------------------------
# Ray casting backends
# ------------------------------------------------------------------
def _cast_rays_trimesh(mesh_path: str,
                       anchor_xyz: np.ndarray,
                       angles: np.ndarray,
                       max_dist: float,
                       mesh_rot_matrix: Optional[np.ndarray] = None,
                       ) -> np.ndarray:
    """Cast horizontal rays using trimesh BVH (fast, correct).

    Returns r_boundary (N,) — distance to first hit per ray, max_dist if miss.
    """
    import trimesh  # noqa: F401

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if mesh_rot_matrix is not None:
        mesh.vertices = mesh.vertices @ np.asarray(
            mesh_rot_matrix, dtype=np.float64)

    n = len(angles)
    origins = np.tile(anchor_xyz.astype(np.float64), (n, 1))
    dirs = np.stack([np.cos(angles), np.zeros(n), np.sin(angles)], axis=1)

    # multiple_hits=True → all intersections; we take the closest per ray.
    locs, idx_ray, _ = mesh.ray.intersects_location(
        ray_origins=origins.astype(np.float64),
        ray_directions=dirs.astype(np.float64),
        multiple_hits=True,
    )

    r = np.full(n, max_dist, dtype=np.float64)
    if len(locs) > 0:
        dx = locs[:, 0] - anchor_xyz[0]
        dz = locs[:, 2] - anchor_xyz[2]
        dist_xz = np.sqrt(dx ** 2 + dz ** 2)
        for hit_r, ir in zip(dist_xz, idx_ray):
            if hit_r < r[ir]:
                r[ir] = float(hit_r)
    return r


def _cast_rays_vertex_fallback(vertices_xyz: np.ndarray,
                                anchor_xyz: np.ndarray,
                                angles: np.ndarray,
                                max_dist: float,
                                ) -> np.ndarray:
    """Approximate ray scan using mesh vertices (no face data needed).

    For each angular bin, the boundary is the MINIMUM horizontal distance to
    any vertex whose projected direction falls in that bin.  Misses are
    interpolated from neighbours.  This is an approximation — thin walls
    between vertex-sparse directions may be missed.
    """
    verts = np.asarray(vertices_xyz, dtype=np.float64).reshape(-1, 3)
    n = len(angles)
    dtheta = 2.0 * np.pi / n

    dx = verts[:, 0] - anchor_xyz[0]
    dz = verts[:, 2] - anchor_xyz[2]
    dist_xz = np.sqrt(dx ** 2 + dz ** 2)

    # Exclude anchor-coincident vertices and those beyond max_dist.
    ok = (dist_xz > 1e-6) & (dist_xz < max_dist * 1.05)
    dx, dz, dist_xz = dx[ok], dz[ok], dist_xz[ok]

    theta = np.arctan2(dz, dx) % (2.0 * np.pi)
    idx = (np.round(theta / dtheta).astype(np.int64)) % n

    r = np.full(n, max_dist, dtype=np.float64)
    for i in range(len(dx)):
        ii = idx[i]
        if dist_xz[i] < r[ii]:
            r[ii] = float(dist_xz[i])

    # Fill bins with no vertex by interpolating from the two nearest filled
    # neighbours (linear interpolation in angle space).
    filled = r < max_dist
    if not filled.all() and filled.any():
        filled_idx = np.where(filled)[0]
        for i in np.where(~filled)[0]:
            # Find closest filled neighbour in each direction.
            diffs = (filled_idx - i) % n
            left_offset = diffs.min()
            right_offset = (n - diffs).min()
            il = (i + left_offset) % n
            ir_ = (i - right_offset) % n
            # Weight by angular distance.
            w_l = right_offset / (left_offset + right_offset)
            w_r = left_offset / (left_offset + right_offset)
            r[i] = w_l * r[il] + w_r * r[ir_]

    return r


# ------------------------------------------------------------------
# XZ downward-projection guard (grid-based, non-convex)
# ------------------------------------------------------------------
class XZProjectionGuard:
    """Fine-grained valid-zone from downward projection of mesh vertices.

    All mesh vertices are projected onto the XZ plane, rasterized onto a
    2-D grid, interior-filled with ``binary_fill_holes``, then eroded by
    ``wall_margin`` via ``distance_transform_edt``.  This preserves concave
    room shapes that a convex hull would incorrectly fill.

    Attributes
    ----------
    valid   : (nx, nz) bool  valid (safe) cells
    origin  : (2,) float     world (x, z) of grid cell (0, 0)
    cell    : float          grid cell size in world units
    """

    def __init__(self, valid: np.ndarray, origin: np.ndarray, cell: float):
        self.valid = valid
        self.origin = np.asarray(origin, dtype=np.float64).reshape(2)
        self.cell = float(cell)
        self.nx, self.nz = valid.shape

    def contains(self, positions_xyz: np.ndarray) -> np.ndarray:
        pos = np.asarray(positions_xyz, dtype=np.float64).reshape(-1, 3)
        ix = np.floor((pos[:, 0] - self.origin[0]) / self.cell).astype(np.int64)
        iz = np.floor((pos[:, 2] - self.origin[1]) / self.cell).astype(np.int64)
        in_bounds = (ix >= 0) & (ix < self.nx) & (iz >= 0) & (iz < self.nz)
        result = np.zeros(len(pos), dtype=bool)
        result[in_bounds] = self.valid[ix[in_bounds], iz[in_bounds]]
        return result


def _build_xz_projection_guard(
    verts_xyz: np.ndarray,
    wall_margin: float,
    cell_ratio: float = 0.01,
    scene_scale: float = 1.0,
) -> Optional["XZProjectionGuard"]:
    """Build a grid-based XZ footprint guard via downward projection.

    Steps:
      1. Project all mesh vertices onto the XZ plane.
      2. Rasterize onto a 2-D grid (cells touched by any vertex are marked).
      3. ``binary_fill_holes`` to fill the enclosed interior.
      4. ``distance_transform_edt`` to measure distance from the boundary.
      5. Valid cells = filled interior AND distance >= wall_margin.

    Returns None on failure (too few points or scipy unavailable).
    """
    from scipy.ndimage import binary_fill_holes, distance_transform_edt

    verts = np.asarray(verts_xyz, dtype=np.float64).reshape(-1, 3)
    xz = verts[:, [0, 2]]  # (N, 2)

    if len(xz) < 4:
        warnings.warn("Too few XZ vertices for projection guard; skipping.",
                      stacklevel=3)
        return None

    cell = float(cell_ratio) * float(scene_scale)
    xmin, zmin = xz.min(axis=0) - cell * 3
    xmax, zmax = xz.max(axis=0) + cell * 3
    nx = int(np.ceil((xmax - xmin) / cell)) + 1
    nz = int(np.ceil((zmax - zmin) / cell)) + 1

    # Rasterize: mark every cell touched by a vertex.
    ix = np.clip(((xz[:, 0] - xmin) / cell).astype(np.int64), 0, nx - 1)
    iz = np.clip(((xz[:, 1] - zmin) / cell).astype(np.int64), 0, nz - 1)
    occ = np.zeros((nx, nz), dtype=bool)
    occ[ix, iz] = True

    # Fill interior enclosed by projected vertices.
    filled = binary_fill_holes(occ)

    # Distance from non-filled (boundary / exterior) cells, in world units.
    dist_cells = distance_transform_edt(filled)
    dist_world = dist_cells * cell

    # Valid = interior AND eroded by wall_margin.
    valid = filled & (dist_world >= wall_margin)

    n_valid = int(valid.sum())
    n_filled = int(filled.sum())
    print(f"  [boundary] XZ projection guard  "
          f"grid=({nx}x{nz})  cell={cell:.4f}  "
          f"filled={n_filled}  valid(after erosion)={n_valid}")

    return XZProjectionGuard(valid, np.array([xmin, zmin]), cell)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def build_horizon_boundary(
    mesh_path: str,
    scene_scale: float,
    n_rays: int = 360,
    wall_margin_ratio: float = 0.05,
    max_dist_ratio: float = 0.8,
    anchor_y: float = 0.0,
    anchor_xz: np.ndarray = None,
    mesh_rot_matrix: np.ndarray = None,
    xz_guard_cell_ratio: float = 0.01,
    ray_extra_margin_ratio: float = 0.02,
) -> HorizonBoundary:
    """Build a polar horizon boundary (ray scan) with XZ projection guard.

    Parameters
    ----------
    mesh_path : path to mesh.ply (binary PLY with face data for trimesh, or
                vertices-only for fallback)
    scene_scale : float  bbox-diagonal scale; sets wall_margin and max_dist.
    n_rays : number of horizontal rays (default 360 -> 1 degree resolution).
    wall_margin_ratio : base safety margin = ratio * scene_scale.
        Used as erosion margin for both the XZ guard and the ray scan.
    ray_extra_margin_ratio : additional margin subtracted from ray-scan
        distances on top of wall_margin (default 0.02 = 2% of scene_scale).
        ray r_safe = r_boundary - (wall_margin + ray_extra_margin).
    max_dist_ratio : rays that hit nothing are treated as hitting at this
        distance (= ratio * scene_scale).
    anchor_y : camera height (y-down, default 0).
    anchor_xz : (2,) anchor (x, z) position (default (0, 0)).
    mesh_rot_matrix : (3, 3) rotation applied to mesh vertices before casting
        (e.g. _PLY_ROT.T to convert sphere-frame mesh -> OpenCV frame).
    xz_guard_cell_ratio : grid cell size for XZ projection guard as ratio of
        scene_scale (default 0.01).  Set to 0 to disable the guard.
    """
    if anchor_xz is None:
        anchor_xz = np.zeros(2, dtype=np.float64)
    anchor_xz = np.asarray(anchor_xz, dtype=np.float64).reshape(2)
    anchor_xyz = np.array([anchor_xz[0], float(anchor_y), anchor_xz[1]],
                          dtype=np.float64)

    wall_margin = float(wall_margin_ratio) * float(scene_scale)
    ray_extra_margin = float(ray_extra_margin_ratio) * float(scene_scale)
    max_dist = float(max_dist_ratio) * float(scene_scale)

    angles = np.linspace(0.0, 2.0 * np.pi, int(n_rays), endpoint=False)

    # Always read vertices -- needed for XZ projection guard and for the
    # vertex-fallback ray scan.
    verts = read_ply_vertices(mesh_path)
    if mesh_rot_matrix is not None:
        verts = verts @ np.asarray(mesh_rot_matrix, dtype=np.float64)

    # Try trimesh BVH ray cast (fast, correct, uses faces).
    r_boundary = None
    try:
        r_boundary = _cast_rays_trimesh(
            mesh_path, anchor_xyz, angles, max_dist, mesh_rot_matrix)
        print(f"  [boundary] ray scan via trimesh  "
              f"(n_rays={n_rays}, hits={int((r_boundary < max_dist).sum())}/"
              f"{n_rays})")
    except Exception as e:
        warnings.warn(f"trimesh ray cast failed ({e}); falling back to "
                      f"vertex-based approximation.", stacklevel=2)

    if r_boundary is None:
        r_boundary = _cast_rays_vertex_fallback(verts, anchor_xyz, angles,
                                                max_dist)
        print(f"  [boundary] ray scan via vertex fallback  "
              f"(n_rays={n_rays}, hits={int((r_boundary < max_dist).sum())}/"
              f"{n_rays})")

    # Ray scan safe zone: pull back by wall_margin + extra collision buffer.
    r_safe = np.maximum(r_boundary - wall_margin - ray_extra_margin, 0.0)

    # XZ downward-projection guard.
    xz_guard = None
    if xz_guard_cell_ratio > 0.0:
        xz_guard = _build_xz_projection_guard(
            verts, wall_margin=wall_margin,
            cell_ratio=xz_guard_cell_ratio, scene_scale=scene_scale)

    return HorizonBoundary(angles, r_boundary, r_safe, anchor_xz,
                           wall_margin, max_dist, xz_guard=xz_guard)
