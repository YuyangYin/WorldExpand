#!/usr/bin/env python3
"""
Visualize trajectories per scene.

Reads step1_all_trajectory_0309.json to obtain, for each scene, the set of
trajectory ids and their frame counts. Actual XYZ coordinates live under
base_path/<scene>/<scene>/<traj_id>/<frame_idx>/<frame_idx>.json (UE Position).

For each scene, plots all trajectories' XY paths together on a single canvas
and saves one PNG per scene.

Extended mode (--neighbors_json):
    Overlays the front/back/left/right neighbor picks produced by
    prepare_history_candidates.py. Each target frame selected by the picker
    is drawn as a yellow dot + tangent arrow; each of its up-to-4 neighbors
    is drawn as a colored arrow (front=red, back=blue, left=green, right=
    magenta) pointing from target to neighbor. This is how you visually
    sanity-check the neighbor policy before kicking off bulk preparation.
"""

# Resolve resource defaults from the external release configuration.
import sys as _we_sys
from pathlib import Path as _WePath
_we_root = next((_p for _p in _WePath(__file__).resolve().parents
                 if (_p / "worldexpand" / "paths.py").is_file()), None)
if _we_root is not None and str(_we_root) not in _we_sys.path:
    _we_sys.path.insert(0, str(_we_root))
from worldexpand.paths import resolve_resource as _release_path

import argparse
import json
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# tqdm for a live progress bar.  If it's not installed we fall back to a
# no-op shim so the script still runs -- callers will just see the periodic
# "[done/total] ..." prints like before.
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:  # pragma: no cover
    _HAS_TQDM = False

    def tqdm(iterable=None, total=None, desc=None, **kwargs):  # type: ignore
        return iterable if iterable is not None else iter(())

# Used when overlaying neighbor picks.
BUCKET_COLOR = {"front": "red", "back": "blue", "left": "green", "right": "magenta"}

# Default cache location for per-scene trajectory dumps (see _load_scene_trajectories).
DEFAULT_CACHE_DIR = _release_path("@repo/outputs/dataset/trajectory_cache")

# If the primary dataset directory isn't available on this machine, transparently
# fall back to the backup directory.  We only switch when (a) the primary path
# literally doesn't exist AND (b) the fallback does -- this keeps behaviour
# strict on correctly-mounted boxes.
_BASE_PATH_FALLBACKS = [
    (_release_path("@repo/data/UE_PanoramicOut"),
     _release_path("@repo/data/UE_PanoramicOut_backup")),
]


def _resolve_base_path(path):
    if path and not os.path.isdir(path):
        for primary, backup in _BASE_PATH_FALLBACKS:
            if path.startswith(primary) and os.path.isdir(backup):
                new_path = backup + path[len(primary):]
                print(f"[base_path] {path} -> {new_path} (fallback)",
                      file=sys.stderr)
                return new_path
    return path


# ---------------------------------------------------------------------------
# Trajectory intersection utilities
# ---------------------------------------------------------------------------
# We treat every trajectory as an XY polyline; segment i is (P_i, P_{i+1}).
# A "real" intersection satisfies both:
#   1. The two segments are NOT neighbors of the same polyline (i.e. they
#      don't share a polyline vertex by construction).
#   2. The crossing point is NOT merely a shared endpoint of the two segs
#      (pure end-to-end touching -- e.g. two trajs departing from the same
#      anchor point -- is excluded per user request).
# Collinear overlaps are counted once (rare in practice).

_EPS = 1e-9


def _orient(ax_, ay_, bx_, by_, cx_, cy_):
    return (bx_ - ax_) * (cy_ - ay_) - (by_ - ay_) * (cx_ - ax_)


def _on_seg(ax_, ay_, bx_, by_, px_, py_):
    return (min(ax_, bx_) - _EPS <= px_ <= max(ax_, bx_) + _EPS and
            min(ay_, by_) - _EPS <= py_ <= max(ay_, by_) + _EPS)


def _pt_eq(x1, y1, x2, y2):
    return abs(x1 - x2) <= _EPS and abs(y1 - y2) <= _EPS


def _share_endpoint(p1, p2, p3, p4):
    """True if the two segments touch only at a shared endpoint."""
    return (_pt_eq(*p1, *p3) or _pt_eq(*p1, *p4) or
            _pt_eq(*p2, *p3) or _pt_eq(*p2, *p4))


def _seg_intersect_point(p1, p2, p3, p4):
    """
    Return (x, y) intersection if segments (p1,p2) and (p3,p4) cross in their
    interiors (or one endpoint lies strictly inside the other segment's
    interior), otherwise None.  Pure shared-endpoint contacts return None.
    """
    if _share_endpoint(p1, p2, p3, p4):
        return None

    x1, y1 = p1; x2, y2 = p2
    x3, y3 = p3; x4, y4 = p4

    d1 = _orient(x3, y3, x4, y4, x1, y1)
    d2 = _orient(x3, y3, x4, y4, x2, y2)
    d3 = _orient(x1, y1, x2, y2, x3, y3)
    d4 = _orient(x1, y1, x2, y2, x4, y4)

    if ((d1 > _EPS and d2 < -_EPS) or (d1 < -_EPS and d2 > _EPS)) and \
       ((d3 > _EPS and d4 < -_EPS) or (d3 < -_EPS and d4 > _EPS)):
        # proper crossing -- solve for parameter t on seg1
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < _EPS:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    # Collinear / T-junction cases: endpoint of one seg lies strictly inside
    # the other (we already excluded shared-endpoint touches above).
    if abs(d1) <= _EPS and _on_seg(x3, y3, x4, y4, x1, y1):
        return (x1, y1)
    if abs(d2) <= _EPS and _on_seg(x3, y3, x4, y4, x2, y2):
        return (x2, y2)
    if abs(d3) <= _EPS and _on_seg(x1, y1, x2, y2, x3, y3):
        return (x3, y3)
    if abs(d4) <= _EPS and _on_seg(x1, y1, x2, y2, x4, y4):
        return (x4, y4)
    return None


def _traj_aabb(xs, ys):
    return (min(xs), min(ys), max(xs), max(ys))


def _aabb_disjoint(a, b):
    return a[2] < b[0] - _EPS or b[2] < a[0] - _EPS or \
           a[3] < b[1] - _EPS or b[3] < a[1] - _EPS


def compute_intersections(traj_xy, keep_points=True):
    """
    Parameters
    ----------
    traj_xy : list[(traj_id:str, xs:list[float], ys:list[float])]
        Only trajectories with >=2 points should be passed in.
    keep_points : bool
        If False, the returned dict's ``cross_points`` / ``self_points``
        entries are empty lists -- we only keep scalar counts. Saves a lot
        of memory & pickle bandwidth when running --stats_only over
        thousands of scenes (point coords are only needed for plotting).

    Returns
    -------
    dict with keys:
        cross_points : list[(x, y, traj_i, traj_j)]     -- cross-trajectory hits
        self_points  : list[(x, y, traj_i)]              -- within-trajectory self-intersections
        n_cross      : int
        n_self       : int
        n_pairs_cross: int   (# distinct traj pairs that intersect at least once)
        n_trajs_self : int   (# trajs with at least one self-intersection)
    """
    # Pre-compute segments + AABBs per trajectory
    segs = []  # list of (traj_id, list[(p1,p2)], aabb)
    for tid, xs, ys in traj_xy:
        if len(xs) < 2:
            continue
        s = [((xs[i], ys[i]), (xs[i + 1], ys[i + 1])) for i in range(len(xs) - 1)]
        segs.append((tid, s, _traj_aabb(xs, ys)))

    cross_points = []
    self_points = []
    pairs_cross = set()
    self_trajs = set()
    n_cross = 0
    n_self = 0

    # --- cross-trajectory ---
    for a in range(len(segs)):
        tid_a, segs_a, box_a = segs[a]
        for b in range(a + 1, len(segs)):
            tid_b, segs_b, box_b = segs[b]
            if _aabb_disjoint(box_a, box_b):
                continue
            hit_pair = False
            for (p1, p2) in segs_a:
                for (p3, p4) in segs_b:
                    pt = _seg_intersect_point(p1, p2, p3, p4)
                    if pt is not None:
                        n_cross += 1
                        if keep_points:
                            cross_points.append((pt[0], pt[1], tid_a, tid_b))
                        hit_pair = True
            if hit_pair:
                pairs_cross.add((tid_a, tid_b))

    # --- self-intersection (non-adjacent segments within same polyline) ---
    for tid, s, _ in segs:
        n = len(s)
        for i in range(n):
            p1, p2 = s[i]
            # j >= i+2 skips adjacent segments (which share a polyline vertex)
            for j in range(i + 2, n):
                p3, p4 = s[j]
                pt = _seg_intersect_point(p1, p2, p3, p4)
                if pt is not None:
                    n_self += 1
                    if keep_points:
                        self_points.append((pt[0], pt[1], tid))
                    self_trajs.add(tid)

    return {
        "cross_points": cross_points,
        "self_points": self_points,
        "n_cross": n_cross,
        "n_self": n_self,
        "n_pairs_cross": len(pairs_cross),
        "n_trajs_self": len(self_trajs),
    }


def _scandir_frame_dirs(traj_dir):
    """Single scandir() pass beats listdir() + per-entry os.path.isfile() by
    a wide margin on COS mounts: d_type is populated in one syscall, so we
    avoid ~N extra stat() round trips per trajectory."""
    try:
        with os.scandir(traj_dir) as it:
            names = [e.name for e in it if e.name.isdigit() and e.is_dir()]
    except OSError:
        return []
    names.sort()
    return names


def load_trajectory_xy(scene_dir, traj_id):
    xs, ys = [], []
    traj_dir = os.path.join(scene_dir, str(traj_id))
    if not os.path.isdir(traj_dir):
        return xs, ys
    for fd in _scandir_frame_dirs(traj_dir):
        fp = os.path.join(traj_dir, fd, f"{fd}.json")
        try:
            with open(fp, "r") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        pos = d.get("Position", {})
        x = pos.get("X"); y = pos.get("Y")
        if x is None or y is None:
            continue
        xs.append(float(x)); ys.append(float(y))
    return xs, ys


def load_trajectory_xy_with_frames(scene_dir, traj_id):
    """Return xs, ys, zs, yaws, frame_names for one trajectory.

    yaw is ``Rotation.Yaw`` (degrees, world frame). UE also stores Pitch/Roll
    but they are always 0 for our panoramic shots, so we drop them on purpose.
    Legacy JSONs without Rotation or without Z silently default to 0.0 so the
    loader stays backwards compatible.
    """
    xs, ys, zs, yaws, frame_names = [], [], [], [], []
    traj_dir = os.path.join(scene_dir, str(traj_id))
    if not os.path.isdir(traj_dir):
        return xs, ys, zs, yaws, frame_names
    for fd in _scandir_frame_dirs(traj_dir):
        fp = os.path.join(traj_dir, fd, f"{fd}.json")
        try:
            with open(fp, "r") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        pos = d.get("Position", {})
        x = pos.get("X"); y = pos.get("Y"); z = pos.get("Z", 0.0)
        if x is None or y is None:
            continue
        rot = d.get("Rotation", {}) or {}
        yaw = rot.get("Yaw", 0.0)
        xs.append(float(x)); ys.append(float(y))
        zs.append(float(z) if z is not None else 0.0)
        yaws.append(float(yaw) if yaw is not None else 0.0)
        frame_names.append(fd)
    return xs, ys, zs, yaws, frame_names


def _build_neighbor_index(neighbors_json):
    """
    Load the prepare_history_candidates.py output into
        dict[scene] -> list[case]
    (case keeps: trajectory, frame, target_position, target_tangent_yaw_deg,
                 history_candidates[ (bucket, scene, trajectory, frame, yaw_rot) ])
    """
    with open(neighbors_json, "r") as f:
        data = json.load(f)
    cases = data.get("cases", data)
    by_scene = {}
    for c in cases:
        by_scene.setdefault(c["scene"], []).append(c)
    return by_scene


# ---------------------------------------------------------------------------
# Per-scene trajectory cache
# ---------------------------------------------------------------------------
# Reading thousands of tiny <frame>.json files from the COS mount is IO-bound
# and painfully slow; every downstream op (visualization, intersection stats)
# wants the *same* XY polylines.  We therefore consolidate every scene into a
# single .npz on fast local storage and transparently reuse it.
#
# Schema (CSR-style, so slicing any traj is zero-copy):
#     traj_ids:    ndarray[str]        (n_traj,)
#     offsets:     ndarray[int64]      (n_traj+1,)   -- xs/ys/zs/yaws/frame_names are flat
#     xs, ys, zs:  ndarray[float32]    (total_frames,)   -- UE uses cm; f32 is plenty
#     yaws:        ndarray[float32]    (total_frames,)   -- Rotation.Yaw in degrees
#     frame_names: ndarray[str]        (total_frames,)
# Presence of the .npz is also the "this scene has been scraped" sentinel
# (empty scenes still get a file with n_traj == 0 to short-circuit retries).
# Cache schema version; bump whenever the on-disk layout changes so stale
# caches are auto-rebuilt instead of silently mis-read.
_CACHE_VERSION = 2


def _scene_cache_path(cache_dir, scene_name):
    return os.path.join(cache_dir, "scene_cache", f"{scene_name}.npz")


def _scrape_scene_from_json(scene_dir, trajectories):
    """Read every <traj>/<frame>/<frame>.json under scene_dir into the CSR
    arrays described above. Pure IO; callers decide whether to cache."""
    def _key(k):
        try:
            return (0, int(k))
        except ValueError:
            return (1, k)

    sorted_ids = sorted(trajectories.keys(), key=_key)

    traj_ids = []
    offsets = [0]
    xs_all, ys_all, zs_all, yaws_all, fn_all = [], [], [], [], []
    for traj_id in sorted_ids:
        xs, ys, zs, yaws, frame_names = load_trajectory_xy_with_frames(scene_dir, traj_id)
        traj_ids.append(str(traj_id))
        xs_all.extend(xs)
        ys_all.extend(ys)
        zs_all.extend(zs)
        yaws_all.extend(yaws)
        fn_all.extend(frame_names)
        offsets.append(len(xs_all))

    return {
        "traj_ids": np.asarray(traj_ids, dtype=object),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "xs": np.asarray(xs_all, dtype=np.float32),
        "ys": np.asarray(ys_all, dtype=np.float32),
        "zs": np.asarray(zs_all, dtype=np.float32),
        "yaws": np.asarray(yaws_all, dtype=np.float32),
        "frame_names": np.asarray(fn_all, dtype=object),
    }


def _save_scene_cache(cache_path, payload):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    # IMPORTANT: np.savez_compressed auto-appends ".npz" to the file path if
    # it doesn't already end in ".npz".  So a tmp path like "foo.npz.tmp"
    # silently becomes "foo.npz.tmp.npz" on disk, and the subsequent
    # os.replace(tmp, final) then dies with ENOENT.  We dodge that by keeping
    # the .npz suffix on the tmp path and tagging it with the pid so parallel
    # workers can't collide either.
    tmp = f"{cache_path}.{os.getpid()}.tmp.npz"
    # Write atomically so a crash mid-save doesn't leave a half-file behind
    # (other workers racing on the same scene would then silently read
    # garbage).  np.savez_compressed honours object arrays via allow_pickle.
    np.savez_compressed(tmp,
                        traj_ids=payload["traj_ids"],
                        offsets=payload["offsets"],
                        xs=payload["xs"],
                        ys=payload["ys"],
                        zs=payload["zs"],
                        yaws=payload["yaws"],
                        frame_names=payload["frame_names"],
                        _version=np.asarray([_CACHE_VERSION], dtype=np.int32))
    try:
        os.replace(tmp, cache_path)
    except OSError:
        # Best-effort cleanup; surface the real error to the caller.
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def _load_scene_cache(cache_path):
    with np.load(cache_path, allow_pickle=True) as z:
        keys = set(z.files)
        # Pre-v2 caches don't have yaws -- raise KeyError so the caller
        # re-scrapes from the source JSONs (no silent zero-fill, otherwise
        # downstream orientation logic would get bad data).
        if "yaws" not in keys:
            raise KeyError("yaws")
        return {
            "traj_ids": z["traj_ids"],
            "offsets": z["offsets"],
            "xs": z["xs"].astype(np.float32, copy=False),
            "ys": z["ys"].astype(np.float32, copy=False),
            "zs": z["zs"].astype(np.float32, copy=False),
            "yaws": z["yaws"].astype(np.float32, copy=False),
            "frame_names": z["frame_names"],
        }


def _payload_to_per_traj(payload):
    """Expand the CSR payload into
        [(traj_id, xs_list, ys_list, zs_list, yaws_list, frame_list), ...]
    so existing downstream code (which was written for per-traj iteration)
    stays untouched (they just ignore yaws/zs if not needed)."""
    traj_ids = payload["traj_ids"]
    offsets = payload["offsets"]
    xs = payload["xs"]; ys = payload["ys"]; zs = payload["zs"]
    yaws = payload["yaws"]
    fn = payload["frame_names"]
    out = []
    for i, tid in enumerate(traj_ids):
        s, e = int(offsets[i]), int(offsets[i + 1])
        out.append((str(tid),
                    xs[s:e].tolist(),
                    ys[s:e].tolist(),
                    zs[s:e].tolist(),
                    yaws[s:e].tolist(),
                    [str(x) for x in fn[s:e]]))
    return out


def load_scene_trajectories(scene_name, scene_dir, trajectories,
                            cache_dir=None, rebuild_cache=False, use_cache=True):
    """
    Returns list[(traj_id, xs, ys, zs, frame_names)] for every trajectory of
    the scene.  Transparently caches the raw scrape in cache_dir to avoid
    hitting the COS mount on subsequent runs.

    - cache miss  -> scrape JSON, write cache (atomic), return.
    - cache hit   -> load cache, return (no JSON touched).
    - rebuild_cache=True always re-scrapes.
    - use_cache=False bypasses both read and write (debugging).
    """
    cache_path = None
    if cache_dir and use_cache:
        cache_path = _scene_cache_path(cache_dir, scene_name)
        if not rebuild_cache and os.path.isfile(cache_path):
            try:
                payload = _load_scene_cache(cache_path)
                return _payload_to_per_traj(payload)
            except (OSError, ValueError, KeyError) as e:
                # Corrupted/old cache -- fall through to re-scrape.
                print(f"[cache] {scene_name} corrupted ({e}); rebuilding",
                      file=sys.stderr)

    payload = _scrape_scene_from_json(scene_dir, trajectories)

    if cache_path is not None:
        try:
            _save_scene_cache(cache_path, payload)
        except OSError as e:
            print(f"[cache] {scene_name} save failed: {e}", file=sys.stderr)

    return _payload_to_per_traj(payload)


def visualize_scene(scene_name, scene_info, base_path, out_dir, overwrite=False,
                    neighbor_cases=None, max_targets=20,
                    count_intersections=False,
                    cache_dir=None, rebuild_cache=False, use_cache=True,
                    cache_only=False, stats_only=False):
    out_path = os.path.join(out_dir, f"{scene_name}.png")
    ix_cache_path = (os.path.join(cache_dir, "ix_stats", f"{scene_name}.json")
                     if cache_dir else None)

    # ------------------------------------------------------------------
    # stats_only fast path:
    #   * never touches COS (requires a pre-populated .npz cache)
    #   * never renders a PNG
    #   * reuses <cache_dir>/ix_stats/<scene>.json if present (resumable)
    # This is what you want when running a full distribution analysis
    # over thousands of cached scenes.
    # ------------------------------------------------------------------
    if stats_only:
        # Re-use previously computed stats when available.
        # Note: we now *also* keep per-intersection details (cross_details)
        # so that `--stats_csv` can dump the full table for step2.2 to
        # consume. Older cache files that predate this field are transparently
        # upgraded by falling through to the recompute branch.
        if ix_cache_path and os.path.isfile(ix_cache_path):
            try:
                with open(ix_cache_path, "r") as f:
                    cached = json.load(f)
                if "cross_details" in cached:
                    return scene_name, "stats-cached", cached
                # legacy cache (no cross_details) -> recompute below
            except (OSError, json.JSONDecodeError):
                pass  # recompute below

        if not cache_dir:
            return scene_name, "stats_only requires --cache_dir", None
        cache_path = _scene_cache_path(cache_dir, scene_name)
        if not os.path.isfile(cache_path):
            return scene_name, "missing cache", None
        try:
            payload = _load_scene_cache(cache_path)
        except (OSError, ValueError, KeyError) as e:
            return scene_name, f"corrupt cache: {e}", None
        per_traj = _payload_to_per_traj(payload)
        # Keep full per-traj info here so we can map each (x, y) back to the
        # nearest integer frame on each of the two owning trajectories.
        traj_full = [(tid, xs, ys, zs, yaws, fn)
                     for tid, xs, ys, zs, yaws, fn in per_traj
                     if len(xs) >= 2]
        traj_xy_list = [(tid, xs, ys) for tid, xs, ys, *_ in traj_full]

        cross_details: list = []
        if not traj_full:
            stats = {"n_cross": 0, "n_self": 0,
                     "n_pairs_cross": 0, "n_trajs_self": 0,
                     "n_trajs": 0,
                     "cross_details": cross_details}
        else:
            # keep_points=True -> we need (x, y, traj_a, traj_b) per hit so
            # step2.2 can pick centers from them without re-running the O(N^2)
            # segment-segment test.
            ix = compute_intersections(traj_xy_list, keep_points=True)

            # Build a tid -> (xs_np, ys_np, frame_names) lookup once.
            tid2poly = {}
            for tid, xs, ys, _zs, _yaws, fn in traj_full:
                tid2poly[str(tid)] = (
                    np.asarray(xs, dtype=np.float64),
                    np.asarray(ys, dtype=np.float64),
                    list(fn),
                )

            def _nearest(tid: str, px: float, py: float):
                xs_np, ys_np, fn_list = tid2poly[str(tid)]
                d2 = (xs_np - px) ** 2 + (ys_np - py) ** 2
                idx = int(np.argmin(d2))
                frame_name = fn_list[idx] if idx < len(fn_list) else ""
                return idx, frame_name

            # Each raw point: (x, y, tid_a, tid_b). Dedupe identical rows
            # (same pair hitting the same vertex twice on adjacent segments).
            seen = set()
            for (x, y, tid_a, tid_b) in ix["cross_points"]:
                key = (round(float(x), 3), round(float(y), 3),
                       str(tid_a), str(tid_b))
                if key in seen:
                    continue
                seen.add(key)
                ia, fa = _nearest(tid_a, float(x), float(y))
                ib, fb = _nearest(tid_b, float(x), float(y))
                cross_details.append({
                    "x": float(x),
                    "y": float(y),
                    "traj_a": str(tid_a),
                    "traj_a_idx": ia,
                    "traj_a_frame": str(fa),
                    "traj_b": str(tid_b),
                    "traj_b_idx": ib,
                    "traj_b_frame": str(fb),
                })

            stats = {
                "n_cross": ix["n_cross"],
                "n_self": ix["n_self"],
                "n_pairs_cross": ix["n_pairs_cross"],
                "n_trajs_self": ix["n_trajs_self"],
                "n_trajs": len(traj_xy_list),
                "cross_details": cross_details,
            }
        if ix_cache_path is not None:
            try:
                os.makedirs(os.path.dirname(ix_cache_path), exist_ok=True)
                with open(ix_cache_path, "w") as f:
                    json.dump(stats, f)
            except OSError as e:
                print(f"[cache] {scene_name} ix_stats save failed: {e}",
                      file=sys.stderr)
        return scene_name, "stats-ok", stats

    # ---- fast path: PNG already rendered, and ix_stats either not requested
    # or already cached -> skip both IO and JSON scrape ----
    if (not cache_only) and os.path.exists(out_path) and not overwrite:
        if not count_intersections:
            return scene_name, "skip", None
        if ix_cache_path and os.path.isfile(ix_cache_path):
            try:
                with open(ix_cache_path, "r") as f:
                    cached = json.load(f)
                # cached contains only the scalar counts (points lists are
                # not re-used when PNG is already rendered).
                return scene_name, "skip", cached
            except (OSError, json.JSONDecodeError):
                pass  # fall through to full recompute

    scene_dir = os.path.join(base_path, scene_name, scene_name)
    if not os.path.isdir(scene_dir):
        return scene_name, f"missing scene dir: {scene_dir}", None

    trajectories = scene_info.get("trajectories", {})
    if not trajectories:
        return scene_name, "no trajectories", None

    # ---- load XY polylines (cached) ----
    per_traj = load_scene_trajectories(
        scene_name, scene_dir, trajectories,
        cache_dir=cache_dir, rebuild_cache=rebuild_cache, use_cache=use_cache)

    if cache_only:
        # We only wanted to populate the cache; skip plotting entirely.
        n_traj = sum(1 for _, xs, _, _, _, _ in per_traj if len(xs) >= 2)
        return scene_name, f"cached ({n_traj} trajs)", None

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap("tab20")
    plotted = 0

    # Keep a (traj_id, frame_name) -> (x, y) index for neighbor arrow drawing
    frame_xy = {}
    # Collected polylines for intersection analysis
    traj_xy_list = []  # list[(traj_id:str, xs, ys)]
    for idx, (traj_id, xs, ys, _zs, _yaws, frame_names) in enumerate(per_traj):
        if len(xs) < 2:
            continue
        for fn, xx, yy in zip(frame_names, xs, ys):
            frame_xy[(str(traj_id), fn)] = (xx, yy)
        color = cmap(idx % 20)
        ax.plot(xs, ys, "-", color=color, linewidth=1.2, alpha=0.85,
                label=f"traj {traj_id}")
        ax.scatter([xs[0]], [ys[0]], color=color, marker="o", s=25,
                   edgecolors="black", linewidths=0.5, zorder=3)
        ax.scatter([xs[-1]], [ys[-1]], color=color, marker="X", s=35,
                   edgecolors="black", linewidths=0.5, zorder=3)
        traj_xy_list.append((str(traj_id), xs, ys))
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return scene_name, "no plottable trajectories", None

    # --- optional: compute & overlay intersections ---
    ix_stats = None
    ix_title = ""
    if count_intersections:
        ix_stats = compute_intersections(traj_xy_list)
        # cross-trajectory intersections: red X
        if ix_stats["cross_points"]:
            cxs = [p[0] for p in ix_stats["cross_points"]]
            cys = [p[1] for p in ix_stats["cross_points"]]
            ax.scatter(cxs, cys, marker="x", color="red", s=60, linewidths=1.6,
                       zorder=6, label=f"cross x-ing ({len(cxs)})")
        # self intersections: blue plus
        if ix_stats["self_points"]:
            sxs = [p[0] for p in ix_stats["self_points"]]
            sys_ = [p[1] for p in ix_stats["self_points"]]
            ax.scatter(sxs, sys_, marker="+", color="blue", s=70, linewidths=1.6,
                       zorder=6, label=f"self x-ing ({len(sxs)})")
        ix_title = (f"  | cross={ix_stats['n_cross']} (pairs={ix_stats['n_pairs_cross']})"
                    f", self={ix_stats['n_self']} (trajs={ix_stats['n_trajs_self']})")

        # persist just the scalar counts (point lists can get big)
        if ix_cache_path is not None:
            try:
                os.makedirs(os.path.dirname(ix_cache_path), exist_ok=True)
                with open(ix_cache_path, "w") as f:
                    json.dump({
                        "n_cross": ix_stats["n_cross"],
                        "n_self": ix_stats["n_self"],
                        "n_pairs_cross": ix_stats["n_pairs_cross"],
                        "n_trajs_self": ix_stats["n_trajs_self"],
                    }, f)
            except OSError as e:
                print(f"[cache] {scene_name} ix_stats save failed: {e}",
                      file=sys.stderr)

    # --- optional: overlay neighbor picks ---
    overlay_msg = ""
    if neighbor_cases:
        cases = neighbor_cases
        if len(cases) > max_targets:
            step = max(1, len(cases) // max_targets)
            cases = cases[::step][:max_targets]
        drew = 0
        for c in cases:
            tp = c.get("target_position") or [None, None]
            tx, ty = tp[0], tp[1]
            if tx is None or ty is None:
                continue
            # target marker + tangent arrow
            ax.scatter([tx], [ty], color="gold", marker="*", s=90,
                       edgecolors="black", linewidths=0.6, zorder=5)
            tyaw = math.radians(float(c.get("target_tangent_yaw_deg", 0.0)))
            arrow_len = _auto_arrow_len(ax)
            ax.annotate("", xy=(tx + arrow_len * math.cos(tyaw), ty + arrow_len * math.sin(tyaw)),
                        xytext=(tx, ty),
                        arrowprops=dict(arrowstyle="->", color="gold", lw=1.2, alpha=0.9),
                        zorder=5)
            # neighbor arrows
            for nb in c.get("history_candidates", []):
                key = (str(nb.get("trajectory")), str(nb.get("frame")))
                nxy = frame_xy.get(key)
                if nxy is None:
                    continue
                nx, ny = nxy
                col = BUCKET_COLOR.get(nb.get("bucket"), "black")
                ax.annotate("", xy=(nx, ny), xytext=(tx, ty),
                            arrowprops=dict(arrowstyle="->", color=col, lw=1.0, alpha=0.8),
                            zorder=4)
                ax.scatter([nx], [ny], color=col, marker="s", s=16,
                           edgecolors="black", linewidths=0.3, zorder=4)
            drew += 1
        overlay_msg = f"  +{drew} targets"

    ax.set_title(f"{scene_name}  ({plotted} trajectories{overlay_msg}{ix_title})")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.3)
    if plotted <= 25:
        ax.legend(fontsize=7, loc="best", ncol=2, framealpha=0.7)

    if neighbor_cases:
        # Add a small legend strip for bucket colors
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color=c, lw=2, label=b) for b, c in BUCKET_COLOR.items()]
        handles.append(Line2D([0], [0], marker="*", color="gold", lw=0, markersize=12,
                              markeredgecolor="black", label="target"))
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return scene_name, "ok", ix_stats


def _auto_arrow_len(ax):
    """Pick a tangent-arrow length that scales with the scene's axis extent."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span = max(abs(x1 - x0), abs(y1 - y0))
    return max(10.0, span * 0.01)


def _worker(args):
    (scene_name, scene_info, base_path, out_dir, overwrite,
     neighbor_cases, max_targets, count_intersections,
     cache_dir, rebuild_cache, use_cache, cache_only, stats_only) = args
    try:
        return visualize_scene(scene_name, scene_info, base_path, out_dir, overwrite,
                               neighbor_cases=neighbor_cases, max_targets=max_targets,
                               count_intersections=count_intersections,
                               cache_dir=cache_dir,
                               rebuild_cache=rebuild_cache,
                               use_cache=use_cache,
                               cache_only=cache_only,
                               stats_only=stats_only)
    except Exception as e:
        return scene_name, f"error: {e}\n{traceback.format_exc()}", None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",
        default=_release_path("@repo/data/trajectory_analysis.json"))
    parser.add_argument("--base_path",
        default=_release_path("@repo/data/UE_PanoramicOut"))
    parser.add_argument("--out_dir",
        default=_release_path("@repo/outputs/dataset/trajectory_visualizations"))
    parser.add_argument("--workers", type=int, default=64,
                        help="Process-pool size. The bottleneck is COS RTT, "
                             "not CPU, so large values (64-128) help a lot "
                             "during --cache_only.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard", type=str, default=None,
                        help="Process only shard i/N of the scene list (e.g. "
                             "'0/3' on machine A, '1/3' on B, '2/3' on C). "
                             "Used to fan out --cache_only across multiple "
                             "machines writing to the same --cache_dir.")
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--neighbors_json", type=str, default=None,
                        help="Optional: output of prepare_history_candidates.py. "
                             "When provided, draws F/B/L/R neighbor arrows on each scene.")
    parser.add_argument("--max_targets", type=int, default=20,
                        help="Cap on how many neighbor targets to draw per scene (for clarity).")
    parser.add_argument("--count_intersections", action="store_true",
                        help="Detect & visualize trajectory self / cross intersections. "
                             "Shared endpoints are NOT counted; adjacent polyline "
                             "segments are excluded from self-intersection.")
    parser.add_argument("--stats_csv", type=str, default=None,
                        help="When --count_intersections / --stats_only is "
                             "set, dump a **per-intersection** CSV here "
                             "(one row per cross point). Columns: "
                             "scene,x,y,traj_a,traj_a_idx,traj_a_frame,"
                             "traj_b,traj_b_idx,traj_b_frame. Downstream "
                             "step2.2_sample_five_view_segments.py consumes "
                             "this file directly.")
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR,
                        help="Fast-disk directory where per-scene trajectory "
                             ".npz caches and ix_stats .json files live. "
                             f"Default: {DEFAULT_CACHE_DIR}")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable the trajectory cache completely (useful "
                             "when debugging scrape logic).")
    parser.add_argument("--rebuild_cache", action="store_true",
                        help="Force re-scrape the raw JSONs and overwrite any "
                             "existing .npz caches.")
    parser.add_argument("--cache_only", action="store_true",
                        help="Just populate the trajectory cache (no PNG, no "
                             "intersection stats). Run this once over all "
                             "scenes to amortise the COS IO cost, then do "
                             "fast visualization / analysis afterwards.")
    parser.add_argument("--stats_only", action="store_true",
                        help="Compute intersection counts **from .npz cache only** "
                             "(no PNG, no COS read). Scenes without a cache "
                             "are skipped. Pair with --stats_csv to dump a "
                             "per-scene distribution.")
    args = parser.parse_args()

    args.base_path = _resolve_base_path(args.base_path)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
    with open(args.json, "r") as f:
        data = json.load(f)

    neighbors_by_scene = {}
    if args.neighbors_json:
        neighbors_by_scene = _build_neighbor_index(args.neighbors_json)
        print(f"Loaded neighbor picks for {len(neighbors_by_scene)} scenes from {args.neighbors_json}")

    scenes = list(data.items())
    # Deterministic order so shard-i and shard-j on different boxes agree on
    # which scene belongs to whom.
    scenes.sort(key=lambda kv: kv[0])
    if args.scene:
        scenes = [(k, v) for k, v in scenes if k == args.scene]
    if args.shard:
        try:
            idx_str, total_str = args.shard.split("/")
            shard_i = int(idx_str); shard_n = int(total_str)
            if shard_n <= 0 or not (0 <= shard_i < shard_n):
                raise ValueError
        except ValueError:
            raise SystemExit(f"--shard must be 'i/N' with 0 <= i < N, got {args.shard!r}")
        scenes = [kv for j, kv in enumerate(scenes) if j % shard_n == shard_i]
        print(f"[shard] {shard_i}/{shard_n} -> {len(scenes)} scenes")
    if args.limit > 0:
        scenes = scenes[: args.limit]

    # --stats_only implies we want intersection counts, and the render path
    # is skipped entirely, so make the flag self-sufficient.
    if args.stats_only:
        args.count_intersections = True

    print(f"Scenes to process: {len(scenes)}  workers={args.workers}")
    use_cache = not args.no_cache
    tasks = [(name, info, args.base_path, args.out_dir, args.overwrite,
              neighbors_by_scene.get(name), args.max_targets,
              args.count_intersections,
              args.cache_dir, args.rebuild_cache, use_cache,
              args.cache_only, args.stats_only)
             for name, info in scenes]

    ok = skip = fail = 0
    all_stats = {}  # scene -> ix_stats

    def _classify(status):
        if status in ("skip", "stats-cached"):
            return "skip"
        if (status.startswith("error")
                or status.startswith("missing")
                or status.startswith("corrupt")
                or status.startswith("stats_only requires")):
            return "fail"
        return "ok"  # "ok", "cached (...)", "stats-ok", "no trajectories", ...

    if args.stats_only:
        pbar_desc = "stats"
    elif args.cache_only:
        pbar_desc = "cache"
    else:
        pbar_desc = "render"

    if args.workers <= 1:
        pbar = tqdm(total=len(tasks), desc=pbar_desc, dynamic_ncols=True)
        for t in tasks:
            name, status, ix_stats = _worker(t)
            cat = _classify(status)
            if cat == "ok": ok += 1
            elif cat == "skip": skip += 1
            else:
                fail += 1
                print(f"[{name}] {status}", file=sys.stderr)
            if ix_stats is not None:
                all_stats[name] = ix_stats
            if _HAS_TQDM:
                pbar.set_postfix(ok=ok, skip=skip, fail=fail, last=name[:20])
                pbar.update(1)
            else:
                print(f"[{ok+skip+fail}/{len(tasks)}] {name}: {status}")
        if _HAS_TQDM:
            pbar.close()
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_worker, t): t[0] for t in tasks}
            done = 0
            pbar = tqdm(total=len(tasks), desc=pbar_desc, dynamic_ncols=True)
            for fut in as_completed(futs):
                name = futs[fut]
                try:
                    _, status, ix_stats = fut.result()
                except Exception as e:
                    status = f"error: {e}"
                    ix_stats = None
                done += 1
                cat = _classify(status)
                if cat == "ok": ok += 1
                elif cat == "skip": skip += 1
                else:
                    fail += 1
                    print(f"[{name}] {status}", file=sys.stderr)
                if ix_stats is not None:
                    all_stats[name] = ix_stats
                if _HAS_TQDM:
                    pbar.set_postfix(ok=ok, skip=skip, fail=fail,
                                     last=name[:20])
                    pbar.update(1)
                elif done % 20 == 0 or done == len(tasks):
                    print(f"[{done}/{len(tasks)}] ok={ok} skip={skip} fail={fail}")
            if _HAS_TQDM:
                pbar.close()

    print(f"Done. ok={ok} skip={skip} fail={fail}  out={args.out_dir}")

    # ---- intersection summary ----
    if args.count_intersections and all_stats:
        total_cross = sum(s["n_cross"] for s in all_stats.values())
        total_self = sum(s["n_self"] for s in all_stats.values())
        total_pairs = sum(s["n_pairs_cross"] for s in all_stats.values())
        total_self_trajs = sum(s["n_trajs_self"] for s in all_stats.values())
        scenes_any = sum(1 for s in all_stats.values()
                         if s["n_cross"] > 0 or s["n_self"] > 0)
        print("\n=== Intersection summary ===")
        print(f"scenes analysed:            {len(all_stats)}")
        print(f"scenes with any x-ing:      {scenes_any}")
        print(f"total cross-traj x-points:  {total_cross}  "
              f"(across {total_pairs} traj-pairs)")
        print(f"total self x-points:        {total_self}  "
              f"(across {total_self_trajs} trajs)")

        # Distribution over scenes (useful to pick thresholds downstream).
        totals = np.asarray([s["n_cross"] + s["n_self"]
                             for s in all_stats.values()], dtype=np.int64)
        cross_arr = np.asarray([s["n_cross"] for s in all_stats.values()],
                               dtype=np.int64)
        self_arr = np.asarray([s["n_self"] for s in all_stats.values()],
                              dtype=np.int64)
        if totals.size:
            def _pctl(arr, q):
                return float(np.percentile(arr, q)) if arr.size else 0.0
            print("\nPer-scene distribution (cross + self):")
            print(f"  mean  ={totals.mean():.2f}  max={int(totals.max())}  "
                  f"p50={_pctl(totals, 50):.1f}  p90={_pctl(totals, 90):.1f}  "
                  f"p99={_pctl(totals, 99):.1f}")
            print("Per-scene distribution (cross only):")
            print(f"  mean  ={cross_arr.mean():.2f}  max={int(cross_arr.max())}  "
                  f"p50={_pctl(cross_arr, 50):.1f}  p90={_pctl(cross_arr, 90):.1f}  "
                  f"p99={_pctl(cross_arr, 99):.1f}")
            print("Per-scene distribution (self only):")
            print(f"  mean  ={self_arr.mean():.2f}  max={int(self_arr.max())}  "
                  f"p50={_pctl(self_arr, 50):.1f}  p90={_pctl(self_arr, 90):.1f}  "
                  f"p99={_pctl(self_arr, 99):.1f}")

        # top-N offenders
        top = sorted(all_stats.items(),
                     key=lambda kv: (kv[1]["n_cross"] + kv[1]["n_self"]),
                     reverse=True)[:10]
        if top:
            print("\nTop-10 scenes by total x-points:")
            print(f"  {'scene':<40s}  cross  self  pairs  trajs_self")
            for name, s in top:
                print(f"  {name:<40s}  {s['n_cross']:5d}  {s['n_self']:5d}  "
                      f"{s['n_pairs_cross']:5d}  {s['n_trajs_self']:10d}")

        if args.stats_csv:
            os.makedirs(os.path.dirname(args.stats_csv) or ".", exist_ok=True)
            # CSV layout = per-intersection details, one row per cross point.
            # Downstream (step2.2_sample_five_view_segments.py) consumes this
            # directly instead of re-running the O(N^2) intersection search.
            #
            # Columns:
            #   scene           : scene name (matches cache/ix_stats key)
            #   x, y            : intersection coordinates (float, world XY)
            #   traj_a / traj_b : the two trajectories that cross here
            #   traj_a_idx      : nearest integer frame-index on traj_a
            #   traj_a_frame    : that frame's folder name (e.g. "000123")
            #   traj_b_idx      : nearest integer frame-index on traj_b
            #   traj_b_frame    : that frame's folder name
            header = ("scene,x,y,traj_a,traj_a_idx,traj_a_frame,"
                      "traj_b,traj_b_idx,traj_b_frame\n")
            n_rows = 0
            with open(args.stats_csv, "w") as f:
                f.write(header)
                for name in sorted(all_stats.keys()):
                    s = all_stats[name]
                    for d in s.get("cross_details", []) or []:
                        f.write(f"{name},{d['x']:.6f},{d['y']:.6f},"
                                f"{d['traj_a']},{d['traj_a_idx']},"
                                f"{d['traj_a_frame']},"
                                f"{d['traj_b']},{d['traj_b_idx']},"
                                f"{d['traj_b_frame']}\n")
                        n_rows += 1
            print(f"\nPer-intersection CSV -> {args.stats_csv}  "
                  f"({n_rows} rows)")


if __name__ == "__main__":
    main()
