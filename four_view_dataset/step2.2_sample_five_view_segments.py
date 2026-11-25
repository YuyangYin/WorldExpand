#!/usr/bin/env python3
"""
step2.2_sample_five_view_segments.py
====================================

从轨迹交叉点出发，为每条轨迹采样最多 5 个 center，每个 center 产出 **五元组**:

    (center, front1, back1, front2, back2)

- ``center``             : 交叉点在"主轨迹"(当前轨迹)上就近取整后的那一帧
- ``front1`` / ``back1`` : 主轨迹上 center 前后 segment_length/2 帧（不足 clamp 到端点）
- ``front2`` / ``back2`` : 主轨迹对方那条交叉轨迹（companion）的交点就近帧的前后两帧

**为什么"按轨迹而不是按 scene"**
    同一 scene 可能有成百上千个 intersection，直接按 scene 抓取会让一条热门轨迹
    贡献大部分数据、制造严重偏斜。以轨迹为单位、每条轨迹最多 5 个 center，并且
    center 之间用贪心的"最大最小距离 (farthest-point)" 采样，能让 5 个 center
    均匀分布在轨迹上。

**对下游的兼容性**
    每个五元组既输出主方向字段 (``directory`` / ``start`` / ``end``) —— 这和
    ``step2.2_sample_segments.py`` 的 case 结构**完全相同**，所以 ``step3.1``、
    ``step3.2``、``step7_add_id.py`` 和 ``pipeline_generate_render_multi_gpu.py``
    都能无改动地继续吃下去；又额外附带 ``views`` 子字段描述另外 4 条分支，供
    Pano_four_full 这边的 dataset 读取。

**工程细节 (符合长期记忆偏好)**
    * **并行**  —— ``ProcessPoolExecutor`` 按 scene 并行；每个 worker 只读本机
      ``cache_dir`` 下的 .npz（没有额外的 COS IO），PSNR 计算是唯一的 IO。
    * **分片**  —— ``--num_machines / --machine_id`` 同 step2.2 原脚本一致，
      scene 列表做确定性 round-robin，保证不同机器不会处理同一 scene。
    * **断点续传**  —— 每个 scene 在 ``<out_dir>/per_scene/<scene>.json`` 写一
      份 atomic 结果；下次启动时 **文件已存在 → 直接跳过**，不会重算 PSNR。
    * **原子写**  —— 写 ``*.tmp`` 再 ``os.replace``，中断也不会留半写文件。

Usage
-----
**推荐用法**：先用 ``step1_visualize_trajectories.py --stats_only --stats_csv`` 算好
交叉点明细 CSV，然后这边直接读入（避免重复运行 O(N^2) 线段相交检测）：

    # step 1 — 交点明细 CSV
    python step1_visualize_trajectories.py --stats_only --workers 64 \\
        --analysis_file   four_view_dataset/data_0519/step0_all_trajectory_0519.json \\
        --base_path       @repo/data/UE_PanoramicOut \\
        --cache_dir       @repo/outputs/dataset/trajectory_cache \\
        --stats_csv       @repo/outputs/dataset/trajectory_cache/intersections.csv

    # step 2 — 五视图采样（直接吃上一步的 CSV）
    python step2.2_sample_five_view_segments.py \\
        --analysis_file      four_view_dataset/data_0519/step0_all_trajectory_0519.json \\
        --intersections_csv  @repo/outputs/dataset/trajectory_cache/intersections.csv \\
        --cache_dir          @repo/outputs/dataset/trajectory_cache \\
        --base_dir           @repo/data/UE_PanoramicOut \\
        --output_dir         ./output/4.22/sampled_five_view_segments \\
        --segment_length 180 --max_centers_per_traj 5 \\
        --psnr_min 8 --psnr_max 18 \\
        --workers 32 --num_machines 4 --machine_id 0

没传 ``--intersections_csv`` 时 fallback 为现场读 ``cache_dir`` 里的 ``.npz`` 并
重算 ``compute_intersections``（结果一样，只是慢一点）。
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


import argparse
import json
import math
import os
import random
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

    def tqdm(iterable=None, total=None, desc=None, **kw):  # type: ignore
        return iterable if iterable is not None else iter(())

# Reuse the battle-tested intersection + cache primitives from the
# visualization script so we don't duplicate logic / diverge in edge cases.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from step1_visualize_trajectories import (  # noqa: E402
    _load_scene_cache,
    _scene_cache_path,
    _payload_to_per_traj,
    compute_intersections,
    _resolve_base_path,
)


# ---------------------------------------------------------------------------
# Load the per-intersection CSV produced by
#   ``step1_visualize_trajectories.py --stats_only --stats_csv ...``
# into dict[scene] -> list[(x, y, traj_a, traj_b)].
# ---------------------------------------------------------------------------
def _load_intersections_csv(path: str):
    """Read a per-intersection CSV and group rows by scene.

    The CSV schema must match what ``step1_visualize_trajectories.py`` writes:
        scene,x,y,traj_a,traj_a_idx,traj_a_frame,traj_b,traj_b_idx,traj_b_frame
    We only actually need (scene, x, y, traj_a, traj_b) here — ``traj_*_idx`` /
    ``traj_*_frame`` are recomputed from the cached polyline anyway so that
    the nearest-frame snap logic stays identical regardless of how the CSV
    rounded its own x/y.
    """
    import csv
    by_scene: Dict[str, List[Tuple[float, float, str, str]]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"scene", "x", "y", "traj_a", "traj_b"}
        if not required.issubset(reader.fieldnames or []):
            raise SystemExit(
                f"--intersections_csv {path} is missing columns. "
                f"expected at least {sorted(required)}, got {reader.fieldnames}")
        for row in reader:
            try:
                scene = row["scene"]
                x = float(row["x"]); y = float(row["y"])
                ta = str(row["traj_a"]); tb = str(row["traj_b"])
            except (KeyError, ValueError):
                continue
            by_scene.setdefault(scene, []).append((x, y, ta, tb))
    return by_scene

# PSNR optional dependency --------------------------------------------------
# skimage is only needed when --psnr_* is active; gracefully degrade otherwise.
try:
    from skimage.metrics import peak_signal_noise_ratio as _psnr_fn  # type: ignore
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

random.seed(42)

DEFAULT_SEGMENT_LENGTH = 180
DEFAULT_MAX_CENTERS = 5
DEFAULT_PSNR_MIN = 8.0
DEFAULT_PSNR_MAX = 18.0
DEFAULT_BASE_DIR = _release_path("@repo/data/UE_PanoramicOut")
DEFAULT_CACHE_DIR = _release_path("@repo/outputs/dataset/trajectory_cache")


# ---------------------------------------------------------------------------
# Path helpers (UE pano layout: <base>/<scene>/<scene>/<traj>/<ffff>/<ffff>.{jpg,exr,json})
# ---------------------------------------------------------------------------
def _frame_str(idx: int) -> str:
    return str(idx).zfill(4)


def get_point_paths(scene: str, traj_id: str, frame_idx: int) -> Dict[str, str]:
    fs = _frame_str(frame_idx)
    rel = os.path.join(scene, scene, str(traj_id), fs)
    return {
        "jpg": os.path.join(rel, f"{fs}.jpg"),
        "exr": os.path.join(rel, f"{fs}.exr"),
        "json": os.path.join(rel, f"{fs}.json"),
    }


def get_absolute_image_path(scene: str, traj_id: str, frame_idx: int, base_dir: str) -> str:
    fs = _frame_str(frame_idx)
    return os.path.join(base_dir, scene, scene, str(traj_id), fs, f"{fs}.jpg")


# ---------------------------------------------------------------------------
# Atomic IO
# ---------------------------------------------------------------------------
def _atomic_write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Geometry helpers for picking frames
# ---------------------------------------------------------------------------
def _nearest_frame_index(xs: np.ndarray, ys: np.ndarray, px: float, py: float) -> int:
    """Return the index in the polyline (xs, ys) whose point is closest to (px, py)."""
    dx = xs - px
    dy = ys - py
    d2 = dx * dx + dy * dy
    return int(np.argmin(d2))


def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _plan_neighbors(center_idx: int, total_frames: int, offset: int) -> Tuple[int, int]:
    """Pick (front, back) frame indices around center_idx, clamped at the endpoints.

    - Ideal offset = ``offset`` (== full ``segment_length`` in frames).
      So ``front`` is ``segment_length`` frames ahead of center, and ``back``
      is ``segment_length`` frames behind — the total span between front
      and back is therefore ``2 * segment_length``.
    - If the trajectory is shorter than ``2 * offset + 1`` we simply return
      ``(total_frames - 1, 0)`` so we span the whole thing (matches the
      requirement "不足的可以直接取最大").
    - ``front_idx > center_idx`` (moves forward along the trajectory) and
      ``back_idx < center_idx`` (moves backward), matching the intuitive
      "look-ahead" / "look-behind" reading of front/back.
    """
    last = total_frames - 1
    if total_frames <= 1:
        return center_idx, center_idx
    if total_frames <= 2 * offset + 1:
        # Degenerate short trajectory: stretch to the endpoints.
        return last, 0
    front = _clamp(center_idx + offset, 0, last)
    back = _clamp(center_idx - offset, 0, last)
    # Avoid front == center (happens when center is already at the end).
    if front == center_idx:
        front = last
    if back == center_idx:
        back = 0
    return front, back


# ---------------------------------------------------------------------------
# Farthest-point sampling over a list of 2D candidate points
# ---------------------------------------------------------------------------
def _farthest_point_sample(points_xy: List[Tuple[float, float]], k: int,
                           seed: int = 42) -> List[int]:
    """Greedy farthest-point sampling.

    Returns indices (into ``points_xy``) of at most ``k`` selected points such
    that the minimum pairwise distance is (greedily) maximized. Deterministic
    given ``seed``.
    """
    n = len(points_xy)
    if n == 0:
        return []
    if n <= k:
        return list(range(n))

    pts = np.asarray(points_xy, dtype=np.float64)
    rng = random.Random(seed)
    first = rng.randrange(n)
    selected = [first]
    # min distance from each point to the current selected set
    min_d = np.linalg.norm(pts - pts[first], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(min_d))
        if min_d[nxt] <= 0:
            break
        selected.append(nxt)
        new_d = np.linalg.norm(pts - pts[nxt], axis=1)
        min_d = np.minimum(min_d, new_d)
    return selected


# ---------------------------------------------------------------------------
# PSNR (optional)
# ---------------------------------------------------------------------------
def _calc_psnr(path_a: str, path_b: str) -> Optional[float]:
    if not _HAS_SKIMAGE:
        return None
    try:
        a = Image.open(path_a).convert("RGB")
        b = Image.open(path_b).convert("RGB")
        if a.size != b.size:
            w = min(a.width, b.width); h = min(a.height, b.height)
            a = a.resize((w, h), Image.LANCZOS)
            b = b.resize((w, h), Image.LANCZOS)
        return float(_psnr_fn(np.array(a), np.array(b), data_range=255))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Build per-trajectory XY arrays + frame-name lookup out of the npz payload
# ---------------------------------------------------------------------------
def _scene_to_traj_tables(per_traj):
    """
    Input : list[(traj_id, xs, ys, zs, yaws, frame_names)] from
            ``_payload_to_per_traj``.
    Output: dict[traj_id] -> {"xs", "ys", "frames"} (all lists of ints/floats
            of length N, N>=2). Trajectories with <2 points are skipped.
    """
    out = {}
    for tid, xs, ys, _zs, _yaws, frame_names in per_traj:
        if len(xs) < 2:
            continue
        out[str(tid)] = {
            "xs": np.asarray(xs, dtype=np.float64),
            "ys": np.asarray(ys, dtype=np.float64),
            # ``frame_names`` in the npz are e.g. "0034"; we keep both the
            # string and the int form — string for path building, int for
            # neighbor offsets / clamping.
            "frame_names": list(frame_names),
        }
    return out


# ---------------------------------------------------------------------------
# Per-trajectory center selection
# ---------------------------------------------------------------------------
def _select_centers_for_traj(this_tid: str,
                             traj_table: Dict[str, Dict],
                             cross_points,
                             max_centers: int,
                             seed: int = 42):
    """
    Collect intersection points that involve trajectory ``this_tid`` and pick
    at most ``max_centers`` of them via farthest-point sampling.

    Returns list of dicts:
        {
            "center_xy":    (x, y),
            "center_idx":   int,                    # nearest point-index in this_tid
            "center_frame": str,                    # '0043' style
            "companion_tid": str,
            "companion_idx":   int,
            "companion_frame": str,
        }
    """
    this_info = traj_table.get(this_tid)
    if this_info is None:
        return []

    # Gather candidate crossings for this trajectory.
    # cross_points entries: (x, y, tid_a, tid_b)
    raw_candidates = []
    for (px, py, ta, tb) in cross_points:
        if ta == this_tid:
            comp = tb
        elif tb == this_tid:
            comp = ta
        else:
            continue
        if comp == this_tid:
            # Self-intersection isn't handled here (self points live in a
            # different bucket in compute_intersections; we deliberately
            # ignore them to keep companion != this).
            continue
        comp_info = traj_table.get(comp)
        if comp_info is None:
            continue
        raw_candidates.append((px, py, comp))

    if not raw_candidates:
        return []

    # ------------------------------------------------------------------
    # Dedup: a pair of trajectories can intersect many times on a polyline
    # (or produce near-duplicate points). Collapse near-duplicates that
    # sit within ``dedup_eps`` in XY so farthest-point doesn't trivially
    # pick clustered points.
    # ------------------------------------------------------------------
    dedup_eps = 1e-3
    kept = []
    for px, py, comp in raw_candidates:
        dup = False
        for (qx, qy, _) in kept:
            if abs(qx - px) < dedup_eps and abs(qy - py) < dedup_eps:
                dup = True; break
        if not dup:
            kept.append((px, py, comp))

    xy_only = [(x, y) for (x, y, _) in kept]
    picked_idx = _farthest_point_sample(xy_only, max_centers, seed=seed)

    picks = []
    for i in picked_idx:
        px, py, comp = kept[i]
        ci = _nearest_frame_index(this_info["xs"], this_info["ys"], px, py)
        comp_info = traj_table[comp]
        cj = _nearest_frame_index(comp_info["xs"], comp_info["ys"], px, py)
        picks.append({
            "center_xy": (float(px), float(py)),
            "center_idx": int(ci),
            "center_frame": str(this_info["frame_names"][ci]),
            "companion_tid": comp,
            "companion_idx": int(cj),
            "companion_frame": str(comp_info["frame_names"][cj]),
        })
    return picks


# ---------------------------------------------------------------------------
# Build full 5-tuple case from a picked center
# ---------------------------------------------------------------------------
def _build_case_from_pick(scene: str,
                          this_tid: str,
                          pick: Dict,
                          traj_table: Dict[str, Dict],
                          segment_length: int,
                          base_dir: str,
                          psnr_min: float,
                          psnr_max: float,
                          enforce_psnr: bool) -> Tuple[Optional[Dict], Dict]:
    """Expand one picked center into a 5-view case. Returns (case_or_None, stats)."""
    stats = {"missing": 0, "psnr_filtered": 0}

    this_info = traj_table[this_tid]
    comp_tid = pick["companion_tid"]
    comp_info = traj_table[comp_tid]

    # Offset = full segment_length frames on each side of the center.
    # i.e. front is segment_length ahead, back is segment_length behind.
    offset = max(1, int(segment_length))

    # --- Main trajectory neighbors ---
    ci = pick["center_idx"]
    this_total = len(this_info["frame_names"])
    front1_idx, back1_idx = _plan_neighbors(ci, this_total, offset)
    center_frame = this_info["frame_names"][ci]
    front1_frame = this_info["frame_names"][front1_idx]
    back1_frame = this_info["frame_names"][back1_idx]

    # --- Companion trajectory neighbors ---
    cj = pick["companion_idx"]
    comp_total = len(comp_info["frame_names"])
    front2_idx, back2_idx = _plan_neighbors(cj, comp_total, offset)
    front2_frame = comp_info["frame_names"][front2_idx]
    back2_frame = comp_info["frame_names"][back2_idx]

    # ------------------------------------------------------------------
    # Build case payload. We keep ``directory/start/end`` pointing at
    # (main trajectory, center, front1) so that the downstream scripts
    # (step3.1 / step3.2 / step7_add_id / pipeline_generate_render_multi_gpu)
    # continue to work without change.
    # ------------------------------------------------------------------
    directory = os.path.join(scene, scene, str(this_tid))
    # Stable case id: {scene}-{main_traj}-{back1_frame}-{front1_frame}.
    #
    # The 5-view case has 4 neighbor frames, so the old
    # ``{scene}-{traj}-{start}-{end}`` convention from step7_add_id.py
    # (which assumed just a start/end pair) is no longer enough. We use the
    # **main-trajectory back1 -> front1** pair instead: within a given main
    # trajectory, the (back1, front1) pair is uniquely determined by the
    # center (and therefore by the intersection), so this stays unique across
    # all picks on that trajectory while still being short and readable.
    case_id = f"{scene}-{this_tid}-{back1_frame}-{front1_frame}"
    case: Dict = {
        "id": case_id,
        "directory": directory,
        "start": get_point_paths(scene, this_tid, ci),
        "end": get_point_paths(scene, this_tid, front1_idx),
        "scene": scene,
        "main_trajectory": str(this_tid),
        "companion_trajectory": str(comp_tid),
        "intersection_xy": list(pick["center_xy"]),
        "metadata": {
            "segment_length": segment_length,
            "neighbor_offset": offset,
            "this_traj_total_frames": this_total,
            "companion_traj_total_frames": comp_total,
            "degenerate_short": this_total <= 2 * offset + 1
                                or comp_total <= 2 * offset + 1,
        },
        # ``views`` fully describes the 5-tuple in a symmetric, easy-to-use
        # shape: 0==center, 1==front1, 2==back1, 3==front2, 4==back2.
        "views": [
            {"role": "center", "trajectory": str(this_tid), "frame": center_frame,
             **get_point_paths(scene, this_tid, ci)},
            {"role": "front1", "trajectory": str(this_tid), "frame": front1_frame,
             **get_point_paths(scene, this_tid, front1_idx)},
            {"role": "back1",  "trajectory": str(this_tid), "frame": back1_frame,
             **get_point_paths(scene, this_tid, back1_idx)},
            {"role": "front2", "trajectory": str(comp_tid), "frame": front2_frame,
             **get_point_paths(scene, comp_tid, front2_idx)},
            {"role": "back2",  "trajectory": str(comp_tid), "frame": back2_frame,
             **get_point_paths(scene, comp_tid, back2_idx)},
        ],
    }

    # ------------------------------------------------------------------
    # PSNR filter (optional).  We check center vs. each neighbor view;
    # only keep cases whose *all* pairs fall inside [psnr_min, psnr_max].
    # ------------------------------------------------------------------
    if enforce_psnr:
        center_path = os.path.join(base_dir, case["views"][0]["jpg"])
        if not os.path.exists(center_path):
            stats["missing"] += 1
            return None, stats

        pair_psnrs = []
        for v in case["views"][1:]:
            other_path = os.path.join(base_dir, v["jpg"])
            if not os.path.exists(other_path):
                stats["missing"] += 1
                return None, stats
            p = _calc_psnr(center_path, other_path)
            if p is None:
                stats["missing"] += 1
                return None, stats
            pair_psnrs.append(p)
            v["psnr_vs_center"] = p

        # Every pair must be in-range.
        if any((p < psnr_min) or (p > psnr_max) for p in pair_psnrs):
            stats["psnr_filtered"] += 1
            return None, stats

        case["metadata"]["psnr_pairs"] = pair_psnrs
        case["metadata"]["psnr_mean"] = float(np.mean(pair_psnrs))

    return case, stats


# ---------------------------------------------------------------------------
# Per-scene worker
# ---------------------------------------------------------------------------
def _process_scene(task):
    (scene_name, cache_dir, base_dir, per_scene_dir,
     segment_length, max_centers, psnr_min, psnr_max, enforce_psnr,
     overwrite, prebuilt_cross_points) = task

    out_path = os.path.join(per_scene_dir, f"{scene_name}.json")
    if (not overwrite) and os.path.exists(out_path):
        # Resume: this scene is already done.
        try:
            with open(out_path, "r") as f:
                d = json.load(f)
            return scene_name, "skip", d.get("cases", []), d.get("stats", {})
        except Exception:
            pass  # corrupt cache → fall through and recompute

    cache_path = _scene_cache_path(cache_dir, scene_name)
    if not os.path.isfile(cache_path):
        return scene_name, "missing_cache", [], {"missing_cache": 1}

    try:
        payload = _load_scene_cache(cache_path)
    except Exception as e:
        return scene_name, f"cache_error:{e}", [], {"cache_error": 1}

    per_traj = _payload_to_per_traj(payload)
    traj_table = _scene_to_traj_tables(per_traj)
    if not traj_table:
        _atomic_write_json(out_path, {"scene": scene_name, "cases": [], "stats": {}})
        return scene_name, "no_trajectories", [], {}

    # Source cross points from the pre-built CSV when available; otherwise
    # compute them on the fly from the cached polylines. Either way we feed
    # the *same* shape ``(x, y, tid_a, tid_b)`` list into the center picker
    # below, and nearest-frame snapping is always recomputed locally so the
    # final frame indices stay identical between the two paths.
    if prebuilt_cross_points is not None:
        cross_points = prebuilt_cross_points
    else:
        traj_xy_list = [(tid, info["xs"].tolist(), info["ys"].tolist())
                        for tid, info in traj_table.items()]
        ix = compute_intersections(traj_xy_list, keep_points=True)
        cross_points = ix["cross_points"]

    if not cross_points:
        _atomic_write_json(out_path, {"scene": scene_name, "cases": [],
                                      "stats": {"cross_points": 0}})
        return scene_name, "no_intersections", [], {"cross_points": 0}

    # Per-trajectory picks.
    all_cases: List[Dict] = []
    agg_stats = {"candidates": 0, "picked": 0,
                 "missing": 0, "psnr_filtered": 0,
                 "cross_points": len(cross_points),
                 "trajs_with_pick": 0}

    for tid in traj_table.keys():
        picks = _select_centers_for_traj(
            tid, traj_table, cross_points,
            max_centers=max_centers,
            seed=abs(hash((scene_name, tid))) % (2**31),
        )
        if not picks:
            continue
        agg_stats["candidates"] += len(picks)
        kept_for_this_traj = 0
        for pick in picks:
            case, pstats = _build_case_from_pick(
                scene_name, tid, pick, traj_table,
                segment_length=segment_length,
                base_dir=base_dir,
                psnr_min=psnr_min, psnr_max=psnr_max,
                enforce_psnr=enforce_psnr,
            )
            agg_stats["missing"] += pstats.get("missing", 0)
            agg_stats["psnr_filtered"] += pstats.get("psnr_filtered", 0)
            if case is not None:
                all_cases.append(case)
                kept_for_this_traj += 1
        if kept_for_this_traj > 0:
            agg_stats["trajs_with_pick"] += 1
            agg_stats["picked"] += kept_for_this_traj

    _atomic_write_json(out_path, {
        "scene": scene_name,
        "cases": all_cases,
        "stats": agg_stats,
    })
    return scene_name, "ok", all_cases, agg_stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Sample 5-view trajectory segments centered at intersections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--analysis_file", type=str, required=True,
                    help="step0/step1 analysis JSON listing every scene + its trajectories. "
                         "REQUIRED — this is the SOLE source of the scene list. "
                         "Use the 5.19 baseline (data_0519/step0_all_trajectory_0519.json). "
                         "The old 3.18/0309 default was removed: silently falling back to it "
                         "pulled in scenes outside the 0519 set and left captions empty downstream.")
    ap.add_argument("--intersections_csv", type=str, default=None,
                    help="Per-intersection CSV produced by "
                         "`step1_visualize_trajectories.py --stats_only --stats_csv`. "
                         "When provided, this script does NOT recompute "
                         "segment-segment intersections; it just reads the "
                         "(scene, x, y, traj_a, traj_b) rows and snaps each "
                         "point to the nearest integer frame on each "
                         "trajectory. Recommended.")
    ap.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR,
                    help=f"Fast-disk cache produced by step1_visualize_trajectories.py "
                         f"(default: {DEFAULT_CACHE_DIR})")
    ap.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR,
                    help=f"Root of UE_PanoramicOut for PSNR IO (default: {DEFAULT_BASE_DIR})")
    ap.add_argument("--output_dir", type=str, required=True,
                    help="Where to write per-scene JSONs and the merged JSON.")
    ap.add_argument("--output_json", type=str, default=None,
                    help="Path to the final merged JSON (default: "
                         "<output_dir>/sampled_five_view_segments{_machine<k>}.json).")

    ap.add_argument("--segment_length", type=int, default=DEFAULT_SEGMENT_LENGTH,
                    help=f"Target segment span; neighbor offset = L/2 (default: {DEFAULT_SEGMENT_LENGTH})")
    ap.add_argument("--max_centers_per_traj", type=int, default=DEFAULT_MAX_CENTERS,
                    help=f"Max 5-view cases per trajectory (default: {DEFAULT_MAX_CENTERS})")

    ap.add_argument("--psnr_min", type=float, default=DEFAULT_PSNR_MIN)
    ap.add_argument("--psnr_max", type=float, default=DEFAULT_PSNR_MAX)
    ap.add_argument("--no_psnr", action="store_true",
                    help="Skip PSNR filtering entirely (useful for debug / cache-less dry runs).")

    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--num_machines", type=int, default=1)
    ap.add_argument("--machine_id", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="Only process the first N scenes after sharding (debug).")
    ap.add_argument("--scene", type=str, default=None,
                    help="Only process a single scene (debug).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute scenes whose per-scene JSON already exists.")
    args = ap.parse_args()

    if args.num_machines < 1 or not (0 <= args.machine_id < args.num_machines):
        raise SystemExit(
            f"bad --num_machines/--machine_id: {args.num_machines}/{args.machine_id}")

    if (not args.no_psnr) and (not _HAS_SKIMAGE):
        print("[Init] skimage not installed; falling back to --no_psnr.",
              file=sys.stderr)
        args.no_psnr = True
    enforce_psnr = not args.no_psnr

    args.base_dir = _resolve_base_path(args.base_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    per_scene_dir = os.path.join(args.output_dir, "per_scene")
    os.makedirs(per_scene_dir, exist_ok=True)

    with open(args.analysis_file, "r") as f:
        data = json.load(f)
    scenes = sorted(list(data.keys()))
    if args.scene:
        scenes = [s for s in scenes if s == args.scene]

    # Machine-level sharding (deterministic round-robin).
    if args.num_machines > 1:
        scenes = [s for i, s in enumerate(scenes) if i % args.num_machines == args.machine_id]
        print(f"[shard] machine {args.machine_id}/{args.num_machines}: "
              f"{len(scenes)} scenes assigned")

    if args.limit > 0:
        scenes = scenes[: args.limit]

    # Optional: pre-built per-intersection CSV. When supplied we skip the
    # O(N^2) per-scene segment-segment check entirely and just look up rows
    # by scene name.
    cross_by_scene: Optional[Dict[str, List[Tuple[float, float, str, str]]]] = None
    if args.intersections_csv:
        if not os.path.isfile(args.intersections_csv):
            raise SystemExit(
                f"--intersections_csv not found: {args.intersections_csv}")
        cross_by_scene = _load_intersections_csv(args.intersections_csv)
        print(f"[Init] loaded intersections CSV: "
              f"{sum(len(v) for v in cross_by_scene.values())} rows "
              f"across {len(cross_by_scene)} scenes")

    print(f"[Init] scenes to process: {len(scenes)} | workers={args.workers} "
          f"| PSNR={'on' if enforce_psnr else 'off'} "
          f"| use_csv={'yes' if cross_by_scene is not None else 'no'}")

    tasks = [(s, args.cache_dir, args.base_dir, per_scene_dir,
              args.segment_length, args.max_centers_per_traj,
              args.psnr_min, args.psnr_max, enforce_psnr,
              args.overwrite,
              # pre-built cross points for this scene, or None to recompute
              cross_by_scene.get(s) if cross_by_scene is not None else None)
             for s in scenes]

    all_cases: List[Dict] = []
    aggregate: Dict[str, int] = {
        "candidates": 0, "picked": 0, "missing": 0,
        "psnr_filtered": 0, "cross_points": 0, "trajs_with_pick": 0,
        "scenes_ok": 0, "scenes_skip": 0, "scenes_fail": 0,
        "scenes_no_intersection": 0, "scenes_missing_cache": 0,
    }

    def _absorb(res):
        scene_name, status, cases, stats = res
        if status == "ok":
            aggregate["scenes_ok"] += 1
        elif status == "skip":
            aggregate["scenes_skip"] += 1
        elif status == "missing_cache":
            aggregate["scenes_missing_cache"] += 1
        elif status == "no_intersections":
            aggregate["scenes_no_intersection"] += 1
        else:
            aggregate["scenes_fail"] += 1
            print(f"[{scene_name}] {status}", file=sys.stderr)
        for k, v in stats.items():
            if isinstance(v, (int, float)) and k in aggregate:
                aggregate[k] += v
        all_cases.extend(cases)

    if args.workers <= 1:
        for t in tqdm(tasks, total=len(tasks), desc="scenes",
                      dynamic_ncols=True):
            try:
                _absorb(_process_scene(t))
            except Exception as e:
                print(f"[{t[0]}] crash: {e}\n{traceback.format_exc()}",
                      file=sys.stderr)
                aggregate["scenes_fail"] += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_process_scene, t): t[0] for t in tasks}
            pbar = tqdm(total=len(tasks), desc="scenes", dynamic_ncols=True)
            for fut in as_completed(futs):
                sname = futs[fut]
                try:
                    _absorb(fut.result())
                except Exception as e:
                    print(f"[{sname}] crash: {e}\n{traceback.format_exc()}",
                          file=sys.stderr)
                    aggregate["scenes_fail"] += 1
                if _HAS_TQDM:
                    pbar.set_postfix(cases=len(all_cases),
                                     ok=aggregate["scenes_ok"],
                                     skip=aggregate["scenes_skip"],
                                     fail=aggregate["scenes_fail"])
                    pbar.update(1)
            if _HAS_TQDM:
                pbar.close()

    # ------------------------------------------------------------------
    # Save merged JSON
    # ------------------------------------------------------------------
    if args.num_machines > 1:
        default_name = f"sampled_five_view_segments_machine{args.machine_id}.json"
    else:
        default_name = "sampled_five_view_segments.json"
    merged_path = args.output_json or os.path.join(args.output_dir, default_name)
    _atomic_write_json(merged_path, {"cases": all_cases,
                                     "summary": aggregate})
    print(f"\n[Done] {len(all_cases)} cases → {merged_path}")
    print("       summary:", json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
