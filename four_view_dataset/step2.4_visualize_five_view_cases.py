#!/usr/bin/env python3
"""
Visualize sampled five-view cases.

For each sampled case produces a 6-panel figure:

    +------------+------------+------------+
    |  center    |  front1    |  back1     |
    +------------+------------+------------+
    |  front2    |  back2     | TRAJ MAP   |
    +------------+------------+------------+

- Panels 1..5 are the five ERP panoramas (center, front1, back1, front2, back2).
- Panel 6 is a *local* XY map of the two participating trajectories, with the
  intersection marked red and the 5 picked frames marked as color-coded dots
  matching the border color of the corresponding panel (so you can trivially
  cross-reference frame <-> position).

Reads the JSON produced by
``step2.2_sample_five_view_segments.py`` / ``step2.3_merge_five_view_segments.py``
(expects a top-level ``cases`` list with ``views`` entries and
``intersection_xy`` + ``main_trajectory`` + ``companion_trajectory``).

Trajectory polylines are read from the npz cache emitted by
``step1_visualize_trajectories.py --cache_only`` (same cache that drives step2.2).

Usage
-----
    python step2.4_visualize_five_view_cases.py \\
        --cases_json /path/to/sampled_five_view_segments.json \\
        --cache_dir  @repo/outputs/dataset/trajectory_cache \\
        --base_dir   @repo/data/UE_PanoramicOut \\
        --out_dir    ./vis_five_view \\
        --num_samples 100 \\
        --workers 8

Options
-------
- ``--num_samples N``   Randomly sample N cases (seeded); pass 0 for all.
                        Default: 100.
- ``--seed``            Sampling seed (default 0).
- ``--scene NAME``      Only visualize cases in this scene.
- ``--cases_per_scene K`` Ensure at most K cases per scene (diversity).
- ``--workers``         Parallel figure rendering.
- ``--overwrite``       Redo figures even if the output PNG already exists
                        (default: skip = resume-friendly).
- ``--include_traj_labels``  Annotate traj ids on the map panel.
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
import os
import random
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

    def tqdm(iterable=None, total=None, desc=None, **kw):  # type: ignore
        return iterable if iterable is not None else iter(())

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

    from step1_visualize_trajectories import (  # noqa: E402
    _load_scene_cache,
    _scene_cache_path,
    _payload_to_per_traj,
    _resolve_base_path,
)

DEFAULT_CACHE_DIR = _release_path("@repo/outputs/dataset/trajectory_cache")
DEFAULT_BASE_DIR = _release_path("@repo/data/UE_PanoramicOut")

ROLE_COLOR = {
    "center": "#ffcc00",  # yellow
    "front1": "#1f77b4",  # blue
    "back1":  "#2ca02c",  # green
    "front2": "#d62728",  # red
    "back2":  "#9467bd",  # purple
}
ROLE_ORDER = ["center", "front1", "back1", "front2", "back2"]


# ---------------------------------------------------------------------------
# Scene cache -> per-traj tables (xs, ys, frame_names)
# ---------------------------------------------------------------------------
def _load_scene_tables(cache_dir: str, scene: str) -> Optional[Dict[str, Dict]]:
    cpath = _scene_cache_path(cache_dir, scene)
    if not os.path.isfile(cpath):
        return None
    try:
        payload = _load_scene_cache(cpath)
    except Exception as e:
        print(f"[{scene}] cache load failed: {e}", file=sys.stderr)
        return None
    per_traj = _payload_to_per_traj(payload)
    out: Dict[str, Dict] = {}
    for tid, xs, ys, _zs, _yaws, frame_names in per_traj:
        out[str(tid)] = {
            "xs": np.asarray(xs, dtype=np.float64),
            "ys": np.asarray(ys, dtype=np.float64),
            "frame_names": list(frame_names),
        }
    return out


# ---------------------------------------------------------------------------
# Image loading (pano ERP can be huge; downscale for plotting)
# ---------------------------------------------------------------------------
def _load_pano(base_dir: str, rel_jpg: str, target_w: int = 1024) -> Optional[np.ndarray]:
    path = os.path.join(base_dir, rel_jpg)
    if not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGB")
        if img.width > target_w:
            new_h = int(round(img.height * target_w / img.width))
            img = img.resize((target_w, new_h), Image.BILINEAR)
        return np.asarray(img)
    except Exception as e:
        print(f"[pano load] {path}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Figure renderer
# ---------------------------------------------------------------------------
def _border_axes(ax, color: str, lw: float = 3.0):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)


def _render_map_panel(ax,
                      case: Dict,
                      traj_tables: Dict[str, Dict],
                      include_traj_labels: bool = True):
    """Draw local XY map with both participating trajectories + 5 frame markers."""
    main_tid = str(case.get("main_trajectory", ""))
    comp_tid = str(case.get("companion_trajectory", ""))

    main_info = traj_tables.get(main_tid)
    comp_info = traj_tables.get(comp_tid)

    # Plot the two polylines (use thin, semi-transparent lines for clarity).
    if main_info is not None:
        ax.plot(main_info["xs"], main_info["ys"], "-",
                color="#444444", linewidth=1.0, alpha=0.75,
                label=f"main={main_tid}")
    if comp_info is not None:
        ax.plot(comp_info["xs"], comp_info["ys"], "--",
                color="#888888", linewidth=1.0, alpha=0.75,
                label=f"comp={comp_tid}")

    # Intersection point.
    ix_xy = case.get("intersection_xy")
    if ix_xy and len(ix_xy) == 2:
        ax.scatter([ix_xy[0]], [ix_xy[1]], marker="x",
                   color="red", s=120, linewidths=2.2,
                   zorder=5, label="intersection")

    # Frame markers (role-colored dots).
    marker_pts = []
    for view in case.get("views", []):
        role = view.get("role")
        tid = str(view.get("trajectory", ""))
        frame = view.get("frame", "")
        info = traj_tables.get(tid)
        if info is None:
            continue
        try:
            idx = info["frame_names"].index(frame)
        except ValueError:
            try:
                idx = int(frame)
            except ValueError:
                continue
        if not (0 <= idx < len(info["xs"])):
            continue
        x = float(info["xs"][idx]); y = float(info["ys"][idx])
        marker_pts.append((x, y))
        color = ROLE_COLOR.get(role, "black")
        ax.scatter([x], [y], s=140,
                   facecolors=color, edgecolors="black", linewidths=1.2,
                   zorder=6)
        ax.annotate(role, (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color="black",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor=color,
                              linewidth=1.0, alpha=0.9))

    # Local zoom: fit to the 5 markers + intersection, with margin.
    anchors = list(marker_pts)
    if ix_xy and len(ix_xy) == 2:
        anchors.append((float(ix_xy[0]), float(ix_xy[1])))
    if anchors:
        xs = np.array([p[0] for p in anchors])
        ys = np.array([p[1] for p in anchors])
        pad = 0.25 * max(float(np.ptp(xs)), float(np.ptp(ys)), 100.0)
        ax.set_xlim(xs.min() - pad, xs.max() + pad)
        ax.set_ylim(ys.min() - pad, ys.max() + pad)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("local trajectory map", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    if include_traj_labels:
        ax.legend(loc="lower right", fontsize=7, framealpha=0.85)


def _render_case(case: Dict,
                 traj_tables: Dict[str, Dict],
                 base_dir: str,
                 out_path: str,
                 pano_target_w: int = 1024,
                 include_traj_labels: bool = True) -> bool:
    """Render one 6-panel figure; return True on success."""
    # Gather panels in a fixed role order so we always show the same layout.
    views_by_role = {v.get("role"): v for v in case.get("views", [])}

    fig, axes = plt.subplots(2, 3, figsize=(18, 7.2))
    axes_flat = axes.flat

    for i, role in enumerate(ROLE_ORDER):
        ax = axes_flat[i]
        view = views_by_role.get(role)
        img = None
        frame = ""
        tid = ""
        if view is not None:
            frame = view.get("frame", "")
            tid = str(view.get("trajectory", ""))
            img = _load_pano(base_dir, view.get("jpg", ""), target_w=pano_target_w)

        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f"{role}\n(missing)", ha="center", va="center",
                    fontsize=14, color="red", transform=ax.transAxes)

        psnr_s = ""
        if view is not None and view.get("psnr_vs_center") is not None:
            psnr_s = f"  psnr={view['psnr_vs_center']:.2f}"
        title = f"{role}  traj={tid} f={frame}{psnr_s}"
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        _border_axes(ax, ROLE_COLOR.get(role, "black"), lw=3.0)

    # Panel 6: local trajectory map.
    map_ax = axes_flat[5]
    _render_map_panel(map_ax, case, traj_tables,
                      include_traj_labels=include_traj_labels)

    # Super-title: case id / main-comp / intersection
    cid = case.get("id") or case.get("scene", "?") + "/" + str(case.get("main_trajectory", "?"))
    ix_xy = case.get("intersection_xy", [None, None])
    ix_s = f"({ix_xy[0]:.1f}, {ix_xy[1]:.1f})" if ix_xy and ix_xy[0] is not None else ""
    fig.suptitle(
        f"{cid}  |  main={case.get('main_trajectory','?')}  "
        f"comp={case.get('companion_trajectory','?')}  "
        f"intersection={ix_s}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = f"{out_path}.tmp.png"
    try:
        fig.savefig(tmp, dpi=110, bbox_inches="tight")
    finally:
        plt.close(fig)
    os.replace(tmp, out_path)
    return True


# ---------------------------------------------------------------------------
# Worker (must be top-level for ProcessPoolExecutor)
# ---------------------------------------------------------------------------
def _worker_render(task):
    (case, cache_dir, base_dir, out_path,
     pano_target_w, include_traj_labels, overwrite) = task
    try:
        if (not overwrite) and os.path.exists(out_path):
            return out_path, "skip", None
        scene = case.get("scene") or case.get("directory", "").split("/")[0]
        traj_tables = _load_scene_tables(cache_dir, scene) or {}
        ok = _render_case(case, traj_tables, base_dir, out_path,
                          pano_target_w=pano_target_w,
                          include_traj_labels=include_traj_labels)
        return out_path, "ok" if ok else "fail", None
    except Exception as e:
        return out_path, "error", f"{e}\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _case_id(c: Dict) -> str:
    if c.get("id"):
        return c["id"]
    # Fallback: synthesize the same id rule that step2.2 now writes,
    # i.e. {scene}-{main_traj}-{back1_frame}-{front1_frame}. The main
    # trajectory's back1 -> front1 pair uniquely identifies the case
    # (it's determined by the center, which is determined by the
    # intersection).
    scene = c.get("scene") or c.get("directory", "unknown").split("/")[0]
    mt = str(c.get("main_trajectory", "?"))
    views = c.get("views", [])
    back1 = next((v for v in views if v.get("role") == "back1"), None)
    front1 = next((v for v in views if v.get("role") == "front1"), None)
    bf = back1.get("frame", "0000") if back1 else "0000"
    ff = front1.get("frame", "0000") if front1 else "0000"
    return f"{scene}-{mt}-{bf}-{ff}"


def main():
    ap = argparse.ArgumentParser(
        description="Visualize five-view cases as 6-panel figures.")
    ap.add_argument("--cases_json", required=True,
                    help="Sampled JSON from step2.2_sample_five_view_segments.py "
                         "or step2.3_merge_five_view_segments.py")
    ap.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR,
                        help="Scene npz cache dir (produced by step1_visualize_trajectories.py).")
    ap.add_argument("--base_dir", default=DEFAULT_BASE_DIR,
                    help="UE_PanoramicOut root (for loading pano jpgs).")
    ap.add_argument("--out_dir", required=True,
                    help="Where to write <case_id>.png figures.")
    ap.add_argument("--num_samples", type=int, default=100,
                    help="Randomly sample N cases (0 = all). Default: 100.")
    ap.add_argument("--cases_per_scene", type=int, default=0,
                    help="If >0, cap the number of cases per scene (diversity).")
    ap.add_argument("--scene", default=None,
                    help="Only visualize cases from this scene.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pano_width", type=int, default=1024,
                    help="Downscale panoramas to this width before plotting.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--include_traj_labels", action="store_true", default=True)
    args = ap.parse_args()

    args.base_dir = _resolve_base_path(args.base_dir)

    with open(args.cases_json, "r") as f:
        blob = json.load(f)
    cases: List[Dict] = blob.get("cases", [])
    print(f"[init] {len(cases)} cases loaded from {args.cases_json}")

    if args.scene:
        cases = [c for c in cases if (c.get("scene") == args.scene or
                                      c.get("directory", "").split("/")[0] == args.scene)]
        print(f"[init] scene={args.scene}: {len(cases)} cases")

    # Optional diversity cap: at most K cases per scene.
    if args.cases_per_scene > 0:
        by_scene = defaultdict(list)
        for c in cases:
            s = c.get("scene") or c.get("directory", "").split("/")[0]
            by_scene[s].append(c)
        capped: List[Dict] = []
        rng = random.Random(args.seed)
        for s, lst in by_scene.items():
            rng.shuffle(lst)
            capped.extend(lst[: args.cases_per_scene])
        cases = capped
        print(f"[init] capped to {args.cases_per_scene}/scene -> {len(cases)} cases")

    # Random sample.
    if args.num_samples > 0 and len(cases) > args.num_samples:
        rng = random.Random(args.seed)
        cases = rng.sample(cases, args.num_samples)
        print(f"[init] sampled {len(cases)} cases (seed={args.seed})")

    os.makedirs(args.out_dir, exist_ok=True)

    tasks = []
    for c in cases:
        cid = _case_id(c)
        out_path = os.path.join(args.out_dir, f"{cid}.png")
        tasks.append((c, args.cache_dir, args.base_dir, out_path,
                      args.pano_width, args.include_traj_labels,
                      args.overwrite))

    n_ok = n_skip = n_fail = 0
    if args.workers <= 1:
        it = tqdm(tasks, total=len(tasks), desc="render")
        for t in it:
            out_path, status, err = _worker_render(t)
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
                print(f"[fail] {out_path}: {err}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_worker_render, t): t[3] for t in tasks}
            pbar = tqdm(total=len(tasks), desc="render", dynamic_ncols=True)
            for fut in as_completed(futs):
                try:
                    out_path, status, err = fut.result()
                except Exception as e:
                    status = "error"; err = str(e); out_path = futs[fut]
                if status == "ok":
                    n_ok += 1
                elif status == "skip":
                    n_skip += 1
                else:
                    n_fail += 1
                    print(f"[fail] {out_path}: {err}", file=sys.stderr)
                if _HAS_TQDM:
                    pbar.set_postfix(ok=n_ok, skip=n_skip, fail=n_fail)
                    pbar.update(1)
            if _HAS_TQDM:
                pbar.close()

    print(f"\n[done] ok={n_ok}  skip={n_skip}  fail={n_fail}  out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
