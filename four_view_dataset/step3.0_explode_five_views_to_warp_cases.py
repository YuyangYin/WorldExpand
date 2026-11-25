#!/usr/bin/env python3
"""
step3.0_explode_five_views_to_warp_cases.py
===========================================

Convert the 5-view JSON produced by
``step2.3_merge_five_view_segments.py`` into a **tuple-v1** schema, where

    1 case = 1 intersection = 5 positions
            (center + front1 + back1 + front2 + back2)

Each "position" only stores ``trajectory + frame``. The jpg / exr / json
paths are NOT serialized — they are reconstructed on the fly by
``paths_for(scene, traj, frame)`` using the canonical UE layout
``{scene}/{scene}/{traj}/{frame}/{frame}.{ext}``. This keeps the JSON
compact *and* human-readable at the same time (~200 B per case, no path
duplication).

The HY-World pipeline (``pipeline_generate_render_multi_gpu.py``) is
taught to recognize this schema and explode each tuple into 4 warp tasks
``(start=center, end=neighbor)`` internally — so step3.0's output JSON is
fed directly to pipeline, no extra flatten step.

Tuple-v1 case layout
--------------------
    {
      "intersection_id":      "Brick_Hs_11-17-0117",
      "scene":                "Brick_Hs_11",
      "main_trajectory":      "17",    # center + front1/back1 live here
      "companion_trajectory": "21",    # front2/back2 live here
      "center_frame":         "0117",

      "center": { "trajectory": "17", "frame": "0117" },
      "front1": { "trajectory": "17", "frame": "0191" },
      "back1":  { "trajectory": "17", "frame": "0000" },
      "front2": { "trajectory": "21", "frame": "0191" },
      "back2":  { "trajectory": "21", "frame": "0000" }
    }

Usage
-----
    python step3.0_explode_five_views_to_warp_cases.py \
        --input_json  sampled_five_view_segments.json \
        --output_json sampled_five_view_segments_tuples.json

Then feed straight to pipeline:

    python pipeline_generate_render_multi_gpu.py \
        --data_json  sampled_five_view_segments_tuples.json \
        --data_root  /path/to/UE_PanoramicOut \
        --output_dir <warp_output_root> \
        --num_gpus 8 --render_engine pytorch3d --shuffle
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

# Shared schema constants + resolvers (single source of truth across
# step3.0 / step3.3 / pipeline / downstream visualizers).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools.tupleio import (  # noqa: E402
    NEIGHBOR_ROLES, COMPANION_ROLES, ALL_POSITIONS,
    DIR_RULE, CANONICAL_KEYS, CANONICAL_FNAMES, FLAGS_LEGEND,
    paths_for, sub_case_basename,
)


def is_tuple_case(c: Dict) -> bool:
    """Duck-type: a tuple-v1 case has all 5 named positions as dicts."""
    if not isinstance(c, dict):
        return False
    return all(isinstance(c.get(k), dict) for k in ALL_POSITIONS)


# Example tuple used for the self-describing JSON header — lets humans grep
# one place to see what the schema looks like without loading 150 MB.
EXAMPLE_TUPLE: Dict = {
    "scene":                "Brick_Hs_11",
    "main_trajectory":      "17",
    "companion_trajectory": "21",
    "center": {"trajectory": "17", "frame": "0117"},
    "front1": {"trajectory": "17", "frame": "0191"},
    "back1":  {"trajectory": "17", "frame": "0000"},
    "front2": {"trajectory": "21", "frame": "0191"},
    "back2":  {"trajectory": "21", "frame": "0000"},
    "_derived": {
        "intersection_id": "Brick_Hs_11-17-0117",
        "center_frame":    "0117",
        "sub_case_dirs": {
            role: sub_case_basename("Brick_Hs_11", "17", role, "0117",
                                    {"front1": "0191", "back1": "0000",
                                     "front2": "0191", "back2": "0000"}[role])
            for role in NEIGHBOR_ROLES
        },
    },
}


def expand_tuple_to_sub_cases(tup: Dict) -> List[Dict]:
    """Turn one tuple-v1 case into 4 ``(start=center, end=neighbor)`` subs.

    Each sub has the exact fields pipeline_generate_render_multi_gpu reads:
    ``id / directory / start{jpg,exr,json} / end{jpg,exr,json}`` plus a few
    provenance fields (``role / scene / main_trajectory / ...``) that the
    pipeline ignores but downstream collectors can use.
    """
    scene = tup["scene"]
    main_traj = str(tup["main_trajectory"])
    comp_traj = str(tup.get("companion_trajectory", ""))
    center = tup["center"]
    center_frame = str(center["frame"])
    parent_id = tup.get("intersection_id") or f"{scene}-{main_traj}-{center_frame}"

    out: List[Dict] = []
    for role in NEIGHBOR_ROLES:
        pos = tup.get(role)
        if not isinstance(pos, dict):
            # This neighbor doesn't exist in this tuple; skip.
            continue
        v_traj = str(pos.get("trajectory",
                             comp_traj if role in COMPANION_ROLES else main_traj))
        v_frame = str(pos["frame"])

        case_id = f"{parent_id}__{role}"
        directory = f"{scene}/{scene}/{main_traj}/{case_id}"
        out.append({
            "id":        case_id,
            "directory": directory,
            "start":     paths_for(scene, str(center.get("trajectory", main_traj)),
                                   center_frame),
            "end":       paths_for(scene, v_traj, v_frame),
            "parent_case_id":       parent_id,
            "role":                 role,
            "scene":                scene,
            "main_trajectory":      main_traj,
            "companion_trajectory": comp_traj,
            "center_frame":         center_frame,
            "view_frame":           v_frame,
        })
    return out


# ---------------------------------------------------------------------------
# Conversion: 5-view parent case  ->  tuple-v1 case
# ---------------------------------------------------------------------------
def _views_by_role(views: List[Dict]) -> Dict[str, Dict]:
    return {v.get("role"): v for v in views if v.get("role")}


def _frame_of(view: Dict) -> Optional[str]:
    """Prefer explicit ``frame`` field; fall back to jpg path tail."""
    if view.get("frame"):
        return str(view["frame"])
    jpg = view.get("jpg", "") or ""
    f = os.path.basename(os.path.dirname(jpg))
    return f or None


def _traj_of(view: Dict, default: str) -> str:
    t = view.get("trajectory")
    return str(t) if t not in (None, "") else default


def parent_to_tuple(p: Dict) -> Optional[Dict]:
    """Build one tuple-v1 case from one step2.3 5-view parent case."""
    by_role = _views_by_role(p.get("views", []))
    center = by_role.get("center")
    if center is None:
        return None

    scene = p.get("scene") or p.get("directory", "").split("/")[0]
    main_traj = str(p.get("main_trajectory",
                          p.get("directory", "").split("/")[-1]))
    comp_traj = str(p.get("companion_trajectory", ""))
    center_frame = _frame_of(center)
    if center_frame is None:
        return None

    tup: Dict = {
        "intersection_id":      f"{scene}-{main_traj}-{center_frame}",
        "scene":                scene,
        "main_trajectory":      main_traj,
        "companion_trajectory": comp_traj,
        "center_frame":         center_frame,
        "center": {
            "trajectory": _traj_of(center, main_traj),
            "frame":      center_frame,
        },
    }
    for role in NEIGHBOR_ROLES:
        v = by_role.get(role)
        if v is None:
            continue
        f = _frame_of(v)
        if f is None:
            continue
        default_traj = comp_traj if role in COMPANION_ROLES else main_traj
        tup[role] = {
            "trajectory": _traj_of(v, default_traj),
            "frame":      f,
        }
    return tup


def convert_parents(parents: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    tuples: List[Dict] = []
    counters = {"ok": 0, "no_center": 0, "missing_roles_total": 0}
    for p in parents:
        tup = parent_to_tuple(p)
        if tup is None:
            counters["no_center"] += 1
            continue
        # Count how many neighbors are missing (rare, but worth reporting).
        counters["missing_roles_total"] += sum(
            1 for r in NEIGHBOR_ROLES if r not in tup
        )
        tuples.append(tup)
        counters["ok"] += 1
    return tuples, counters


# ---------------------------------------------------------------------------
# Output writer: outer shell indented, one tuple per line (grep / head friendly)
# ---------------------------------------------------------------------------
def _write_json_pretty_jsonl(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    cases = payload.get("cases", [])
    summary = payload.get("summary", {})
    with open(tmp, "w") as f:
        f.write("{\n")
        f.write('  "cases": [\n')
        n = len(cases)
        for i, c in enumerate(cases):
            # One tuple per line; spaces after separators keep it readable.
            line = json.dumps(c, ensure_ascii=False)
            sep = "," if i < n - 1 else ""
            f.write(f"    {line}{sep}\n")
        f.write("  ],\n")
        f.write('  "summary": '
                + json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        f.write("}\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Convert step2.3 5-view JSON to tuple-v1 "
                    "(1 intersection = 1 tuple with 5 positions).")
    ap.add_argument("--input_json", required=True,
                    help="sampled_five_view_segments.json from step2.3.")
    ap.add_argument("--output_json", required=True,
                    help="Tuple-v1 JSON path.")
    ap.add_argument("--max_cases", type=int, default=0,
                    help="If >0, keep only the first N parents (debug).")
    args = ap.parse_args()

    with open(args.input_json, "r") as f:
        data = json.load(f)
    parents = data.get("cases", [])
    if args.max_cases > 0:
        parents = parents[: args.max_cases]
    print(f"[init] loaded {len(parents)} 5-view parent cases")

    tuples, stats = convert_parents(parents)
    print(f"[done] {stats['ok']} tuples emitted, "
          f"{stats['no_center']} parents dropped (no center), "
          f"{stats['missing_roles_total']} (tuple, role) slots missing")

    summary = {
        "schema":         "tuple-v1",
        "description":    (
            "1 case = 1 intersection with 5 positions "
            "(center + front1 + back1 + front2 + back2). Each position "
            "only stores trajectory+frame; pipeline reconstructs "
            "jpg/exr/json on the fly via paths_for(scene, traj, frame)."
        ),
        "dir_rule":       DIR_RULE,
        "example_case":   EXAMPLE_TUPLE,
        "neighbor_roles": list(NEIGHBOR_ROLES),
        "companion_roles": sorted(COMPANION_ROLES),
        "counters":       stats,
        "source":         os.path.abspath(args.input_json),
    }
    _write_json_pretty_jsonl(args.output_json,
                             {"cases": tuples, "summary": summary})
    print(f"[done] -> {args.output_json}")


if __name__ == "__main__":
    main()
