#!/usr/bin/env python3
"""
tupleio.py — shared helpers for the 5-view / warp pipeline schemas.

One source of truth for:

* NEIGHBOR_ROLES / COMPANION_ROLES / ALL_POSITIONS
* ``paths_for(scene, traj, frame)``     — UE per-frame layout
* ``sub_case_basename(scene, main_traj, role, cf, vf)``
                                        — warp pipeline output directory
* CANONICAL_FILES (ordered list)        — the 4 files that may live in a
                                          sub-case dir, in the exact order
                                          used by the flags bitmask
* ``flags_encode / flags_decode``       — 4-char ``'0'/'1'`` string
* ``expand_warp_output(flags_str,
      scene, main, cf, role, vf)``      — reconstruct the old verbose
                                          ``warp_output`` dict on the fly
                                          (for downstream callers that
                                          still want ``wo['warp_png']``)
* ``view_warp_files(view, role, tup)``  — schema-agnostic accessor that
                                          works for both the old 798 MB
                                          verbose JSON and the new flags
                                          JSON

Keep in sync with:
    step3.0_explode_five_views_to_warp_cases.py
    step3.3_collect_warp_outputs.py
    HY-World-1.5/pipeline_generate_render_multi_gpu.py::_expand_tuple_case
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Role constants (5 positions per intersection).
# ---------------------------------------------------------------------------
NEIGHBOR_ROLES: Tuple[str, ...] = ("front1", "back1", "front2", "back2")
COMPANION_ROLES: frozenset = frozenset({"front2", "back2"})
ALL_POSITIONS: Tuple[str, ...] = ("center",) + NEIGHBOR_ROLES

# ---------------------------------------------------------------------------
# Canonical files a sub-case dir MAY contain, in the *exact order* used by
# the flags bitmask. Changing this list is a breaking change — old JSONs'
# flags would misalign. Add new entries ONLY at the end.
#
# NOTE: ``*_pcd`` / ``depth_pcd`` family files produced by the pytorch3d
# branch of the warp pipeline are intentionally NOT tracked here. Training
# only consumes ``warp / mask / concat`` (+ optional ``global_mesh``).
# ---------------------------------------------------------------------------
CANONICAL_FILES: Tuple[Tuple[str, str], ...] = (
    ("warp_png",              "warp.png"),
    ("mask_png",              "mask.png"),
    ("concat_png",            "concat.png"),
    ("global_mesh_ply",       "global_mesh.ply"),
)
CANONICAL_KEYS:    Tuple[str, ...] = tuple(k for k, _ in CANONICAL_FILES)
CANONICAL_FNAMES:  Tuple[str, ...] = tuple(n for _, n in CANONICAL_FILES)
N_CANONICAL: int = len(CANONICAL_FILES)

# ``warp.png`` must exist for a sub-case to count as ``status=ok``.
# Indexed into the bitmask.
OK_PROBE_INDICES: Tuple[int, ...] = (0,)  # warp_png

# Mapping kept around for scripts that still want a dict form.
CANONICAL_FILES_DICT: Dict[str, str] = dict(CANONICAL_FILES)

# ---------------------------------------------------------------------------
# UE frame layout + warp sub-case directory rule.
# ---------------------------------------------------------------------------
def paths_for(scene: str, traj: str, frame: str) -> Dict[str, str]:
    """Canonical relative paths for one UE frame.

    Layout: ``{scene}/{scene}/{traj}/{frame}/{frame}.{ext}``.
    """
    rel = f"{scene}/{scene}/{traj}/{frame}"
    return {
        "jpg":  f"{rel}/{frame}.jpg",
        "exr":  f"{rel}/{frame}.exr",
        "json": f"{rel}/{frame}.json",
    }


def sub_case_basename(scene: str, main_traj: str, role: str,
                      center_frame: str, view_frame: str) -> str:
    """Warp pipeline's sub-case directory basename.

    Layout (relative to ``--warp_output_root``):

        {scene}/{scene}/{main_traj}/{scene}-{main_traj}-{cf}__{role}/{cf}_{vf}
    """
    case_id = f"{scene}-{main_traj}-{center_frame}__{role}"
    directory = f"{scene}/{scene}/{main_traj}/{case_id}"
    return f"{directory}/{center_frame}_{view_frame}"


# Human-readable rule block — embedded at the top of every tuple / warp
# JSON so the file is self-describing.
DIR_RULE: Dict[str, str] = {
    "frame_file":   "{scene}/{scene}/{traj}/{frame}/{frame}.{jpg|exr|json}",
    "sub_case_dir": "{warp_root}/{scene}/{scene}/{main}/"
                    "{scene}-{main}-{cf}__{role}/{cf}_{vf}",
    "role_trajectory": "{'center':'main','front1':'main','back1':'main',"
                       "'front2':'companion','back2':'companion'}",
    "intersection_id": "{scene}-{main}-{cf}",
}


# ---------------------------------------------------------------------------
# Flags bitmask — 9-char ``'0'/'1'`` string, aligned with CANONICAL_FILES.
# ---------------------------------------------------------------------------
def flags_encode(present: Iterable[bool]) -> str:
    """``[True, False, ...]`` -> ``"101...."`` (length = N_CANONICAL)."""
    out = []
    for i, p in enumerate(present):
        if i >= N_CANONICAL:
            break
        out.append("1" if p else "0")
    while len(out) < N_CANONICAL:
        out.append("0")
    return "".join(out)


def flags_from_filenames(filenames_on_disk: Iterable[str]) -> str:
    """Set of filenames present in a sub-case dir -> flags bitmask."""
    present_set = set(filenames_on_disk)
    return flags_encode(fn in present_set for fn in CANONICAL_FNAMES)


def flags_is_ok(flags: str) -> bool:
    """``status=ok`` iff ``warp.png`` exists."""
    if not flags:
        return False
    for i in OK_PROBE_INDICES:
        if i < len(flags) and flags[i] == "1":
            return True
    return False


def flags_to_dict(flags: str) -> Dict[str, bool]:
    """``"1101"`` -> ``{warp_png: True, mask_png: True, ...}``."""
    return {
        k: (i < len(flags) and flags[i] == "1")
        for i, k in enumerate(CANONICAL_KEYS)
    }


def flags_present_keys(flags: str) -> List[str]:
    return [k for k, p in flags_to_dict(flags).items() if p]


FLAGS_LEGEND: str = (
    f"{N_CANONICAL}-char string; position i is '1' iff CANONICAL_FILES[i] "
    "exists in case_dir. Order: "
    + ", ".join(f"{i}={k}" for i, k in enumerate(CANONICAL_KEYS))
    + ". status='ok' iff flags[0]=='1' (warp.png)."
)


# ---------------------------------------------------------------------------
# Verbose ``warp_output`` reconstruction (lazy / on-demand).
#
# Old 798 MB JSON stored this dict verbatim on every view. New JSON stores
# only the flags string; this helper rebuilds the dict on the fly so
# downstream callers that still want ``wo["warp_png"]`` keep working.
# ---------------------------------------------------------------------------
def expand_warp_output(flags: str,
                       scene: str, main_traj: str, role: str,
                       center_frame: str, view_frame: str
                       ) -> Dict[str, str]:
    """Reconstruct the old-style ``warp_output`` dict from a flags string.

    ``case_dir`` + ``status`` are always set. Each canonical key present
    in flags is added with its full relative path ``{case_dir}/{fname}``.
    """
    case_dir = sub_case_basename(scene, main_traj, role,
                                 center_frame, view_frame)
    out: Dict[str, str] = {
        "case_dir": case_dir,
        "status":   "ok" if flags_is_ok(flags) else "missing",
    }
    for i, (key, fname) in enumerate(CANONICAL_FILES):
        if i < len(flags) and flags[i] == "1":
            out[key] = f"{case_dir}/{fname}"
    return out


# ---------------------------------------------------------------------------
# Compatibility accessor: return the (relative) warp file path for a given
# canonical key, working for BOTH schemas:
#
# * Old (798 MB) — view["warp_output"] is a full dict with direct keys
# * New (flags)  — view["warp_output"] is {"f": "1101..."} and we rebuild
#                  the path on the fly using the parent tuple's fields
# ---------------------------------------------------------------------------
def view_warp_file(view: Dict,
                   canonical_key: str,
                   *,
                   scene: Optional[str] = None,
                   main_traj: Optional[str] = None,
                   role: Optional[str] = None,
                   center_frame: Optional[str] = None) -> Optional[str]:
    """Return relative path for ``canonical_key`` in ``view['warp_output']``.

    Parameters that are only needed for the new flags schema (``scene`` /
    ``main_traj`` / ``role`` / ``center_frame``) can be passed through
    from the parent tuple when the caller has them. If None, we best-
    effort pull them from the view itself.
    """
    wo = view.get("warp_output") or {}
    if not wo:
        return None

    # Fast path: old verbose format.
    direct = wo.get(canonical_key)
    if direct:
        return direct

    # New flags format?
    flags = wo.get("f") or wo.get("flags")
    if not flags:
        return None

    # Need (scene, main, role, cf, vf) to rebuild the case_dir.
    role = role or view.get("role")
    view_frame = view.get("frame") or view.get("view_frame")
    if role is None or view_frame is None:
        return None
    if scene is None or main_traj is None or center_frame is None:
        # Last resort: the view itself sometimes carries these (pipeline
        # attaches them in sub-cases).
        scene = scene or view.get("scene")
        main_traj = main_traj or view.get("main_trajectory")
        center_frame = center_frame or view.get("center_frame")
    if not (scene and main_traj and center_frame):
        return None

    i = CANONICAL_KEYS.index(canonical_key) if canonical_key in CANONICAL_KEYS else -1
    if i < 0 or i >= len(flags) or flags[i] != "1":
        return None
    case_dir = sub_case_basename(scene, str(main_traj), role,
                                 str(center_frame), str(view_frame))
    fname = CANONICAL_FNAMES[i]
    return f"{case_dir}/{fname}"


# ---------------------------------------------------------------------------
# Tiny iteration helper: walks ``data["cases"]`` and also supports the
# still-embryonic ``scenes`` grouped layout in case we add it later.
# ---------------------------------------------------------------------------
def iter_cases(doc: Dict):
    """Yield tuple cases from either ``{cases: [...]}`` or ``{scenes: {...}}``."""
    if "cases" in doc and isinstance(doc["cases"], list):
        for c in doc["cases"]:
            yield c
        return
    scenes = doc.get("scenes")
    if isinstance(scenes, dict):
        for scene, tuples in scenes.items():
            for t in tuples:
                # Re-inflate minimal fields if grouped format omitted them.
                if "scene" not in t:
                    t = {**t, "scene": scene}
                yield t


__all__ = [
    "NEIGHBOR_ROLES", "COMPANION_ROLES", "ALL_POSITIONS",
    "CANONICAL_FILES", "CANONICAL_KEYS", "CANONICAL_FNAMES",
    "N_CANONICAL", "OK_PROBE_INDICES", "CANONICAL_FILES_DICT",
    "DIR_RULE", "FLAGS_LEGEND",
    "paths_for", "sub_case_basename",
    "flags_encode", "flags_from_filenames",
    "flags_is_ok", "flags_to_dict", "flags_present_keys",
    "expand_warp_output", "view_warp_file", "iter_cases",
]
