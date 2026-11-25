"""Path / sequence-list utilities for the 0519 NVSPano data pipeline.

Sequence list format (``$SEQUENCE_LIST_0519``)
----------------------------------------------
Each line is ``<scene>/<seq_id>[/<version>]`` where:

* ``scene``    — top-level UE scene folder name (e.g. ``AIUE_V02_003_SY_01``).
* ``seq_id``   — sub-scene trajectory id (integer string).
* ``version``  — optional: ``v2`` / ``v3`` for the new renders, or absent
  (empty third segment, equivalent to ``""``) for the older renders.

Frame filesystem layout
-----------------------
::

    # versioned (v2 / v3):
    <DATA_ROOT>/<scene>/<version>/<scene>/<seq_id>/<frame:04d>/<frame:04d>.{jpg,exr,json}
    # un-versioned (legacy):
    <DATA_ROOT>/<scene>/<scene>/<seq_id>/<frame:04d>/<frame:04d>.{jpg,exr,json}

Caption layout (0519 dataset)
-----------------------------
The 0519 caption directory is **version-aware**: versioned renders carry a
``_v<n>`` suffix in the filename and *do not* share captions with legacy::

    <CAPTION_DIR>/<scene>/<seq_id>_v3.json   # v3 trajectories
    <CAPTION_DIR>/<scene>/<seq_id>_v2.json   # v2 trajectories  (when present)
    <CAPTION_DIR>/<scene>/<seq_id>.json      # legacy (un-versioned) trajectories

We therefore look the filename up by ``(seq_id, version)`` with a small
priority list (see :func:`caption_candidates`) and treat the first
existing file as the canonical caption.

Public helpers
--------------
* :func:`parse_sequence_line`     — one line -> ``(scene, seq_id, version)``
* :func:`read_sequence_list`      — file -> ``list[(scene, seq_id, version)]``
* :func:`read_scene_list`         — file -> ``list[scene]``
* :func:`traj_dir_rel`            — relative dir under DATA_ROOT for a (scene, seq, ver)
* :func:`traj_dir_abs`            — absolute dir for a (scene, seq, ver)
* :func:`frame_paths_rel`         — relative jpg/exr/json paths for a single frame
* :func:`frame_paths_abs`         — absolute jpg/exr/json paths for a single frame
* :func:`caption_path`            — absolute caption JSON path (single, version-aware)
* :func:`caption_candidates`      — priority-ordered list of caption paths to try
* :func:`find_caption`            — first existing caption path (or ``None``)
* :func:`case_directory_field`    — value to put under ``case["directory"]``
* :func:`case_uid`                — stable, unique case id (used by JSON output + cache keys)
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Sequence-list parsing
# -----------------------------------------------------------------------------

SequenceEntry = Tuple[str, str, str]  # (scene, seq_id, version); version="" if absent


def parse_sequence_line(line: str) -> Optional[SequenceEntry]:
    """Parse one ``<scene>/<seq>[/<version>]`` line.

    Returns ``None`` for blank lines or malformed entries. Trailing empty
    third segments (``"scene/seq/"`` after a stray slash) are treated the
    same as the un-versioned form (``"scene/seq"``).
    """
    s = line.strip()
    if not s:
        return None
    parts = s.split("/")
    if len(parts) < 2:
        return None
    scene = parts[0].strip()
    seq_id = parts[1].strip()
    version = parts[2].strip() if len(parts) >= 3 else ""
    if not scene or not seq_id:
        return None
    return scene, seq_id, version


def read_sequence_list(path: str) -> List[SequenceEntry]:
    """Read the sequence list, returning ``[(scene, seq_id, version), ...]``.

    Preserves file order and de-duplicates exact-triple repeats (rare but
    cheap to guard against).
    """
    seen: "OrderedDict[SequenceEntry, None]" = OrderedDict()
    with open(path, "r") as f:
        for raw in f:
            ent = parse_sequence_line(raw)
            if ent is not None:
                seen.setdefault(ent, None)
    return list(seen.keys())


def read_scene_list(path: str) -> List[str]:
    """Read a one-scene-per-line whitelist, preserving order, de-duped."""
    seen: "OrderedDict[str, None]" = OrderedDict()
    with open(path, "r") as f:
        for raw in f:
            s = raw.strip()
            if s:
                seen.setdefault(s, None)
    return list(seen.keys())


# -----------------------------------------------------------------------------
# Path construction (single source of truth)
# -----------------------------------------------------------------------------

def traj_dir_rel(scene: str, seq_id: str, version: str = "") -> str:
    """Relative trajectory directory (under ``DATA_ROOT``).

    Examples
    --------
    >>> traj_dir_rel("S", "0", "v3")
    'S/v3/S/0'
    >>> traj_dir_rel("S", "0", "")
    'S/S/0'
    """
    if version:
        return f"{scene}/{version}/{scene}/{seq_id}"
    return f"{scene}/{scene}/{seq_id}"


def traj_dir_abs(scene: str, seq_id: str, version: str, data_root: str) -> str:
    return os.path.join(data_root, traj_dir_rel(scene, seq_id, version))


def frame_paths_rel(
    scene: str, seq_id: str, version: str, frame_idx: int
) -> Dict[str, str]:
    """Relative jpg/exr/json paths for ``frame_idx``.

    All three live under the same per-frame folder with the same basename.
    """
    fstr = f"{frame_idx:04d}"
    base = f"{traj_dir_rel(scene, seq_id, version)}/{fstr}"
    return {
        "jpg": f"{base}/{fstr}.jpg",
        "exr": f"{base}/{fstr}.exr",
        "json": f"{base}/{fstr}.json",
    }


def frame_paths_abs(
    scene: str, seq_id: str, version: str, frame_idx: int, data_root: str
) -> Dict[str, str]:
    rel = frame_paths_rel(scene, seq_id, version, frame_idx)
    return {k: os.path.join(data_root, v) for k, v in rel.items()}


def caption_path(
    scene: str, seq_id: str, caption_dir: str, version: str = ""
) -> str:
    """Single, version-aware caption JSON path (best-guess, no existence check).

    * ``version="v3"`` -> ``<caption_dir>/<scene>/<seq_id>_v3.json``
    * ``version="v2"`` -> ``<caption_dir>/<scene>/<seq_id>_v2.json``
    * ``version=""``   -> ``<caption_dir>/<scene>/<seq_id>.json``  (legacy)

    Use :func:`caption_candidates` if you want to try the *other* variants
    as a fallback (some scenes only ship one variant on disk).
    """
    suffix = f"_{version}" if version else ""
    return os.path.join(caption_dir, scene, f"{seq_id}{suffix}.json")


# Priority order for fallback lookups. ``v3`` is the newest / most populated
# format in the 0519 release, so it goes first when no version is known.
_CAPTION_VERSION_FALLBACK = ("v3", "v2", "")


def caption_candidates(
    scene: str, seq_id: str, version: str, caption_dir: str
) -> List[str]:
    """Return all plausible caption paths in priority order, de-duped.

    The first entry is always the version-matched path; the remaining
    entries are fallbacks so an v3-only or legacy-only caption is still
    found when the caller's ``version`` is wrong/missing.
    """
    base = os.path.join(caption_dir, scene)
    seen: "OrderedDict[str, None]" = OrderedDict()
    seen.setdefault(caption_path(scene, seq_id, caption_dir, version), None)
    for ver in _CAPTION_VERSION_FALLBACK:
        if ver == version:
            continue
        seen.setdefault(caption_path(scene, seq_id, caption_dir, ver), None)
    return list(seen.keys())


def find_caption(
    scene: str, seq_id: str, version: str, caption_dir: str
) -> Optional[str]:
    """Return the first existing caption path among the candidates, else ``None``.

    Performs at most a handful of ``os.path.isfile`` probes (1 hit in the
    common case, 2–3 in fallback cases).
    """
    for p in caption_candidates(scene, seq_id, version, caption_dir):
        if os.path.isfile(p):
            return p
    return None


# -----------------------------------------------------------------------------
# Case id / directory helpers (referenced by stepc / stepe / training)
# -----------------------------------------------------------------------------

def case_directory_field(scene: str, seq_id: str, version: str) -> str:
    """Value to store under ``case["directory"]``.

    Kept structurally compatible with the legacy trajectory manifest schema
    (``scene/scene/seq``) by interleaving ``version`` in the v2/v3 case so
    that downstream ``directory.split("/")[0]`` still yields ``scene``
    (which stepf relies on for the train/test scene split).
    """
    return traj_dir_rel(scene, seq_id, version)


def case_uid(
    scene: str, seq_id: str, version: str, start_idx: int, end_idx: int
) -> str:
    """Stable unique case identifier (used in metadata and as cache key).

    Format: ``<scene>__<seq>__<version_or_legacy>__<start>_<end>``.
    """
    ver_tag = version or "legacy"
    return f"{scene}__{seq_id}__{ver_tag}__{start_idx:04d}_{end_idx:04d}"
