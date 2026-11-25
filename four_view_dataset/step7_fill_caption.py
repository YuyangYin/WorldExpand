#!/usr/bin/env python3
"""Step 7: Backfill empty ``caption`` fields in already-split JSONs.

The four-view JSONs produced upstream (train.json / test.json /
test_subset.json …) can carry empty ``caption`` strings when the default
``frame_long_long`` caption source (``long_long_caption``) is missing.  This
step fills those captions in place from the OLDER, shorter per-trajectory
caption.

Caption resolution reuses the canonical v2 resolver
the version-aware ``dataset_utils.paths.find_caption`` helper:

  v3     -> {caption_dir}/{scene}/{main_trajectory}_v3.json
  v2     -> {caption_dir}/{scene}/{main_trajectory}_v2.json
  legacy -> {caption_dir}/{scene}/{main_trajectory}.json

with automatic fallback across the other variants, and reads the
``long caption`` / ``long_caption`` field.  (The plain
``{scene}/{traj}.json`` layout that step4's ``traj_long`` assumed misses every
v3/v2 scene — which is why captions came back all-empty.)

Input is a DIRECTORY: every ``*.json`` under it that looks like a case list
({"cases": [...]} or a bare [...] of case dicts) is processed.  Non-case JSONs
(e.g. split_info.json) are skipped.  By default only EMPTY captions are filled;
existing non-empty captions are left untouched (use --overwrite-existing to
refill all).

Usage:
    python four_view_dataset/step7_fill_caption.py \\
        --input_dir   four_view_dataset/data_0519/split \\
        --caption_dir data/pano_captions \\
        --workers 32
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
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

# Reuse the canonical version-aware caption resolver (run from repo root).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from dataset_utils.paths import find_caption  # noqa: E402

DEFAULT_INPUT_DIR = (_release_path("@repo/"
                     "four_view_dataset/data_0519/split"))
DEFAULT_CAPTION_DIR = (_release_path("@repo/data/pano_captions"))


def _extract_cases(payload: Any):
    """Return (cases_list, container_kind). kind: 'dict' | 'list' | None."""
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        return payload["cases"], "dict"
    if isinstance(payload, list):
        return payload, "list"
    return None, None


def _looks_like_cases(cases: List[Any]) -> bool:
    return bool(cases) and isinstance(cases[0], dict) and "scene" in cases[0]


def _cap_key(case: Dict[str, Any]) -> Tuple[str, str, str]:
    """Dedup / lookup key: (scene, seq_id=main_trajectory, version)."""
    scene = str(case.get("scene", ""))
    seq_id = str(case.get("main_trajectory", ""))
    version = str(case.get("version", ""))
    if version == "legacy":
        version = ""
    return (scene, seq_id, version)


def _read_caption_text(path: str) -> str:
    """Read the long caption text from a resolved caption json; '' on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return ""
    text = d.get("long caption") or d.get("long_caption") or ""
    return str(text).strip() if text else ""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input_dir", default=DEFAULT_INPUT_DIR,
                    help="Directory whose *.json case files get caption-filled "
                         "in place.")
    ap.add_argument("--glob", default="*.json",
                    help="Glob (relative to --input_dir) selecting files "
                         "(default: *.json).")
    ap.add_argument("--caption_dir", default=DEFAULT_CAPTION_DIR,
                    help="Root of pano_captions. Path: {dir}/{scene}/"
                         "{main_trajectory}[_v3|_v2].json")
    ap.add_argument("--workers", type=int, default=32,
                    help="Threads for parallel caption file reads (IO bound).")
    ap.add_argument("--overwrite-existing", action="store_true",
                    help="Refill ALL captions, not just empty ones.")
    ap.add_argument("--output_dir", default=None,
                    help="Write filled files here instead of editing in place "
                         "(mirrors input filenames).")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.glob)))
    if not files:
        print(f"[skip] no files match {args.glob} under {args.input_dir}")
        sys.exit(0)
    print(f"[init] caption_dir={args.caption_dir}")
    print(f"[init] {len(files)} candidate file(s) under {args.input_dir}", flush=True)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1) Load all files, gather the unique caption lookups needed -------
    loaded: List[Tuple[str, Any, List[Dict], str]] = []  # (path, payload, cases, kind)
    needed: set = set()
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"  [skip] {os.path.basename(path)}: not JSON ({e})")
            continue
        cases, kind = _extract_cases(payload)
        if cases is None or not _looks_like_cases(cases):
            print(f"  [skip] {os.path.basename(path)}: not a case list")
            continue
        loaded.append((path, payload, cases, kind))
        for c in cases:
            if args.overwrite_existing or not str(c.get("caption", "")).strip():
                needed.add(_cap_key(c))

    if not loaded:
        print("[done] nothing to fill (no case-list files).")
        sys.exit(0)

    print(f"[init] {len(needed)} unique (scene/traj/version) caption lookups",
          flush=True)

    # ---- 2) Resolve unique captions in parallel (version-aware + fallback) -
    def _resolve(key: Tuple[str, str, str]) -> Tuple[Tuple, str]:
        scene, seq_id, version = key
        p = find_caption(scene, seq_id, version, args.caption_dir)
        return key, (_read_caption_text(p) if p else "")

    cap_map: Dict[Tuple, str] = {}
    needed_list = list(needed)
    if needed_list:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            for key, cap in ex.map(_resolve, needed_list, chunksize=64):
                cap_map[key] = cap
    resolved_nonempty = sum(1 for v in cap_map.values() if v)
    print(f"[resolve] {resolved_nonempty}/{len(cap_map)} lookups returned a caption",
          flush=True)

    # ---- 3) Apply + write ---------------------------------------------------
    grand_filled = grand_still_empty = grand_total = 0
    for path, payload, cases, kind in loaded:
        filled = still_empty = 0
        for c in cases:
            has = str(c.get("caption", "")).strip()
            if has and not args.overwrite_existing:
                continue
            cap = cap_map.get(_cap_key(c), "")
            if cap:
                c["caption"] = cap
                filled += 1
            elif not has:
                still_empty += 1
        grand_filled += filled
        grand_still_empty += still_empty
        grand_total += len(cases)

        if kind == "dict":
            payload["cases"] = cases          # preserve sibling keys (summary…)
            out_payload = payload
        else:
            out_payload = cases
        out_path = (os.path.join(args.output_dir, os.path.basename(path))
                    if args.output_dir else path)
        tmp = out_path + f".{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, out_path)
        print(f"  [{os.path.basename(path)}] cases={len(cases)} "
              f"filled=+{filled} still_empty={still_empty} -> {out_path}")

    print(f"\n[done] total cases={grand_total} filled=+{grand_filled} "
          f"still_empty={grand_still_empty}")


if __name__ == "__main__":
    main()
