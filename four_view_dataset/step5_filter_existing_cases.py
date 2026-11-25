#!/usr/bin/env python3
"""Step 5: Filter four-view cases whose on-disk frame files are missing.

Runs BEFORE the train/test split (step6) so both splits are drawn from a
clean, fully-present pool.  Input is the full step4 JSON
(``step4_total.json``); output is the filtered full pool that step6 then
splits.

Each case holds 5 viewpoints (1 ``start`` + 4 ``views[*].end``).  All file
paths in the JSON are relative to ``--data-root``.  A case is kept only if
*every* required file of *all 5* viewpoints exists; if any one is missing the
whole case is dropped.

Required files per viewpoint (matches what the online dataloader actually
reads in training/data/datasets_pano2pano_four_view_online.py):
  - start view : jpg + exr + json
  - end views  : jpg + json   (end exr is never read, so it is not checked)

Run this on a host where ``--data-root`` is mounted.

Usage:
    python four_view_dataset/step5_filter_existing_cases.py \\
        --input   four_view_dataset/data/step4_total.json \\
        --output  four_view_dataset/data/step4_total_filtered.json \\
        --data-root @repo/data/UE_PanoramicOut_backup \\
        --workers 64
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
import os
import sys
from concurrent.futures import ThreadPoolExecutor

DEFAULT_DATA_ROOT = _release_path("@repo/data/UE_PanoramicOut_backup")


def required_paths(case):
    """Yield (relative_path) for every file that must exist for this case."""
    start = case["start"]
    start_jpg = start["jpg"]
    yield start_jpg
    yield start.get("exr") or start_jpg.replace(".jpg", ".exr")
    yield start.get("json") or start_jpg.replace(".jpg", ".json")
    for v in case["views"]:
        end = v["end"]
        end_jpg = end["jpg"]
        yield end_jpg
        yield end.get("json") or end_jpg.replace(".jpg", ".json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=_release_path("@repo/"
                    "four_view_dataset/data/step4_total.json"),
                    help="Input four-view JSON ({'cases': [...]}), "
                         "typically step4_total.json.")
    ap.add_argument("--output", default=None,
                    help="Output JSON. Default: <input>_filtered.json")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                    help="Root that case paths are relative to.")
    ap.add_argument("--workers", type=int, default=64,
                    help="Thread count for parallel os.path.exists (IO bound).")
    ap.add_argument("--dropped-out", default=None,
                    help="Optional JSON to write the dropped cases for inspection.")
    args = ap.parse_args()

    out_path = args.output or os.path.splitext(args.input)[0] + "_filtered.json"

    with open(args.input) as f:
        data = json.load(f)
    cases = data["cases"]
    print(f"[load] {len(cases)} cases from {args.input}", flush=True)

    # Dedup the file paths across cases so shared frames are stat'd once.
    exists_cache = {}
    all_rel = set()
    for c in cases:
        all_rel.update(required_paths(c))
    print(f"[scan] {len(all_rel)} unique file paths to check under {args.data_root}",
          flush=True)

    rel_list = list(all_rel)

    def check(rel):
        return rel, os.path.exists(os.path.join(args.data_root, rel))

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for rel, ok in ex.map(check, rel_list, chunksize=256):
            exists_cache[rel] = ok
            done += 1
            if done % 100000 == 0:
                print(f"  checked {done}/{len(rel_list)}", flush=True)

    kept, dropped = [], []
    for c in cases:
        if all(exists_cache[r] for r in required_paths(c)):
            kept.append(c)
        else:
            dropped.append(c)

    missing_files = sum(1 for ok in exists_cache.values() if not ok)
    print(f"[result] kept={len(kept)} dropped={len(dropped)} "
          f"(missing_unique_files={missing_files})", flush=True)

    with open(out_path, "w") as f:
        json.dump({"cases": kept}, f, ensure_ascii=False, indent=2)
    print(f"[write] filtered JSON -> {out_path}", flush=True)

    if args.dropped_out:
        with open(args.dropped_out, "w") as f:
            json.dump({"cases": dropped}, f, ensure_ascii=False, indent=2)
        print(f"[write] dropped cases -> {args.dropped_out}", flush=True)


if __name__ == "__main__":
    main()
