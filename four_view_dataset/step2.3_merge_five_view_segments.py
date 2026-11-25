#!/usr/bin/env python3
"""Merge multi-machine output of ``step2.2_sample_five_view_segments.py``.

Accepts either:
    * per-machine top-level JSONs   (``sampled_five_view_segments_machine{k}.json``)
    * the per-scene directory       (``<input_dir>/per_scene/*.json``)

and writes a single deduplicated merged JSON. ``per_scene/`` is the source of
truth for resume-ability (the top-level JSONs are just snapshots at run end);
when both are present we prefer ``per_scene/`` since it always reflects the
latest successful runs on every machine.

Usage
-----
Typical 4-machine run:

    python step2.3_merge_five_view_segments.py \\
        --input_dir  ./output/4.22/sampled_five_view_segments \\
        --num_machines 4

Falling back to per-machine JSONs (e.g. if ``per_scene/`` was wiped):

    python step2.3_merge_five_view_segments.py \\
        --input_dir  ./output/4.22/sampled_five_view_segments \\
        --num_machines 4 \\
        --no_per_scene
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple


def _case_dedup_key(case: Dict) -> str:
    """Identify a unique case by its 5-view JPG paths.

    Two cases are "the same" iff the 5 jpgs are identical, regardless of
    which machine produced them. We sort path strings so the key is
    invariant w.r.t. accidental view-ordering differences.
    """
    jpgs = sorted(v.get("jpg", "") for v in case.get("views", []))
    return "|".join(jpgs)


def _read_json_cases(fp: str) -> Tuple[str, List[Dict], str]:
    """Worker: read one JSON file, return (path, cases, err)."""
    try:
        with open(fp, "r") as f:
            d = json.load(f)
        return fp, d.get("cases", []), ""
    except Exception as e:  # noqa: BLE001
        return fp, [], str(e)


def _load_files_parallel(files: List[str], workers: int, tag: str) -> List[Dict]:
    out: List[Dict] = []
    if not files:
        return out
    workers = max(1, min(workers, len(files)))
    t0 = time.time()
    done = 0
    total = len(files)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_read_json_cases, fp) for fp in files]
        for fut in as_completed(futs):
            fp, cases, err = fut.result()
            done += 1
            if err:
                print(f"  [warn] unreadable {fp}: {err}")
            else:
                out.extend(cases)
            if done % 100 == 0 or done == total:
                print(f"  [{tag}] {done}/{total} files  "
                      f"({len(out)} cases so far, "
                      f"{time.time()-t0:.1f}s)")
    print(f"  [{tag}] loaded {len(out)} cases from {total} files "
          f"in {time.time()-t0:.1f}s")
    return out


def _load_per_scene(per_scene_dir: str, workers: int) -> List[Dict]:
    files = sorted(glob.glob(os.path.join(per_scene_dir, "*.json")))
    return _load_files_parallel(files, workers, "per_scene")


def _load_machine_jsons(input_dir: str, num_machines: int,
                        workers: int) -> List[Dict]:
    files = []
    for mid in range(num_machines):
        fp = os.path.join(
            input_dir, f"sampled_five_view_segments_machine{mid}.json")
        if not os.path.exists(fp):
            print(f"  [warn] missing {fp}")
            continue
        files.append(fp)
    return _load_files_parallel(files, workers, "machine")


def main():
    ap = argparse.ArgumentParser(description="Merge 5-view sample JSONs.")
    ap.add_argument("--input_dir", required=True,
                    help="Directory produced by step2.2_sample_five_view_segments.py")
    ap.add_argument("--num_machines", type=int, default=1,
                    help="For fallback per-machine JSON loading.")
    ap.add_argument("--output_json", type=str, default=None,
                    help="Default: <input_dir>/sampled_five_view_segments.json")
    ap.add_argument("--no_per_scene", action="store_true",
                    help="Ignore per_scene/ and only merge sampled_...machine{k}.json")
    ap.add_argument("--workers", type=int, default=16,
                    help="Parallel worker processes for JSON loading.")
    ap.add_argument("--indent", type=int, default=-1,
                    help="JSON indent for output. Three modes:\n"
                         "  N > 0  : standard json.dump(indent=N) — prettiest "
                         "but VERY slow for million-case files.\n"
                         "  N == 0 : single compact line — fastest but "
                         "unreadable.\n"
                         "  N == -1: pretty outer shell + one case per line "
                         "(recommended default): head/tail/grep-friendly, "
                         "nearly as fast as indent=0.")
    args = ap.parse_args()

    per_scene_dir = os.path.join(args.input_dir, "per_scene")
    use_per_scene = (not args.no_per_scene) and os.path.isdir(per_scene_dir)

    print(f"[merge] input_dir={args.input_dir}  use_per_scene={use_per_scene}  "
          f"workers={args.workers}")
    t_start = time.time()
    if use_per_scene:
        cases = _load_per_scene(per_scene_dir, args.workers)
    else:
        cases = _load_machine_jsons(args.input_dir, args.num_machines,
                                    args.workers)

    # Dedup across machines (same case can appear if scenes were re-assigned).
    t_dedup = time.time()
    seen = {}
    for c in cases:
        seen[_case_dedup_key(c)] = c
    merged = list(seen.values())
    print(f"[merge] {len(cases)} raw  →  {len(merged)} unique  "
          f"(dedup {time.time()-t_dedup:.1f}s)")

    output_json = args.output_json or os.path.join(
        args.input_dir, "sampled_five_view_segments.json")
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    tmp = f"{output_json}.tmp"
    t_write = time.time()

    payload = {"cases": merged, "summary": {"total": len(merged)}}

    if args.indent is not None and args.indent > 0:
        # Standard pretty dump. Slow for big files but the output is as
        # human-friendly as it gets.
        with open(tmp, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=args.indent)
    elif args.indent == 0:
        # Compact single line. Fastest, unreadable.
        with open(tmp, "w") as f:
            json.dump(payload, f, ensure_ascii=False)
    else:
        # indent == -1 (default): pretty outer shell + ONE case per line.
        # - Still valid JSON (parses with json.load just fine).
        # - head/tail/grep/less all work line-by-line.
        # - Speed ≈ compact mode: every case is a single json.dumps() call,
        #   no recursive pretty-printing.
        with open(tmp, "w") as f:
            f.write("{\n")
            # --- cases array, one case per line ---
            f.write('  "cases": [\n')
            n = len(merged)
            for i, c in enumerate(merged):
                line = json.dumps(c, ensure_ascii=False)
                if i < n - 1:
                    f.write(f"    {line},\n")
                else:
                    f.write(f"    {line}\n")
            f.write("  ],\n")
            # --- summary (small, pretty it inline) ---
            summary_s = json.dumps(payload["summary"], ensure_ascii=False,
                                   indent=2)
            # re-indent the summary block by 2 spaces so it lines up
            summary_s = "\n".join("  " + ln if ln else ln
                                  for ln in summary_s.split("\n"))
            f.write(f'  "summary": {summary_s.lstrip()}\n')
            f.write("}\n")

    os.replace(tmp, output_json)
    sz = os.path.getsize(output_json) / (1024 * 1024)
    print(f"[merge] wrote {output_json}  ({sz:.1f} MB, "
          f"write {time.time()-t_write:.1f}s, "
          f"total {time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
