#!/usr/bin/env python3
"""
Step 6: Split four-view-v1 train/test JSON by scene.

Runs AFTER step5 (missing-file filter), so the input here is the filtered
full pool (e.g. ``step4_total_filtered.json``) and both splits are guaranteed
to reference on-disk frames only.

Splits are performed at the *scene* level so all cases belonging to the same
scene end up in the same split.  Scene name is read from each case's ``scene``
field (produced by step4_build_four_view_json.py); falls back to the first
component of ``directory`` if ``scene`` is absent.

By default, a fixed number of scenes (``--num_test_scenes``, default 20) is
held out as the test set.  If ``--num_test_scenes <= 0`` the script falls back
to the proportional ``--test_ratio``.

Outputs (under ``--output_dir``):
  train.json            all train cases  ({"cases": [...]})
  test.json             all test cases
  test_subset.json      balanced subset from test (--test_subset_size, default 200)
  train_subset_N.json   optional balanced subset from train (--subset_size / --subset_ratio)

Usage:
    python four_view_dataset/step6_split_train_test.py \\
        --input_json  four_view_dataset/data/step4_total_filtered.json \\
        --output_dir  four_view_dataset/data/split \\
        --num_test_scenes 20 \\
        --seed 42
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "cases" in raw:
        return raw["cases"], True
    if isinstance(raw, list):
        return raw, False
    raise ValueError(
        f"Unexpected JSON format in {path}. "
        "Expected {\"cases\": [...]} or [...]."
    )


def _save_json(data: list, path: Path, wrap_cases: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cases": data} if wrap_cases else data
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _scene_of(item: dict) -> str:
    """Return the scene name for a four-view-v1 case."""
    # Prefer the explicit 'scene' field added by step4_build_four_view_json.py.
    s = item.get("scene")
    if s:
        return str(s)
    # Fallback: first component of 'directory'  (e.g. "SceneName/v3/...")
    return item.get("directory", "unknown").split("/")[0]


# ---------------------------------------------------------------------------
# Balanced subset sampling (same algorithm as reference step6)
# ---------------------------------------------------------------------------

def _sample_balanced(data: list, target: int, seed: int = 42) -> list:
    """Sample ``target`` items from ``data``, balanced across scenes."""
    rng = random.Random(seed)

    groups: dict = defaultdict(list)
    for item in data:
        groups[_scene_of(item)].append(item)

    num_scenes = len(groups)
    if num_scenes == 0:
        return []

    target = min(target, len(data))
    base = target // num_scenes

    sampled: dict = {}
    budget = target
    surplus_scenes = []

    for scene in sorted(groups):
        items = groups[scene]
        alloc = min(base, len(items))
        sampled[scene] = rng.sample(items, alloc)
        budget -= alloc
        if len(items) > alloc:
            surplus_scenes.append(scene)

    if budget > 0:
        surplus_info = [
            (s, len(groups[s]) - len(sampled[s]))
            for s in surplus_scenes
        ]
        surplus_info.sort(key=lambda x: -x[1])
        for i, (scene, avail) in enumerate(surplus_info):
            if budget <= 0:
                break
            left = len(surplus_info) - i
            extra = min(avail, math.ceil(budget / left), budget)
            already = {id(x) for x in sampled[scene]}
            pool = [x for x in groups[scene] if id(x) not in already]
            sampled[scene].extend(rng.sample(pool, extra))
            budget -= extra

    result = []
    for scene in sorted(sampled):
        result.extend(sampled[scene])
        print(f"  {scene}: {len(sampled[scene])} / {len(groups[scene])}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split four-view-v1 JSON into train / test by scene."
    )
    parser.add_argument("--input_json",      required=True,
                        help="Input JSON (filtered step4 pool, e.g. "
                             "step4_total_filtered.json from step5).")
    parser.add_argument("--output_dir",      required=True,
                        help="Directory for output split files.")
    parser.add_argument("--num_test_scenes", type=int, default=20,
                        help="Fixed number of scenes held out for test "
                             "(default: 20).  Use <= 0 to switch to --test_ratio.")
    parser.add_argument("--test_ratio",      type=float, default=0.02,
                        help="Fraction of scenes for test when "
                             "--num_test_scenes <= 0 (default: 0.02).")
    parser.add_argument("--test_subset_size", type=int, default=200,
                        help="Balanced test_subset.json size (default: 200). "
                             "Set 0 to skip.")
    parser.add_argument("--subset_size",     type=int, default=None,
                        help="Sample a fixed-size balanced subset from train "
                             "(mutually exclusive with --subset_ratio).")
    parser.add_argument("--subset_ratio",    type=float, default=None,
                        help="Sample a proportional balanced subset from train "
                             "(mutually exclusive with --subset_size).")
    parser.add_argument("--seed",            type=int, default=42,
                        help="Random seed (default: 42).")
    args = parser.parse_args()

    if args.subset_size is not None and args.subset_ratio is not None:
        parser.error("--subset_size and --subset_ratio are mutually exclusive.")

    random.seed(args.seed)
    output_dir = Path(args.output_dir)

    # ------------------------------------------------------------------ #
    # Load                                                                 #
    # ------------------------------------------------------------------ #
    data, wrap_cases = _load_json(args.input_json)
    print(f"Total cases: {len(data)}")

    # ------------------------------------------------------------------ #
    # Group by scene                                                       #
    # ------------------------------------------------------------------ #
    scene_groups: dict = defaultdict(list)
    for item in data:
        scene_groups[_scene_of(item)].append(item)

    all_scenes = sorted(scene_groups)
    num_scenes = len(all_scenes)

    if args.num_test_scenes is not None and args.num_test_scenes > 0:
        n_test_scenes = min(args.num_test_scenes, num_scenes)
        split_mode = f"--num_test_scenes={args.num_test_scenes}"
    else:
        n_test_scenes = max(1, math.ceil(num_scenes * args.test_ratio))
        split_mode = f"--test_ratio={args.test_ratio}"

    print(f"Total scenes: {num_scenes}")
    print(f"Split mode:   {split_mode} → {n_test_scenes} test scene(s)")

    # ------------------------------------------------------------------ #
    # Random scene-level split                                             #
    # ------------------------------------------------------------------ #
    test_scenes  = set(random.sample(all_scenes, n_test_scenes))
    train_scenes = set(all_scenes) - test_scenes

    train_data, test_data = [], []
    for scene in all_scenes:
        (test_data if scene in test_scenes else train_data).extend(
            scene_groups[scene]
        )

    print(f"\nTrain: {len(train_scenes)} scenes, {len(train_data)} cases")
    print(f"Test:  {len(test_scenes)} scenes,  {len(test_data)} cases")

    print(f"\nTest scenes ({n_test_scenes}):")
    for scene in sorted(test_scenes):
        print(f"  {scene}: {len(scene_groups[scene])} cases")

    # ------------------------------------------------------------------ #
    # Save train / test                                                    #
    # ------------------------------------------------------------------ #
    _save_json(train_data, output_dir / "train.json", wrap_cases)
    _save_json(test_data,  output_dir / "test.json",  wrap_cases)
    print(f"\nSaved train.json ({len(train_data)} cases)")
    print(f"Saved test.json  ({len(test_data)} cases)")

    # ------------------------------------------------------------------ #
    # Balanced test subset                                                 #
    # ------------------------------------------------------------------ #
    if args.test_subset_size > 0:
        n = args.test_subset_size
        print(f"\nSampling test_subset ({n} cases, balanced across scenes):")
        test_subset = _sample_balanced(test_data, n, seed=args.seed)
        _save_json(test_subset, output_dir / "test_subset.json", wrap_cases)
        print(f"Saved test_subset.json ({len(test_subset)} cases)")

    # ------------------------------------------------------------------ #
    # Optional balanced train subset                                       #
    # ------------------------------------------------------------------ #
    if args.subset_size is not None or args.subset_ratio is not None:
        if args.subset_ratio is not None:
            subset_n = max(1, int(len(train_data) * args.subset_ratio))
        else:
            subset_n = args.subset_size
        subset_n = min(subset_n, len(train_data))
        print(f"\nSampling train_subset ({subset_n} cases, balanced across scenes):")
        train_subset = _sample_balanced(train_data, subset_n, seed=args.seed)
        _save_json(train_subset,
                   output_dir / f"train_subset_{subset_n}.json", wrap_cases)
        print(f"Saved train_subset_{subset_n}.json ({len(train_subset)} cases)")

    print("\nDone.")


if __name__ == "__main__":
    main()
