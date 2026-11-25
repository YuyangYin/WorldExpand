
# Resolve resource defaults from the external release configuration.
import sys as _we_sys
from pathlib import Path as _WePath
_we_root = next((_p for _p in _WePath(__file__).resolve().parents
                 if (_p / "worldexpand" / "paths.py").is_file()), None)
if _we_root is not None and str(_we_root) not in _we_sys.path:
    _we_sys.path.insert(0, str(_we_root))
from worldexpand.paths import resolve_resource as _release_path

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial

from tqdm import tqdm

DEFAULT_BASE_PATH = _release_path("@repo/data/UE_PanoramicOut")


def read_case_names_from_file(input_file):
    """Read case names from input file (one case name per line)"""
    case_names = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                case_names.append(line)
    return case_names


def read_case_trajectories_from_file(input_file):
    """Read case_name/traj_id pairs from input file.

    Each line format: {case_name}/{traj_id}
    e.g. xerichammockbrush_yzy_16/19

    Returns: dict mapping case_name -> list of traj_ids (int)
    """
    case_trajs = defaultdict(list)
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('/')
            if len(parts) >= 2:
                case_name = parts[0]
                traj_id = parts[1]
                if traj_id.isdigit():
                    case_trajs[case_name].append(int(traj_id))
    # Sort trajectory ids for each case
    for case_name in case_trajs:
        case_trajs[case_name] = sorted(set(case_trajs[case_name]))
    return dict(case_trajs)


def scan_case_names_from_dir(base_path):
    """Scan case names by listing subdirectories under base_path"""
    case_names = []
    for item in sorted(os.listdir(base_path)):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            case_names.append(item)
    return case_names


def analyze_case(args_tuple):
    """
    Analyze a single case to get trajectory count and frame lengths.
    Directory structure: {base_path}/{case_name}/{case_name}/{traj_id}/{frame_id}/files
    Returns: (case_name, trajectory_count, trajectory_frames_dict, frame_lengths_dict)

    args_tuple can be:
      (base_path, case_name)              - scan all trajectories
      (base_path, case_name, traj_list)   - only analyze specified trajectories
    """
    if len(args_tuple) == 3:
        base_path, case_name, specified_trajs = args_tuple
    else:
        base_path, case_name = args_tuple
        specified_trajs = None

    case_path = os.path.join(base_path, case_name, case_name)

    if not os.path.exists(case_path):
        return case_name, None, None, None

    if specified_trajs is not None:
        # Only analyze specified trajectories
        trajectory_dirs = sorted(specified_trajs)
    else:
        # Scan all trajectory subdirectories
        trajectory_dirs = []
        try:
            for item in os.listdir(case_path):
                item_path = os.path.join(case_path, item)
                if os.path.isdir(item_path) and item.isdigit():
                    trajectory_dirs.append(int(item))
        except Exception:
            return case_name, None, None, None
        trajectory_dirs.sort()

    if not trajectory_dirs:
        return case_name, 0, {}, {}

    trajectory_count = len(trajectory_dirs)

    trajectory_frames = {}
    frame_lengths = defaultdict(list)

    for traj_num in trajectory_dirs:
        traj_path = os.path.join(case_path, str(traj_num))

        frame_dirs = []
        try:
            for item in os.listdir(traj_path):
                item_path = os.path.join(traj_path, item)
                if os.path.isdir(item_path) and item.isdigit() and len(item) == 4:
                    frame_dirs.append(int(item))
        except Exception:
            continue

        if frame_dirs:
            max_frame = max(frame_dirs)
            trajectory_frames[str(traj_num)] = max_frame
            frame_lengths[max_frame].append(traj_num)

    return case_name, trajectory_count, trajectory_frames, dict(frame_lengths)


def main():
    parser = argparse.ArgumentParser(description='Analyze panoramic data cases')
    parser.add_argument('--base_path', type=str, default=DEFAULT_BASE_PATH,
                        help='Base directory containing case folders '
                             '(default: @repo/data/UE_PanoramicOut)')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Path to a text file. Supports two formats: '
                             '1) one case_name per line; '
                             '2) case_name/traj_id per line (e.g. xerichammockbrush_yzy_16/19). '
                             'If not provided, cases are auto-scanned from base_path.')
    parser.add_argument('--output_file', type=str,
                        default=_release_path('@repo/data/analysis_results.json'),
                        help='Path to output JSON file for results')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count)')
    args = parser.parse_args()

    # Determine case list and optional per-case trajectory filter
    case_traj_map = None  # None means analyze all trajectories per case
    if args.input_file:
        # Detect format: check if lines contain '/'
        with open(args.input_file, 'r') as f:
            sample_lines = [l.strip() for l in f.readlines()[:20] if l.strip()]
        has_slash = any('/' in l for l in sample_lines)

        if has_slash:
            # New format: case_name/traj_id per line
            case_traj_map = read_case_trajectories_from_file(args.input_file)
            case_names = sorted(case_traj_map.keys())
        else:
            # Old format: one case_name per line
            case_names = read_case_names_from_file(args.input_file)
    else:
        case_names = scan_case_names_from_dir(args.base_path)

    total_cases = len(case_names)
    num_workers = args.workers if args.workers else cpu_count()

    # Build task arguments
    if case_traj_map is not None:
        task_args = [(args.base_path, cn, case_traj_map[cn]) for cn in case_names]
        total_trajs = sum(len(v) for v in case_traj_map.values())
        print(f"Input: {total_cases} cases, {total_trajs} trajectories (from case/traj format)")
    else:
        task_args = [(args.base_path, cn) for cn in case_names]
        print(f"Input: {total_cases} cases (scan all trajectories)")

    # Analyze cases in parallel with tqdm progress bar
    results = {}
    failed_count = 0
    inconsistent_count = 0

    with Pool(processes=num_workers) as pool:
        for case_name, traj_count, trajectory_frames, frame_lengths in tqdm(
            pool.imap_unordered(analyze_case, task_args),
            total=total_cases,
            desc="Analyzing cases",
        ):
            if traj_count is None:
                failed_count += 1
                continue

            unique_lengths = list(frame_lengths.keys()) if frame_lengths else []
            if len(unique_lengths) > 1:
                inconsistent_count += 1

            results[case_name] = {
                'trajectory_count': traj_count,
                'trajectory_frames': trajectory_frames,
                'frame_lengths': frame_lengths,
            }

    # Build output data
    output_data = {}
    for case_name in case_names:
        if case_name in results:
            data = results[case_name]
            output_data[case_name] = {
                'trajectory_count': data['trajectory_count'],
                'trajectories': data['trajectory_frames'],
            }

    # Save results
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Overall statistics
    all_frame_lengths = []
    total_trajectory_count = 0
    for data in results.values():
        total_trajectory_count += data['trajectory_count']
        for frame_len in data['trajectory_frames'].values():
            all_frame_lengths.append(frame_len)

    length_distribution = defaultdict(int)
    for fl in all_frame_lengths:
        length_distribution[fl] += 1
    # Sort by frame length
    sorted_dist = sorted(length_distribution.items(), key=lambda x: x[0])

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Total cases:        {total_cases}")
    print(f"  Analyzed:           {len(results)}")
    print(f"  Failed:             {failed_count}")
    print(f"  Inconsistent:       {inconsistent_count}")
    print(f"  Total trajectories: {total_trajectory_count}")
    print(f"  Total with frames:  {len(all_frame_lengths)}")
    if all_frame_lengths:
        print(f"  Min frame length:   {min(all_frame_lengths)}")
        print(f"  Max frame length:   {max(all_frame_lengths)}")
        print(f"  Avg frame length:   {sum(all_frame_lengths) / len(all_frame_lengths):.1f}")
    print(f"{'='*60}")
    print(f"  Frame length distribution:")
    for length, count in sorted_dist:
        pct = count / len(all_frame_lengths) * 100
        bar = '█' * int(pct / 2)
        print(f"    {length:>6d} frames: {count:>6d} trajectories ({pct:5.1f}%) {bar}")
    print(f"{'='*60}")
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
