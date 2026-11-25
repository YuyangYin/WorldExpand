#!/usr/bin/env python3
"""step4_build_four_view_json.py
================================

把 step3.0_tuples.json（tuple-v1 schema）转换成下游训练可用的 four-view JSON。

输入 case（step3.0 格式）：
    {intersection_id, scene, main_trajectory, companion_trajectory,
     center_frame,
     center/front1/back1/front2/back2: {trajectory, frame}}

输出 case（目标格式）：
    {case_id, directory, scene, caption, version,
     main_trajectory, companion_trajectory, center_frame,
     start: {jpg, exr, json},          # 相对 DATA_ROOT
     start_yaw: float,                  # 从 center.json 读取
     views: [
       {role, trajectory,
        end: {jpg, exr, json},
        metadata: {end_yaw_deg, end_position_recentered: {X,Y,Z},
                   distance_cm, end_yaw_raw_deg}}
     ]}

关键设计
--------
1. 版本探测（v3 → v2 → legacy）：probe center 帧 .jpg 是否存在。
2. 路径规则与 dataset_utils/paths.py 一致。
3. Caption（--caption_mode，默认 frame_long_long）：
   frame_long_long：{caption_dir}/{scene}/(version/){scene}/{traj}/{frame}.json
                    → long_long_caption（读 main_trajectory 的 center_frame）。
   traj_long（旧）：{caption_dir}/{scene}/{traj}.json → long_caption / long caption。
   两种模式缺失都不删 case，置 ""。
4. Pose：从帧 .json 读 Position (cm) + Rotation.Yaw (deg)，yaw-only recenter。

三层磁盘 Cache（用于 debug 断点续传）
--------------------------------------
每步骤的计算结果写入 --cache_dir 下的三个 JSON 文件：

  version_cache.json   key: "scene/main_traj/center_frame"
                        val: "v3"|"v2"|""|"__missing__"

  pose_cache.json      key: "scene/version/traj/frame"
                        val: {"pos":{X,Y,Z}, "rot":{Pitch,Yaw,Roll}} | null（读取失败）

  caption_cache.json   key: frame_long_long 模式 "scene/version/traj/frame"
                             traj_long 模式      "scene/traj"
                        val: "caption text" 或 ""（空串表示文件缺失，已缓存，避免重读）

Workers 启动时通过全局变量拿到已有 cache dict，处理时先查 cache，
命中直接用；未命中才读磁盘，并将新结果附在返回值里带回主进程。
主进程合并后，在脚本结束时原子写回三个 cache 文件。

CLI
---
::

    python step4_build_four_view_json.py \\
        --input_json  $OUT/step3.0_tuples.json \\
        --data_root   $DATA_ROOT \\
        --caption_dir $CAPTION_DIR \\
        --output_json $OUT/step4_total.json \\
        --workers 32

    # Debug：指定 cache 目录 + 只处理前 200 条
    python step4_build_four_view_json.py \\
        --input_json  $OUT/step3.0_tuples.json \\
        --data_root   $DATA_ROOT \\
        --caption_dir $CAPTION_DIR \\
        --output_json /tmp/step4_debug.json \\
        --cache_dir   /tmp/step4_cache \\
        --workers 8 --limit 200
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# 路径工具（与 dataset_utils/paths.py 保持逻辑一致）
# ---------------------------------------------------------------------------

_VERSION_PROBE_ORDER = ("v3", "v2", "")
# sentinel：已探测过但三种版本都不存在
_VERSION_MISSING = "__missing__"


def _traj_dir_rel(scene: str, traj: str, version: str) -> str:
    if version:
        return f"{scene}/{version}/{scene}/{traj}"
    return f"{scene}/{scene}/{traj}"


def _frame_paths_rel(scene: str, traj: str, version: str, frame: str) -> Dict[str, str]:
    base = f"{_traj_dir_rel(scene, traj, version)}/{frame}/{frame}"
    return {"jpg": f"{base}.jpg", "exr": f"{base}.exr", "json": f"{base}.json"}


def _frame_paths_abs(scene: str, traj: str, version: str, frame: str,
                     data_root: str) -> Dict[str, str]:
    rel = _frame_paths_rel(scene, traj, version, frame)
    return {k: os.path.join(data_root, v) for k, v in rel.items()}


def _detect_version_raw(scene: str, traj: str, frame: str, data_root: str) -> str:
    """实际 probe 磁盘，返回 'v3'/'v2'/'' 或 _VERSION_MISSING。不访问 cache。"""
    for ver in _VERSION_PROBE_ORDER:
        jpg = os.path.join(data_root, _frame_paths_rel(scene, traj, ver, frame)["jpg"])
        if os.path.isfile(jpg):
            return ver
    return _VERSION_MISSING


def _case_directory(scene: str, traj: str, version: str) -> str:
    return _traj_dir_rel(scene, traj, version)


def _case_id(scene: str, main_traj: str, comp_traj: str, center_frame: str) -> str:
    return f"{scene}__{main_traj}__{comp_traj}__{center_frame}"


# ---------------------------------------------------------------------------
# Caption 工具
#
# 两种 caption 模式（--caption_mode）：
#
#   frame_long_long（默认）：
#     路径：{caption_dir}/{scene}/(version/){scene}/{traj}/{frame_id}.json
#           （与帧数据路径同构、版本感知，但 caption 文件直接是 {frame_id}.json）
#     字段：long_long_caption
#     读 main_trajectory 的 center_frame（start 帧）的 caption。
#
#   traj_long（旧约定，与 step4_build_total_json.py 一致）：
#     路径：{caption_dir}/{scene}/{traj}.json   （无版本后缀）
#     字段：long_caption 或 long caption
#     只读 main_trajectory 的 caption。
#
# 两种模式都：缺失不删 case，置 ""。
# ---------------------------------------------------------------------------

CAPTION_MODE_FRAME = "frame_long_long"
CAPTION_MODE_TRAJ = "traj_long"


def _caption_path(caption_dir: str, scene: str, traj: str,
                  version: str, frame: str, mode: str) -> str:
    """根据 caption 模式返回 caption JSON 路径。"""
    if mode == CAPTION_MODE_TRAJ:
        return os.path.join(caption_dir, scene, f"{traj}.json")
    # frame_long_long：版本感知，文件名直接是 {frame}.json
    return os.path.join(caption_dir, _traj_dir_rel(scene, traj, version), f"{frame}.json")


def _nearest_caption_frame(traj_dir: str, frame: str) -> Optional[str]:
    """在 caption 轨迹目录里找数值上最接近 frame 的 frame_id（不含扩展名）。
    目录不存在/空返回 None。平居时取较小 frame_id（确定性）。"""
    try:
        target = int(frame)
    except (TypeError, ValueError):
        return None
    best_name = None
    best_diff = None
    try:
        with os.scandir(traj_dir) as it:
            for e in it:
                if not e.name.endswith(".json"):
                    continue
                stem = e.name[:-5]
                try:
                    val = int(stem)
                except ValueError:
                    continue
                diff = abs(val - target)
                if best_diff is None or diff < best_diff or (diff == best_diff and val < int(best_name)):
                    best_diff = diff
                    best_name = stem
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return None
    return best_name


def _read_caption_field(p: str, mode: str) -> str:
    """读单个 caption json 的文本字段；失败返回 ''。"""
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        if mode == CAPTION_MODE_TRAJ:
            cap = d.get("long_caption") or d.get("long caption") or ""
        else:
            cap = d.get("long_long_caption") or ""
        return str(cap).strip() if cap else ""
    except Exception:
        return ""


def _load_caption_raw(caption_dir: str, scene: str, traj: str,
                      version: str, frame: str, mode: str) -> str:
    """实际读磁盘；找不到或解析失败时返回 ''。不访问 cache。

    frame_long_long 模式：若精确 {frame}.json 不存在，则在该轨迹目录里
    “就近”取数值最接近的 frame_id 的 caption；仍无则返回 ''。
    traj_long 模式不变。"""
    if not caption_dir:
        return ""
    p = _caption_path(caption_dir, scene, traj, version, frame, mode)
    if os.path.isfile(p):
        cap = _read_caption_field(p, mode)
        if cap:
            return cap
    # traj_long 没有“就近”语义，直接返回上面结果
    if mode == CAPTION_MODE_TRAJ:
        return ""
    # frame_long_long 回退：在轨迹目录找就近帧
    traj_dir = os.path.join(caption_dir, _traj_dir_rel(scene, traj, version))
    near = _nearest_caption_frame(traj_dir, frame)
    if near is None or near == frame:
        return ""
    return _read_caption_field(os.path.join(traj_dir, f"{near}.json"), mode)


def _caption_cache_key(scene: str, traj: str, version: str,
                       frame: str, mode: str) -> str:
    """caption cache key；frame 模式带版本+帧，traj 模式仅 scene/traj。"""
    if mode == CAPTION_MODE_TRAJ:
        return f"{scene}/{traj}"
    ver_tag = version if version else "legacy"
    return f"{scene}/{ver_tag}/{traj}/{frame}"


# ---------------------------------------------------------------------------
# Pose 工具
# ---------------------------------------------------------------------------

def _load_pose_raw(json_path: str) -> Optional[Dict[str, Any]]:
    """实际读帧 .json；失败返回 None，成功返回 {"pos":{X,Y,Z}, "rot":{...}}。
    不访问 cache。
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        pos = d["Position"]
        rot = d["Rotation"]
        return {
            "pos": {k: float(pos[k]) for k in ("X", "Y", "Z")},
            "rot": {k: float(rot[k]) for k in ("Pitch", "Yaw", "Roll")},
        }
    except Exception:
        return None


def _pose_cache_key(scene: str, version: str, traj: str, frame: str) -> str:
    ver_tag = version if version != _VERSION_MISSING else "missing"
    return f"{scene}/{ver_tag}/{traj}/{frame}"


# ---------------------------------------------------------------------------
# 坐标 recenter（yaw-only，与原始轨迹采样脚本一致）
# ---------------------------------------------------------------------------

def _recenter_end_pose(
    start_pos: Dict[str, float], start_rot: Dict[str, float],
    end_pos: Dict[str, float],   end_rot: Dict[str, float],
) -> Tuple[Dict[str, float], float]:
    y1 = math.radians(float(start_rot.get("Yaw", 0.0)))
    cos_y1, sin_y1 = math.cos(y1), math.sin(y1)
    dx = float(end_pos["X"]) - float(start_pos["X"])
    dy = float(end_pos["Y"]) - float(start_pos["Y"])
    dz = float(end_pos["Z"]) - float(start_pos["Z"])
    dx_local = cos_y1 * dx + sin_y1 * dy
    dy_local = -sin_y1 * dx + cos_y1 * dy
    raw_delta = float(end_rot.get("Yaw", 0.0)) - float(start_rot.get("Yaw", 0.0))
    end_yaw_delta = ((raw_delta + 180.0) % 360.0) - 180.0
    return {"X": dx_local, "Y": dy_local, "Z": dz}, end_yaw_delta


def _l2_cm(p1: Dict[str, float], p2: Dict[str, float]) -> float:
    return math.sqrt(
        (p1["X"] - p2["X"]) ** 2 +
        (p1["Y"] - p2["Y"]) ** 2 +
        (p1["Z"] - p2["Z"]) ** 2
    )


# ---------------------------------------------------------------------------
# DiskCache —— 轻量 JSON cache，供主进程持久化用
# ---------------------------------------------------------------------------

class DiskCache:
    """内存 dict + 原子写 JSON 文件。

    仅在主进程里直接调用 save()；子进程把新条目附在返回值里，
    由主进程 merge() 后统一 save()，避免多进程并发写。
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.data: Dict[str, Any] = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                print(f"  [cache] 已加载 {len(self.data)} 条 ← {path}")
            except Exception as e:
                print(f"  [cache] 读取失败（{e}），从空 cache 开始：{path}")

    def get(self, key: str, sentinel: Any = object()) -> Any:
        return self.data.get(key, sentinel)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def merge(self, new_entries: Dict[str, Any]) -> int:
        """合并子进程返回的新条目，返回新增数量。"""
        added = 0
        for k, v in new_entries.items():
            if k not in self.data:
                self.data[k] = v
                added += 1
        return added

    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + f".{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def __len__(self) -> int:
        return len(self.data)


# ---------------------------------------------------------------------------
# Worker 全局状态（每个子进程私有）
# ---------------------------------------------------------------------------

_W_DATA_ROOT: str = ""
_W_CAPTION_DIR: str = ""
# 子进程持有的只读 cache 快照（在 init 时拷入，子进程只读不写）
_W_VERSION_CACHE: Dict[str, str] = {}
_W_POSE_CACHE: Dict[str, Any] = {}
_W_CAPTION_CACHE: Dict[str, str] = {}
# pos_unit: "cm"（默认，不做处理）或 "m"（recentered XYZ ÷ 100）
_W_POS_UNIT: str = "cm"
# force_version: 非空时跳过磁盘探测，直接使用该版本字符串
_W_FORCE_VERSION: str = ""
# skip_pose: True 时跳过磁盘位姿读取，用零值填充
_W_SKIP_POSE: bool = False
# caption_mode: "frame_long_long"（默认）或 "traj_long"
_W_CAPTION_MODE: str = CAPTION_MODE_FRAME

NEIGHBOR_ROLES = ("front1", "back1", "front2", "back2")


def _worker_init(data_root: str, caption_dir: str,
                 version_snap: Optional[Dict], pose_snap: Optional[Dict],
                 caption_snap: Optional[Dict],
                 pos_unit: str = "cm",
                 force_version: str = "",
                 skip_pose: bool = False,
                 caption_mode: str = CAPTION_MODE_FRAME) -> None:
    global _W_DATA_ROOT, _W_CAPTION_DIR
    global _W_VERSION_CACHE, _W_POSE_CACHE, _W_CAPTION_CACHE
    global _W_POS_UNIT, _W_FORCE_VERSION, _W_SKIP_POSE, _W_CAPTION_MODE
    _W_DATA_ROOT     = data_root
    _W_CAPTION_DIR   = caption_dir
    # 仅在显式传入时覆盖；fork 模式下传 None，沿用父进程 copy-on-write
    # 继承的全局 cache，避免把 ~80MB cache pickle 给每个 worker（否则
    # worker 启动会卡在 init 很久）。
    if version_snap is not None:
        _W_VERSION_CACHE = version_snap
    if pose_snap is not None:
        _W_POSE_CACHE = pose_snap
    if caption_snap is not None:
        _W_CAPTION_CACHE = caption_snap
    _W_POS_UNIT      = pos_unit
    _W_FORCE_VERSION = force_version
    _W_SKIP_POSE     = skip_pose
    _W_CAPTION_MODE  = caption_mode


def _worker_fn(tup: Dict[str, Any]) -> Tuple[
    Optional[Dict[str, Any]],   # output case（None = 跳过）
    str,                        # status
    Dict[str, Dict],            # new cache entries: {"version":{}, "pose":{}, "caption":{}}
]:
    """子进程处理函数。cache 命中直接用，未命中写入 new_cache 带回主进程。"""
    new_cache: Dict[str, Dict] = {"version": {}, "pose": {}, "caption": {}}

    scene      = tup["scene"]
    main_traj  = str(tup["main_trajectory"])
    comp_traj  = str(tup.get("companion_trajectory", ""))
    center_frame = str(tup["center"]["frame"])

    # ── 步骤 1：版本探测（查 cache / probe 磁盘 / force_version 直接跳过）────
    ver_key = f"{scene}/{main_traj}/{center_frame}"
    if _W_FORCE_VERSION:
        # --force_version 指定时完全跳过磁盘探测
        version = _W_FORCE_VERSION
    elif ver_key in _W_VERSION_CACHE:
        version = _W_VERSION_CACHE[ver_key]
    else:
        version = _detect_version_raw(scene, main_traj, center_frame, _W_DATA_ROOT)
        new_cache["version"][ver_key] = version

    if version == _VERSION_MISSING:
        return None, "version_not_found", new_cache

    # ── 步骤 2：center Pose（查 cache / 读磁盘 / skip_pose 零值）────────
    _ZERO_POS = {"X": 0.0, "Y": 0.0, "Z": 0.0}
    _ZERO_ROT = {"Pitch": 0.0, "Yaw": 0.0, "Roll": 0.0}
    if _W_SKIP_POSE:
        pose_raw = {"pos": _ZERO_POS, "rot": _ZERO_ROT}
    else:
        start_abs = _frame_paths_abs(scene, main_traj, version, center_frame, _W_DATA_ROOT)
        pose_key  = _pose_cache_key(scene, version, main_traj, center_frame)
        if pose_key in _W_POSE_CACHE:
            pose_raw = _W_POSE_CACHE[pose_key]
        else:
            pose_raw = _load_pose_raw(start_abs["json"])
            new_cache["pose"][pose_key] = pose_raw  # None 也缓存（避免重复读必定失败的文件）

        if pose_raw is None:
            return None, "center_pose_fail", new_cache

    start_pos = pose_raw["pos"]
    start_rot = pose_raw["rot"]
    start_yaw = float(start_rot.get("Yaw", 0.0))

    # ── 步骤 3：Caption（查 cache / 读磁盘）──────────────────────────────
    # main trajectory 的 center_frame caption（只读 main，不读 companion）
    cap_key = _caption_cache_key(scene, main_traj, version, center_frame, _W_CAPTION_MODE)
    if cap_key in _W_CAPTION_CACHE:
        caption = _W_CAPTION_CACHE[cap_key]
    else:
        caption = _load_caption_raw(_W_CAPTION_DIR, scene, main_traj,
                                    version, center_frame, _W_CAPTION_MODE)
        new_cache["caption"][cap_key] = caption

    # ── 步骤 4：Views（各 neighbor 的路径 + Pose + recenter）─────────────
    start_rel = _frame_paths_rel(scene, main_traj, version, center_frame)
    views: List[Dict[str, Any]] = []

    for role in NEIGHBOR_ROLES:
        pos_info = tup.get(role)
        if not isinstance(pos_info, dict):
            continue
        v_traj  = str(pos_info.get("trajectory", ""))
        v_frame = str(pos_info["frame"])
        if not v_traj:
            v_traj = comp_traj if role in ("front2", "back2") else main_traj

        end_rel = _frame_paths_rel(scene, v_traj, version, v_frame)

        # Pose cache（skip_pose 时直接用零值，不读磁盘）
        if _W_SKIP_POSE:
            end_pose_raw = {"pos": _ZERO_POS, "rot": _ZERO_ROT}
        else:
            end_pose_key = _pose_cache_key(scene, version, v_traj, v_frame)
            if end_pose_key in _W_POSE_CACHE:
                end_pose_raw = _W_POSE_CACHE[end_pose_key]
            else:
                end_abs = _frame_paths_abs(scene, v_traj, version, v_frame, _W_DATA_ROOT)
                end_pose_raw = _load_pose_raw(end_abs["json"])
                new_cache["pose"][end_pose_key] = end_pose_raw

        if end_pose_raw is None:
            views.append({"role": role, "trajectory": v_traj,
                          "end": end_rel, "metadata": None})
            continue

        end_pos = end_pose_raw["pos"]
        end_rot = end_pose_raw["rot"]
        recentered, yaw_delta = _recenter_end_pose(start_pos, start_rot,
                                                   end_pos, end_rot)
        distance_cm = _l2_cm(start_pos, end_pos)

        # --pos_unit m：recentered XYZ 由 cm 转为 m（÷100）
        if _W_POS_UNIT == "m":
            recentered = {k: v / 100.0 for k, v in recentered.items()}

        views.append({
            "role":       role,
            "trajectory": v_traj,
            "end":        end_rel,
            "metadata": {
                "end_yaw_deg":              yaw_delta,
                "end_yaw_raw_deg":          float(end_rot.get("Yaw", 0.0)),
                "end_position_recentered":  recentered,
                "distance_cm":              distance_cm,
            },
        })

    # ── 步骤 5：组装输出 case ────────────────────────────────────────────
    ver_tag = version if version else "legacy"
    out_case: Dict[str, Any] = {
        "case_id":              _case_id(scene, main_traj, comp_traj, center_frame),
        "intersection_id":      tup.get("intersection_id", ""),
        "directory":            _case_directory(scene, main_traj, version),
        "scene":                scene,
        "version":              ver_tag,
        "caption":              caption,
        "main_trajectory":      main_traj,
        "companion_trajectory": comp_traj,
        "center_frame":         center_frame,
        "start":                start_rel,
        "start_yaw":            start_yaw,
        "views":                views,
    }
    return out_case, "ok", new_cache


# ---------------------------------------------------------------------------
# 多机分片工具
# ---------------------------------------------------------------------------

def _shard_suffix(machine_id: int, num_machines: int) -> str:
    """分片文件后缀；单机（num_machines<=1）返回空串（沿用原文件名）。"""
    if num_machines <= 1:
        return ""
    return f".shard{machine_id:02d}-of-{num_machines:02d}"


def _with_suffix(path: str, suffix: str) -> str:
    """在扩展名前插入 suffix，例如 a/b.json + .shard00-of-04 → a/b.shard00-of-04.json。"""
    if not suffix:
        return path
    stem, ext = os.path.splitext(path)
    return f"{stem}{suffix}{ext}"


def _merge_shards(args: argparse.Namespace) -> None:
    """合并模式：读取全部分片输出 + cache，去重合并成最终 output_json / cache。

    分片输出文件名规则与计算模式一致（``_with_suffix`` / ``_shard_suffix``），
    cases 按 case_id 去重，summary.counters / version_dist 重新汇总。
    各机的三个 cache 文件 union 回无后缀的 version/pose/caption_cache.json，
    方便后续单机重跑直接命中。
    """
    if args.num_machines < 2:
        print("[merge] num_machines<2，无需合并。")
        return

    cache_dir = args.cache_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.output_json)), "_step4_cache"
    )

    # ── 1) 合并 cases（去重） ────────────────────────────────────────────
    all_cases: List[Dict[str, Any]] = []
    seen_ids: set = set()
    dup = 0
    counters: Counter = Counter()
    missing: List[str] = []
    for i in range(args.num_machines):
        part = _with_suffix(args.output_json, _shard_suffix(i, args.num_machines))
        if not os.path.isfile(part):
            missing.append(part)
            continue
        with open(part, "r", encoding="utf-8") as f:
            d = json.load(f)
        for c in d.get("cases", []):
            cid = c.get("case_id")
            if cid in seen_ids:
                dup += 1
                continue
            seen_ids.add(cid)
            all_cases.append(c)
        for k, v in (d.get("summary", {}).get("counters", {}) or {}).items():
            counters[k] += int(v)
    if missing:
        print(f"[merge][warn] {len(missing)} 个分片输出缺失（跳过）：")
        for m in missing:
            print(f"    {m}")
    print(f"[merge] 合并 {args.num_machines} 个分片 → {len(all_cases)} 条 case"
          f"（去重移除 {dup} 条）")

    # ── 2) union 三个 cache（写回无后缀文件） ────────────────────────────
    def _union_cache(name: str) -> int:
        base = DiskCache(os.path.join(cache_dir, f"{name}.json"))
        for i in range(args.num_machines):
            p = os.path.join(cache_dir, f"{name}{_shard_suffix(i, args.num_machines)}.json")
            if not os.path.isfile(p):
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    base.merge(json.load(f))
            except Exception as e:
                print(f"[merge][warn] cache 读取失败 {p}：{e}")
        base.save()
        return len(base)

    if os.path.isdir(cache_dir):
        nv = _union_cache("version_cache")
        npose = _union_cache("pose_cache")
        ncap = _union_cache("caption_cache")
        print(f"[merge] cache union：version {nv}  pose {npose}  caption {ncap}")

    # ── 3) 写最终输出 ────────────────────────────────────────────────────
    ver_dist: Counter = Counter()
    caption_hit = 0
    for c in all_cases:
        ver_dist[c.get("version", "")] += 1
        if c.get("caption"):
            caption_hit += 1
    summary = {
        "schema": "four-view-v1",
        "total_cases": len(all_cases),
        "input_source": os.path.abspath(args.input_json),
        "data_root": args.data_root,
        "caption_dir": args.caption_dir or "",
        "caption_mode": args.caption_mode,
        "cache_dir": cache_dir,
        "version_dist": dict(ver_dist),
        "caption_hit": caption_hit,
        "counters": dict(counters),
        "pos_unit": args.pos_unit,
        "merged_from_shards": args.num_machines,
        "shards_missing": missing,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    tmp = args.output_json + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"cases": all_cases, "summary": summary},
                  f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, args.output_json)
    print(f"\n[merge][done] → {args.output_json}  ({len(all_cases)} cases，"
          f"版本分布 {dict(ver_dist)}，有 caption {caption_hit})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="step3.0_tuples.json → four-view training JSON"
    )
    ap.add_argument("--input_json",  required=True,
                    help="step3.0_tuples.json（tuple-v1 schema）。")
    ap.add_argument("--data_root",   required=True,
                    help="UE_PanoramicOut 挂载根目录。")
    ap.add_argument("--caption_dir", default="",
                    help="caption 根目录；留空则 caption=''。\n"
                         "frame_long_long 模式下指向 ...pano_chinese_captions/UE_PanoramicOut/")
    ap.add_argument("--caption_mode",
                    choices=[CAPTION_MODE_FRAME, CAPTION_MODE_TRAJ],
                    default=CAPTION_MODE_FRAME,
                    help="caption 取值模式：\n"
                         "  frame_long_long（默认）：\n"
                         "    {caption_dir}/{scene}/(version/){scene}/{traj}/{frame}.json\n"
                         "    读字段 long_long_caption（main_traj 的 center_frame）。\n"
                         "  traj_long（旧约定）：\n"
                         "    {caption_dir}/{scene}/{traj}.json，读 long_caption / long caption。")
    ap.add_argument("--output_json", required=True,
                    help="输出 JSON 路径。")
    ap.add_argument("--cache_dir",   default="",
                    help="三个 cache 文件的存放目录。\n"
                         "默认：<output_json 同级>/_step4_cache/\n"
                         "文件：version_cache.json / pose_cache.json / caption_cache.json")
    ap.add_argument("--workers", type=int, default=16,
                    help="并行 worker 数（默认 16；1 = 串行，方便 pdb 调试）。")
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理前 N 条（>0 时生效；debug 用）。")
    ap.add_argument("--overwrite", action="store_true",
                    help="输出已存在时强制覆盖。")
    ap.add_argument("--pos_unit", choices=["cm", "m"], default="cm",
                    help="end_position_recentered 的坐标单位。\n"
                         "  cm（默认）：保持 UE5 原始单位（厘米）。\n"
                         "  m          ：除以 100，转换为米。\n"
                         "distance_cm 字段始终保持厘米，不受影响。")
    ap.add_argument("--force_version", default="",
                    choices=["", "v3", "v2", "legacy"],
                    help="跳过磁盘版本探测，直接使用指定版本字符串构造路径。\n"
                         "在数据盘不可访问的环境（如纯路径生成/调试）中使用。\n"
                         "留空（默认）= 正常 probe 磁盘。")
    ap.add_argument("--skip_pose", action="store_true",
                    help="跳过所有 .json 位姿文件读取，用零值填充位姿字段。\n"
                         "配合 --force_version 在数据盘不可访问时生成结构正确的 JSON。\n"
                         "注意：输出的位姿数值无意义，仅用于路径/格式验证。")
    # ── 多机分片（默认 num_machines=1 即单机，行为完全不变）──────────────
    ap.add_argument("--num_machines", type=int, default=1,
                    help="多机分片总数（默认 1=单机）。>1 时每台只处理 1/N 分片\n"
                         "（strided 切片），输出写到 <output_json> 加 .shardXX-of-NN 后缀，\n"
                         "cache 文件同样带该后缀，互不覆盖。")
    ap.add_argument("--machine_id", type=int, default=0,
                    help="本机分片编号，0..num_machines-1。每台用不同值各跑一次。")
    ap.add_argument("--merge", action="store_true",
                    help="合并模式：不做任何计算，读取全部 .shardXX-of-NN 分片输出\n"
                         "（及其 cache）去重合并成最终 --output_json 与统一 cache。\n"
                         "需与 --num_machines / --output_json / --cache_dir 一致。")
    args = ap.parse_args()

    # ── 分片参数校验 ──────────────────────────────────────────────────────
    if args.num_machines < 1:
        ap.error("--num_machines 必须 >= 1")
    if not (0 <= args.machine_id < args.num_machines):
        ap.error(f"--machine_id 必须在 [0, {args.num_machines}) 内，当前 {args.machine_id}")

    # ── 合并模式：读分片 → 合并 → 写最终输出，提前返回 ───────────────────
    if args.merge:
        _merge_shards(args)
        return

    # 本机分片输出路径（单机时即 --output_json 原值）
    shard_suffix = _shard_suffix(args.machine_id, args.num_machines)
    out_path = _with_suffix(args.output_json, shard_suffix)

    # ── 输出检查 ─────────────────────────────────────────────────────────
    if os.path.isfile(out_path) and not args.overwrite:
        print(f"[skip] {out_path} 已存在，加 --overwrite 强制重跑。")
        sys.exit(0)

    # ── Cache 目录 ────────────────────────────────────────────────────────
    cache_dir = args.cache_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.output_json)), "_step4_cache"
    )
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[cache] 目录：{cache_dir}")

    # 多机时 cache 文件带分片后缀，各机互不覆盖；merge 模式再 union 回无后缀文件。
    ver_cache  = DiskCache(os.path.join(cache_dir, f"version_cache{shard_suffix}.json"))
    pose_cache = DiskCache(os.path.join(cache_dir, f"pose_cache{shard_suffix}.json"))
    cap_cache  = DiskCache(os.path.join(cache_dir, f"caption_cache{shard_suffix}.json"))

    # ── 读输入 ────────────────────────────────────────────────────────────
    print(f"[init] 读取 {args.input_json} …")
    with open(args.input_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    tuples: List[Dict[str, Any]] = raw.get("cases", [])
    if args.limit > 0:
        tuples = tuples[: args.limit]
    print(f"[init] 共 {len(tuples)} 条 tuple-v1 case。")

    # 多机分片：strided 切片（每台取 machine_id, machine_id+N, ... ），负载均衡。
    if args.num_machines > 1:
        _n_all = len(tuples)
        tuples = tuples[args.machine_id::args.num_machines]
        print(f"[shard] machine {args.machine_id}/{args.num_machines}: "
              f"本机 {len(tuples)}/{_n_all} 条（strided 切片）→ {out_path}", flush=True)

    # ── 并行处理 ──────────────────────────────────────────────────────────
    out_cases: List[Dict[str, Any]] = []
    counters: Counter = Counter()
    # 新 cache 条目（从子进程收集）
    new_ver:  Dict[str, str] = {}
    new_pose: Dict[str, Any] = {}
    new_cap:  Dict[str, str] = {}

    def _collect(case, status, nc):
        counters[status] += 1
        if case is not None:
            out_cases.append(case)
        new_ver.update(nc.get("version", {}))
        new_pose.update(nc.get("pose", {}))
        new_cap.update(nc.get("caption", {}))

    print(f"[init] pos_unit={args.pos_unit}  "
          f"(end_position_recentered 单位：{'cm' if args.pos_unit == 'cm' else 'm（÷100）'})")
    if args.force_version:
        print(f"[init] force_version={args.force_version}  (跳过磁盘版本探测)")
    if args.skip_pose:
        print(f"[init] skip_pose=True  (位姿字段用零值填充，不读磁盘)")
    print(f"[init] caption_mode={args.caption_mode}", flush=True)

    if args.workers <= 1:
        # 串行：直接在主进程运行，方便 pdb
        _worker_init(
            args.data_root, args.caption_dir,
            ver_cache.data, pose_cache.data, cap_cache.data,
            args.pos_unit, args.force_version, args.skip_pose, args.caption_mode,
        )
        for tup in tqdm(tuples, desc="step4[serial]"):
            case, status, nc = _worker_fn(tup)
            _collect(case, status, nc)
    else:
        # 把大体积只读 cache 直接挂到父进程全局；ProcessPoolExecutor 在
        # Linux 上默认 fork，子进程通过 copy-on-write 免费继承这些全局，
        # 不必把 ~80MB 的 pose/version cache pickle 给每个 worker。
        global _W_VERSION_CACHE, _W_POSE_CACHE, _W_CAPTION_CACHE
        _W_VERSION_CACHE = ver_cache.data
        _W_POSE_CACHE    = pose_cache.data
        _W_CAPTION_CACHE = cap_cache.data

        # cache 传 None → worker init 不覆盖 fork 继承的全局
        _init_args = (
            args.data_root, args.caption_dir,
            None, None, None,
            args.pos_unit, args.force_version, args.skip_pose, args.caption_mode,
        )
        print(f"[init] 启动 {args.workers} 个 worker（fork 继承 cache，无需 pickle）…",
              flush=True)
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=_init_args,
        ) as pool:
            # 用 submit 保留顺序索引，按输入顺序收集结果（不用 as_completed），
            # 确保两次相同输入产出相同顺序的 JSON。
            print(f"[init] 提交 {len(tuples)} 个任务…", flush=True)
            futs = [pool.submit(_worker_fn, tup) for tup in tuples]
            for fut in tqdm(futs, total=len(futs), desc="step4", mininterval=0.5):
                case, status, nc = fut.result()
                _collect(case, status, nc)

    # ── 合并 + 保存 cache ─────────────────────────────────────────────────
    added_ver  = ver_cache.merge(new_ver)
    added_pose = pose_cache.merge(new_pose)
    added_cap  = cap_cache.merge(new_cap)
    print(f"\n[cache] 新增条目：version +{added_ver}  pose +{added_pose}  caption +{added_cap}")
    print(f"[cache] 总计：version {len(ver_cache)}  pose {len(pose_cache)}  caption {len(cap_cache)}")
    ver_cache.save()
    pose_cache.save()
    cap_cache.save()
    print(f"[cache] 已写回 {cache_dir}")

    # ── 统计 ──────────────────────────────────────────────────────────────
    total = len(tuples)
    ok = counters["ok"]
    print(f"\n[done] {ok}/{total} 条成功")
    for k, v in sorted(counters.items()):
        if k != "ok":
            print(f"  {k}: {v}")

    ver_dist: Counter = Counter()
    caption_hit = 0
    for c in out_cases:
        ver_dist[c["version"]] += 1
        if c["caption"]:
            caption_hit += 1
    print(f"\n版本分布：{dict(ver_dist)}")
    print(f"有 caption：{caption_hit}/{ok} ({100*caption_hit/max(ok,1):.1f}%)")

    # ── 写输出（one-case-per-line，原子写）───────────────────────────────
    summary = {
        "schema": "four-view-v1",
        "total_cases": ok,
        "input_source": os.path.abspath(args.input_json),
        "data_root": args.data_root,
        "caption_dir": args.caption_dir or "",
        "caption_mode": args.caption_mode,
        "cache_dir": cache_dir,
        "version_dist": dict(ver_dist),
        "caption_hit": caption_hit,
        "counters": dict(counters),
        "fields": [
            "case_id", "intersection_id", "directory",
            "scene", "version", "caption",
            "main_trajectory", "companion_trajectory", "center_frame",
            "start (jpg/exr/json rel-path)", "start_yaw",
            "views[{role, trajectory, end{jpg/exr/json}, metadata{...}}]",
        ],
        "path_rule": {
            "v3/v2":  "{scene}/{version}/{scene}/{traj}/{frame}/{frame}.{ext}",
            "legacy": "{scene}/{scene}/{traj}/{frame}/{frame}.{ext}",
        },
        "pos_unit": args.pos_unit,
        "metadata_fields": [
            "end_yaw_deg (wrapped to [-180,180])",
            "end_yaw_raw_deg",
            f"end_position_recentered (X/Y/Z {args.pos_unit}, start-local yaw frame)",
            "distance_cm (XYZ L2 in cm, always cm regardless of --pos_unit)",
        ],
        "cache_files": {
            "version": os.path.join(cache_dir, f"version_cache{shard_suffix}.json"),
            "pose":    os.path.join(cache_dir, f"pose_cache{shard_suffix}.json"),
            "caption": os.path.join(cache_dir, f"caption_cache{shard_suffix}.json"),
        },
    }
    if args.num_machines > 1:
        summary["shard"] = {"machine_id": args.machine_id,
                            "num_machines": args.num_machines}

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"cases": out_cases, "summary": summary},
                  f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, out_path)
    print(f"\n[done] → {out_path}")
    if args.num_machines > 1:
        print(f"[shard] 本机分片完成；全部 {args.num_machines} 台跑完后，用 "
              f"--merge 合并：\n"
              f"        python {os.path.basename(__file__)} --merge "
              f"--num_machines {args.num_machines} "
              f"--output_json {args.output_json} --cache_dir {cache_dir} "
              f"--input_json {args.input_json} --data_root {args.data_root}")


if __name__ == "__main__":
    main()
