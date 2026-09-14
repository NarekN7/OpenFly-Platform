"""Load eval run artifacts and join predictions with console.log poses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SAMPLE_HEADER_RE = re.compile(
    r"^Sample (\d+): \[([^\]]+)\] -> \[([^\]]+)\], initial heading: ([-\d.eE+]+)\s*$"
)
FLOAT_LIST_RE = re.compile(r"[-\d.eE+]+")

_REPO_ROOT = Path(__file__).resolve().parents[2]
GT_X9_JSON_CANDIDATES = (
    _REPO_ROOT / "data_curated/seenx9.json",
    Path("/nfs/np/mnt/xtb/vln/data_curated/seenx9.json"),
    Path("/mnt/xtb/vln/data_curated/seenx9.json"),
)


@dataclass
class SamplePose:
    sample_index: int
    start_xyz: List[float]
    target_xyz: List[float]
    start_yaw: float


@dataclass
class TrajectoryRecord:
    sample_index: int
    environment: str
    image_path: str
    gpt_instruction: str
    predicted_actions: List[int]
    gt_actions: List[int]
    num_steps: int
    final_distance: float
    success: int
    spl: float
    osr_hit: int
    stopped_by_model: bool
    hit_max_steps: bool
    image_error: bool
    start_xyz: List[float] = field(default_factory=list)
    target_xyz: List[float] = field(default_factory=list)
    start_yaw: float = 0.0


def _parse_float_list(text: str) -> List[float]:
    return [float(x) for x in FLOAT_LIST_RE.findall(text)]


def parse_console_log(console_path: Path) -> Dict[int, SamplePose]:
    poses: Dict[int, SamplePose] = {}
    if not console_path.is_file():
        return poses

    with console_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = SAMPLE_HEADER_RE.match(line.strip())
            if not m:
                continue
            idx = int(m.group(1))
            start_xyz = _parse_float_list(m.group(2))
            target_xyz = _parse_float_list(m.group(3))
            start_yaw = float(m.group(4))
            if len(start_xyz) < 3 or len(target_xyz) < 3:
                continue
            poses[idx] = SamplePose(
                sample_index=idx,
                start_xyz=start_xyz[:3],
                target_xyz=target_xyz[:3],
                start_yaw=start_yaw,
            )
    return poses


def resolve_run_root(run_root: Path) -> Path:
    """Return directory containing env_*/predictions.json (supports closed_loop/ layout)."""
    if not run_root.is_dir():
        return run_root
    nested = run_root / "closed_loop"
    if nested.is_dir() and discover_env_dirs(nested):
        return nested
    if discover_env_dirs(run_root):
        return run_root
    return run_root


def discover_env_dirs(run_root: Path) -> List[Path]:
    if not run_root.is_dir():
        return []
    env_dirs = []
    for child in sorted(run_root.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.startswith("env_"):
            continue
        if (child / "predictions.json").is_file():
            env_dirs.append(child)
    return env_dirs


def load_env_records(env_dir: Path) -> List[TrajectoryRecord]:
    predictions_path = env_dir / "predictions.json"
    console_path = env_dir / "console.log"
    if not predictions_path.is_file():
        return []

    with predictions_path.open("r", encoding="utf-8") as f:
        predictions = json.load(f)

    poses = parse_console_log(console_path)
    records: List[TrajectoryRecord] = []

    for row in predictions:
        sample_index = int(row["sample_index"])
        pose = poses.get(sample_index)
        record = TrajectoryRecord(
            sample_index=sample_index,
            environment=str(row.get("environment", env_dir.name)),
            image_path=str(row.get("image_path", "")),
            gpt_instruction=str(row.get("gpt_instruction", "")),
            predicted_actions=[int(a) for a in row.get("predicted_actions", [])],
            gt_actions=[int(a) for a in row.get("gt_actions", [])],
            num_steps=int(row.get("num_steps", 0)),
            final_distance=float(row.get("final_distance", 0.0)),
            success=int(row.get("success", 0)),
            spl=float(row.get("spl", 0.0)),
            osr_hit=int(row.get("osr_hit", 0)),
            stopped_by_model=bool(row.get("stopped_by_model", False)),
            hit_max_steps=bool(row.get("hit_max_steps", False)),
            image_error=bool(row.get("image_error", False)),
        )
        if pose is not None:
            record.start_xyz = pose.start_xyz
            record.target_xyz = pose.target_xyz
            record.start_yaw = pose.start_yaw
        records.append(record)

    return records


def resolve_gt_x9_json(gt_json_path: Optional[str] = None) -> Optional[Path]:
    if gt_json_path:
        path = Path(gt_json_path).expanduser()
        return path if path.is_file() else None
    for candidate in GT_X9_JSON_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def load_gt_action_index(gt_json: Path) -> Dict[str, List[int]]:
    with gt_json.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    index: Dict[str, List[int]] = {}
    for row in rows:
        image_path = row.get("image_path")
        action = row.get("action")
        if not image_path or not isinstance(action, list):
            continue
        index[str(image_path)] = [int(a) for a in action]
    return index


def apply_gt_x9_overlay(
    records: List[TrajectoryRecord],
    gt_index: Dict[str, List[int]],
) -> Tuple[int, int]:
    """Replace gt_actions from tier2/x9 JSON keyed by image_path. Returns (matched, missing)."""
    matched = 0
    missing = 0
    for record in records:
        gt = gt_index.get(record.image_path)
        if gt is None:
            missing += 1
            continue
        record.gt_actions = gt
        matched += 1
    return matched, missing


def logged_eval_json_from_run(run_root: Path) -> Optional[str]:
    resolved = resolve_run_root(run_root)
    for env_dir in discover_env_dirs(resolved):
        metrics = load_metrics(env_dir)
        if metrics and metrics.get("eval_json"):
            return str(metrics["eval_json"])
    return None


def load_eval_run(
    run_root: Path,
    gt_json_path: Optional[str] = None,
    use_gt_x9: bool = True,
) -> Tuple[List[TrajectoryRecord], dict]:
    records: List[TrajectoryRecord] = []
    resolved = resolve_run_root(run_root)
    for env_dir in discover_env_dirs(resolved):
        records.extend(load_env_records(env_dir))

    meta = {
        "logged_eval_json": logged_eval_json_from_run(run_root),
        "gt_x9_json": None,
        "gt_x9_matched": 0,
        "gt_x9_missing": 0,
    }
    if use_gt_x9:
        gt_json = resolve_gt_x9_json(gt_json_path)
        if gt_json is not None:
            gt_index = load_gt_action_index(gt_json)
            matched, missing = apply_gt_x9_overlay(records, gt_index)
            meta["gt_x9_json"] = str(gt_json.resolve())
            meta["gt_x9_matched"] = matched
            meta["gt_x9_missing"] = missing
    return records, meta


def load_eval_run_records(run_root: Path, **kwargs) -> List[TrajectoryRecord]:
    records, _ = load_eval_run(run_root, **kwargs)
    return records


def load_metrics(env_dir: Path) -> Optional[dict]:
    metrics_path = env_dir / "metrics.json"
    if not metrics_path.is_file():
        return None
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_run(run_root: Path) -> dict:
    resolved = resolve_run_root(run_root)
    env_dirs = discover_env_dirs(resolved)
    summary = {
        "run_root": str(run_root),
        "resolved_root": str(resolved),
        "num_envs": len(env_dirs),
        "envs": [],
        "total_samples": 0,
    }
    for env_dir in env_dirs:
        records = load_env_records(env_dir)
        metrics = load_metrics(env_dir)
        env_summary = {
            "name": env_dir.name,
            "num_samples": len(records),
            "metrics": metrics,
        }
        summary["envs"].append(env_summary)
        summary["total_samples"] += len(records)
    return summary
