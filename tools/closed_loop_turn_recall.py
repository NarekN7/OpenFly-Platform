#!/usr/bin/env python3
"""GT left/right turn recall from closed-loop predictions.json (post-hoc, no inference)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _action_in(actions: Optional[Sequence[int]], action_id: int) -> bool:
    if not actions:
        return False
    return action_id in actions


def _recall_for_action(rows: List[Dict[str, Any]], action_id: int) -> Dict[str, Any]:
    n_cases = 0
    n_success = 0
    for row in rows:
        if row.get("image_error"):
            continue
        gt = row.get("gt_actions")
        pred = row.get("predicted_actions")
        if not _action_in(gt, action_id):
            continue
        n_cases += 1
        if _action_in(pred, action_id):
            n_success += 1
    recall = (n_success / n_cases) if n_cases else None
    return {"n_cases": n_cases, "n_success": n_success, "recall": recall}


def turn_recall_from_predictions(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Trajectory-level presence recall: GT contains 2/3 -> pred contains 2/3."""
    left = _recall_for_action(rows, 2)
    right = _recall_for_action(rows, 3)
    return {"left": left, "right": right}


def _load_predictions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return data


def turn_recall_from_run_root(closed_loop_root: Path) -> Dict[str, Any]:
    """Aggregate turn recall from closed_loop/<env>/predictions.json."""
    if not closed_loop_root.is_dir():
        return {"overall": turn_recall_from_predictions([]), "envs": {}}

    all_rows: List[Dict[str, Any]] = []
    envs: Dict[str, Any] = {}
    for env_dir in sorted(closed_loop_root.iterdir()):
        if not env_dir.is_dir():
            continue
        pred_path = env_dir / "predictions.json"
        if not pred_path.is_file():
            continue
        rows = _load_predictions(pred_path)
        envs[env_dir.name] = turn_recall_from_predictions(rows)
        all_rows.extend(rows)

    return {"overall": turn_recall_from_predictions(all_rows), "envs": envs}


def format_recall_line(side: Dict[str, Any], label: str) -> str:
    recall = side.get("recall")
    n_success = side.get("n_success", 0)
    n_cases = side.get("n_cases", 0)
    if recall is None:
        return f"{label}=— (0 cases)"
    return f"{label}={100.0 * recall:.1f}% ({n_success}/{n_cases})"
