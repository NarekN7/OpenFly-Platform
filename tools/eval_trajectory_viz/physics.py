"""Trajectory replay physics matching train/eval.py (tier2 action vocab).

Action 10 (F9) distance is configurable:
  mult=9 → 27 m (train/eval.py default)
  mult=6 → 18 m (eval_f9_18m.py / f918m runs)
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence

SUCCESS_RADIUS_M = 20.0
STEP_SIZE = 3.0
# Default matches train/eval.py (9 * 3 m = 27 m).
DEFAULT_F9_STEP_MULT = 9
F9_18M_STEP_MULT = 6  # 6 * 3 m = 18 m

_F9_STEP_MULT = DEFAULT_F9_STEP_MULT


def calculate_distance(point1: Sequence[float], point2: Sequence[float]) -> float:
    return math.sqrt(
        (point2[0] - point1[0]) ** 2
        + (point2[1] - point1[1]) ** 2
        + (point2[2] - point1[2]) ** 2
    )


def set_f9_step_mult(mult: int) -> None:
    """Set process-wide F9 (action 10) step multiplier used by replay/capture."""
    global _F9_STEP_MULT
    _F9_STEP_MULT = int(mult)


def get_f9_step_mult() -> int:
    return _F9_STEP_MULT


def f9_distance_m(mult: Optional[int] = None) -> float:
    m = _F9_STEP_MULT if mult is None else int(mult)
    return STEP_SIZE * m


def get_pose_after_make_action(
    new_pose: Sequence[float],
    action: int,
    *,
    f9_step_mult: Optional[int] = None,
) -> List[float]:
    x, y, z, yaw = new_pose
    step_size = STEP_SIZE
    f9_mult = _F9_STEP_MULT if f9_step_mult is None else int(f9_step_mult)

    if action == 0:
        pass
    elif action == 1:
        x += step_size * math.cos(yaw)
        y += step_size * math.sin(yaw)
    elif action == 2:
        yaw += math.radians(30)
    elif action == 3:
        yaw -= math.radians(30)
    elif action == 4:
        z += step_size
    elif action == 5:
        z -= step_size
    elif action == 6:
        x -= step_size * math.sin(yaw)
        y += step_size * math.cos(yaw)
    elif action == 7:
        x += step_size * math.sin(yaw)
        y -= step_size * math.cos(yaw)
    elif action == 8:
        x += step_size * math.cos(yaw) * 2
        y += step_size * math.sin(yaw) * 2
    elif action == 9:
        x += step_size * math.cos(yaw) * 3
        y += step_size * math.sin(yaw) * 3
    elif action == 10:
        x += step_size * math.cos(yaw) * f9_mult
        y += step_size * math.sin(yaw) * f9_mult

    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    return [x, y, z, yaw]


def replay_trajectory(
    actions: Iterable[int],
    start_xyz: Sequence[float],
    start_yaw: float,
    *,
    f9_step_mult: Optional[int] = None,
) -> List[List[float]]:
    """Return pose list including start; each pose is [x, y, z, yaw]."""
    pose = [start_xyz[0], start_xyz[1], start_xyz[2], start_yaw]
    path = [pose.copy()]
    for action in actions:
        if action == 0:
            break
        pose = get_pose_after_make_action(pose, int(action), f9_step_mult=f9_step_mult)
        path.append(pose.copy())
    return path


def first_divergence_step(gt_actions: Sequence[int], pred_actions: Sequence[int]) -> int | None:
    """Return 0-based step index where action lists first differ, or None if identical."""
    for i, (gt, pred) in enumerate(zip(gt_actions, pred_actions)):
        if gt != pred:
            return i
    if len(gt_actions) != len(pred_actions):
        return min(len(gt_actions), len(pred_actions))
    return None
