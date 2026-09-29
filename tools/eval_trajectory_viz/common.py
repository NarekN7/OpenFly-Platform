"""Shared helpers for eval trajectory viz (grid + frame viewer)."""

from __future__ import annotations

from urllib.parse import urlencode

from loader import TrajectoryRecord

# Streamlit URL path for pages/1_Frame_Viewer.py
FRAME_VIEWER_PATH = "Frame_Viewer"


def viewer_relative_url(environment: str, sample_index: int) -> str:
    """URL path that opens Frame Viewer (st.link_button opens it in a new tab)."""
    return "/" + FRAME_VIEWER_PATH + "?" + urlencode(
        {"env": environment, "sample": str(int(sample_index))}
    )


ACTION_LABELS = {
    0: "S",
    1: "F1",
    2: "L",
    3: "R",
    4: "U",
    5: "D",
    8: "F2",
    9: "F3",
    10: "F9",
}


def action_label(action: int) -> str:
    return ACTION_LABELS.get(int(action), str(action))


def format_actions(actions: list[int], max_items: int = 12) -> str:
    labels = [action_label(a) for a in actions]
    if len(labels) <= max_items:
        return "[" + ", ".join(labels) + "]"
    head = labels[:max_items]
    return "[" + ", ".join(head) + f"] ... (+{len(labels) - max_items})"


def record_to_dict(r: TrajectoryRecord) -> dict:
    return {
        "sample_index": r.sample_index,
        "environment": r.environment,
        "image_path": r.image_path,
        "gpt_instruction": r.gpt_instruction,
        "predicted_actions": r.predicted_actions,
        "gt_actions": r.gt_actions,
        "num_steps": r.num_steps,
        "final_distance": r.final_distance,
        "success": r.success,
        "spl": r.spl,
        "osr_hit": r.osr_hit,
        "stopped_by_model": r.stopped_by_model,
        "hit_max_steps": r.hit_max_steps,
        "image_error": r.image_error,
        "start_xyz": r.start_xyz,
        "target_xyz": r.target_xyz,
        "start_yaw": r.start_yaw,
    }


def dict_to_record(d: dict) -> TrajectoryRecord:
    return TrajectoryRecord(**d)


def record_key(d: dict) -> str:
    return f"{d['environment']}:{d['sample_index']}:{d['image_path']}"


def pitch_for_image_path(image_path: str) -> float:
    return -45.0 if "high" in image_path else 0.0
