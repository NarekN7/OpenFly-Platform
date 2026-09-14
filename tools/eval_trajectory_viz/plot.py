"""Render top-down pseudo maps for GT vs predicted trajectories."""

from __future__ import annotations

import io
from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from physics import (
    DEFAULT_F9_STEP_MULT,
    SUCCESS_RADIUS_M,
    first_divergence_step,
    get_f9_step_mult,
    replay_trajectory,
)

# Fixed output size for every cell in the grid (pixels).
FIGSIZE = (3.0, 3.0)
DPI = 100
OUTPUT_PX = int(FIGSIZE[0] * DPI)
VIEW_LIMIT = 1.08  # normalized coords: content fits in [-VIEW_LIMIT, VIEW_LIMIT]


def _xy_path(poses: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.array([p[0] for p in poses], dtype=float)
    ys = np.array([p[1] for p in poses], dtype=float)
    return xs, ys


def _square_viewport(
    all_x: np.ndarray,
    all_y: np.ndarray,
    target_xyz: Sequence[float],
    *,
    padding_frac: float = 0.14,
) -> Tuple[float, float, float]:
    """Return center (cx, cy) and half-span for a square viewport."""
    pad = SUCCESS_RADIUS_M + 5.0
    x_min = float(min(all_x.min(), target_xyz[0] - pad))
    x_max = float(max(all_x.max(), target_xyz[0] + pad))
    y_min = float(min(all_y.min(), target_xyz[1] - pad))
    y_max = float(max(all_y.max(), target_xyz[1] + pad))

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    half = max(x_max - x_min, y_max - y_min) * 0.5
    half = max(half, SUCCESS_RADIUS_M + 8.0)
    half *= 1.0 + padding_frac
    return cx, cy, half


def _normalize_xy(xs: np.ndarray, ys: np.ndarray, cx: float, cy: float, half: float) -> Tuple[np.ndarray, np.ndarray]:
    return (xs - cx) / half, (ys - cy) / half


def _normalize_point(x: float, y: float, cx: float, cy: float, half: float) -> Tuple[float, float]:
    return (x - cx) / half, (y - cy) / half


def render_pseudo_map(
    start_xyz: Sequence[float],
    target_xyz: Sequence[float],
    start_yaw: float,
    gt_actions: Sequence[int],
    pred_actions: Sequence[int],
    success: int,
    *,
    current_step: Optional[int] = None,
    pred_f9_step_mult: Optional[int] = None,
    gt_f9_step_mult: int = DEFAULT_F9_STEP_MULT,
    figsize: Tuple[float, float] = FIGSIZE,
    dpi: int = DPI,
) -> bytes:
    # GT from seenx9 is planned with default F9=27 m; preds may use 18 m (f918m).
    pred_mult = get_f9_step_mult() if pred_f9_step_mult is None else int(pred_f9_step_mult)
    gt_path = replay_trajectory(gt_actions, start_xyz, start_yaw, f9_step_mult=gt_f9_step_mult)
    pred_path = replay_trajectory(pred_actions, start_xyz, start_yaw, f9_step_mult=pred_mult)

    gt_x, gt_y = _xy_path(gt_path)
    pred_x, pred_y = _xy_path(pred_path)

    all_x = np.concatenate([gt_x, pred_x, [start_xyz[0], target_xyz[0]]])
    all_y = np.concatenate([gt_y, pred_y, [start_xyz[1], target_xyz[1]]])
    cx, cy, half = _square_viewport(all_x, all_y, target_xyz)

    gt_x, gt_y = _normalize_xy(gt_x, gt_y, cx, cy, half)
    pred_x, pred_y = _normalize_xy(pred_x, pred_y, cx, cy, half)
    start_nx, start_ny = _normalize_point(start_xyz[0], start_xyz[1], cx, cy, half)
    target_nx, target_ny = _normalize_point(target_xyz[0], target_xyz[1], cx, cy, half)
    circle_r = SUCCESS_RADIUS_M / half

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="#ffffff")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="#f8fafc")

    ax.plot(gt_x, gt_y, color="#2563eb", linewidth=2.0, solid_capstyle="round", zorder=2)
    ax.scatter(gt_x, gt_y, color="#2563eb", s=14, zorder=3)

    pred_color = "#16a34a" if success else "#dc2626"
    ax.plot(pred_x, pred_y, color=pred_color, linewidth=2.0, solid_capstyle="round", zorder=2)
    ax.scatter(pred_x, pred_y, color=pred_color, s=14, zorder=3)

    div_step = first_divergence_step(gt_actions, pred_actions)
    if div_step is not None and div_step + 1 < len(pred_path):
        div_pose = pred_path[div_step + 1]
        div_nx, div_ny = _normalize_point(div_pose[0], div_pose[1], cx, cy, half)
        ax.scatter(
            [div_nx],
            [div_ny],
            s=110,
            facecolors="none",
            edgecolors="#f59e0b",
            linewidths=2.2,
            zorder=5,
        )

    ax.scatter(
        [start_nx],
        [start_ny],
        marker="^",
        s=90,
        color="#22c55e",
        edgecolors="black",
        linewidths=0.6,
        zorder=6,
    )
    ax.scatter(
        [target_nx],
        [target_ny],
        marker="*",
        s=160,
        color="#ef4444",
        edgecolors="black",
        linewidths=0.6,
        zorder=6,
    )

    if current_step is not None and 0 <= current_step < len(pred_path):
        cur = pred_path[current_step]
        cur_nx, cur_ny = _normalize_point(cur[0], cur[1], cx, cy, half)
        ax.scatter(
            [cur_nx],
            [cur_ny],
            s=180,
            facecolors="#a855f7",
            edgecolors="white",
            linewidths=1.5,
            zorder=7,
            marker="o",
        )

    success_circle = Circle(
        (target_nx, target_ny),
        circle_r,
        fill=False,
        linestyle="--",
        linewidth=1.4,
        edgecolor="#ef4444",
        alpha=0.85,
        zorder=1,
    )
    ax.add_patch(success_circle)

    lim = VIEW_LIMIT
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(1.0)

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches=None,
        pad_inches=0,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    buf.seek(0)

    # Hard guarantee: every tile is exactly OUTPUT_PX x OUTPUT_PX.
    try:
        from PIL import Image

        img = Image.open(buf).convert("RGBA")
        if img.size != (OUTPUT_PX, OUTPUT_PX):
            img = img.resize((OUTPUT_PX, OUTPUT_PX), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        buf.seek(0)
        return buf.read()


def render_pseudo_map_from_record(record, current_step: Optional[int] = None, **kwargs) -> Optional[bytes]:
    if not record.start_xyz or not record.target_xyz:
        return None
    return render_pseudo_map(
        record.start_xyz,
        record.target_xyz,
        record.start_yaw,
        record.gt_actions,
        record.predicted_actions,
        record.success,
        current_step=current_step,
        **kwargs,
    )
