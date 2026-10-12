#!/usr/bin/env python3
"""Terminal-to-landmark distance distribution for the active training set.

Training JSON (from current Qwen x9 sbatch files):
  data_curated/trainx9_curated_0.json

Each traj ends with stop (action 0). The terminal point is pos[-1] (pose after the
pre-last move, when stop is issued). The geographic target building is
aim_landmark.position in the matching pose.jsonl under data_wo_annotation.

Note: closed-loop eval NE in train/eval.py uses goal=pos[-1], so expert NE vs the
eval goal is identically 0. This script measures distance to the landmark target.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_pose_jsonl(text: str) -> list[dict]:
    objs: list[dict] = []
    brace = 0
    start = 0
    for i, ch in enumerate(text):
        if ch == "{":
            if brace == 0:
                start = i
            brace += 1
        elif ch == "}":
            brace -= 1
            if brace == 0:
                try:
                    objs.append(json.loads(text[start : i + 1]))
                except json.JSONDecodeError:
                    pass
    return objs


def altitude_bucket(image_path: str) -> str:
    # .../astar_data/<high|medium|low>_<short|average|long>/...
    parts = image_path.split("/")
    if len(parts) >= 3:
        split = parts[2]
        for a in ("high", "medium", "low"):
            if split.startswith(a):
                return a
    return "unknown"


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_json",
        type=Path,
        default=repo / "data_curated" / "trainx9_curated_0.json",
    )
    parser.add_argument(
        "--pose_root",
        type=Path,
        default=repo / "data_wo_annotation",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=repo / "slurm_logs" / "train_terminal_ne",
    )
    parser.add_argument("--success_m", type=float, default=20.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.train_json) as f:
        train = json.load(f)

    d3: list[float] = []
    dxy: list[float] = []
    dz: list[float] = []
    envs: list[str] = []
    alts: list[str] = []
    start_goal: list[float] = []
    missing_pose = 0
    missing_aim = 0

    for t in train:
        ip = t["image_path"]
        pose_path = args.pose_root / ip / "pose.jsonl"
        if not pose_path.is_file():
            missing_pose += 1
            continue
        objs = parse_pose_jsonl(pose_path.read_text())
        aims = [o["aim_landmark"] for o in objs if "aim_landmark" in o]
        if not aims or "position" not in aims[-1]:
            missing_aim += 1
            continue
        term = np.asarray(t["pos"][-1], dtype=float)
        aim = np.asarray(aims[-1]["position"], dtype=float)
        start = np.asarray(t["pos"][0], dtype=float)
        d3.append(float(np.linalg.norm(term - aim)))
        dxy.append(float(np.linalg.norm(term[:2] - aim[:2])))
        dz.append(float(abs(term[2] - aim[2])))
        envs.append(ip.split("/")[0])
        alts.append(altitude_bucket(ip))
        start_goal.append(float(np.linalg.norm(start - term)))

    d3_a = np.asarray(d3, dtype=float)
    dxy_a = np.asarray(dxy, dtype=float)
    dz_a = np.asarray(dz, dtype=float)
    sg_a = np.asarray(start_goal, dtype=float)
    thr = float(args.success_m)

    def summarize(arr: np.ndarray) -> dict:
        qs = np.percentile(arr, [0, 1, 5, 25, 50, 75, 95, 99, 100]).tolist()
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "p1": qs[1],
            "p5": qs[2],
            "p25": qs[3],
            "median": qs[4],
            "p75": qs[5],
            "p95": qs[6],
            "p99": qs[7],
            "max": qs[8],
            f"pct_lt_{thr:g}m": float((arr < thr).mean() * 100.0),
            f"pct_ge_{thr:g}m": float((arr >= thr).mean() * 100.0),
            "n_ge_50m": int((arr >= 50).sum()),
            "n_ge_100m": int((arr >= 100).sum()),
        }

    by_env = {}
    for e in sorted(set(envs)):
        m = np.asarray([x == e for x in envs])
        by_env[e] = {
            "n": int(m.sum()),
            "xy_pct_lt_20m": float((dxy_a[m] < thr).mean() * 100.0),
            "xy_median": float(np.median(dxy_a[m])),
            "d3_pct_lt_20m": float((d3_a[m] < thr).mean() * 100.0),
            "d3_median": float(np.median(d3_a[m])),
        }

    by_alt = {}
    for a in ("low", "medium", "high", "unknown"):
        m = np.asarray([x == a for x in alts])
        if not m.any():
            continue
        by_alt[a] = {
            "n": int(m.sum()),
            "xy_pct_lt_20m": float((dxy_a[m] < thr).mean() * 100.0),
            "xy_median": float(np.median(dxy_a[m])),
            "d3_pct_lt_20m": float((d3_a[m] < thr).mean() * 100.0),
            "d3_median": float(np.median(d3_a[m])),
        }

    summary = {
        "train_json": str(args.train_json),
        "pose_root": str(args.pose_root),
        "n_train": len(train),
        "n_analyzed": int(d3_a.size),
        "missing_pose": missing_pose,
        "missing_aim": missing_aim,
        "success_threshold_m": thr,
        "note": (
            "Distance = ||terminal pos[-1] - aim_landmark.position||. "
            "Eval NE in train/eval.py uses goal=pos[-1] (expert NE=0)."
        ),
        "terminal_to_landmark_3d": summarize(d3_a),
        "terminal_to_landmark_xy": summarize(dxy_a),
        "terminal_to_landmark_abs_z": summarize(dz_a),
        "start_to_terminal_3d": summarize(sg_a),
        "by_env": by_env,
        "by_altitude": by_alt,
        "env_counts": dict(Counter(envs)),
        "altitude_counts": dict(Counter(alts)),
    }

    summary_path = args.out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # --- plots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.hist(dxy_a, bins=80, color="#2c7fb8", edgecolor="white", linewidth=0.3)
    ax.axvline(thr, color="#d7301f", linestyle="--", linewidth=2, label=f"{thr:g} m")
    ax.set_xlabel("XY distance terminal → landmark (m)")
    ax.set_ylabel("Trajectories")
    ax.set_title("Horizontal (XY) terminal–landmark distance")
    ax.legend()

    ax = axes[0, 1]
    ax.hist(d3_a, bins=80, color="#41b6c4", edgecolor="white", linewidth=0.3)
    ax.axvline(thr, color="#d7301f", linestyle="--", linewidth=2, label=f"{thr:g} m")
    ax.set_xlabel("3D distance terminal → landmark (m)")
    ax.set_ylabel("Trajectories")
    ax.set_title("3D terminal–landmark distance")
    ax.legend()

    ax = axes[1, 0]
    for arr, label, color in (
        (dxy_a, "XY", "#2c7fb8"),
        (d3_a, "3D", "#41b6c4"),
    ):
        xs = np.sort(arr)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, color=color, linewidth=2, label=label)
    ax.axvline(thr, color="#d7301f", linestyle="--", linewidth=2, label=f"{thr:g} m")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of terminal–landmark distance")
    ax.set_xlim(0, max(120, float(np.percentile(d3_a, 99))))
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    # outlier-focused: fraction above thresholds
    thresholds = [10, 20, 30, 40, 50, 75, 100]
    pct_xy = [(dxy_a >= t).mean() * 100 for t in thresholds]
    pct_3d = [(d3_a >= t).mean() * 100 for t in thresholds]
    x = np.arange(len(thresholds))
    w = 0.38
    ax.bar(x - w / 2, pct_xy, width=w, color="#2c7fb8", label="XY ≥ thr")
    ax.bar(x + w / 2, pct_3d, width=w, color="#41b6c4", label="3D ≥ thr")
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in thresholds])
    ax.set_xlabel("Threshold (m)")
    ax.set_ylabel("% of trajectories")
    ax.set_title("Outlier rates (share beyond threshold)")
    ax.legend()

    fig.suptitle(
        f"Train terminal→landmark distances\n{args.train_json.name} (n={d3_a.size})",
        fontsize=12,
    )
    fig.tight_layout()
    hist_path = args.out_dir / "terminal_landmark_distance_hist_cdf.png"
    fig.savefig(hist_path, dpi=160)
    plt.close(fig)

    # by altitude
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    order = [a for a in ("low", "medium", "high") if a in by_alt]
    data_xy = [dxy_a[np.asarray([x == a for x in alts])] for a in order]
    data_3d = [d3_a[np.asarray([x == a for x in alts])] for a in order]
    axes[0].boxplot(data_xy, tick_labels=order, showfliers=False)
    axes[0].axhline(thr, color="#d7301f", linestyle="--", linewidth=1.5)
    axes[0].set_title("XY distance by altitude split")
    axes[0].set_ylabel("meters")
    axes[1].boxplot(data_3d, tick_labels=order, showfliers=False)
    axes[1].axhline(thr, color="#d7301f", linestyle="--", linewidth=1.5)
    axes[1].set_title("3D distance by altitude split")
    axes[1].set_ylabel("meters")
    fig.tight_layout()
    box_path = args.out_dir / "terminal_landmark_by_altitude.png"
    fig.savefig(box_path, dpi=160)
    plt.close(fig)

    # scatter start-goal vs terminal-landmark XY
    fig, ax = plt.subplots(figsize=(7, 5.5))
    # subsample for readability
    rng = np.random.default_rng(0)
    if dxy_a.size > 8000:
        idx = rng.choice(dxy_a.size, size=8000, replace=False)
    else:
        idx = np.arange(dxy_a.size)
    ax.scatter(sg_a[idx], dxy_a[idx], s=4, alpha=0.25, c="#2c7fb8", linewidths=0)
    ax.axhline(thr, color="#d7301f", linestyle="--", linewidth=1.5, label=f"XY={thr:g}m")
    ax.set_xlabel("Start → terminal Euclidean distance (m)")
    ax.set_ylabel("Terminal → landmark XY distance (m)")
    ax.set_title("Episode length vs terminal–landmark standoff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    scatter_path = args.out_dir / "start_goal_vs_terminal_landmark_xy.png"
    fig.savefig(scatter_path, dpi=160)
    plt.close(fig)

    print(f"Wrote {summary_path}")
    print(f"Wrote {hist_path}")
    print(f"Wrote {box_path}")
    print(f"Wrote {scatter_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
