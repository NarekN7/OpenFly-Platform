#!/usr/bin/env python3
"""Plot train+val loss for one run, or compare baseline / lr4x / lr16x on shared axes."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS: Dict[str, Tuple[str, str]] = {
    "baseline (2e-5)": (
        "/mnt/weka/nnurijanyan/checkpoints/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h0-10ep/runs/May20_05-06-33_gpu05",
        "#16a34a",
    ),
    "lr4x (8e-5)": (
        "/mnt/weka/nnurijanyan/checkpoints/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h0-10ep-lr4x/runs/May20_13-49-39_gpu01",
        "#2563eb",
    ),
    "lr16x (3.2e-4)": (
        "/mnt/weka/nnurijanyan/checkpoints/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h0-10ep-lr16x/runs/May20_13-52-23_gpu05",
        "#dc2626",
    ),
}


def _load_series(logdir: str, primary: str, fallback: str) -> Tuple[List[int], List[float]]:
    ea = EventAccumulator(logdir)
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    tag = primary if primary in tags else (fallback if fallback in tags else None)
    if tag is None:
        return [], []
    events = ea.Scalars(tag)
    return [int(e.step) for e in events], [float(e.value) for e in events]


def _ylim(*value_lists: List[float]) -> float:
    all_v: List[float] = []
    for vl in value_lists:
        all_v.extend(vl)
    if not all_v:
        return 1.0
    return min(1.2, float(np.percentile(all_v, 99)) * 1.05)


def plot_single(logdir: str, out: Path, *, title: str, train_color: str = "#2563eb") -> Path:
    train_s, train_v = _load_series(logdir, "loss/train", "train/loss")
    val_s, val_v = _load_series(logdir, "loss/validation", "eval/loss")
    if not train_s and not val_s:
        raise SystemExit(f"No loss scalars in {logdir}")

    fig, ax = plt.subplots(figsize=(12, 5), dpi=120)
    if train_s:
        ax.plot(train_s, train_v, color=train_color, linewidth=1.2, alpha=0.85, label="train")
    if val_s:
        ax.plot(val_s, val_v, color="#f59e0b", linewidth=0, marker="o", markersize=6, label="validation")
    ax.set_ylim(0, _ylim(train_v, val_v))
    ax.set_xlim(0, max(train_s[-1] if train_s else 0, val_s[-1] if val_s else 0))
    ax.set_xlabel("step")
    ax.set_ylabel("CE loss")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_compare(out: Path, *, runs: Dict[str, Tuple[str, str]] | None = None) -> Path:
    runs = runs or RUNS
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120, sharex=True)
    ax_train, ax_val = axes
    all_train_v: List[float] = []
    all_val_v: List[float] = []
    xmax = 0

    for label, (logdir, color) in runs.items():
        train_s, train_v = _load_series(logdir, "loss/train", "train/loss")
        val_s, val_v = _load_series(logdir, "loss/validation", "eval/loss")
        all_train_v.extend(train_v)
        all_val_v.extend(val_v)
        if train_s:
            ax_train.plot(train_s, train_v, color=color, linewidth=1.1, alpha=0.9, label=label)
            xmax = max(xmax, train_s[-1])
        if val_s:
            ax_val.plot(
                val_s,
                val_v,
                color=color,
                linewidth=0,
                marker="o",
                markersize=5,
                label=label,
            )
            xmax = max(xmax, val_s[-1])

    ymax = _ylim(all_train_v, all_val_v)
    for ax in axes:
        ax.set_ylim(0, ymax)
        ax.set_xlim(0, xmax)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    ax_train.set_ylabel("CE loss")
    ax_train.set_title("Training loss (all LRs)")
    ax_val.set_title("Validation loss (all LRs)")

    fig.suptitle("Qwen3-VL-4B LR sweep — same step axis", fontsize=11, y=1.02)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_compare_overlay(out: Path, *, runs: Dict[str, Tuple[str, str]] | None = None) -> Path:
    """Train solid + val dashed per LR, all on one axes."""
    runs = runs or RUNS
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=120)
    all_v: List[float] = []
    xmax = 0

    for label, (logdir, color) in runs.items():
        train_s, train_v = _load_series(logdir, "loss/train", "train/loss")
        val_s, val_v = _load_series(logdir, "loss/validation", "eval/loss")
        all_v.extend(train_v)
        all_v.extend(val_v)
        if train_s:
            ax.plot(train_s, train_v, color=color, linewidth=1.1, alpha=0.85, label=f"{label} train")
            xmax = max(xmax, train_s[-1])
        if val_s:
            ax.plot(
                val_s,
                val_v,
                color=color,
                linewidth=1.5,
                linestyle="--",
                marker="o",
                markersize=4,
                alpha=0.95,
                label=f"{label} val",
            )
            xmax = max(xmax, val_s[-1])

    ax.set_ylim(0, _ylim(all_v))
    ax.set_xlim(0, xmax)
    ax.set_xlabel("step")
    ax.set_ylabel("CE loss")
    ax.set_title("LR sweep: train (solid) + validation (dashed) on one plot")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["single", "compare", "overlay", "all"],
        default="all",
        help="single=one run train+val; compare=2 panels; overlay=one panel; all=lr16x + compare + overlay",
    )
    p.add_argument("--logdir", help="For mode=single")
    p.add_argument("--out", type=Path, default=Path("slurm_logs"))
    p.add_argument("--title", default="loss")
    args = p.parse_args()
    out_dir = Path(args.out)

    if args.mode in ("single", "all"):
        logdir = args.logdir or RUNS["lr16x (3.2e-4)"][0]
        path = plot_single(
            logdir,
            out_dir / "lr16x_loss_train_val.png",
            title="Qwen3-VL-4B lr16x (3.2e-4) — train + validation",
            train_color="#dc2626",
        )
        print(path.resolve())

    if args.mode in ("compare", "all"):
        path = plot_compare(out_dir / "lr_sweep_train_val_compare.png")
        print(path.resolve())

    if args.mode in ("overlay", "all"):
        path = plot_compare_overlay(out_dir / "lr_sweep_train_val_overlay.png")
        print(path.resolve())


if __name__ == "__main__":
    main()
