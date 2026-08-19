#!/usr/bin/env python3
"""Plot train + eval loss on one matplotlib figure."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _load(ea: EventAccumulator, primary: str, fallback: str):
    tags = set(ea.Tags().get("scalars", []))
    tag = primary if primary in tags else (fallback if fallback in tags else None)
    if tag is None:
        return [], []
    events = ea.Scalars(tag)
    return [int(e.step) for e in events], [float(e.value) for e in events]


def plot_loss_png(logdir: str, out: Path, *, title: str) -> Path:
    ea = EventAccumulator(logdir)
    ea.Reload()
    train_s, train_v = _load(ea, "loss/train", "train/loss")
    val_s, val_v = _load(ea, "loss/validation", "eval/loss")
    if not train_s and not val_s:
        raise SystemExit(f"No loss scalars in {logdir}")

    fig, ax = plt.subplots(figsize=(12, 5), dpi=120)
    if train_s:
        ax.plot(train_s, train_v, color="#2563eb", linewidth=1.2, alpha=0.85, label="train (every 10 steps)")
    if val_s:
        ax.plot(
            val_s,
            val_v,
            color="#f59e0b",
            linewidth=0,
            marker="o",
            markersize=6,
            label="validation (every 500 steps)",
        )

    all_v = train_v + val_v
    p99 = np.percentile(all_v, 99)
    ax.set_ylim(0, min(1.2, float(p99) * 1.05))
    xmax = max(train_s[-1] if train_s else 0, val_s[-1] if val_s else 0)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("step")
    ax.set_ylabel("CE loss")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="loss")
    args = p.parse_args()
    path = plot_loss_png(args.logdir, Path(args.out), title=args.title)
    print(path.resolve())


if __name__ == "__main__":
    main()
