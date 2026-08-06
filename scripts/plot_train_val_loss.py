#!/usr/bin/env python3
"""Write a simple train+val loss PNG from TensorBoard event files (no server needed)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        default="/mnt/weka/nnurijanyan/checkpoints/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h0-10ep/runs/May20_05-06-33_gpu05",
    )
    parser.add_argument("--out", type=str, default="/home/nnurijanyan/OpenFly-Platform/slurm_logs/baseline_loss_train_val.png")
    args = parser.parse_args()

    ea = EventAccumulator(args.logdir)
    ea.Reload()

    def series(tag: str):
        return [(e.step, e.value) for e in ea.Scalars(tag)]

    train = series("loss/train") if ea.Scalars("loss/train") else series("train/loss")
    val = series("loss/validation") if ea.Scalars("loss/validation") else series("eval/loss")

    fig, ax = plt.subplots(figsize=(10, 5))
    if train:
        ax.plot([s for s, _ in train], [v for _, v in train], label="train", alpha=0.85)
    if val:
        ax.plot([s for s, _ in val], [v for _, v in val], "o-", label="validation", markersize=4)
    ax.set_xlabel("step")
    ax.set_ylabel("CE loss")
    ax.set_title("Qwen3-VL-4B baseline (bs1-ga4, LR 2e-5)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
