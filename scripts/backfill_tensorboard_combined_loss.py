#!/usr/bin/env python3
"""Backfill loss/train + loss/validation and a TB multiline chart into an existing run."""
from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

CHART_TITLE = "train + validation CE loss"


def _series(ea: EventAccumulator, primary: str, fallback: str):
    tags = set(ea.Tags().get("scalars", []))
    tag = primary if primary in tags else (fallback if fallback in tags else None)
    if tag is None:
        return []
    return [(int(e.step), float(e.value)) for e in ea.Scalars(tag)]


def backfill(logdir: Path, *, dry_run: bool = False, force: bool = False) -> None:
    logdir = logdir.resolve()
    ea = EventAccumulator(str(logdir))
    ea.Reload()

    train = _series(ea, "loss/train", "train/loss")
    val = _series(ea, "loss/validation", "eval/loss")
    if not train and not val:
        raise SystemExit(f"No train/loss or eval/loss scalars in {logdir}")

    if not force and "loss/train" in ea.Tags().get("scalars", []):
        print(f"Skip {logdir}: loss/train already present (use --force to append again)")
        return

    if dry_run:
        print(f"Would write {len(train)} train + {len(val)} val points to {logdir}")
        return

    writer = SummaryWriter(log_dir=str(logdir))
    writer.add_custom_scalars_multilinechart(
        ["loss/train", "loss/validation"],
        category="loss",
        title=CHART_TITLE,
    )
    for step, value in train:
        writer.add_scalar("loss/train", value, step)
    for step, value in val:
        writer.add_scalar("loss/validation", value, step)
    writer.flush()
    writer.close()
    print(f"Backfilled {len(train)} loss/train + {len(val)} loss/validation → {logdir}")
    print("TensorBoard: open CUSTOM SCALARS (or Time Series, filter 'loss/') and open the combined chart.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logdir", type=str, required=True, help="Single run directory (contains events.out.tfevents.*)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="Append even if loss/train exists")
    args = p.parse_args()
    backfill(Path(args.logdir), dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
