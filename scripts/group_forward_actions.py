#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def _iter_trajectory_dirs_under(root: Path) -> Iterator[Path]:
    if not root.is_dir():
        return
    for p in root.rglob("restructured.jsonl"):
        if p.is_file():
            yield p.parent


def _load_single_jsonl_row(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        line = f.readline()
    if not line:
        raise ValueError(f"Empty file: {path}")
    return json.loads(line)


def _write_single_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write("\n")


def _group_forward_actions(actions: list[int]) -> tuple[list[int], list[int]]:
    """
    Group consecutive forward actions (1) left-to-right:
      - Prefer 9 (= 3x forward) whenever possible
      - Then 8 (= 2x forward)
      - Then 1

    Returns:
      (new_actions, kept_original_indices)
    """
    if any(a in (8, 9) for a in actions):
        raise ValueError("Input action list unexpectedly already contains 8 or 9.")

    new_actions: list[int] = []
    kept: list[int] = []

    i = 0
    n = len(actions)
    while i < n:
        a = actions[i]
        if a != 1:
            new_actions.append(a)
            kept.append(i)
            i += 1
            continue

        j = i
        while j < n and actions[j] == 1:
            j += 1
        run_len = j - i

        while run_len > 0:
            if run_len >= 3:
                new_actions.append(9)
                kept.append(i)
                i += 3
                run_len -= 3
            elif run_len == 2:
                new_actions.append(8)
                kept.append(i)
                i += 2
                run_len -= 2
            else:
                new_actions.append(1)
                kept.append(i)
                i += 1
                run_len -= 1

    return new_actions, kept


def _build_grouped_row(row: dict[str, Any]) -> dict[str, Any]:
    required = ["image_path", "gpt_instruction", "action", "index_list", "pos", "yaw"]
    for k in required:
        if k not in row:
            raise ValueError(f"Missing key {k!r} in row")

    actions = row["action"]
    index_list = row["index_list"]
    pos = row["pos"]
    yaw = row["yaw"]

    if not isinstance(actions, list) or not all(isinstance(a, int) for a in actions):
        raise ValueError("action must be list[int]")
    if not isinstance(index_list, list) or not all(isinstance(x, str) for x in index_list):
        raise ValueError("index_list must be list[str]")
    if not isinstance(pos, list) or not all(isinstance(p, list) and len(p) == 3 for p in pos):
        raise ValueError("pos must be list[list[float]] with len=3")
    if not isinstance(yaw, list) or not all(isinstance(v, (int, float)) for v in yaw):
        raise ValueError("yaw must be list[float]")
    if not (len(actions) == len(index_list) == len(pos) == len(yaw)):
        raise ValueError("Length mismatch between action/index_list/pos/yaw")

    new_actions, kept = _group_forward_actions(actions)
    new_row = dict(row)
    new_row["action"] = new_actions
    new_row["index_list"] = [index_list[i] for i in kept]
    new_row["pos"] = [pos[i] for i in kept]
    new_row["yaw"] = [float(yaw[i]) for i in kept]

    if not (len(new_row["action"]) == len(new_row["index_list"]) == len(new_row["pos"]) == len(new_row["yaw"])):
        raise AssertionError("Post-grouping length mismatch")

    return new_row


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Group consecutive forward actions in restructured.jsonl.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trajectory-dir", type=Path, help="Trajectory directory containing restructured.jsonl.")
    group.add_argument("--root", type=Path, help="Root to scan for restructured.jsonl (e.g. data_wo_annotation).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing grouped.jsonl.")
    parser.add_argument("--progress-every", type=int, default=500, help="Print progress every N trajectories in batch.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    def convert_one(traj_dir: Path) -> bool:
        in_path = traj_dir / "restructured.jsonl"
        out_path = traj_dir / "grouped.jsonl"
        if not in_path.is_file():
            raise FileNotFoundError(f"Missing {in_path}")
        if out_path.exists() and not args.overwrite:
            return False
        row = _load_single_jsonl_row(in_path)
        grouped = _build_grouped_row(row)
        _write_single_jsonl_row(out_path, grouped)
        return True

    if args.trajectory_dir is not None:
        wrote = convert_one(args.trajectory_dir)
        print(f"{'WROTE' if wrote else 'SKIP'} {args.trajectory_dir / 'grouped.jsonl'}")
        return 0

    root = args.root
    assert root is not None
    traj_dirs = list(_iter_trajectory_dirs_under(root))
    total = len(traj_dirs)
    print(f"Found {total} trajectories under {root}", flush=True)
    wrote = skipped = errors = 0
    for i, d in enumerate(traj_dirs, start=1):
        try:
            if convert_one(d):
                wrote += 1
            else:
                skipped += 1
        except Exception as e:
            errors += 1
            print(f"[ERROR] {d}: {e}", flush=True)
        if args.progress_every > 0 and (i == 1 or i % args.progress_every == 0 or i == total):
            print(f"Progress {i}/{total} | wrote={wrote} skipped={skipped} errors={errors}", flush=True)
    print(f"Done. Wrote={wrote} Skipped={skipped} Errors={errors}", flush=True)
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

