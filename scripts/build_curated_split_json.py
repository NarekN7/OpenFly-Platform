#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator


REQUIRED_KEYS = ("image_path", "gpt_instruction", "action", "index_list", "pos", "yaw")


def _iter_trajectory_dirs_under(root: Path) -> Iterator[Path]:
    if not root.is_dir():
        return
    for p in root.rglob("grouped.jsonl"):
        if p.is_file():
            yield p.parent


def _load_annotation_image_paths(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array at top-level in {path}")
    out: set[str] = set()
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Bad row type at {path}[{i}]: {type(row).__name__}")
        img = row.get("image_path")
        if not isinstance(img, str) or not img:
            raise ValueError(f"Missing/invalid image_path at {path}[{i}]")
        out.add(img)
    return out


def _load_single_jsonl_row(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        line = f.readline()
    if not line:
        raise ValueError(f"Empty file: {path}")
    return json.loads(line)


class _JsonArrayWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._f = path.open("w", encoding="utf-8")
        self._first = True
        self._f.write("[\n")

    def write_obj(self, obj: dict[str, Any]) -> None:
        if not self._first:
            self._f.write(",\n")
        self._f.write(json.dumps(obj, ensure_ascii=False))
        self._first = False

    def close(self) -> None:
        self._f.write("\n]\n")
        self._f.close()

    def __enter__(self) -> "_JsonArrayWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _validate_grouped_row(row: dict[str, Any], *, src: Path) -> None:
    for k in REQUIRED_KEYS:
        if k not in row:
            raise ValueError(f"Missing key {k!r} in {src}")
    a = row["action"]
    idx = row["index_list"]
    pos = row["pos"]
    yaw = row["yaw"]
    if not isinstance(a, list) or not all(isinstance(x, int) for x in a):
        raise ValueError(f"Invalid action list in {src}")
    if not isinstance(idx, list) or not all(isinstance(x, str) for x in idx):
        raise ValueError(f"Invalid index_list in {src}")
    if not isinstance(pos, list) or not all(isinstance(p, list) and len(p) == 3 for p in pos):
        raise ValueError(f"Invalid pos in {src}")
    if not isinstance(yaw, list) or not all(isinstance(v, (int, float)) for v in yaw):
        raise ValueError(f"Invalid yaw in {src}")
    if not (len(a) == len(idx) == len(pos) == len(yaw)):
        raise ValueError(f"Length mismatch action/index_list/pos/yaw in {src}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate grouped.jsonl trajectories into curated train/seen JSON arrays."
    )
    parser.add_argument("--root", type=Path, default=Path("data_wo_annotation"), help="Root containing trajectories.")
    parser.add_argument(
        "--annotation-train",
        type=Path,
        default=Path("Annotation/train.json"),
        help="Annotation train.json path.",
    )
    parser.add_argument(
        "--annotation-seen",
        type=Path,
        default=Path("Annotation/seen.json"),
        help="Annotation seen.json path.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data_curated"), help="Output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing curated output files.")
    parser.add_argument("--progress-every", type=int, default=1000, help="Print progress every N trajectories.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, process only first N trajectories (for testing).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    train_set = _load_annotation_image_paths(args.annotation_train)
    seen_set = _load_annotation_image_paths(args.annotation_seen)
    overlap = train_set.intersection(seen_set)
    if overlap:
        raise ValueError(f"train/seen image_path sets overlap (example: {next(iter(overlap))!r})")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / "train_curated.json"
    seen_out = out_dir / "seen_curated.json"
    missing_log = out_dir / "missing_image_paths.log"

    if not args.overwrite:
        for p in (train_out, seen_out, missing_log):
            if p.exists():
                raise FileExistsError(f"{p} exists; pass --overwrite to replace.")

    traj_dirs = sorted(_iter_trajectory_dirs_under(args.root), key=lambda p: p.as_posix())
    total = len(traj_dirs)
    if args.limit and args.limit > 0:
        traj_dirs = traj_dirs[: args.limit]
    print(f"Found {total} grouped trajectories under {args.root}", flush=True)
    if args.limit and args.limit > 0:
        print(f"Limiting to first {len(traj_dirs)} trajectories", flush=True)

    wrote_train = 0
    wrote_seen = 0
    missing = 0
    errors = 0

    with (
        _JsonArrayWriter(train_out) as train_writer,
        _JsonArrayWriter(seen_out) as seen_writer,
        missing_log.open("w", encoding="utf-8") as miss_f,
    ):
        for i, d in enumerate(traj_dirs, start=1):
            grouped_path = d / "grouped.jsonl"
            try:
                row = _load_single_jsonl_row(grouped_path)
                if not isinstance(row, dict):
                    raise ValueError("row is not an object")
                _validate_grouped_row(row, src=grouped_path)
                img = row.get("image_path")
                if not isinstance(img, str) or not img:
                    raise ValueError("missing image_path")

                if img in train_set:
                    train_writer.write_obj(row)
                    wrote_train += 1
                elif img in seen_set:
                    seen_writer.write_obj(row)
                    wrote_seen += 1
                else:
                    miss_f.write(img + "\n")
                    missing += 1
            except Exception as e:
                errors += 1
                print(f"[ERROR] {grouped_path}: {e}", file=sys.stderr, flush=True)

            if args.progress_every > 0 and (i == 1 or i % args.progress_every == 0 or i == len(traj_dirs)):
                print(
                    f"Progress {i}/{len(traj_dirs)} | "
                    f"train={wrote_train} seen={wrote_seen} missing={missing} errors={errors}",
                    flush=True,
                )

    print(
        f"Done. train={wrote_train} seen={wrote_seen} missing={missing} errors={errors} "
        f"(processed={len(traj_dirs)})",
        flush=True,
    )
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

