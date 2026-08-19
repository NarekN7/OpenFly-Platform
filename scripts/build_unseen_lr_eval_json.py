#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REQUIRED_KEYS = ("image_path", "gpt_instruction", "action", "index_list", "pos", "yaw")
LEFT_ACTION = 2
RIGHT_ACTION = 3


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


def _validate_row(row: dict[str, Any], *, src: str) -> None:
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


def _make_skill_sample(row: dict[str, Any], end: int) -> dict[str, Any]:
    return {
        "image_path": row["image_path"],
        "gpt_instruction": row["gpt_instruction"],
        "action": row["action"][:end],
        "index_list": row["index_list"][:end],
        "pos": row["pos"][:end],
        "yaw": row["yaw"][:end],
    }


def _lr_crops(row: dict[str, Any], action_id: int) -> list[dict[str, Any]]:
    """Crops ending with a single (non-consecutive) left/right action; need history."""
    actions = row["action"]
    out: list[dict[str, Any]] = []
    for t, a in enumerate(actions):
        if a != action_id:
            continue
        if t < 1:
            continue
        # Reject consecutive same turn at end: [..., 2, 2] / [..., 3, 3]
        if actions[t - 1] == action_id:
            continue
        out.append(_make_skill_sample(row, t + 1))
    return out


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _JsonArrayWriter(path) as writer:
        for row in rows:
            writer.write_obj(row)


def main(argv: Iterable[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Sample left/right skill eval crops from out-of-train validation trajectories."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo / "data_curated" / "validationx9.json",
        help="Out-of-train validation trajectories.",
    )
    parser.add_argument(
        "--left-output",
        type=Path,
        default=repo / "data_curated" / "left_evaluation_unseen.json",
    )
    parser.add_argument(
        "--right-output",
        type=Path,
        default=repo / "data_curated" / "right_evaluation_unseen.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo / "data_curated" / "left_right_evaluation_unseen_manifest.json",
    )
    parser.add_argument("--n-each", type=int, default=100, help="Trajectories per side.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(list(argv) if argv is not None else None)

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {args.input}")

    rng = random.Random(args.seed)

    # One representative crop per trajectory per side (first valid crop after shuffle of crops).
    left_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    right_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for i, row in enumerate(data):
        _validate_row(row, src=f"{args.input}[{i}]")
        img = row["image_path"]
        for sample in _lr_crops(row, LEFT_ACTION):
            left_by_path[img].append(sample)
        for sample in _lr_crops(row, RIGHT_ACTION):
            right_by_path[img].append(sample)

    left_paths = sorted(left_by_path.keys())
    right_paths = sorted(right_by_path.keys())
    if len(left_paths) < args.n_each:
        raise RuntimeError(f"Need {args.n_each} left trajectories, only {len(left_paths)} available")
    if len(right_paths) < args.n_each:
        raise RuntimeError(f"Need {args.n_each} right trajectories, only {len(right_paths)} available")

    # Prefer disjoint trajectory sets for left vs right when possible.
    chosen_left_paths = set(rng.sample(left_paths, args.n_each))
    remaining_right = [p for p in right_paths if p not in chosen_left_paths]
    if len(remaining_right) >= args.n_each:
        chosen_right_paths = set(rng.sample(remaining_right, args.n_each))
    else:
        # Fall back: fill with any right-capable trajectories.
        chosen_right_paths = set(remaining_right)
        extra = [p for p in right_paths if p not in chosen_right_paths]
        need = args.n_each - len(chosen_right_paths)
        chosen_right_paths.update(rng.sample(extra, need))

    left_out: list[dict[str, Any]] = []
    for p in sorted(chosen_left_paths):
        crops = left_by_path[p]
        left_out.append(rng.choice(crops))

    right_out: list[dict[str, Any]] = []
    for p in sorted(chosen_right_paths):
        crops = right_by_path[p]
        right_out.append(rng.choice(crops))

    rng.shuffle(left_out)
    rng.shuffle(right_out)

    # Final checks
    for side, rows, aid in (("left", left_out, LEFT_ACTION), ("right", right_out, RIGHT_ACTION)):
        if len(rows) != args.n_each:
            raise AssertionError(f"{side}: expected {args.n_each}, got {len(rows)}")
        if len({r["image_path"] for r in rows}) != args.n_each:
            raise AssertionError(f"{side}: non-unique image_path")
        for r in rows:
            if r["action"][-1] != aid:
                raise AssertionError(f"{side}: bad end action {r['action'][-1]}")
            if len(r["action"]) < 2:
                raise AssertionError(f"{side}: empty history")
            if r["action"][-2] == aid:
                raise AssertionError(f"{side}: consecutive end turn {r['action']}")

    _write_json(args.left_output, left_out)
    _write_json(args.right_output, right_out)

    overlap_paths = {r["image_path"] for r in left_out} & {r["image_path"] for r in right_out}
    manifest = {
        "seed": args.seed,
        "n_each": args.n_each,
        "source": str(args.input),
        "left_output": str(args.left_output),
        "right_output": str(args.right_output),
        "available_left_trajectories": len(left_paths),
        "available_right_trajectories": len(right_paths),
        "shared_image_paths_between_left_and_right_outputs": len(overlap_paths),
        "filter": "skip consecutive same L/R at crop end; require history len>=1",
    }
    with args.manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(
        f"Wrote left={len(left_out)} -> {args.left_output}; "
        f"right={len(right_out)} -> {args.right_output}; "
        f"shared_paths={len(overlap_paths)}; manifest={args.manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
