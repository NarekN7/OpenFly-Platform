#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REQUIRED_KEYS = ("image_path", "gpt_instruction", "action", "index_list", "pos", "yaw")
DEFAULT_TARGET_ACTIONS = frozenset({2, 3})


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


def _parse_action_ids(raw: str) -> frozenset[int]:
    ids = frozenset(int(x.strip()) for x in raw.split(",") if x.strip())
    if not ids:
        raise ValueError("--actions must contain at least one integer action id")
    return ids


def _validate_source_row(row: dict[str, Any], *, src: str) -> None:
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


def _validate_output_row(row: dict[str, Any], *, src: str, target_actions: frozenset[int]) -> None:
    _validate_source_row(row, src=src)
    if row["action"][-1] not in target_actions:
        raise ValueError(
            f"Output row must end with one of {sorted(target_actions)} in {src}"
        )


def _make_skill_sample(row: dict[str, Any], end: int) -> dict[str, Any]:
    return {
        "image_path": row["image_path"],
        "gpt_instruction": row["gpt_instruction"],
        "action": row["action"][:end],
        "index_list": row["index_list"][:end],
        "pos": row["pos"][:end],
        "yaw": row["yaw"][:end],
    }


def expand_skill_samples(
    row: dict[str, Any],
    target_actions: frozenset[int],
    *,
    last_only: bool = False,
) -> list[dict[str, Any]]:
    actions = row["action"]
    if last_only:
        if not actions or actions[-1] not in target_actions:
            return []
        return [_make_skill_sample(row, len(actions))]

    out: list[dict[str, Any]] = []
    for t, action_id in enumerate(actions):
        if action_id not in target_actions:
            continue
        # Skip consecutive repeated skill actions (keep only the first of a run),
        # e.g. drop samples ending in [..., 2, 2] / [..., 3, 3].
        if t > 0 and actions[t - 1] == action_id:
            continue
        out.append(_make_skill_sample(row, t + 1))
    return out


def main(argv: Iterable[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Expand curated trajectories into skill training samples for target actions."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo / "data_curated" / "trainx9_curated.json",
        help="Source curated JSON array.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "data_curated" / "trainx9_left_right_curated.json",
        help="Output JSON array path.",
    )
    parser.add_argument(
        "--actions",
        type=str,
        default="2,3",
        help="Comma-separated target action ids to extract (default: 2,3 for left/right).",
    )
    parser.add_argument(
        "--last-only",
        action="store_true",
        help="Only keep trajectories whose final action is in --actions (for stop/skill-at-suffix).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, process only first N source trajectories (for testing).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print progress every N source trajectories.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    target_actions = _parse_action_ids(args.actions)

    if not args.input.is_file():
        raise FileNotFoundError(f"Missing input file: {args.input}")

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array at top-level in {args.input}")

    if args.limit and args.limit > 0:
        data = data[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    wrote = 0
    skipped_no_match = 0
    errors = 0

    with _JsonArrayWriter(args.output) as writer:
        for i, row in enumerate(data, start=1):
            src = f"{args.input}[{i - 1}]"
            try:
                if not isinstance(row, dict):
                    raise ValueError(f"Bad row type: {type(row).__name__}")
                _validate_source_row(row, src=src)
                samples = expand_skill_samples(row, target_actions, last_only=args.last_only)
                if not samples:
                    skipped_no_match += 1
                    continue
                for sample in samples:
                    _validate_output_row(sample, src=src, target_actions=target_actions)
                    writer.write_obj(sample)
                    wrote += 1
            except Exception as e:
                errors += 1
                print(f"[ERROR] {src}: {e}", file=sys.stderr, flush=True)

            if args.progress_every > 0 and (i == 1 or i % args.progress_every == 0 or i == len(data)):
                print(
                    f"Progress {i}/{len(data)} | wrote={wrote} skipped_no_match={skipped_no_match} errors={errors}",
                    flush=True,
                )

    print(
        f"Done. source={len(data)} output={wrote} skipped_no_match={skipped_no_match} "
        f"errors={errors} actions={sorted(target_actions)} -> {args.output}",
        flush=True,
    )
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
