#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REQUIRED_KEYS = ("image_path", "gpt_instruction", "action", "index_list", "pos", "yaw")
LEFT_ACTION = 2
RIGHT_ACTION = 3
OTHER_ACTIONS = (0, 1, 4, 5, 8, 9, 10)
LEFT_QUOTA = 150
RIGHT_QUOTA = 150
OTHER_QUOTA_EACH = 100
TOTAL = LEFT_QUOTA + RIGHT_QUOTA + OTHER_QUOTA_EACH * len(OTHER_ACTIONS)


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
    if not a:
        raise ValueError(f"Empty action list in {src}")
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


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array at top-level in {path}")
    return data


def _sample_skill_rows(
    rows: list[dict[str, Any]],
    *,
    quota: int,
    expected_end: int,
    used_paths: set[str],
    rng: random.Random,
    src: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        _validate_row(row, src=f"{src}[{i}]")
        if row["action"][-1] != expected_end:
            raise ValueError(f"Expected end action {expected_end} in {src}[{i}], got {row['action'][-1]}")
        # Need at least one history action before the label.
        if len(row["action"]) < 2:
            continue
        img = row["image_path"]
        if not isinstance(img, str) or not img:
            raise ValueError(f"Invalid image_path in {src}[{i}]")
        if img in used_paths:
            continue
        candidates.append(row)

    if len(candidates) < quota:
        raise RuntimeError(f"Need {quota} unique paths from {src}, only {len(candidates)} available")

    chosen = rng.sample(candidates, quota)
    out: list[dict[str, Any]] = []
    for row in chosen:
        img = row["image_path"]
        used_paths.add(img)
        # Keep full already-cropped skill sample as-is.
        out.append(
            {
                "image_path": row["image_path"],
                "gpt_instruction": row["gpt_instruction"],
                "action": list(row["action"]),
                "index_list": list(row["index_list"]),
                "pos": [list(p) for p in row["pos"]],
                "yaw": list(row["yaw"]),
            }
        )
    return out


def _build_other_candidates(
    rows: list[dict[str, Any]],
    *,
    src: str,
) -> dict[int, list[dict[str, Any]]]:
    """One crop candidate per (image_path, action_id): first occurrence of that action in a run."""
    by_action: dict[int, list[dict[str, Any]]] = defaultdict(list)
    # Prefer one representative crop per trajectory per action id.
    seen_path_action: set[tuple[str, int]] = set()

    for i, row in enumerate(rows):
        _validate_row(row, src=f"{src}[{i}]")
        img = row["image_path"]
        if not isinstance(img, str) or not img:
            raise ValueError(f"Invalid image_path in {src}[{i}]")
        actions = row["action"]
        for t, action_id in enumerate(actions):
            if action_id not in OTHER_ACTIONS:
                continue
            # Need at least one history action before the label (end >= 2).
            if t < 1:
                continue
            if actions[t - 1] == action_id:
                continue
            key = (img, action_id)
            if key in seen_path_action:
                continue
            seen_path_action.add(key)
            by_action[action_id].append(_make_skill_sample(row, t + 1))

    return by_action


def _sample_other_rows(
    by_action: dict[int, list[dict[str, Any]]],
    *,
    used_paths: set[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for action_id in OTHER_ACTIONS:
        pool = [c for c in by_action.get(action_id, []) if c["image_path"] not in used_paths]
        if len(pool) < OTHER_QUOTA_EACH:
            raise RuntimeError(
                f"Need {OTHER_QUOTA_EACH} unique trajectories ending crop with action {action_id}, "
                f"only {len(pool)} available after excluding used paths"
            )
        chosen = rng.sample(pool, OTHER_QUOTA_EACH)
        for sample in chosen:
            used_paths.add(sample["image_path"])
            out.append(sample)
    return out


def main(argv: Iterable[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build a 1000-sample in-train overfitting probe JSON (unique trajectories)."
    )
    parser.add_argument(
        "--left",
        type=Path,
        default=repo / "data_curated" / "trainx9_left_curated.json",
        help="Left skill train JSON (ends with 2).",
    )
    parser.add_argument(
        "--right",
        type=Path,
        default=repo / "data_curated" / "trainx9_right_curated.json",
        help="Right skill train JSON (ends with 3).",
    )
    parser.add_argument(
        "--full-train",
        type=Path,
        default=repo / "data_curated" / "trainx9_curated.json",
        help="Full train trajectories for non-L/R crops.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "data_curated" / "trainx9_intrain_eval_1000.json",
        help="Output probe JSON path.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo / "data_curated" / "trainx9_intrain_eval_1000_manifest.json",
        help="Companion manifest path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if TOTAL != 1000:
        raise AssertionError(f"Quotas must sum to 1000, got {TOTAL}")

    for p in (args.left, args.right, args.full_train):
        if not p.is_file():
            raise FileNotFoundError(f"Missing input file: {p}")

    rng = random.Random(args.seed)
    used_paths: set[str] = set()

    print(f"Loading left from {args.left}", flush=True)
    left_rows = _load_json_array(args.left)
    print(f"Loading right from {args.right}", flush=True)
    right_rows = _load_json_array(args.right)
    print(f"Loading full train from {args.full_train}", flush=True)
    full_rows = _load_json_array(args.full_train)

    print(f"Sampling {LEFT_QUOTA} left / {RIGHT_QUOTA} right...", flush=True)
    left_samples = _sample_skill_rows(
        left_rows,
        quota=LEFT_QUOTA,
        expected_end=LEFT_ACTION,
        used_paths=used_paths,
        rng=rng,
        src=str(args.left),
    )
    right_samples = _sample_skill_rows(
        right_rows,
        quota=RIGHT_QUOTA,
        expected_end=RIGHT_ACTION,
        used_paths=used_paths,
        rng=rng,
        src=str(args.right),
    )

    print("Building non-L/R crop candidates...", flush=True)
    by_action = _build_other_candidates(full_rows, src=str(args.full_train))
    for aid in OTHER_ACTIONS:
        print(f"  action {aid}: {len(by_action.get(aid, []))} trajectory candidates", flush=True)

    print(f"Sampling {OTHER_QUOTA_EACH} each for {list(OTHER_ACTIONS)}...", flush=True)
    other_samples = _sample_other_rows(by_action, used_paths=used_paths, rng=rng)

    selected = left_samples + right_samples + other_samples
    rng.shuffle(selected)

    if len(selected) != TOTAL:
        raise AssertionError(f"Expected {TOTAL} samples, got {len(selected)}")
    if len({r["image_path"] for r in selected}) != TOTAL:
        raise AssertionError("image_path uniqueness violated")
    if any(len(r["action"]) < 2 for r in selected):
        raise AssertionError("Found samples with empty history (action length < 2)")

    ends = Counter(r["action"][-1] for r in selected)
    expected_ends = {LEFT_ACTION: LEFT_QUOTA, RIGHT_ACTION: RIGHT_QUOTA}
    for aid in OTHER_ACTIONS:
        expected_ends[aid] = OTHER_QUOTA_EACH
    if dict(ends) != expected_ends:
        raise AssertionError(f"End-action histogram mismatch: got {dict(ends)} expected {expected_ends}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with _JsonArrayWriter(args.output) as writer:
        for row in selected:
            _validate_row(row, src="selected")
            writer.write_obj(row)

    manifest = {
        "seed": args.seed,
        "total": TOTAL,
        "quotas": {
            "left_2": LEFT_QUOTA,
            "right_3": RIGHT_QUOTA,
            "other_each": OTHER_QUOTA_EACH,
            "other_actions": list(OTHER_ACTIONS),
        },
        "sources": {
            "left": str(args.left),
            "right": str(args.right),
            "full_train": str(args.full_train),
        },
        "output": str(args.output),
        "end_action_counts": {str(k): v for k, v in sorted(ends.items())},
        "unique_image_paths": TOTAL,
    }
    with args.manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {TOTAL} samples -> {args.output}", flush=True)
    print(f"Manifest -> {args.manifest}", flush=True)
    print(f"end_action_counts={dict(sorted(ends.items()))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
