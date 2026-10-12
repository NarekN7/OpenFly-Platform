#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

REQUIRED_KEYS = ("image_path", "gpt_instruction", "action", "index_list", "pos", "yaw")
STOP_ACTION = 0


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


def _stop_crop(row: dict[str, Any]) -> dict[str, Any] | None:
    """Full trajectory crop ending with stop (action 0 at suffix)."""
    actions = row["action"]
    if not actions or actions[-1] != STOP_ACTION:
        return None
    if len(actions) < 2:
        return None
    end = len(actions)
    return {
        "image_path": row["image_path"],
        "gpt_instruction": row["gpt_instruction"],
        "action": actions[:end],
        "index_list": row["index_list"][:end],
        "pos": row["pos"][:end],
        "yaw": row["yaw"][:end],
    }


def main(argv: Iterable[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Sample stop (0) skill eval crops from out-of-train validation/test trajectories."
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Preset: validation (validationx9.json) or test (seen_curated.json).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Override input JSON (default depends on --split).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
    )
    parser.add_argument("--n", type=int, default=100, help="Number of trajectories to sample.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.split == "test":
        default_input = repo / "data_curated" / "seen_curated.json"
        default_output = repo / "data_curated" / "stop_evaluation_skill_test.json"
        default_manifest = repo / "data_curated" / "stop_evaluation_skill_test_manifest.json"
    else:
        default_input = repo / "data_curated" / "validationx9.json"
        default_output = repo / "data_curated" / "stop_evaluation_skill_validation.json"
        default_manifest = repo / "data_curated" / "stop_evaluation_skill_validation_manifest.json"

    input_path = args.input or default_input
    output_path = args.output or default_output
    manifest_path = args.manifest or default_manifest

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {input_path}")

    candidates: list[dict[str, Any]] = []
    for i, row in enumerate(data):
        _validate_row(row, src=f"{input_path}[{i}]")
        sample = _stop_crop(row)
        if sample is not None:
            candidates.append(sample)

    if len(candidates) < args.n:
        raise RuntimeError(f"Need {args.n} stop trajectories, only {len(candidates)} available")

    rng = random.Random(args.seed)
    chosen = rng.sample(candidates, args.n)
    rng.shuffle(chosen)

    if len({r["image_path"] for r in chosen}) != args.n:
        raise AssertionError("Non-unique image_path in sampled stop eval set")
    for r in chosen:
        if r["action"][-1] != STOP_ACTION:
            raise AssertionError(f"Bad end action {r['action'][-1]}")
        if len(r["action"]) < 2:
            raise AssertionError("Empty history before stop")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _JsonArrayWriter(output_path) as writer:
        for row in chosen:
            writer.write_obj(row)

    manifest = {
        "seed": args.seed,
        "n": args.n,
        "split": args.split,
        "source": str(input_path),
        "output": str(output_path),
        "available_stop_trajectories": len(candidates),
        "filter": "trajectory must end with action 0; require history len>=1",
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote stop={len(chosen)} -> {output_path}; manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
