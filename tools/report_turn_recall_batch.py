#!/usr/bin/env python3
"""Batch report: GT left/right turn recall for multiple combined eval runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from closed_loop_turn_recall import format_recall_line, turn_recall_from_run_root

DEFAULT_ROOTS = [
    (
        "stop0 (213924)",
        "eval_runs/interleave_2b_nosfx_stop0_train213924_ckptlast_20260817_162311",
    ),
    (
        "stop5 (213925)",
        "eval_runs/interleave_2b_nosfx_stop5_train213925_ckptlast_20260817_224827",
    ),
    (
        "stop10 (213926)",
        "eval_runs/interleave_2b_nosfx_stop10_train213926_ckptlast_20260818_050319",
    ),
    (
        "stop20 (213927)",
        "eval_runs/interleave_2b_nosfx_stop20_train213927_ckptlast_20260818_083724",
    ),
]


def _pct_str(recall: float | None) -> str:
    if recall is None:
        return "—"
    return f"{100.0 * recall:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch GT turn recall report")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="OpenFly-Platform repo root",
    )
    parser.add_argument(
        "--update-combined-summary",
        action="store_true",
        help="Merge turn_recall into each run's combined_summary.json if present",
    )
    args = parser.parse_args()
    repo = args.repo_root

    print("=== Closed-loop GT turn recall (trajectory presence) ===")
    print("| Model | Left recall (2) | Right recall (3) |")
    print("|-------|----------------:|-----------------:|")

    for label, rel in DEFAULT_ROOTS:
        root = repo / rel
        closed = root / "closed_loop"
        turn = turn_recall_from_run_root(closed)
        overall = turn.get("overall", {})
        left = overall.get("left", {})
        right = overall.get("right", {})

        left_cell = (
            f"{_pct_str(left.get('recall'))} ({left.get('n_success', 0)}/{left.get('n_cases', 0)})"
            if left.get("n_cases", 0) > 0
            else "— (0 cases)"
        )
        right_cell = (
            f"{_pct_str(right.get('recall'))} ({right.get('n_success', 0)}/{right.get('n_cases', 0)})"
            if right.get("n_cases", 0) > 0
            else "— (0 cases)"
        )
        print(f"| {label} | {left_cell} | {right_cell} |")

        if args.update_combined_summary:
            summary_path = root / "combined_summary.json"
            if summary_path.is_file():
                data = json.loads(summary_path.read_text(encoding="utf-8"))
                if "closed_loop" not in data:
                    data["closed_loop"] = {"root": str(closed)}
                data["closed_loop"]["turn_recall"] = turn
                summary_path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"  updated {summary_path}")

    print()
    print("Detail (first model):")
    first_root = repo / DEFAULT_ROOTS[0][1] / "closed_loop"
    first_turn = turn_recall_from_run_root(first_root)
    ov = first_turn.get("overall", {})
    print(
        "  "
        + format_recall_line(ov.get("left", {}), "left(2)")
        + "  "
        + format_recall_line(ov.get("right", {}), "right(3)")
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
