#!/usr/bin/env python3
"""Aggregate metrics from a combined eval run directory into one summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from closed_loop_turn_recall import format_recall_line, turn_recall_from_run_root


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def _first_existing_dir(root: Path, *names: str) -> Path:
    for name in names:
        cand = root / name
        if cand.is_dir():
            return cand
    return root / names[0]


def _attach_lr(out: Dict[str, Any], key: str, lr_root: Path) -> None:
    out[key] = {"root": str(lr_root)}
    lr_sum = _read_json(lr_root / "summary.json")
    if lr_sum:
        out[key]["summary"] = lr_sum
        out[key]["overall_accuracy"] = lr_sum.get("overall_accuracy")


def _attach_stop(out: Dict[str, Any], key: str, stop_root: Path) -> None:
    out[key] = {"root": str(stop_root)}
    stop_sum = _read_json(stop_root / "summary.json")
    if stop_sum:
        out[key]["summary"] = stop_sum
        out[key]["accuracy"] = stop_sum.get("accuracy")


def _print_lr(result: Dict[str, Any], key: str, label: str) -> None:
    lr_acc = result.get(key, {}).get("overall_accuracy")
    if lr_acc is not None:
        lr_sum = result[key].get("summary", {})
        sets = lr_sum.get("sets", {})
        left = sets.get("left", {})
        right = sets.get("right", {})
        print(
            f"{label}: overall={_pct(lr_acc)} "
            f"({lr_sum.get('overall_n_correct')}/{lr_sum.get('overall_n_scored')})  "
            f"left={_pct(left.get('accuracy', 0))}  right={_pct(right.get('accuracy', 0))}"
        )
    else:
        print(f"{label}: (missing summary)")


def _print_stop(result: Dict[str, Any], key: str, label: str) -> None:
    stop_acc = result.get(key, {}).get("accuracy")
    if stop_acc is not None:
        s = result[key].get("summary", {})
        print(
            f"{label}: {_pct(stop_acc)} "
            f"({s.get('n_correct')}/{s.get('n_scored')})"
        )
    else:
        print(f"{label}: (missing summary)")


def aggregate(root: Path, checkpoint: str) -> Dict[str, Any]:
    closed_root = root / "closed_loop"
    lr_val_root = _first_existing_dir(root, "skill_lr_validation", "skill_lr_unseen")
    stop_val_root = _first_existing_dir(root, "skill_stop_validation", "skill_stop_unseen")
    lr_test_root = root / "skill_lr_test"
    stop_test_root = root / "skill_stop_test"
    intrain_root = root / "skill_intrain"

    out: Dict[str, Any] = {
        "checkpoint": checkpoint,
        "combined_root": str(root),
        "closed_loop": {"root": str(closed_root), "envs": {}},
        "skill_intrain": {"root": str(intrain_root)},
    }

    env_metrics: List[Dict[str, Any]] = []
    if closed_root.is_dir():
        for env_dir in sorted(closed_root.iterdir()):
            if not env_dir.is_dir():
                continue
            met = _read_json(env_dir / "metrics.json")
            if met is None:
                continue
            row = {
                "env": env_dir.name,
                "mean_success_rate": met.get("mean_success_rate"),
                "mean_navigation_error": met.get("mean_navigation_error"),
                "mean_oracle_success_rate": met.get("mean_oracle_success_rate"),
                "n_trajectories": met.get("n_trajectories"),
                "metrics_path": str(env_dir / "metrics.json"),
            }
            out["closed_loop"]["envs"][env_dir.name] = row
            env_metrics.append(row)

    if env_metrics:
        srs = [r["mean_success_rate"] for r in env_metrics if r.get("mean_success_rate") is not None]
        nes = [r["mean_navigation_error"] for r in env_metrics if r.get("mean_navigation_error") is not None]
        osrs = [r["mean_oracle_success_rate"] for r in env_metrics if r.get("mean_oracle_success_rate") is not None]
        out["closed_loop"]["avg"] = {
            "mean_success_rate": sum(srs) / len(srs) if srs else None,
            "mean_navigation_error": sum(nes) / len(nes) if nes else None,
            "mean_oracle_success_rate": sum(osrs) / len(osrs) if osrs else None,
            "n_envs": len(env_metrics),
        }

    turn_recall = turn_recall_from_run_root(closed_root)
    if turn_recall.get("overall") or turn_recall.get("envs"):
        out["closed_loop"]["turn_recall"] = turn_recall

    _attach_lr(out, "skill_lr_validation", lr_val_root)
    _attach_stop(out, "skill_stop_validation", stop_val_root)
    _attach_lr(out, "skill_lr_test", lr_test_root)
    _attach_stop(out, "skill_stop_test", stop_test_root)

    # Keep legacy keys so older consumers of combined_summary.json still work.
    out["skill_lr_unseen"] = out["skill_lr_validation"]
    out["skill_stop_unseen"] = out["skill_stop_validation"]

    intrain_sum = _read_json(intrain_root / "summary.json")
    if intrain_sum:
        out["skill_intrain"]["summary"] = intrain_sum
        out["skill_intrain"]["accuracy"] = intrain_sum.get("accuracy")

    return out


def print_table(result: Dict[str, Any]) -> None:
    print("=== Combined eval summary ===")
    print(f"checkpoint: {result.get('checkpoint')}")
    print(f"root: {result.get('combined_root')}")
    print()

    cl = result.get("closed_loop", {})
    avg = cl.get("avg")
    if avg:
        print("Closed-loop (avg over envs):")
        print(
            f"  SR={_pct(avg['mean_success_rate'])}  "
            f"NE={avg['mean_navigation_error']:.2f}m  "
            f"OSR={_pct(avg['mean_oracle_success_rate'])}  "
            f"envs={avg['n_envs']}"
        )
        for env, row in sorted(cl.get("envs", {}).items()):
            print(
                f"    {env}: SR={_pct(row['mean_success_rate'])}  "
                f"NE={row['mean_navigation_error']:.2f}m  "
                f"OSR={_pct(row['mean_oracle_success_rate'])}  n={row.get('n_trajectories')}"
            )
    else:
        print("Closed-loop: (no per-env metrics found)")

    turn_recall = cl.get("turn_recall", {}).get("overall")
    if turn_recall:
        left = turn_recall.get("left", {})
        right = turn_recall.get("right", {})
        print(
            "Closed-loop GT turn recall: "
            f"{format_recall_line(left, 'left(2)')}  "
            f"{format_recall_line(right, 'right(3)')}"
        )
    print()

    _print_lr(result, "skill_lr_validation", "Skill LR validation")
    _print_stop(result, "skill_stop_validation", "Skill stop validation")
    _print_lr(result, "skill_lr_test", "Skill LR test")
    _print_stop(result, "skill_stop_test", "Skill stop test")

    intrain_acc = result.get("skill_intrain", {}).get("accuracy")
    if intrain_acc is not None:
        s = result["skill_intrain"].get("summary", {})
        print(
            f"In-train crops: {_pct(intrain_acc)} "
            f"({s.get('n_correct')}/{s.get('n_scored')})"
        )
    else:
        print("In-train crops: (missing summary)")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate combined eval run metrics")
    parser.add_argument("--root", required=True, help="Combined eval root directory")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path for metadata")
    parser.add_argument(
        "--write-json",
        default="",
        help="If set, write combined_summary.json to this path",
    )
    args = parser.parse_args()

    root = Path(args.root)
    result = aggregate(root, args.checkpoint)
    print_table(result)

    if args.write_json:
        out_path = Path(args.write_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
