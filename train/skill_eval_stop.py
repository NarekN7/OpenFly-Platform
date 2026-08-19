#!/usr/bin/env python3
"""
Offline stop skill probe for Qwen3-VL (tier-2 / x9).

Same protocol as train/skill_eval_left_right.py; success when pred == 0 (stop).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from skill_eval_left_right import _evaluate_json  # noqa: E402
from qwen3_vl_interleaved_common import (  # noqa: E402
    load_model,
    load_processor,
    parse_image_roots,
    resolve_system_prompt,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Offline stop skill eval for Qwen3-VL")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get(
            "OPENFLY_EVAL_QWEN3_CHECKPOINT",
            "/nfs/np/mnt/xtb/vln/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h16-x9-0-lrb20-10ep/checkpoint-last",
        ),
    )
    parser.add_argument(
        "--json",
        default=os.environ.get(
            "OPENFLY_SKILL_STOP_JSON",
            str(root / "skill_eval" / "stop_evaluation_skill_validation.json"),
        ),
    )
    parser.add_argument(
        "--image-root",
        default=os.environ.get("OPENFLY_SKILL_IMAGE_ROOT", "/mnt/xtb/vln/train_curated"),
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("OPENFLY_SKILL_EVAL_OUT_DIR", ""),
    )
    parser.add_argument("--device", default=os.environ.get("OPENFLY_QWEN_DEVICE", "cuda:0"))
    parser.add_argument(
        "--temporal-past",
        type=int,
        default=int(os.environ.get("OPENFLY_QWEN_TEMPORAL_HISTORY_PAST", "16")),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("OPENFLY_QWEN_MAX_NEW_TOKENS", "16")),
    )
    parser.add_argument("--attn", default=os.environ.get("OPENFLY_QWEN_ATTN", "sdpa"))
    parser.add_argument("--limit", type=int, default=int(os.environ.get("OPENFLY_SKILL_EVAL_LIMIT", "0")))
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(args.out_dir) if args.out_dir else Path("eval_runs") / "skill_stop_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_roots = parse_image_roots(Path(args.image_root))
    jpath = Path(args.json)

    print(
        f"Skill stop eval\n"
        f"  checkpoint={args.checkpoint}\n"
        f"  image_roots={[str(r) for r in image_roots]}\n"
        f"  alignment: interleaved frame→action (GT history), temporal_past={args.temporal_past}\n"
        f"  image_preprocessing: native PIL + checkpoint processor\n"
        f"  json={jpath}\n"
        f"  out={out_dir}",
        flush=True,
    )

    with jpath.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {jpath}")

    n_stop_gt = sum(1 for item in data if int(item["action"][-1]) == 0)
    print(f"  dataset: n={len(data)} trajectories with gt_stop_final={n_stop_gt}", flush=True)

    processor = load_processor(args.checkpoint)
    model, device = load_model(args.checkpoint, args.device, args.attn)
    system_prompt = resolve_system_prompt()

    print(f"=== stop: {jpath} n={len(data)} ===", flush=True)
    preds, metrics = _evaluate_json(
        data=data,
        model=model,
        processor=processor,
        device=device,
        image_roots=image_roots,
        temporal_past=args.temporal_past,
        system_prompt=system_prompt,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )
    metrics["eval_json"] = str(jpath)
    metrics["checkpoint"] = args.checkpoint
    metrics["image_root"] = str(image_roots[0])
    metrics["skill"] = "stop"
    metrics["target_action"] = 0
    metrics["n_gt_stop_final"] = n_stop_gt

    pred_path = out_dir / "predictions.json"
    met_path = out_dir / "metrics.json"
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    with met_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Wrote {pred_path}", flush=True)
    print(f"Wrote {met_path} accuracy={metrics['accuracy']:.4f}", flush=True)

    summary: Dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "eval_json": str(jpath),
        "skill": "stop",
        "target_action": 0,
        "accuracy": metrics["accuracy"],
        "n_correct": metrics["n_correct"],
        "n_scored": metrics["n_scored"],
        "n_errors": metrics["n_errors"],
        "n_gt_stop_final": n_stop_gt,
        "pred_action_counts": metrics.get("pred_action_counts", {}),
        "metrics_path": str(met_path),
        "predictions_path": str(pred_path),
    }
    sum_path = out_dir / "summary.json"
    with sum_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote {sum_path}", flush=True)
    print(
        f"STOP accuracy={summary['accuracy']:.4f} "
        f"({summary['n_correct']}/{summary['n_scored']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
