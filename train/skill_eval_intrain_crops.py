#!/usr/bin/env python3
"""
Offline in-train crop fit probe for Qwen3-VL (tier-2 / x9).

Aligned with scripts/qwen3_vl_sft_nosfx.py via train/qwen3_vl_interleaved_common.py.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from qwen3_vl_interleaved_common import (
    build_interleaved_messages,
    interleaved_window_lo,
    load_model,
    load_processor,
    load_trajectory_pil_frames,
    parse_image_roots,
    predict_action,
    processor_image_size,
    resolve_prompt_suffix,
    resolve_system_prompt,
)


def _evaluate_json(
    *,
    data: List[Dict[str, Any]],
    model,
    processor,
    device: str,
    image_roots: Sequence[Path],
    temporal_past: int,
    system_prompt: str,
    max_new_tokens: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    n_ok = 0
    n_err = 0
    pred_counts: Counter = Counter()
    conf: Counter = Counter()
    per_gt: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n": 0, "n_correct": 0})

    for i, item in enumerate(data):
        image_path = item["image_path"]
        actions = [int(a) for a in item["action"]]
        index_list = item["index_list"]
        instruction = item.get("gpt_instruction", "")
        gt = int(actions[-1])
        t = len(actions) - 1
        try:
            all_frames = load_trajectory_pil_frames(image_roots, image_path, index_list)
            if len(all_frames) != len(actions):
                raise ValueError(
                    f"frame/action length mismatch: frames={len(all_frames)} actions={len(actions)}"
                )
            lo = interleaved_window_lo(t, temporal_past)
            pils = list(all_frames[lo : t + 1])
            window_past_actions = list(actions[lo:t])
            messages = build_interleaved_messages(
                system_prompt,
                instruction,
                resolve_prompt_suffix(),
                pils,
                window_past_actions,
            )
            pred, raw = predict_action(model, processor, messages, device, max_new_tokens)
            correct = 1 if pred == gt else 0
            n_ok += correct
            pred_counts[pred] += 1
            conf[(gt, pred)] += 1
            gk = str(gt)
            per_gt[gk]["n"] += 1
            per_gt[gk]["n_correct"] += correct
            predictions.append(
                {
                    "sample_index": i,
                    "image_path": image_path,
                    "gt_action": gt,
                    "pred_action": pred,
                    "correct": correct,
                    "prefix_actions": list(actions[:-1]),
                    "window_past_actions": window_past_actions,
                    "n_images": len(pils),
                    "window_lo": lo,
                    "timestep": t,
                    "chat_layout": "interleaved",
                    "raw_decode": raw,
                }
            )
        except Exception as e:
            n_err += 1
            predictions.append(
                {
                    "sample_index": i,
                    "image_path": image_path,
                    "gt_action": gt,
                    "pred_action": None,
                    "correct": 0,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            print(f"[{i}] ERROR {image_path}: {e}", flush=True)

        if (i + 1) % 25 == 0 or i + 1 == len(data):
            scored = i + 1 - n_err
            acc = (n_ok / scored) if scored else 0.0
            print(
                f"progress {i + 1}/{len(data)} correct={n_ok} scored={scored} "
                f"acc={acc:.4f} errors={n_err}",
                flush=True,
            )

    scored = len(data) - n_err
    per_gt_action_accuracy: Dict[str, Dict[str, float]] = {}
    for gk, stats in sorted(per_gt.items(), key=lambda x: int(x[0])):
        n = stats["n"]
        nc = stats["n_correct"]
        per_gt_action_accuracy[gk] = {
            "n": n,
            "n_correct": nc,
            "accuracy": (nc / n) if n else 0.0,
        }

    metrics: Dict[str, Any] = {
        "n_samples": len(data),
        "n_scored": scored,
        "n_errors": n_err,
        "n_correct": n_ok,
        "accuracy": (n_ok / scored) if scored else 0.0,
        "per_gt_action_accuracy": per_gt_action_accuracy,
        "pred_action_counts": {str(k): int(v) for k, v in sorted(pred_counts.items())},
        "confusion_gt_pred": {f"{a}->{b}": int(c) for (a, b), c in sorted(conf.items())},
        "temporal_history_past": temporal_past,
        "images_per_step": temporal_past + 1,
        "chat_layout": "interleaved",
        "history_actions": "gt",
        "processor_image_size": processor_image_size(processor),
        "image_preprocessing": "native_pil_plus_processor",
        "action_vocab": "tier2",
        "image_roots": [str(r) for r in image_roots],
    }
    return predictions, metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Offline in-train crop fit eval for Qwen3-VL")
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
            "OPENFLY_INTRAIN_EVAL_JSON",
            str(Path(__file__).resolve().parents[1] / "skill_eval" / "trainx9_intrain_eval_500.json"),
        ),
    )
    parser.add_argument(
        "--image-root",
        default=os.environ.get("OPENFLY_SKILL_IMAGE_ROOT", "/nfs/np/mnt/xtb/vln/train_curated"),
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

    out_dir = Path(args.out_dir) if args.out_dir else Path("eval_runs") / "skill_intrain_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_roots = parse_image_roots(Path(args.image_root))
    jpath = Path(args.json)

    print(
        f"In-train crop fit eval\n"
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

    rows = data[:args.limit] if args.limit > 0 else data
    n_available = sum(
        1
        for item in rows
        if all(
            any((root / item["image_path"] / f"{idx}.png").is_file() for root in image_roots)
            for idx in item["index_list"]
        )
    )
    print(f"  frame_coverage={n_available}/{len(rows)} crops resolvable", flush=True)

    processor = load_processor(args.checkpoint)
    model, device = load_model(args.checkpoint, args.device, args.attn)
    system_prompt = resolve_system_prompt()

    print(f"=== intrain: {jpath} n={len(rows)} ===", flush=True)
    preds, metrics = _evaluate_json(
        data=rows,
        model=model,
        processor=processor,
        device=device,
        image_roots=image_roots,
        temporal_past=args.temporal_past,
        system_prompt=system_prompt,
        max_new_tokens=args.max_new_tokens,
    )
    metrics["eval_json"] = str(jpath)
    metrics["checkpoint"] = args.checkpoint
    metrics["image_root"] = str(image_roots[0])
    metrics["frame_coverage_preflight"] = {"n_available": n_available, "n_total": len(rows)}

    pred_path = out_dir / "predictions.json"
    met_path = out_dir / "metrics.json"
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    with met_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Wrote {pred_path}", flush=True)
    print(f"Wrote {met_path} accuracy={metrics['accuracy']:.4f}", flush=True)

    summary = {
        "checkpoint": args.checkpoint,
        "eval_json": str(jpath),
        "accuracy": metrics["accuracy"],
        "n_correct": metrics["n_correct"],
        "n_scored": metrics["n_scored"],
        "n_errors": metrics["n_errors"],
        "per_gt_action_accuracy": metrics["per_gt_action_accuracy"],
        "metrics_path": str(met_path),
        "predictions_path": str(pred_path),
    }
    sum_path = out_dir / "summary.json"
    with sum_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote {sum_path}", flush=True)
    print(
        f"OVERALL accuracy={summary['accuracy']:.4f} "
        f"({summary['n_correct']}/{summary['n_scored']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
