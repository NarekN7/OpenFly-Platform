#!/usr/bin/env python3
"""
Offline left/right skill probe for Qwen3-VL (tier-2 / x9).

Aligned with scripts/qwen3_vl_sft_nosfx.py via train/qwen3_vl_interleaved_common.py.

For each trajectory of length n, evaluate only the final timestep t=n-1:
  - Load GT frames via image_path + index_list (native PIL; processor resize)
  - Interleaved USER[frame]→ASSISTANT[gt_action] for s in [lo, t-1], then
    USER[frame_t] with instruction on the first window turn only (no last-turn
    suffix; no left-pad; lo = max(0, t - past))
  - Success = 1 iff predicted action equals action[-1] (2=left, 3=right)
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
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
    limit: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = data[:limit] if limit > 0 else data
    predictions: List[Dict[str, Any]] = []
    n_ok = 0
    n_err = 0
    pred_counts: Counter = Counter()
    conf: Counter = Counter()

    for i, item in enumerate(rows):
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

        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            scored = i + 1 - n_err
            acc = (n_ok / scored) if scored else 0.0
            print(
                f"progress {i + 1}/{len(rows)} correct={n_ok} scored={scored} "
                f"acc={acc:.4f} errors={n_err}",
                flush=True,
            )

    scored = len(rows) - n_err
    metrics: Dict[str, Any] = {
        "n_samples": len(rows),
        "n_scored": scored,
        "n_errors": n_err,
        "n_correct": n_ok,
        "accuracy": (n_ok / scored) if scored else 0.0,
        "pred_action_counts": {str(k): int(v) for k, v in sorted(pred_counts.items())},
        "confusion_gt_pred": {f"{a}->{b}": int(c) for (a, b), c in sorted(conf.items())},
        "temporal_history_past": temporal_past,
        "chat_layout": "interleaved",
        "history_actions": "gt",
        "processor_image_size": processor_image_size(processor),
        "image_preprocessing": "native_pil_plus_processor",
        "action_vocab": "tier2",
        "image_roots": [str(r) for r in image_roots],
    }
    return predictions, metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Offline left/right skill eval for Qwen3-VL")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get(
            "OPENFLY_EVAL_QWEN3_CHECKPOINT",
            "/nfs/np/mnt/xtb/vln/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h16-x9-0-lrb20-10ep/checkpoint-last",
        ),
    )
    parser.add_argument(
        "--json",
        action="append",
        dest="jsons",
        default=None,
        help="Skill JSON path; can repeat. Env OPENFLY_SKILL_EVAL_JSONS=comma-separated also works.",
    )
    parser.add_argument(
        "--image-root",
        default=os.environ.get("OPENFLY_SKILL_IMAGE_ROOT", "/mnt/xtb/vln/train_curated"),
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("OPENFLY_SKILL_EVAL_OUT_DIR", ""),
        help="Parent output dir; per-json subdirs left/right are created from filename.",
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

    jsons = list(args.jsons or [])
    if not jsons:
        env_j = os.environ.get("OPENFLY_SKILL_EVAL_JSONS", "").strip()
        if env_j:
            jsons = [p.strip() for p in env_j.split(",") if p.strip()]
    if not jsons:
        root = Path(__file__).resolve().parents[1]
        jsons = [
            str(root / "skill_eval" / "left_evaluation_skill_validation.json"),
            str(root / "skill_eval" / "right_evaluation_skill_validation.json"),
        ]

    out_parent = Path(args.out_dir) if args.out_dir else Path("eval_runs") / "skill_lr_run"
    out_parent.mkdir(parents=True, exist_ok=True)
    image_roots = parse_image_roots(Path(args.image_root))

    print(
        f"Skill left/right eval\n"
        f"  checkpoint={args.checkpoint}\n"
        f"  image_roots={[str(r) for r in image_roots]}\n"
        f"  alignment: interleaved frame→action (GT history), temporal_past={args.temporal_past} "
        f"(no left-pad; up to {args.temporal_past + 1} frames)\n"
        f"  image_preprocessing: native PIL + checkpoint processor (no cv2 pre-resize)\n"
        f"  jsons={jsons}\n"
        f"  out={out_parent}",
        flush=True,
    )

    processor = load_processor(args.checkpoint)
    model, device = load_model(args.checkpoint, args.device, args.attn)
    system_prompt = resolve_system_prompt()

    summary: Dict[str, Any] = {"checkpoint": args.checkpoint, "sets": {}}
    total_correct = 0
    total_scored = 0

    for jp in jsons:
        jpath = Path(jp)
        tag = jpath.stem.replace("validationx9_", "")
        if "left" in tag:
            tag = "left"
        elif "right" in tag:
            tag = "right"
        out_dir = out_parent / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        with jpath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list in {jpath}")

        print(f"=== {tag}: {jpath} n={len(data)} ===", flush=True)
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

        pred_path = out_dir / "predictions.json"
        met_path = out_dir / "metrics.json"
        with pred_path.open("w", encoding="utf-8") as f:
            json.dump(preds, f, ensure_ascii=False, indent=2)
        with met_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Wrote {pred_path}", flush=True)
        print(f"Wrote {met_path} accuracy={metrics['accuracy']:.4f}", flush=True)

        summary["sets"][tag] = {
            "accuracy": metrics["accuracy"],
            "n_correct": metrics["n_correct"],
            "n_scored": metrics["n_scored"],
            "n_errors": metrics["n_errors"],
            "metrics_path": str(met_path),
        }
        total_correct += metrics["n_correct"]
        total_scored += metrics["n_scored"]

    summary["overall_accuracy"] = (total_correct / total_scored) if total_scored else 0.0
    summary["overall_n_correct"] = total_correct
    summary["overall_n_scored"] = total_scored
    sum_path = out_parent / "summary.json"
    with sum_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote {sum_path}", flush=True)
    print(
        f"OVERALL accuracy={summary['overall_accuracy']:.4f} "
        f"({total_correct}/{total_scored})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
