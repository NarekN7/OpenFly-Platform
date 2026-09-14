#!/usr/bin/env python3
"""Offline L/R evaluation with streamed evidence for loss/accuracy comparisons."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

from qwen3_vl_interleaved_common import (
    DEFAULT_VLN_SYSTEM_PROMPT,
    VLN_ACTION_PARSER,
    VLN_ALLOWED_ACTION_IDS,
    action_generation_policy,
    action_output_format_valid,
    action_stopping_criteria,
    build_interleaved_messages,
    interleaved_window_lo,
    load_model,
    load_processor,
    load_trajectory_pil_frames,
    parse_vln_action_id,
    predict_action,
    processor_image_size,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def validate_rows(rows: list[dict], label: str) -> None:
    if not rows:
        raise ValueError(f"Empty {label} split")
    for i, row in enumerate(rows):
        actions = row["action"]
        indices = row["index_list"]
        if not actions or len(actions) != len(indices):
            raise ValueError(f"{label}[{i}]: empty or misaligned action/index_list")
        if any(type(a) is not int or a not in VLN_ALLOWED_ACTION_IDS for a in actions):
            raise ValueError(f"{label}[{i}]: unexpected action IDs")
        if actions[-1] not in (2, 3):
            raise ValueError(f"{label}[{i}]: final action must be left or right")
        if not isinstance(row["gpt_instruction"], str) or not row["image_path"]:
            raise ValueError(f"{label}[{i}]: invalid instruction or image_path")


def summarize(rows: list[dict]) -> dict:
    valid = [r for r in rows if "error" not in r]
    result = {"n_samples": len(rows), "n_scored": len(valid), "n_errors": len(rows) - len(valid)}
    result["mean_final_action_nll"] = (
        sum(r["final_action_nll"] for r in valid) / len(valid) if valid else None
    )
    for key in ("raw_first_token_correct", "generation_first_token_correct", "parsed_correct", "strict_correct"):
        scored = [r for r in valid if r[key] is not None]
        count = sum(r[key] for r in scored)
        result[key + "_count"] = count
        result[key + "_accuracy"] = count / len(scored) if scored else None
        result[key + "_accuracy_all_samples"] = count / len(rows) if scored else None
    format_rows = [r for r in valid if r["strict_correct"] is not None]
    result["n_format_scored"] = len(format_rows)
    result["n_invalid_strict_outputs"] = sum(r["strict_action"] is None for r in format_rows) if format_rows else None
    result["n_invalid_actions"] = sum(r["parsed_action"] is None for r in valid)
    result["action_parser"] = VLN_ACTION_PARSER
    result["n_correct_first_token_but_wrong_parsed"] = sum(
        r["raw_first_token_correct"] and not r["parsed_correct"] for r in valid
    )
    confusion: dict[str, int] = {}
    for row in valid:
        key = f"{row['gt_action']}->{row['parsed_action']}"
        confusion[key] = confusion.get(key, 0) + 1
    result["confusion_gt_parsed"] = confusion
    return result


def score_sample(model, processor, messages: list[dict], gt: int, device: str, max_new_tokens: int) -> dict:
    tok = processor.tokenizer
    target_ids = tok.encode(str(gt), add_special_tokens=False)
    if len(target_ids) != 1:
        raise ValueError(f"L/R diagnostic requires one-token labels, got {target_ids}")
    target = target_ids[0]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [p["image"] for m in messages for p in m["content"] if p["type"] == "image"]
    inputs = processor(text=text, images=images, return_tensors="pt", padding=False, truncation=False)
    input_hash = hashlib.sha256(inputs["input_ids"].numpy().tobytes()).hexdigest()
    pixel_hash = hashlib.sha256(inputs["pixel_values"].numpy().tobytes()).hexdigest()
    grid = inputs["image_grid_thw"].tolist()
    inputs = {
        k: (v.to(device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(device))
        for k, v in inputs.items()
    }
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            return_dict_in_generate=True,
            output_logits=True,
            stopping_criteria=action_stopping_criteria(tok, inputs["input_ids"].shape[1]),
        )
    token_ids = output.sequences[0, inputs["input_ids"].shape[1]:].tolist()
    logits = output.logits[0][0].float()
    log_probs = logits.log_softmax(-1)
    raw = tok.decode(token_ids, skip_special_tokens=True)
    parsed = parse_vln_action_id(raw)
    format_valid = action_output_format_valid(raw, parsed)
    format_scored = format_valid is not None
    strict = parsed if format_valid else None
    top_values, top_ids = log_probs.topk(8)
    raw_top = int(logits.argmax().item())
    return {
        "input_ids_sha256": input_hash,
        "pixel_values_sha256": pixel_hash,
        "image_grid_thw": grid,
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "generated_token_ids": token_ids,
        "raw_decode": raw,
        "raw_decode_with_special_tokens": tok.decode(token_ids, skip_special_tokens=False),
        "raw_first_token_id": raw_top,
        "raw_first_token_text": tok.decode([raw_top]),
        "generation_first_token_id": token_ids[0],
        "target_token_id": target,
        "final_action_nll": float(-log_probs[target].item()),
        "p_correct": float(log_probs[target].exp().item()),
        "target_token_rank": int((logits > logits[target]).sum().item()) + 1,
        "top_first_tokens": [
            {"id": int(i), "text": tok.decode([i]), "probability": float(v.exp().item())}
            for v, i in zip(top_values, top_ids.tolist())
        ],
        "parsed_action": parsed,
        "action_valid": parsed is not None,
        "output_format_valid": format_valid,
        "generation_policy": action_generation_policy(),
        "strict_action": strict,
        "raw_first_token_correct": raw_top == target,
        "generation_first_token_correct": token_ids[0] == target,
        "parsed_correct": parsed == gt,
        "strict_correct": (strict == gt) if format_scored else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temporal-past", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--full-response", action="store_true", help="Disable action stopping to measure unconstrained response format")
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument("--check-reference", action="store_true", help="Compare first sample of each split to the existing evaluator")
    args = parser.parse_args()
    if args.full_response:
        os.environ["OPENFLY_QWEN_STOP_AFTER_ACTION"] = "0"
    if args.temporal_past < 0 or args.max_new_tokens < 1 or args.limit_per_split < 0:
        parser.error("Invalid history, generation length, or sample limit")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    datasets = {}
    for split, path in (("train", args.train_json), ("validation", args.validation_json)):
        rows = json.loads(path.read_text())
        validate_rows(rows, split)
        datasets[split] = rows[:args.limit_per_split] if args.limit_per_split else rows
    args.out_dir.mkdir(parents=True, exist_ok=False)
    processor = load_processor(args.checkpoint)
    model, device = load_model(args.checkpoint, args.device, "sdpa")
    metadata = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "python": sys.version,
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "transformers", "Pillow")},
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "processor_class": type(processor).__name__,
        "image_processor_class": type(processor.image_processor).__name__,
        "processor_image_size": processor_image_size(processor),
        "system_prompt": DEFAULT_VLN_SYSTEM_PROMPT.strip(),
        "prompt_suffix": "",
        "history_actions": "ground_truth",
        "generation_config": model.generation_config.to_dict(),
        "generation_overrides": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "action_parser": VLN_ACTION_PARSER,
        "generation_policy": action_generation_policy(),
        "format_metric_policy": "Exact-format metrics are measured only with full-response generation; null when action stopping is enabled.",
        "invalid_action_policy": "Count as incorrect; included in n_scored. Strict output format is reported separately.",
        "nll_definition": "Full-vocabulary negative log probability of the final L/R token from the first generation forward pass, before logits processing; GT history; EOS excluded.",
        "error_policy": "Record every sample; report scored-only and all-sample accuracy; exit nonzero if any sample fails.",
        "source_sha256": {str(p): sha256_file(p) for p in (args.train_json, args.validation_json, Path(__file__), Path(__file__).with_name("qwen3_vl_interleaved_common.py"))},
    }
    write_json(args.out_dir / "run_config.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    all_metrics = {}
    any_errors = False
    for split, data in datasets.items():
        records = []
        split_dir = args.out_dir / split
        split_dir.mkdir()
        with (split_dir / "predictions.jsonl").open("x", buffering=1) as stream:
            for i, item in enumerate(data):
                actions = item["action"]
                t = len(actions) - 1
                lo = interleaved_window_lo(t, args.temporal_past)
                row = {"split": split, "sample_index": i, "image_path": item["image_path"],
                       "index_list": item["index_list"], "gt_action": actions[-1], "timestep": t,
                       "window_lo": lo, "window_past_actions": actions[lo:t], "n_images": t - lo + 1}
                start = time.monotonic()
                try:
                    # Match the existing evaluator's full-trajectory loading/error behavior.
                    frames = load_trajectory_pil_frames([args.image_root], item["image_path"], item["index_list"])
                    messages = build_interleaved_messages(DEFAULT_VLN_SYSTEM_PROMPT, item["gpt_instruction"], "", frames[lo:t+1], actions[lo:t])
                    row.update(score_sample(model, processor, messages, actions[-1], device, args.max_new_tokens))
                    if args.check_reference and i == 0:
                        pred, raw = predict_action(model, processor, messages, device, args.max_new_tokens)
                        if (pred, raw) != (row["parsed_action"], row["raw_decode"]):
                            raise RuntimeError(f"Reference parity failed: reference={(pred, raw)!r}, diagnostic={(row['parsed_action'], row['raw_decode'])!r}")
                        row["reference_parity"] = True
                        print(f"{split}: reference evaluator parity PASS", flush=True)
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    row["traceback"] = traceback.format_exc()
                    any_errors = True
                    print(f"{split}[{i}] ERROR: {row['error']}", flush=True)
                row["seconds"] = time.monotonic() - start
                records.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                if (i + 1) % 25 == 0 or i + 1 == len(data):
                    metrics = {"overall": summarize(records)}
                    for name, action in (("left", 2), ("right", 3)):
                        action_rows = [r for r in records if r["gt_action"] == action]
                        metrics[name] = summarize(action_rows)
                    write_json(split_dir / "metrics.json", metrics)
                    print(f"{split} {i+1}/{len(data)} {json.dumps(metrics['overall'])}", flush=True)
        all_metrics[split] = metrics
        write_json(args.out_dir / "summary.json", all_metrics)
    write_json(args.out_dir / "status.json", {"state": "failed" if any_errors else "complete", "n_errors": sum(m["overall"]["n_errors"] for m in all_metrics.values())})
    return 1 if any_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
