#!/usr/bin/env python3
"""
Exact overfit-8 final-step probe matching scripts/qwen3_vl_sft_overfit.py training.

What matches training:
  - DEFAULT_VLN_SYSTEM_PROMPT
  - temporal_history_past = 16 => 17 frames, left-padded for short trajectories
  - single-turn: user(images + instruction) -> assistant(action id)
  - same chat template / processor path as the SFT collator
  - greedy next-token after the assistant generation prompt (same position training CE targets)

This historical single-turn probe shares action parsing/stopping with the
interleaved evaluators; its image/message layout remains specific to overfit training.
First-token accuracy compares vocabulary token IDs. Actions 1 and 10 share
their first Qwen token, so this metric cannot distinguish those two actions.

Example (on np):
  python eval_overfit8_exact.py \\
    --model_dir /mnt/xtb/vln/qwen3-vl-2b-vln-overfit8-lrb-fixlt/checkpoint-last \\
    --eval_json /mnt/xtb/vln/qwen3-vl-2b-vln-overfit8-lrb-fixlt/overfit8_data/eval_8.json \\
    --frames_root /mnt/xtb/vln/train_curated
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Add the repo root, not train/: train/datasets must not shadow Hugging Face datasets.
REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train.qwen3_vl_interleaved_common import (
    DEFAULT_VLN_SYSTEM_PROMPT,
    VLN_ACTION_PARSER,
    VLN_ALLOWED_ACTION_IDS,
    action_generation_policy,
    action_output_format_valid,
    action_stopping_criteria,
    parse_vln_action_id,
)

TEMPORAL_HISTORY_PAST = 16


def frame_paths_for_timestep(
    traj_dir: Path,
    index_list: Sequence[str],
    timestep: int,
    temporal_history_past: int,
) -> List[Path]:
    """Same packing as qwen3_vl_sft_overfit._frame_paths_for_timestep."""
    past = temporal_history_past
    lo = max(0, timestep - past)
    idxs = list(index_list[lo : timestep + 1])
    target = past + 1
    while len(idxs) < target:
        idxs.insert(0, idxs[0])
    paths = [traj_dir / f"{idx}.png" for idx in idxs]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frames under {traj_dir}: {missing[:5]}")
    return paths


def build_messages(item: Dict[str, Any], frames_root: Path) -> List[Dict[str, Any]]:
    """Same sample pack as VlnOverfitMixedDataset._pack (always last timestep)."""
    image_path = item["image_path"]
    instruction = item["gpt_instruction"]
    actions = list(item["action"])
    index_list = [str(x) for x in item["index_list"]]
    if not actions or len(actions) != len(index_list):
        raise ValueError(f"Bad action/index_list for {image_path}")

    t = len(actions) - 1
    traj_dir = frames_root / image_path
    paths = frame_paths_for_timestep(traj_dir, index_list, t, TEMPORAL_HISTORY_PAST)

    user_content: List[Dict[str, Any]] = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        user_content.append({"type": "image", "image": im})
    user_content.append({"type": "text", "text": instruction})

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": DEFAULT_VLN_SYSTEM_PROMPT.strip()}],
        },
        {"role": "user", "content": user_content},
    ]


@torch.inference_mode()
def predict_one(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    messages: List[Dict[str, Any]],
    *,
    gt_action: int,
    max_new_tokens: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Inference aligned with training:
      apply_chat_template(..., add_generation_prompt=True)
      then greedy decode.

    Also reports single-step argmax at the first generated position (the token
    training last_token / weighted CE actually pushes on for the first action digit).
    """
    if max_new_tokens < 2:
        raise ValueError("max_new_tokens must be >= 2 to distinguish action 10 from 1")
    if gt_action not in VLN_ALLOWED_ACTION_IDS:
        raise ValueError(f"Invalid target action: {gt_action}")
    target_ids = processor.tokenizer.encode(str(gt_action), add_special_tokens=False)
    if not target_ids:
        raise ValueError(f"Empty tokenization for action {gt_action}")
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    images: List[Image.Image] = []
    for msg in messages:
        for part in msg["content"]:
            if isinstance(part, dict) and part.get("type") == "image":
                images.append(part["image"])

    inputs = processor(
        text=prompt_text,
        images=images,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # Single next-token argmax (teacher-free, same prefix training uses before the action).
    out_logits = model(**{k: v for k, v in inputs.items() if k != "labels"}).logits
    next_id = int(out_logits[0, -1].argmax(dim=-1).item())
    next_tok = processor.tokenizer.decode([next_id], skip_special_tokens=False)
    next_txt = processor.tokenizer.decode([next_id], skip_special_tokens=True)
    del out_logits

    in_len = int(inputs["input_ids"].shape[1])
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        stopping_criteria=action_stopping_criteria(processor.tokenizer, in_len),
    )
    new_ids = gen[0, in_len:]
    gen_text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
    pred = parse_vln_action_id(gen_text)
    format_valid = action_output_format_valid(gen_text, pred)
    correct = pred == gt_action

    return {
        "gen_text": gen_text,
        "pred": pred,
        "ok": correct,
        "action_valid": pred is not None,
        "output_format_valid": format_valid,
        "strict_correct": (correct and format_valid) if format_valid is not None else None,
        "generated_token_ids": new_ids.tolist(),
        "action_parser": VLN_ACTION_PARSER,
        "generation_policy": action_generation_policy(),
        "invalid_action_policy": "count_as_incorrect",
        "target_token_ids": target_ids,
        "first_token_id": next_id,
        "first_token_raw": next_tok,
        "first_token_text": next_txt,
        # Kept as a diagnostic field; this parsed prefix is not a full action prediction for 10.
        "pred_from_first_token": parse_vln_action_id(next_txt),
        "ok_first_token": next_id == target_ids[0],
        "n_images": len(images),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    format_rows = [r for r in rows if r["output_format_valid"] is not None]
    n_correct = sum(r["ok"] for r in rows)
    n_first_correct = sum(r["ok_first_token"] for r in rows)
    n_invalid_format = None
    strict_accuracy = None
    if format_rows:
        n_invalid_format = sum(not r["output_format_valid"] for r in format_rows)
        strict_accuracy = sum(r["strict_correct"] for r in format_rows) / len(format_rows)
    return {
        "n_total": n,
        "n_scored": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n if n else None,
        "n_first_token_correct": n_first_correct,
        "first_token_accuracy": n_first_correct / n if n else None,
        "first_token_metric": "full_vocabulary_argmax_equals_first_target_token_id",
        "n_invalid_actions": sum(not r["action_valid"] for r in rows),
        "n_format_scored": len(format_rows),
        "n_invalid_format": n_invalid_format,
        "strict_accuracy": strict_accuracy,
        "action_parser": VLN_ACTION_PARSER,
        "generation_policy": action_generation_policy(),
        "invalid_action_policy": "count_as_incorrect",
        "message_protocol": "overfit_single_turn_padded_history",
        "temporal_history_past": TEMPORAL_HISTORY_PAST,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_dir",
        type=str,
        default="/mnt/xtb/vln/qwen3-vl-2b-vln-overfit8-lrb-fixlt/checkpoint-last",
        help="HF checkpoint directory with model weights (including sharded saves) and processor files",
    )
    ap.add_argument(
        "--eval_json",
        type=str,
        default="/mnt/xtb/vln/qwen3-vl-2b-vln-overfit8-lrb-fixlt/overfit8_data/eval_8.json",
        help="Overfit eval_8.json (4 general + 2 left + 2 right)",
    )
    ap.add_argument(
        "--frames_root",
        type=str,
        default="/mnt/xtb/vln/train_curated",
        help="Root containing env_airsim_*/... png trees",
    )
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--full-response", action="store_true", help="Disable action stopping to measure output format")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument(
        "--out_json",
        type=str,
        default="",
        help="Write per-sample results JSON and a companion .summary.json with metrics/protocol",
    )
    args = ap.parse_args()
    if args.max_new_tokens < 2:
        ap.error("--max_new_tokens must be >= 2 to distinguish action 10 from 1")
    if args.full_response:
        os.environ["OPENFLY_QWEN_STOP_AFTER_ACTION"] = "0"

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = Path(args.model_dir)
    eval_json = Path(args.eval_json)
    frames_root = Path(args.frames_root)
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    if not eval_json.is_file():
        raise FileNotFoundError(eval_json)
    if not frames_root.is_dir():
        raise FileNotFoundError(frames_root)
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 8:
        raise ValueError("Expected eval_8.json with exactly 8 trajectory objects")
    for i, item in enumerate(data):
        actions, indices = item["action"], item["index_list"]
        if not actions or len(actions) != len(indices):
            raise ValueError(f"Row {i}: empty or misaligned action/index_list")
        if any(type(a) is not int or a not in VLN_ALLOWED_ACTION_IDS for a in actions):
            raise ValueError(f"Row {i}: expected tier-2 integer action IDs {sorted(VLN_ALLOWED_ACTION_IDS)}")

    print(f"model_dir   = {model_dir}")
    print(f"eval_json   = {eval_json}")
    print(f"frames_root = {frames_root}")
    print(f"device={device} dtype={args.dtype}")
    print(f"temporal_history_past={TEMPORAL_HISTORY_PAST}")
    print(f"action_parser={VLN_ACTION_PARSER} generation_policy={action_generation_policy()}")

    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    roles = ["general"] * 4 + ["left"] * 2 + ["right"] * 2
    rows: List[Dict[str, Any]] = []

    print()
    print(f"{'#':>2}  {'role':7s}  {'GT':>3}  {'Pred':>4}  {'OK':>2}  {'1stTok':>8}  {'OK1':>3}  gen_text")
    print("-" * 90)

    for i, item in enumerate(data):
        gt = int(item["action"][-1])
        messages = build_messages(item, frames_root)
        out = predict_one(
            model,
            processor,
            messages,
            gt_action=gt,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        pred = out["pred"]
        ok = out["ok"]
        ok1 = out["ok_first_token"]
        row = {
            **out,
            "i": i,
            "role": roles[i],
            "image_path": item["image_path"],
            "gt": gt,
            "action": item["action"],
            "index_list": item["index_list"],
        }
        rows.append(row)
        print(
            f"{i:2d}  {roles[i]:7s}  {gt:3d}  {str(pred):>4s}  {'Y' if ok else 'N':>2}  "
            f"{out['first_token_raw']!r:>8s}  {'Y' if ok1 else 'N':>3}  {out['gen_text']!r}"
        )

    metrics = summarize(rows)
    print("-" * 90)
    print(f"action accuracy:       {metrics['n_correct']}/8 = {100.0 * metrics['accuracy']:.1f}%")
    print(f"first-token accuracy:  {metrics['n_first_token_correct']}/8 = {100.0 * metrics['first_token_accuracy']:.1f}%")
    print("First-token accuracy compares token IDs; it cannot distinguish actions 1 and 10.")
    print(f"invalid actions: {metrics['n_invalid_actions']}/8")
    if metrics["n_format_scored"]:
        print(f"invalid output format: {metrics['n_invalid_format']}/8; strict accuracy: {metrics['strict_accuracy']:.1%}")
    else:
        print("Output format is unmeasured with action stopping; use --full-response to measure it.")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary_path = out_path.with_suffix(".summary.json")
        summary_path.write_text(
            json.dumps({**vars(args), **metrics}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {out_path} and {summary_path}")


if __name__ == "__main__":
    main()
