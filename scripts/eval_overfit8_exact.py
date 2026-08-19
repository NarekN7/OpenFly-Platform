#!/usr/bin/env python3
"""
Exact overfit-8 final-step probe matching scripts/qwen3_vl_sft_overfit.py training.

What matches training:
  - DEFAULT_VLN_SYSTEM_PROMPT
  - temporal_history_past = 16  => up to 17 frames at last crop timestep
  - single-turn: user(images + instruction) -> assistant(action id)
  - same chat template / processor path as the SFT collator
  - greedy next-token after the assistant generation prompt (same position training CE targets)

Example (on np):
  python eval_overfit8_exact.py \\
    --model_dir /mnt/xtb/vln/qwen3-vl-2b-vln-overfit8-lrb-fixlt/checkpoint-last \\
    --eval_json /mnt/xtb/vln/qwen3-vl-2b-vln-overfit8-lrb-fixlt/overfit8_data/eval_8.json \\
    --frames_root /mnt/xtb/vln/train_curated
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

DEFAULT_VLN_SYSTEM_PROMPT = """You are an AI assistant controlling a flying drone. Navigate using the current camera view and the human instruction by replying with exactly one action id from 0 to 10 (digits only, no other text). Action meanings:
0. Stop
1. Move forward (×1)
2. Turn left (~30°)
3. Turn right (~30°)
4. Move up
5. Move down
8. Move forward (×2)
9. Move forward (×3)
10. Move forward (×9)
"""

TEMPORAL_HISTORY_PAST = 16
ALLOWED = {0, 1, 2, 3, 4, 5, 8, 9, 10}


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


def parse_action_id(text: str) -> int | None:
    """Parse first 0-10 action id from model text (digits only expected)."""
    text = text.strip()
    # Prefer whole-string / leading number matches.
    m = re.match(r"^\s*(10|[0-598])\b", text)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(10|[0-598])\b", text)
    if m:
        return int(m.group(1))
    return None


@torch.inference_mode()
def predict_one(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    messages: List[Dict[str, Any]],
    *,
    max_new_tokens: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Inference aligned with training:
      apply_chat_template(..., add_generation_prompt=True)
      then greedy decode.

    Also reports single-step argmax at the first generated position (the token
    training last_token / weighted CE actually pushes on for the action digit).
    """
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

    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    in_len = int(inputs["input_ids"].shape[1])
    new_ids = gen[0, in_len:]
    gen_text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
    pred = parse_action_id(gen_text)
    pred_from_argmax = parse_action_id(next_txt)

    return {
        "gen_text": gen_text,
        "pred": pred,
        "first_token_id": next_id,
        "first_token_raw": next_tok,
        "first_token_text": next_txt,
        "pred_from_first_token": pred_from_argmax,
        "n_images": len(images),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_dir",
        type=str,
        default="/mnt/xtb/vln/qwen3-vl-2b-vln-overfit8-lrb-fixlt/checkpoint-last",
        help="HF checkpoint dir with model.safetensors + processor files",
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
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument(
        "--out_json",
        type=str,
        default="",
        help="Optional path to write per-sample results JSON",
    )
    args = ap.parse_args()

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

    print(f"model_dir   = {model_dir}")
    print(f"eval_json   = {eval_json}")
    print(f"frames_root = {frames_root}")
    print(f"device={device} dtype={args.dtype}")
    print(f"temporal_history_past={TEMPORAL_HISTORY_PAST}")

    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    data = json.loads(eval_json.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 8:
        raise ValueError(f"Expected eval_8.json with 8 items, got {type(data)} len={getattr(data, '__len__', lambda: '?')()}")

    roles = ["general"] * 4 + ["left"] * 2 + ["right"] * 2
    rows: List[Dict[str, Any]] = []
    n_ok = 0
    n_ok_first = 0

    print()
    print(f"{'#':>2}  {'role':7s}  {'GT':>3}  {'Pred':>4}  {'OK':>2}  {'1stTok':>6}  {'OK1':>3}  gen_text")
    print("-" * 90)

    for i, item in enumerate(data):
        gt = int(item["action"][-1])
        messages = build_messages(item, frames_root)
        out = predict_one(
            model,
            processor,
            messages,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        pred = out["pred"]
        pred1 = out["pred_from_first_token"]
        ok = pred == gt
        ok1 = pred1 == gt
        n_ok += int(ok)
        n_ok_first += int(ok1)
        row = {
            "i": i,
            "role": roles[i],
            "image_path": item["image_path"],
            "gt": gt,
            "pred": pred,
            "ok": ok,
            "pred_from_first_token": pred1,
            "ok_first_token": ok1,
            "n_images": out["n_images"],
            "first_token_raw": out["first_token_raw"],
            "gen_text": out["gen_text"],
            "action": item["action"],
            "index_list": item["index_list"],
        }
        rows.append(row)
        print(
            f"{i:2d}  {roles[i]:7s}  {gt:3d}  {str(pred):>4s}  {'Y' if ok else 'N':>2}  "
            f"{str(pred1):>6s}  {'Y' if ok1 else 'N':>3}  {out['gen_text']!r}"
        )

    print("-" * 90)
    print(f"generate accuracy:     {n_ok}/8 = {100.0 * n_ok / 8:.1f}%")
    print(f"first-token accuracy:  {n_ok_first}/8 = {100.0 * n_ok_first / 8:.1f}%")
    print("(first-token = greedy argmax at the same position training CE targets)")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
