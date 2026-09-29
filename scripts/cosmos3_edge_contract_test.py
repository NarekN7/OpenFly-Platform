#!/usr/bin/env python3
"""Real Cosmos3-Edge processor/model forward-backward contract check."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open

from cosmos3_edge_vl_sft import (
    COSMOS3_EDGE_REPO,
    Cosmos3EdgeModelFactory,
    Cosmos3EdgeProcessorFactory,
    Cosmos3EdgeTrajectoryCollator,
)
from qwen3_vl_sft import DEFAULT_VLN_SYSTEM_PROMPT, VlnTrajectoryCropDataset, WeightedTrainer


def _gradient_norm(module: torch.nn.Module) -> float:
    total = 0.0
    count = 0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().float().norm().item())
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite gradient in {type(module).__name__}")
        total += value
        count += 1
    if count == 0 or total <= 0:
        raise RuntimeError(f"No nonzero gradients in {type(module).__name__}")
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=COSMOS3_EDGE_REPO)
    parser.add_argument(
        "--train_json",
        default="/home/nnurijanyan/OpenFly-Platform/data_curated/trainx9_curated_0.json",
    )
    parser.add_argument(
        "--frames_root", default="/mnt/weka/nnurijanyan/data/vln/train_curated"
    )
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--temporal_history_past", type=int, default=16)
    parser.add_argument("--min_pixels", type=int, default=784)
    parser.add_argument("--max_pixels", type=int, default=57600)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("The real forward/backward contract test requires a CUDA GPU")

    processor = Cosmos3EdgeProcessorFactory.from_pretrained(args.model_name_or_path)
    processor.image_processor.size = {
        **dict(processor.image_processor.size),
        "shortest_edge": args.min_pixels,
        "longest_edge": args.max_pixels,
    }
    collator = Cosmos3EdgeTrajectoryCollator(
        processor=processor,
        max_length=16384,
        system_prompt=DEFAULT_VLN_SYSTEM_PROMPT,
        supervise_eos=True,
    )

    dataset = VlnTrajectoryCropDataset(
        json_path=args.train_json,
        frames_root=args.frames_root,
        temporal_history_past=args.temporal_history_past,
        deterministic=True,
        verify_images_exist=True,
    )
    row = dataset[args.sample_index]
    batch = collator([row])
    if "mm_token_type_ids" not in batch:
        raise RuntimeError("Processor omitted required mm_token_type_ids")
    supervised = int((batch["labels"] != -100).sum().item())
    if supervised <= 0:
        raise RuntimeError("Collator produced no supervised action labels")

    # Exercise every legal action through the real tokenizer/template.
    first_image = next(
        part["image_path"]
        for message in row["messages"]
        for part in message.get("content", [])
        if isinstance(part, dict) and part.get("type") == "image"
    )
    for action in (0, 1, 2, 3, 4, 5, 8, 9, 10):
        probe = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image_path": first_image},
                        {"type": "text", "text": "Navigate."},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": str(action)}]},
            ],
            "loss_mode": "last_token",
        }
        probe_batch = collator([probe])
        decoded = processor.tokenizer.decode(
            probe_batch["labels"][0][probe_batch["labels"][0] != -100].tolist(),
            skip_special_tokens=True,
        ).strip()
        if decoded != str(action):
            raise RuntimeError(f"Real tokenizer action mismatch: {action=} {decoded=}")

    model = Cosmos3EdgeModelFactory.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train().cuda()

    device_batch = {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    trainer = object.__new__(WeightedTrainer)
    trainer.loss_type = "weighted"
    trainer.im_start_id = collator._im_start_id
    trainer.assistant_id = collator._assistant_id
    trainer.im_end_id = collator._im_end_id
    trainer.newline_id = collator._newline_id
    trainer.supervise_eos = True
    trainer._loss_mask_dumped = False
    loss = trainer.compute_loss(model, device_batch)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss: {loss}")
    loss.backward()
    gradient_norm_sums = {
        "vision": _gradient_norm(model.model.visual),
        "projector": _gradient_norm(model.model.projector),
        "language_model": _gradient_norm(model.model.language_model),
        "lm_head": _gradient_norm(model.lm_head),
    }

    with tempfile.TemporaryDirectory(prefix="cosmos3-edge-contract-") as tmp_dir:
        model.save_pretrained(tmp_dir)
        processor.save_pretrained(tmp_dir)
        index_path = Path(tmp_dir) / "model.safetensors.index.json"
        weight_path = Path(tmp_dir) / "model.safetensors"
        if not weight_path.is_file() and not index_path.is_file():
            raise RuntimeError(f"save_pretrained wrote no weights under {tmp_dir}")
        if index_path.is_file():
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map") or {}
            for shard in sorted(set(weight_map.values())):
                with safe_open(str(Path(tmp_dir) / shard), framework="pt", device="cpu") as tensors:
                    if not list(tensors.keys()):
                        raise RuntimeError(f"Empty shard: {shard}")
        del model
        torch.cuda.empty_cache()
        reloaded = Cosmos3EdgeModelFactory.from_pretrained(tmp_dir, torch_dtype=torch.bfloat16)
        reloaded.eval().cuda()
        with torch.no_grad():
            reload_out = reloaded(
                **{
                    key: value
                    for key, value in device_batch.items()
                    if key not in {"labels", "loss_mode"} and isinstance(value, torch.Tensor)
                }
            )
        if reload_out.logits is None or not torch.isfinite(reload_out.logits).all():
            raise RuntimeError("Reloaded checkpoint produced non-finite logits")
        del reloaded
        torch.cuda.empty_cache()

    result = {
        "loss": float(loss.detach().float().item()),
        "seq_len": int(batch["input_ids"].shape[1]),
        "supervised_tokens": supervised,
        "sample": row.get("traj_meta"),
        "reload_ok": True,
        "gradient_norm_sums": gradient_norm_sums,
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
