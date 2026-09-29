"""Cosmos3-Edge inference that matches the Qwen interleaved VLN eval contract.

Uses the existing vln interpreter. The pinned cosmos-framework checkout is added
to sys.path; torch, torchvision, and transformers are not upgraded. The processor
module is loaded by file path so the package ``__init__`` does not import the
unused multistorage client.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from qwen3_vl_interleaved_common import (
    action_stopping_criteria,
    parse_vln_action_id,
)

OPENFLY_BACKEND = "cosmos3_edge"
_DEFAULT_FRAMEWORK_ROOT = Path("/auto/home/nareknurijanyan/cosmos-framework")
_MIN_PIXELS = 784
_MAX_PIXELS = 57600


def framework_root() -> Path:
    raw = os.environ.get("COSMOS_FRAMEWORK_ROOT", "").strip()
    return Path(raw) if raw else _DEFAULT_FRAMEWORK_ROOT


def ensure_framework_on_path() -> Path:
    root = framework_root()
    if not (root / "cosmos_framework").is_dir():
        raise FileNotFoundError(
            f"Cosmos framework checkout not found at {root}. "
            "Set COSMOS_FRAMEWORK_ROOT to the pinned commit checkout."
        )
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


def is_cosmos3_edge_checkpoint(ckpt: str) -> bool:
    config_path = Path(ckpt) / "config.json"
    if not config_path.is_file():
        return False
    try:
        model_type = json.loads(config_path.read_text(encoding="utf-8")).get("model_type")
    except (OSError, ValueError):
        return False
    return model_type == "cosmos3_edge"


def _load_processor_builder():
    ensure_framework_on_path()
    path = (
        framework_root()
        / "cosmos_framework"
        / "data"
        / "generator"
        / "processors"
        / "cosmos3_edge_processing.py"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing Cosmos processor module: {path}")
    spec = importlib.util.spec_from_file_location("openfly_cosmos3_edge_processing", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_cosmos3_edge_processor


def load_cosmos3_processor(ckpt: str) -> Any:
    builder = _load_processor_builder()
    processor = builder(str(Path(ckpt)))
    size = dict(getattr(processor.image_processor, "size", {}) or {})
    print(
        f"Cosmos3-Edge processor loaded from {ckpt} "
        f"image_size={size}",
        flush=True,
    )
    return processor


def load_cosmos3_model(ckpt: str, device: str, attn: str) -> Any:
    ensure_framework_on_path()
    from cosmos_framework.model.generator.reasoner.cosmos3_edge import (
        Cosmos3EdgeForConditionalGeneration,
    )

    attn_implementation = (attn or "sdpa").strip() or "sdpa"
    model = Cosmos3EdgeForConditionalGeneration.from_pretrained(
        str(Path(ckpt)),
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    if os.environ.get("OPENFLY_QWEN_DEVICE_MAP", "").strip():
        raise ValueError("Cosmos3-Edge eval does not support OPENFLY_QWEN_DEVICE_MAP")
    model = model.to(device)
    model.eval()
    model._openfly_backend = OPENFLY_BACKEND
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Cosmos3-Edge model loaded from {ckpt} device={device} "
        f"attn={attn_implementation} params={n_params:,}",
        flush=True,
    )
    return model


def _pixel_budget(processor: Any) -> Tuple[int, int]:
    size = dict(getattr(getattr(processor, "image_processor", None), "size", {}) or {})
    min_pixels = int(size.get("shortest_edge", _MIN_PIXELS))
    max_pixels = int(size.get("longest_edge", _MAX_PIXELS))
    return min_pixels, max_pixels


def messages_with_pixel_budget(messages: Sequence[Dict[str, Any]], processor: Any) -> List[Dict[str, Any]]:
    """Match the training collator: per-image min/max pixels from the processor size."""
    min_pixels, max_pixels = _pixel_budget(processor)
    copied: List[Dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            copied.append(dict(message))
            continue
        new_content = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                updated = dict(part)
                updated["min_pixels"] = min_pixels
                updated["max_pixels"] = max_pixels
                new_content.append(updated)
            else:
                new_content.append(part)
        copied.append({**message, "content": new_content})
    return copied


def _batch_tensors(outputs: Any) -> Dict[str, torch.Tensor]:
    if hasattr(outputs, "items"):
        raw = dict(outputs)
    else:
        raw = dict(outputs)
    batched: Dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            try:
                value = torch.as_tensor(value)
            except (TypeError, ValueError, RuntimeError):
                continue
        if key in {"input_ids", "attention_mask"} and value.ndim == 1:
            value = value.unsqueeze(0)
        batched[key] = value
    if "input_ids" not in batched:
        raise RuntimeError(f"Cosmos processor did not return input_ids; keys={list(batched)}")
    return batched


def predict_action_cosmos(
    model,
    processor,
    messages: Sequence[Dict[str, Any]],
    device: str,
    max_new_tokens: int,
) -> Tuple[Optional[int], str]:
    """Greedy action generation with thinking disabled, matching training targets."""
    if max_new_tokens < 2:
        raise ValueError("Full navigation needs max_new_tokens >= 2 for action 10")
    model.eval()
    prepared = messages_with_pixel_budget(messages, processor)
    outputs = processor.apply_chat_template(
        prepared,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
    inputs = _batch_tensors(outputs)
    # The reasoner forward reads vision tensors and token ids. Extra processor
    # keys (mm token types) are not model inputs.
    keep = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    inputs = {key: value for key, value in inputs.items() if key in keep}

    def _to_dev(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_floating_point():
            return tensor.to(device, dtype=torch.bfloat16)
        return tensor.to(device)

    inputs = {key: _to_dev(value) for key, value in inputs.items()}
    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    prompt_len = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=pad_id,
            stopping_criteria=action_stopping_criteria(tok, prompt_len),
        )
    text_out = tok.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)
    return parse_vln_action_id(text_out), text_out
