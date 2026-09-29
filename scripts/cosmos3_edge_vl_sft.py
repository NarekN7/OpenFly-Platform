#!/usr/bin/env python3
"""Cosmos3-Edge VLN SFT with the exact OpenFly Qwen LRB training semantics.

The dataset, skill mixing, weighted/last-token loss, Trainer configuration, and
best/last checkpoint callbacks are intentionally imported from qwen3_vl_sft.
Only model loading and multimodal collation are replaced for Cosmos3-Edge.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from filelock import FileLock
from huggingface_hub import snapshot_download

import qwen3_vl_sft as openfly_sft


COSMOS3_EDGE_REPO = "nvidia/Cosmos3-Edge"
COSMOS3_EDGE_REVISION = "344d602b128d1bbdacb43b08d0a3626f46343e29"

# The public omni repository is about 36 GB. Reasoner SFT needs only these
# assets. The indexed shards contain both generator and reasoner tensors;
# NVIDIA's loader reads only keys listed by the root index.
COSMOS3_EDGE_ALLOW_PATTERNS = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "transformer/*.safetensors",
    "vision_encoder/model.safetensors",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)

_RESOLVED_MODEL_PATHS: Dict[str, str] = {}


def _revision_for(repo_id: str) -> str | None:
    explicit = os.environ.get("COSMOS3_EDGE_MODEL_REVISION", "").strip()
    if explicit:
        return explicit
    if repo_id == COSMOS3_EDGE_REPO:
        return COSMOS3_EDGE_REVISION
    return None


def _resolve_model_path(model_name_or_path: str) -> str:
    """Resolve and selectively download the pinned official snapshot once per process."""
    path = Path(model_name_or_path).expanduser()
    if path.is_dir():
        return str(path.resolve())
    if model_name_or_path in _RESOLVED_MODEL_PATHS:
        return _RESOLVED_MODEL_PATHS[model_name_or_path]

    cache_root = openfly_sft._hf_hub_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    safe_name = model_name_or_path.replace("/", "__").replace(":", "_")
    lock_path = cache_root / f".openfly_cosmos3_edge_{safe_name}.lock"
    with FileLock(str(lock_path), timeout=7200):
        resolved = snapshot_download(
            repo_id=model_name_or_path,
            revision=_revision_for(model_name_or_path),
            allow_patterns=list(COSMOS3_EDGE_ALLOW_PATTERNS),
        )
    _RESOLVED_MODEL_PATHS[model_name_or_path] = resolved
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(
            f"[cosmos3-edge] pinned snapshot ready: repo={model_name_or_path!r} "
            f"revision={_revision_for(model_name_or_path)!r} path={resolved}",
            flush=True,
        )
    return resolved


def _prefetch_cosmos3_edge(model_name_or_path: str) -> None:
    _resolve_model_path(model_name_or_path)


class Cosmos3EdgeProcessorFactory:
    """AutoProcessor-compatible factory backed by NVIDIA's corrected processor."""

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **_: Any):
        from cosmos_framework.data.generator.processors.cosmos3_edge_processing import (
            build_cosmos3_edge_processor,
        )

        return build_cosmos3_edge_processor(_resolve_model_path(model_name_or_path))


def _has_top_level_weights(path: Path) -> bool:
    return any(
        (path / name).is_file()
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    ) and not (path / "transformer").is_dir()


class Cosmos3EdgeModelFactory:
    """Qwen model-factory compatible adapter using NVIDIA's indexed loader."""

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs: Any,
    ):
        # Importing registers model_type=cosmos3_edge with HF Auto classes.
        from cosmos_framework.model.generator.reasoner.cosmos3_edge import (
            Cosmos3EdgeConfig,
            Cosmos3EdgeForConditionalGeneration,
        )
        from cosmos_framework.model.generator.utils.safetensors_loader import load_vlm_model

        model_path = Path(_resolve_model_path(model_name_or_path))
        attn_implementation = os.environ.get(
            "COSMOS3_EDGE_ATTN_IMPLEMENTATION", "sdpa"
        ).strip()

        # OpenFly's own canonical checkpoints can use normal HF loading.
        if _has_top_level_weights(model_path):
            kwargs.pop("trust_remote_code", None)
            kwargs.pop("device_map", None)
            kwargs.pop("torch_dtype", None)
            return Cosmos3EdgeForConditionalGeneration.from_pretrained(
                str(model_path),
                dtype=torch_dtype,
                attn_implementation=attn_implementation,
                **kwargs,
            )

        # The official omni snapshot is not a normal HF VLM checkpoint. Its
        # root index names tensors inside mixed generator shards. NVIDIA's
        # loader remaps only reasoner keys and verifies the SigLIP2 shard hash.
        config = Cosmos3EdgeConfig.from_pretrained(str(model_path))
        config._attn_implementation = attn_implementation
        config.text_config._attn_implementation = attn_implementation
        config.vision_config._attn_implementation = attn_implementation
        model = Cosmos3EdgeForConditionalGeneration(config)
        model.to(dtype=torch_dtype)
        loaded = load_vlm_model(
            model=model,
            checkpoint_path=str(model_path),
            credential_path=None,
            parallel_dims=None,
        )
        if not loaded:
            raise RuntimeError("NVIDIA Cosmos3-Edge loader returned no reasoner tensors")
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_vision = sum(p.numel() for p in model.model.visual.parameters() if not p.requires_grad)
            print(
                f"[cosmos3-edge] loaded {len(loaded):,} tensors; "
                f"trainable parameters={trainable:,}; frozen vision={frozen_vision:,}; "
                f"attention={attn_implementation}",
                flush=True,
            )
        if any(not p.requires_grad for p in model.parameters()):
            # Official NVIDIA Edge SFT freezes SigLIP2. OpenFly LRB60 keeps the
            # full reasoner trainable, matching the Qwen run.
            for parameter in model.parameters():
                parameter.requires_grad = True
        return model


def _as_batch_tensor(value: Any, *, name: str) -> torch.Tensor:
    """Normalize processor outputs to a batched tensor ``[1, ...]`` or ``[N, ...]``."""
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.ndim == 0:
        raise ValueError(f"Processor returned a scalar for {name}")
    if value.ndim == 1 and name in {"input_ids", "attention_mask", "mm_token_type_ids", "labels"}:
        value = value.unsqueeze(0)
    return value


class Cosmos3EdgeTrajectoryCollator:
    """Cosmos processor adapter preserving OpenFly's action-only label contract."""

    def __init__(
        self,
        processor,
        max_length: int,
        system_prompt: str = "",
        supervise_eos: bool = True,
    ) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = int(max_length)
        self.system_prompt = system_prompt.strip()
        self.supervise_eos = bool(supervise_eos)

        def token_ids(text: str) -> tuple[int, ...]:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                raise RuntimeError(f"Expected at least one token for {text!r}")
            return tuple(int(token_id) for token_id in ids)

        def single_token_id(text: str) -> int:
            ids = token_ids(text)
            if len(ids) != 1:
                raise RuntimeError(f"Expected one token for {text!r}, got ids={ids}")
            return ids[0]

        self._im_start_id = single_token_id("<|im_start|>")
        self._im_end_id = single_token_id("<|im_end|>")
        # Cosmos's tokenizer splits "assistant" into ["ass", "istant"], unlike
        # Qwen's one-token role name. The shared trainer only needs the first
        # token to locate <|im_start|>assistant spans; label construction below
        # verifies and skips the complete role-token sequence.
        self._assistant_ids = token_ids("assistant")
        self._assistant_id = self._assistant_ids[0]
        self._newline_id = single_token_id("\n")
        self._think_start_id = single_token_id("<think>")
        self._think_end_id = single_token_id("</think>")

    def _prepend_system(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.system_prompt:
            return list(messages)
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            *messages,
        ]

    def _materialize_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        materialized, _ = openfly_sft._deep_copy_messages_replace_images(messages)
        # Per-image overrides use the exact Qwen LRB60 numeric pixel budget,
        # while Cosmos's processor handles its required 32-pixel divisibility.
        size = dict(getattr(self.processor.image_processor, "size", {}) or {})
        min_pixels = size.get("shortest_edge")
        max_pixels = size.get("longest_edge")
        for message in materialized:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get("type") == "image":
                    if min_pixels is not None:
                        part["min_pixels"] = int(min_pixels)
                    if max_pixels is not None:
                        part["max_pixels"] = int(max_pixels)
        return materialized

    def _label_assistant_actions(
        self,
        input_ids: torch.Tensor,
        messages: List[Dict[str, Any]],
    ) -> torch.Tensor:
        labels = torch.full_like(input_ids, -100)
        spans = openfly_sft._qwen3_vl_assistant_supervision_spans(
            input_ids,
            self._im_start_id,
            self._assistant_id,
            self._im_end_id,
        )
        expected_actions = [
            str(part["text"])
            for message in messages
            if message.get("role") == "assistant"
            for part in message.get("content", [])
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        if len(spans) != len(expected_actions):
            raise ValueError(
                f"Cosmos assistant span mismatch: found {len(spans)}, "
                f"expected {len(expected_actions)}"
            )

        excluded = {
            self._im_start_id,
            self._assistant_id,
            self._newline_id,
            self._think_start_id,
            self._think_end_id,
        }
        ids = input_ids[0]
        for span, expected_action in zip(spans, expected_actions):
            start, end = span
            role_start = start + 1
            role_end = role_start + len(self._assistant_ids)
            if tuple(int(ids[pos].item()) for pos in range(role_start, role_end)) != self._assistant_ids:
                raise ValueError(
                    "Cosmos assistant span has an unexpected role-token sequence: "
                    f"expected={self._assistant_ids}, "
                    f"actual={tuple(int(ids[pos].item()) for pos in range(role_start, role_end))}"
                )
            action_positions: List[int] = []
            for pos in range(role_end, end):
                token_id = int(ids[pos].item())
                if token_id in excluded or token_id == self._im_end_id:
                    continue
                action_positions.append(pos)
            decoded = self.tokenizer.decode(
                [int(ids[p].item()) for p in action_positions],
                skip_special_tokens=True,
            ).strip()
            if decoded != expected_action:
                raise ValueError(
                    "Cosmos action mask did not isolate the expected assistant action: "
                    f"expected={expected_action!r}, decoded={decoded!r}, "
                    f"positions={action_positions}"
                )
            for pos in action_positions:
                labels[0, pos] = ids[pos]
            if self.supervise_eos:
                eos_pos = end - 1
                if int(ids[eos_pos].item()) != self._im_end_id:
                    raise ValueError("Cosmos assistant span does not terminate in <|im_end|>")
                labels[0, eos_pos] = ids[eos_pos]
        return labels

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        per_item: List[Dict[str, torch.Tensor]] = []
        loss_modes: List[str] = []
        for item in batch:
            loss_modes.append(str(item.get("loss_mode", "weighted")))
            messages = self._prepend_system(item["messages"])
            materialized = self._materialize_messages(messages)
            outputs = self.processor.apply_chat_template(
                materialized,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
                padding=False,
                truncation=False,
                return_mm_token_type_ids=True,
            )
            item_out: Dict[str, torch.Tensor] = {}
            for key, value in dict(outputs).items():
                if value is None:
                    continue
                try:
                    item_out[key] = _as_batch_tensor(value, name=key)
                except (TypeError, ValueError, RuntimeError):
                    continue
            input_ids = item_out["input_ids"]
            seq_len = int(input_ids.shape[1])
            if (
                self.max_length > 0
                and seq_len > self.max_length
                and int(os.environ.get("RANK", "0")) == 0
            ):
                print(
                    f"[Cosmos3EdgeTrajectoryCollator] warning: seq_len={seq_len} "
                    f"> max_length hint={self.max_length}; reduce temporal history if OOM",
                    flush=True,
                )
            item_out["labels"] = self._label_assistant_actions(input_ids, messages)
            per_item.append(item_out)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        max_len = max(int(item["input_ids"].shape[1]) for item in per_item)
        sequence_keys = {"input_ids", "attention_mask", "mm_token_type_ids"}
        batched: Dict[str, torch.Tensor] = {}
        keys = set().union(*(item.keys() for item in per_item))
        for key in sorted(keys):
            if key == "labels":
                continue
            columns = []
            for item in per_item:
                tensor = item[key]
                if not isinstance(tensor, torch.Tensor):
                    continue
                if key in sequence_keys and int(tensor.shape[-1]) < max_len:
                    padding = max_len - int(tensor.shape[-1])
                    fill = int(pad_id) if key == "input_ids" else 0
                    tensor = F.pad(tensor, (0, padding), value=fill)
                columns.append(tensor)
            if columns:
                batched[key] = torch.cat(columns, dim=0)

        labels = []
        for item in per_item:
            tensor = item["labels"]
            if int(tensor.shape[1]) < max_len:
                tensor = F.pad(tensor, (0, max_len - int(tensor.shape[1])), value=-100)
            labels.append(tensor)
        batched["labels"] = torch.cat(labels, dim=0)
        batched["loss_mode"] = loss_modes
        return batched


def main() -> None:
    # Keep the entire proven OpenFly training path; replace only its three
    # Qwen-specific integration points.
    openfly_sft.AutoProcessor = Cosmos3EdgeProcessorFactory
    openfly_sft.Qwen3VLForConditionalGeneration = Cosmos3EdgeModelFactory
    openfly_sft.Qwen3VlTrajectoryCollator = Cosmos3EdgeTrajectoryCollator
    openfly_sft._prefetch_hub_repo_serially = _prefetch_cosmos3_edge
    if os.environ.get("COSMOS3_EDGE_DUMP_LOSS_MASK", "").strip():
        os.environ["QWEN3VL_DUMP_LOSS_MASK"] = os.environ[
            "COSMOS3_EDGE_DUMP_LOSS_MASK"
        ]
    openfly_sft.main()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
