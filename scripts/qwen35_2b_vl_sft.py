#!/usr/bin/env python3
"""Qwen3.5-2B VLN SFT with the exact OpenFly Qwen LRB training semantics.

The dataset, skill mixing, weighted/last-token loss, Trainer configuration, and
best/last checkpoint callbacks are imported from qwen3_vl_sft. Only the model
class and multimodal collation are replaced for Qwen3.5.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

import qwen3_vl_sft as openfly_sft


QWEN35_2B_REPO = "Qwen/Qwen3.5-2B"
# A 16,255-token outlier (44k-char instruction) asked backward for 15.09 GiB and
# OOMed an H100 that still had a normal ~1,200-token sample's footprint.
# 4,096 keeps that fp32 vocab gradient under ~4 GiB. Measured lengths: 17 frames
# are ~1,200 tokens, a 4,211-char instruction is 2,633, and only a handful of
# instructions in the whole mix are long enough to cross 4,096.
MAX_SAFE_SEQUENCE_LENGTH = 4096
# Long enough for a normal instruction (p99 is under ~1,100 chars) and short
# enough that 17 frames plus this text stay well under MAX_SAFE_SEQUENCE_LENGTH.
_INSTRUCTION_CHAR_CAP = 1500


def _as_batch_tensor(value: Any, *, name: str) -> torch.Tensor:
    """Normalize processor outputs to a batched tensor ``[1, ...]`` or ``[N, ...]``."""
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.ndim == 0:
        raise ValueError(f"Processor returned a scalar for {name}")
    if value.ndim == 1 and name in {"input_ids", "attention_mask", "mm_token_type_ids", "labels"}:
        value = value.unsqueeze(0)
    return value


def _token_ids(tokenizer, text: str) -> tuple[int, ...]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise RuntimeError(f"Expected at least one token for {text!r}")
    return tuple(int(token_id) for token_id in ids)


def _ids_match_at(ids: Sequence[int], pos: int, needle: Sequence[int]) -> bool:
    end = pos + len(needle)
    if end > len(ids):
        return False
    return tuple(int(ids[pos + offset]) for offset in range(len(needle))) == tuple(needle)


class Qwen35ModelFactory:
    """Qwen3-VL factory-compatible loader for Qwen3.5 conditional generation."""

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs: Any,
    ):
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("device_map", None)
        kwargs.pop("torch_dtype", None)
        load_kwargs = dict(kwargs)
        try:
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_name_or_path,
                dtype=torch_dtype,
                **load_kwargs,
            )
        except TypeError:
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                **load_kwargs,
            )
        if any(not parameter.requires_grad for parameter in model.parameters()):
            for parameter in model.parameters():
                parameter.requires_grad = True
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"[qwen35-2b] loaded {model_name_or_path!r}; "
                f"trainable parameters={trainable:,}",
                flush=True,
            )
        return model


class Qwen35TrajectoryCollator:
    """Qwen3.5 processor adapter preserving OpenFly's action-only label contract."""

    def __init__(
        self,
        processor,
        max_length: int,
        system_prompt: str = "",
        supervise_eos: bool = True,
        max_safe_sequence_length: int = MAX_SAFE_SEQUENCE_LENGTH,
    ) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = int(max_length)
        self.max_safe_sequence_length = int(max_safe_sequence_length)
        self.system_prompt = system_prompt.strip()
        self.supervise_eos = bool(supervise_eos)

        def single_token_id(text: str) -> int:
            ids = _token_ids(self.tokenizer, text)
            if len(ids) != 1:
                raise RuntimeError(f"Expected one token for {text!r}, got ids={ids}")
            return ids[0]

        self._im_start_id = single_token_id("<|im_start|>")
        self._im_end_id = single_token_id("<|im_end|>")
        self._assistant_ids = _token_ids(self.tokenizer, "assistant")
        self._assistant_id = self._assistant_ids[0]
        self._newline_id = single_token_id("\n")
        self._think_start_ids = _token_ids(self.tokenizer, "<think>")
        self._think_end_ids = _token_ids(self.tokenizer, "</think>")

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
                f"Qwen3.5 assistant span mismatch: found {len(spans)}, "
                f"expected {len(expected_actions)}"
            )

        ids = input_ids[0]
        id_list = [int(token_id.item()) for token_id in ids]
        for span, expected_action in zip(spans, expected_actions):
            start, end = span
            role_start = start + 1
            role_end = role_start + len(self._assistant_ids)
            if tuple(id_list[role_start:role_end]) != self._assistant_ids:
                raise ValueError(
                    "Qwen3.5 assistant span has an unexpected role-token sequence: "
                    f"expected={self._assistant_ids}, "
                    f"actual={tuple(id_list[role_start:role_end])}"
                )
            action_positions: List[int] = []
            pos = role_end
            while pos < end:
                token_id = id_list[pos]
                if token_id == self._im_end_id:
                    break
                if _ids_match_at(id_list, pos, self._think_start_ids):
                    pos += len(self._think_start_ids)
                    continue
                if _ids_match_at(id_list, pos, self._think_end_ids):
                    pos += len(self._think_end_ids)
                    continue
                if token_id == self._newline_id:
                    pos += 1
                    continue
                action_positions.append(pos)
                pos += 1
            decoded = self.tokenizer.decode(
                [id_list[p] for p in action_positions],
                skip_special_tokens=True,
            ).strip()
            if decoded != expected_action:
                raise ValueError(
                    "Qwen3.5 action mask did not isolate the expected assistant action: "
                    f"expected={expected_action!r}, decoded={decoded!r}, "
                    f"positions={action_positions}"
                )
            for pos in action_positions:
                labels[0, pos] = ids[pos]
            if self.supervise_eos:
                eos_pos = end - 1
                if int(ids[eos_pos].item()) != self._im_end_id:
                    raise ValueError("Qwen3.5 assistant span does not terminate in <|im_end|>")
                labels[0, eos_pos] = ids[eos_pos]
        return labels

    def _tokenize(self, messages: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        outputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
            processor_kwargs={
                "padding": False,
                "truncation": False,
                "return_mm_token_type_ids": True,
            },
        )
        item_out: Dict[str, torch.Tensor] = {}
        for key, value in dict(outputs).items():
            if value is None:
                continue
            try:
                item_out[key] = _as_batch_tensor(value, name=key)
            except (TypeError, ValueError, RuntimeError):
                continue
        return item_out

    def _shorten_first_user_text(
        self, messages: List[Dict[str, Any]], char_limit: int
    ) -> List[Dict[str, Any]]:
        shortened = copy.deepcopy(messages)
        for message in shortened:
            content = message.get("content")
            if message.get("role") != "user" or not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text":
                    continue
                text = str(part.get("text", ""))
                if len(text) > char_limit:
                    part["text"] = text[:char_limit]
            break
        return shortened

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        per_item: List[Dict[str, torch.Tensor]] = []
        loss_modes: List[str] = []
        for item in batch:
            loss_modes.append(str(item.get("loss_mode", "weighted")))
            messages = self._materialize_messages(self._prepend_system(item["messages"]))
            item_out = self._tokenize(messages)
            seq_len = int(item_out["input_ids"].shape[1])
            if self.max_safe_sequence_length > 0 and seq_len > self.max_safe_sequence_length:
                # The frames and action labels stay. Only an extreme instruction
                # can push a 17-frame sample into the 15 GiB backward allocation.
                messages = self._shorten_first_user_text(messages, _INSTRUCTION_CHAR_CAP)
                item_out = self._tokenize(messages)
                new_len = int(item_out["input_ids"].shape[1])
                print(
                    f"[Qwen35TrajectoryCollator] seq_len={seq_len} exceeds "
                    f"{self.max_safe_sequence_length}; shortened the instruction "
                    f"to {_INSTRUCTION_CHAR_CAP} chars and kept the trajectory "
                    f"(seq_len={new_len})",
                    flush=True,
                )
                if new_len > self.max_safe_sequence_length:
                    raise RuntimeError(
                        f"Trajectory is still seq_len={new_len} after shortening the "
                        f"instruction to {_INSTRUCTION_CHAR_CAP} chars "
                        f"(limit {self.max_safe_sequence_length})."
                    )
                seq_len = new_len
            if (
                self.max_length > 0
                and seq_len > self.max_length
                and int(os.environ.get("RANK", "0")) == 0
            ):
                print(
                    f"[Qwen35TrajectoryCollator] warning: seq_len={seq_len} "
                    f"> max_length hint={self.max_length}; reduce temporal history if OOM",
                    flush=True,
                )
            item_out["labels"] = self._label_assistant_actions(item_out["input_ids"], messages)
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
    openfly_sft.Qwen3VLForConditionalGeneration = Qwen35ModelFactory
    openfly_sft.Qwen3VlTrajectoryCollator = Qwen35TrajectoryCollator
    if os.environ.get("QWEN35_DUMP_LOSS_MASK", "").strip():
        os.environ["QWEN3VL_DUMP_LOSS_MASK"] = os.environ["QWEN35_DUMP_LOSS_MASK"]
    openfly_sft.main()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
