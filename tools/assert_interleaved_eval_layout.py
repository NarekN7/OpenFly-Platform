#!/usr/bin/env python3
"""Assert eval interleaved message layout matches nosfx VlnTrajectoryCropDataset._turn_messages."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from qwen3_vl_interleaved_common import (  # noqa: E402
    build_interleaved_messages,
    interleaved_window_lo,
    resolve_prompt_suffix,
)


def _training_turn_messages(
    instruction: str,
    actions: Sequence[int],
    t: int,
    past: int,
) -> List[Dict[str, Any]]:
    """Mirror scripts/qwen3_vl_sft_nosfx.py VlnTrajectoryCropDataset._turn_messages."""
    lo = max(0, t - past)
    messages: List[Dict[str, Any]] = []
    for s in range(lo, t + 1):
        user_content: List[Dict[str, Any]] = [{"type": "image", "image_path": f"frame_{s}.png"}]
        if s == lo:
            user_content.append({"type": "text", "text": instruction})
        messages.append({"role": "user", "content": user_content})
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": str(int(actions[s]))}]}
        )
    return messages


def _strip_image_payload(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content", [])
        new_content: List[Dict[str, Any]] = []
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    new_content.append({"type": "image", "image": "<img>"})
                else:
                    new_content.append(dict(part))
        out.append({"role": msg["role"], "content": new_content})
    return out


def _inference_messages(
    instruction: str,
    actions: Sequence[int],
    t: int,
    past: int,
) -> List[Dict[str, Any]]:
    lo = interleaved_window_lo(t, past)
    dummy_images = [f"img_{i}" for i in range(lo, t + 1)]
    window_past = list(actions[lo:t])
    return build_interleaved_messages(
        "system prompt",
        instruction,
        resolve_prompt_suffix(),
        dummy_images,
        window_past,
    )


def main() -> int:
    instruction = "Fly to the building."
    actions = list(range(20))
    past = 16
    errors: List[str] = []

    for t in [0, 3, 15, 19]:
        train_msgs = _training_turn_messages(instruction, actions, t, past)
        infer_msgs = _inference_messages(instruction, actions, t, past)

        # Training includes final assistant; inference omits it.
        train_infer_prefix = _strip_image_payload(train_msgs[:-1])
        infer_stripped = _strip_image_payload(infer_msgs[1:])  # skip system in infer

        if train_infer_prefix != infer_stripped:
            errors.append(f"t={t}: message layout mismatch")
            errors.append(f"  train={train_infer_prefix}")
            errors.append(f"  infer={infer_stripped}")

    if errors:
        print("FAIL: interleaved layout mismatch", flush=True)
        for line in errors:
            print(line, flush=True)
        return 1

    print(
        "OK: eval interleaved layout matches nosfx _turn_messages (minus final assistant)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
