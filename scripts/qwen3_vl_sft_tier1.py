#!/usr/bin/env python3
"""
Frozen tier-1 (pre-x9) VLN SFT entrypoint.

Patches action vocabulary 0-9 and the historical system prompt used for 4B_h16_NEW_BEST
training, then delegates to `qwen3_vl_sft.main()` for all trainer/dataset/checkpoint logic.

Use `qwen3_vl_sft.py` for x9 tier-2 training (action 10). Do not merge tier-1 constants back
into that script.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure scripts/ is importable when launched via accelerate from repo root.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import qwen3_vl_sft as sft

VLN_ALLOWED_ACTION_IDS = frozenset({0, 1, 2, 3, 4, 5, 8, 9})

DEFAULT_VLN_SYSTEM_PROMPT = """You are an AI assistant controlling a flying drone. Navigate using the current camera view and the human instruction by replying with exactly one digit from 0 to 9 (no other text). Action meanings:
0. Stop
1. Move forward
2. Turn left (~30°)
3. Turn right (~30°)
4. Move up
5. Move down
6. Move left (strafe)
7. Move right (strafe)
8. Move forward (longer step)
9. Move forward (longest step)
"""

sft.VLN_ALLOWED_ACTION_IDS = VLN_ALLOWED_ACTION_IDS
sft.DEFAULT_VLN_SYSTEM_PROMPT = DEFAULT_VLN_SYSTEM_PROMPT


def main() -> None:
    sft.main()


if __name__ == "__main__":
    main()
