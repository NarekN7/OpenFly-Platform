#!/usr/bin/env python3
"""Qwen3.5-2B overfit-8 check on the production trainer.

The 8 fixed crops (4 general weighted, 2 left and 2 right last-token) come from
the Qwen 3 overfit. Model class, collator, and loss stay the Qwen 3.5 pipeline
in qwen35_2b_vl_sft.py / qwen3_vl_sft.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

import qwen3_vl_sft as openfly_sft
from qwen35_2b_vl_sft import Qwen35ModelFactory, Qwen35TrajectoryCollator


class VlnOverfitMixedDataset(Dataset):
    """Fixed concat of general + left + right crops for overfit verification.

    Epoch length = len(general) + len(left) + len(right). Every sample always
    supervises the last timestep of its (already-cropped) trajectory:
      - general indices: loss_mode=weighted
      - left/right indices: loss_mode=last_token
    No random crop/shift and no skill-pool resampling.
    """

    def __init__(
        self,
        *,
        general_json: str,
        skill_json_left: str,
        skill_json_right: str,
        frames_root: str,
        temporal_history_past: int,
        verify_images_exist: bool = True,
        chat_window_turns: int = 1,
        max_window_sample_attempts: int = 64,
        max_trajectories: Optional[int] = None,
        debug_samples: Optional[int] = None,
        skill_mix_rate: float = 0.0,
        deterministic: bool = True,
    ) -> None:
        del (
            chat_window_turns,
            max_window_sample_attempts,
            max_trajectories,
            debug_samples,
            skill_mix_rate,
            deterministic,
        )
        self.frames_root = Path(frames_root)
        self.temporal_history_past = int(temporal_history_past)
        self.verify_images_exist = bool(verify_images_exist)

        self.general = self._load_trajs(general_json, expected_last=None)
        self.skill_left = self._load_trajs(skill_json_left, expected_last=2)
        self.skill_right = self._load_trajs(skill_json_right, expected_last=3)
        if not self.general:
            raise RuntimeError(f"No general trajectories in {general_json}")
        if not self.skill_left:
            raise RuntimeError(f"No left skill trajectories in {skill_json_left}")
        if not self.skill_right:
            raise RuntimeError(f"No right skill trajectories in {skill_json_right}")

        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[VlnOverfitMixedDataset] general={len(self.general)} "
                f"left={len(self.skill_left)} right={len(self.skill_right)} "
                f"epoch_len={len(self)} (concat-all, always last-t)",
                flush=True,
            )

    @staticmethod
    def _load_trajs(json_path: str, expected_last: Optional[int]) -> List[Dict[str, Any]]:
        path = Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing json: {path}")
        out: List[Dict[str, Any]] = []
        for traj in openfly_sft._iter_trajectories(path):
            actions = traj.get("action", [])
            index_list = traj.get("index_list", [])
            if not actions or not index_list or len(actions) != len(index_list):
                continue
            if expected_last is not None and int(actions[-1]) != expected_last:
                continue
            out.append(traj)
        return out

    def __len__(self) -> int:
        return len(self.general) + len(self.skill_left) + len(self.skill_right)

    def _pack(
        self,
        traj: Dict[str, Any],
        *,
        loss_mode: str,
        role: str,
    ) -> Dict[str, Any]:
        image_path = traj["image_path"]
        instruction = traj["gpt_instruction"]
        actions: List[int] = list(traj["action"])
        index_list: List[str] = [str(x) for x in traj["index_list"]]
        t = len(actions) - 1
        traj_dir = self.frames_root / image_path
        past = self.temporal_history_past
        lo = max(0, t - past)
        messages: List[Dict[str, Any]] = []
        for s in range(lo, t + 1):
            path = str(traj_dir / f"{index_list[s]}.png")
            if self.verify_images_exist and not Path(path).is_file():
                raise FileNotFoundError(f"Missing frame under {traj_dir} for timestep {s}")
            user_content: List[Dict[str, Any]] = [{"type": "image", "image_path": path}]
            if s == lo:
                user_content.append({"type": "text", "text": instruction})
            messages.append({"role": "user", "content": user_content})
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": str(int(actions[s]))}]}
            )
        return {
            "messages": messages,
            "traj_meta": {
                "image_path": image_path,
                "start_idx": t,
                "crop_length": 1,
                "t": t,
                "role": role,
            },
            "loss_mode": loss_mode,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        n_g = len(self.general)
        n_l = len(self.skill_left)
        if idx < n_g:
            return self._pack(self.general[idx], loss_mode="weighted", role="general")
        skill_idx = idx - n_g
        if skill_idx < n_l:
            return self._pack(self.skill_left[skill_idx], loss_mode="last_token", role="left")
        return self._pack(self.skill_right[skill_idx - n_l], loss_mode="last_token", role="right")


def main() -> None:
    openfly_sft.VlnOverfitMixedDataset = VlnOverfitMixedDataset
    openfly_sft.Qwen3VLForConditionalGeneration = Qwen35ModelFactory
    openfly_sft.Qwen3VlTrajectoryCollator = Qwen35TrajectoryCollator
    if os.environ.get("QWEN35_DUMP_LOSS_MASK", "").strip():
        os.environ["QWEN3VL_DUMP_LOSS_MASK"] = os.environ["QWEN35_DUMP_LOSS_MASK"]
    if "--overfit_concat" not in sys.argv:
        sys.argv.append("--overfit_concat")
    openfly_sft.main()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
