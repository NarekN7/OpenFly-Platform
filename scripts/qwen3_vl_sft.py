import argparse
import glob
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# Avoid the `checkpoint-*` prefix for best-val weights: HF epoch rotation globs that pattern.
BEST_EVAL_DIRNAME = "eval-best"

from filelock import FileLock
from peft import LoraConfig, get_peft_model

# Matches discrete IDs in `train/eval.py` `convert_to_action_id` / `getPoseAfterMakeAction`.
VLN_ALLOWED_ACTION_IDS = frozenset({0, 1, 2, 3, 4, 5, 8, 9, 10})

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


def _hf_hub_cache_root() -> Path:
    hub = os.environ.get("HF_HUB_CACHE", "").strip()
    if hub:
        return Path(hub)
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    return Path(hf_home) / "hub"


def _prefetch_hub_repo_serially(repo_id: str) -> None:
    """
    Serialize Hugging Face Hub snapshot downloads across processes.

    With `accelerate launch` + many ranks, concurrent `from_pretrained` on the same sharded repo
    can race on the disk cache and raise OSError (e.g. missing model-00002-of-00002.safetensors).
    """
    rid = repo_id.strip()
    if not rid:
        return
    exp = Path(rid).expanduser()
    if exp.is_dir() or exp.is_file():
        return
    if rid.startswith((".", "/", "~", "http://", "https://")):
        return
    if "/" not in rid:
        return

    cache_root = _hf_hub_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    safe = rid.replace("/", "__").replace(":", "_")
    lock_path = cache_root / f".openfly_prefetch_{safe}.lock"

    from huggingface_hub import snapshot_download

    lock = FileLock(str(lock_path), timeout=7200)
    with lock:
        snapshot_path = snapshot_download(repo_id=rid)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"[hub] snapshot ready for {rid!r} at {snapshot_path} (serialized prefetch)")


def _to_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_str in ("fp16", "float16"):
        return torch.float16
    if dtype_str in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def _iter_trajectories(json_path_p: Path) -> List[Dict[str, Any]]:
    raw = json_path_p.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {json_path_p}")
        return [x for x in data if isinstance(x, dict)]

    trajs: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.endswith(","):
            line = line[:-1]
        trajs.append(json.loads(line))
    return trajs


def _poisson_shift_weights(max_shift: int, lam: float = 1.0) -> np.ndarray:
    """PMF on {1, ..., max_shift} for Poisson(λ), renormalized (excluding 0)."""
    if max_shift <= 0:
        return np.array([])
    xs = np.arange(1, max_shift + 1, dtype=np.float64)
    probs = np.array([(lam**x) * math.exp(-lam) / math.gamma(x + 1) for x in xs], dtype=np.float64)
    probs = probs / probs.sum()
    return probs


def _window_frame_paths_exist(
    traj_dir: Path,
    index_list: Sequence[str],
    start_idx: int,
    crop_length: int,
    temporal_history_past: int,
) -> bool:
    """True iff every PNG path needed for timesteps start_idx..start_idx+crop_length-1 exists on disk."""
    for t in range(start_idx, start_idx + crop_length):
        paths = _frame_paths_for_timestep(
            traj_dir, index_list, t, temporal_history_past, verify=False
        )
        for p in paths:
            if not Path(p).is_file():
                return False
    return True


def _frame_paths_for_timestep(
    traj_dir: Path,
    index_list: Sequence[str],
    timestep: int,
    temporal_history_past: int,
    verify: bool,
) -> List[str]:
    """
    Up to (temporal_history_past + 1) PNG paths aligned with trajectory timestep `t`.

    `index_list[t]` is an opaque frame key (often non-consecutive, e.g. 2,5,8); chronology follows
    **list order**, not numeric order. Files are `{key}.png` under `traj_dir`. Only keys present in
    `index_list` are loaded; extra PNGs in the folder are ignored (supports regrouped trajectories).
    Left-pad by repeating the earliest key in the sliced window until length is temporal_history_past+1.
    """
    past = temporal_history_past
    lo = max(0, timestep - past)
    idxs = list(index_list[lo : timestep + 1])
    target = past + 1
    while len(idxs) < target:
        idxs.insert(0, idxs[0])
    paths = [str(traj_dir / f"{idx}.png") for idx in idxs]
    if verify and not all(Path(p).exists() for p in paths):
        raise FileNotFoundError(f"Missing frame under {traj_dir} for timestep {timestep}")
    return paths


def _deep_copy_messages_replace_images(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    """Clone chat messages, load PNG paths into PIL RGB, return (messages_for_template, flat_image_list)."""
    flat: List[Image.Image] = []
    out_msgs: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        new_content: List[Dict[str, Any]] = []
        if isinstance(content, str):
            new_content = [{"type": "text", "text": content}]
        else:
            for part in content:
                if part.get("type") == "image":
                    p = part.get("image_path")
                    if not p:
                        raise ValueError("image_path missing for image part")
                    im = Image.open(p).convert("RGB")
                    flat.append(im)
                    new_content.append({"type": "image", "image": im})
                else:
                    new_content.append(dict(part))
        out_msgs.append({"role": role, "content": new_content})
    return out_msgs, flat


def _qwen3_vl_assistant_supervision_spans(
    input_ids: torch.Tensor,
    im_start_id: int,
    assistant_id: int,
    im_end_id: int,
) -> List[Tuple[int, int]]:
    """
    Token spans [start, end) for each assistant turn in a Qwen3 chat sequence, aligned with
    <|im_start|>assistant ... <|redacted_im_end|> blocks in `input_ids` (batch dim 1).
    """
    ids = input_ids[0].tolist()
    n = len(ids)
    spans: List[Tuple[int, int]] = []
    i = 0
    while i < n - 1:
        if ids[i] == im_start_id and ids[i + 1] == assistant_id:
            j = i + 2
            while j < n and ids[j] != im_end_id:
                j += 1
            if j >= n:
                break
            spans.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return spans


class VlnTrajectoryCropDataset(Dataset):
    """
    One row per trajectory: `__getitem__(i)` always uses `self.trajectories[i]` (one representative
    per trajectory index per logical pass through the dataset).

    Each sample is a single-turn chat:
      optional system (handled by collator)
      user(images + instruction + suffix) -> assistant(action_id)

    We sample a window anchored at the trajectory suffix (with random left shift), then sample a
    single timestep `t` from within that window. This keeps exactly one sample per trajectory index
    per epoch while matching eval prompt formatting.

    Window: crop_length = min(max_crop_length, len(actions)), anchored at trajectory suffix, then random left shift
    (uniform or Poisson PMF) with 10% probability of zero shift when shifting is optional.

    If a sampled window references a missing PNG, the same trajectory is **resampled** (new random
    shift) up to `max_window_sample_attempts` times — we do not substitute a different trajectory index.
    """

    def __init__(
        self,
        json_path: str,
        frames_root: str,
        max_crop_length: int = 17,
        chat_window_turns: int = 1,
        crop_shift_sampling: str = "uniform",
        temporal_history_past: int = 16,
        no_shift_probability: float = 0.1,
        prompt_suffix: str = "\nNext action id (0-10): ",
        verify_images_exist: bool = False,
        max_window_sample_attempts: int = 256,
        max_trajectories: Optional[int] = None,
        debug_samples: Optional[int] = None,
        deterministic: bool = False,
    ) -> None:
        if max_crop_length < 1:
            raise ValueError("max_crop_length must be >= 1")
        if chat_window_turns != 1:
            raise ValueError("This dataset currently supports only chat_window_turns=1 (single-turn samples).")
        if temporal_history_past < 0:
            raise ValueError("temporal_history_past must be >= 0")
        if crop_shift_sampling not in ("uniform", "pmf"):
            raise ValueError("crop_shift_sampling must be 'uniform' or 'pmf'")
        if max_window_sample_attempts < 1:
            raise ValueError("max_window_sample_attempts must be >= 1")

        self.frames_root = Path(frames_root)
        json_path_p = Path(json_path)
        if not json_path_p.exists():
            raise FileNotFoundError(f"Missing json: {json_path_p}")
        if not self.frames_root.exists():
            raise FileNotFoundError(f"Missing frames root: {self.frames_root}")

        self.max_crop_length = max_crop_length
        self.chat_window_turns = chat_window_turns
        self.crop_shift_sampling = crop_shift_sampling
        self.temporal_history_past = temporal_history_past
        self.no_shift_probability = no_shift_probability
        self.prompt_suffix = prompt_suffix
        self.verify_images_exist = verify_images_exist
        self.max_window_sample_attempts = max_window_sample_attempts
        self.deterministic = deterministic

        self.trajectories: List[Dict[str, Any]] = []
        for traj in _iter_trajectories(json_path_p):
            actions = traj.get("action", [])
            index_list = traj.get("index_list", [])
            if not actions or not index_list or len(actions) != len(index_list):
                continue
            self.trajectories.append(traj)
            if max_trajectories is not None and len(self.trajectories) >= max_trajectories:
                break
            if debug_samples is not None and len(self.trajectories) >= debug_samples:
                break

        if len(self.trajectories) == 0:
            raise RuntimeError("No valid trajectories in dataset (check json and action/index_list alignment).")

        bad_actions: set[int] = set()
        for traj in self.trajectories:
            for a in traj["action"]:
                if a not in VLN_ALLOWED_ACTION_IDS:
                    bad_actions.add(int(a))
        if bad_actions and int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[VlnTrajectoryCropDataset] warning: unexpected action ids {sorted(bad_actions)} "
                f"(expected {sorted(VLN_ALLOWED_ACTION_IDS)}) in {json_path_p}"
            )

    def __len__(self) -> int:
        return len(self.trajectories)

    def _sample_start_shift(self, start_idx: int) -> int:
        if start_idx <= 0:
            return 0
        if random.random() < self.no_shift_probability:
            return 0
        choices = np.arange(1, start_idx + 1)
        if self.crop_shift_sampling == "uniform":
            return int(np.random.choice(choices))
        probs = _poisson_shift_weights(start_idx, lam=1.0)
        return int(np.random.choice(choices, p=probs))

    def _turn_messages(
        self,
        instruction: str,
        action_t: int,
        index_list: List[str],
        traj_dir: Path,
        t: int,
    ) -> List[Dict[str, Any]]:
        paths = _frame_paths_for_timestep(
            traj_dir,
            index_list,
            t,
            self.temporal_history_past,
            self.verify_images_exist,
        )
        user_content: List[Dict[str, Any]] = [{"type": "image", "image_path": p} for p in paths]
        user_content.append({"type": "text", "text": f"{instruction}{self.prompt_suffix}"})
        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": str(action_t)}]},
        ]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        traj = self.trajectories[idx]
        image_path = traj["image_path"]
        instruction = traj["gpt_instruction"]
        actions: List[int] = list(traj["action"])
        index_list: List[str] = list(traj["index_list"])
        traj_dir = self.frames_root / image_path

        base_crop_length = min(self.max_crop_length, len(actions))

        def _turn_paths_exist(t: int) -> bool:
            paths = _frame_paths_for_timestep(
                traj_dir, index_list, t, self.temporal_history_past, verify=False
            )
            return all(Path(p).is_file() for p in paths)

        def pack(start_idx: int, crop_length: int, t: int) -> Dict[str, Any]:
            return {
                "messages": self._turn_messages(instruction, actions[t], index_list, traj_dir, t),
                "traj_meta": {
                    "image_path": image_path,
                    "start_idx": start_idx,
                    "crop_length": crop_length,
                    "t": t,
                },
            }

        def _try_deterministic_single_turn(crop_length: int) -> Optional[Tuple[int, int, int]]:
            # Suffix-only windows fail when index_list outlasts PNG exports (common on val).
            # Use the latest timestep anywhere in the trajectory that has frames on disk.
            for t in range(len(actions) - 1, -1, -1):
                if not _turn_paths_exist(t):
                    continue
                cl = min(crop_length, t + 1)
                start_idx = t + 1 - cl
                return (start_idx, cl, t)
            return None

        def _try_sample_single_turn(crop_length: int) -> Optional[Tuple[int, int, int]]:
            suffix_start_local = len(actions) - crop_length
            if suffix_start_local <= 0:
                start_idx = 0
                # Sample t inside [0..crop_length-1]
                for _ in range(self.max_window_sample_attempts):
                    t = int(np.random.randint(start_idx, start_idx + crop_length))
                    if _turn_paths_exist(t):
                        return (start_idx, crop_length, t)
                return None

            for _ in range(self.max_window_sample_attempts):
                shift = self._sample_start_shift(suffix_start_local)
                start_idx = suffix_start_local - shift
                t = int(np.random.randint(start_idx, start_idx + crop_length))
                if _turn_paths_exist(t):
                    return (start_idx, crop_length, t)
            return None

        try_single_turn = (
            _try_deterministic_single_turn if self.deterministic else _try_sample_single_turn
        )

        # Preferred: requested crop length; stochastic train samples random shift/t, val uses suffix/final t.
        sampled = try_single_turn(base_crop_length)
        if sampled is not None:
            start_idx, crop_length, t = sampled
            return pack(start_idx, crop_length, t)

        # Fallback: shorten crop (stochastic) or scan latest valid t (deterministic).
        for crop_length in range(base_crop_length - 1, 0, -1):
            sampled = try_single_turn(crop_length)
            if sampled is not None:
                start_idx, crop_length, t = sampled
                return pack(start_idx, crop_length, t)

        if not self.deterministic:
            sampled = _try_deterministic_single_turn(base_crop_length)
            if sampled is not None:
                start_idx, crop_length, t = sampled
                return pack(start_idx, crop_length, t)

        raise RuntimeError(
            f"Trajectory idx={idx} image_path={image_path!r}: no valid timestep found on this trajectory "
            f"for any crop_length in [1..{base_crop_length}] under {traj_dir}. "
            "Repair JSON/index_list vs disk exports."
        )


class VlnMixedGeneralSkillDataset(Dataset):
    """Concat general trajectory-crop samples with balanced left/right skill samples.

    Length is deterministic: len(general) + 2 * int(len(general) * skill_mix_rate / 2).

    - General samples: existing `VlnTrajectoryCropDataset` sampling + weighted loss.
    - Skill samples: equal left/right counts; force last timestep and last-token-only loss.
      Epoch skill layout: first R_each from left pool, next R_each from right pool.
    """

    @staticmethod
    def _load_skill_pool(skill_json: str, expected_last: int) -> List[Dict[str, Any]]:
        skill_path = Path(skill_json)
        if not skill_path.is_file():
            raise FileNotFoundError(f"Missing skill_json: {skill_path}")
        skill: List[Dict[str, Any]] = []
        for traj in _iter_trajectories(skill_path):
            actions = traj.get("action", [])
            index_list = traj.get("index_list", [])
            if not actions or not index_list or len(actions) != len(index_list):
                continue
            if int(actions[-1]) != expected_last:
                continue
            skill.append(traj)
        if not skill:
            raise RuntimeError(f"No valid skill trajectories with last action {expected_last} in {skill_path}")
        return skill

    def __init__(
        self,
        *,
        general_json: str,
        skill_json_left: str,
        skill_json_right: str,
        skill_mix_rate: float,
        frames_root: str,
        max_crop_length: int,
        chat_window_turns: int,
        crop_shift_sampling: str,
        temporal_history_past: int,
        no_shift_probability: float,
        prompt_suffix: str = "\nNext action id (0-10): ",
        verify_images_exist: bool,
        max_window_sample_attempts: int,
        max_trajectories: Optional[int],
        debug_samples: Optional[int],
        deterministic: bool,
    ) -> None:
        if deterministic:
            raise ValueError("VlnMixedGeneralSkillDataset is intended for training only (deterministic=False).")
        if skill_mix_rate <= 0:
            raise ValueError("skill_mix_rate must be > 0 for mixed dataset")

        self.general = VlnTrajectoryCropDataset(
            json_path=general_json,
            frames_root=frames_root,
            max_crop_length=max_crop_length,
            chat_window_turns=chat_window_turns,
            crop_shift_sampling=crop_shift_sampling,
            temporal_history_past=temporal_history_past,
            no_shift_probability=no_shift_probability,
            prompt_suffix=prompt_suffix,
            verify_images_exist=verify_images_exist,
            max_window_sample_attempts=max_window_sample_attempts,
            max_trajectories=max_trajectories,
            debug_samples=debug_samples,
            deterministic=False,
        )

        self.skill_left = self._load_skill_pool(skill_json_left, expected_last=2)
        self.skill_right = self._load_skill_pool(skill_json_right, expected_last=3)
        self.skill_mix_rate = float(skill_mix_rate)
        self.R_each = int(len(self.general) * self.skill_mix_rate / 2)
        self.R = 2 * self.R_each
        if self.R_each <= 0:
            raise RuntimeError(
                f"skill_mix_rate too small: len(general)={len(self.general)} rate={self.skill_mix_rate} "
                f"-> R_each={self.R_each}"
            )
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[VlnMixedGeneralSkillDataset] general={len(self.general)} "
                f"left_pool={len(self.skill_left)} right_pool={len(self.skill_right)} "
                f"rate={self.skill_mix_rate} R_each={self.R_each} extra_per_epoch={self.R} "
                f"epoch_len={len(self)}",
                flush=True,
            )

    def __len__(self) -> int:
        return len(self.general) + self.R

    def _skill_item(self, traj: Dict[str, Any], skill_name: str) -> Dict[str, Any]:
        image_path = traj["image_path"]
        instruction = traj["gpt_instruction"]
        actions: List[int] = list(traj["action"])
        index_list: List[str] = list(traj["index_list"])
        t = len(actions) - 1
        traj_dir = self.general.frames_root / image_path
        messages = self.general._turn_messages(instruction, int(actions[t]), index_list, traj_dir, t)
        return {
            "messages": messages,
            "traj_meta": {
                "image_path": image_path,
                "start_idx": t,
                "crop_length": 1,
                "t": t,
                "skill": skill_name,
            },
            "loss_mode": "last_token",
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < len(self.general):
            out = self.general[idx]
            out["loss_mode"] = "weighted"
            return out

        skill_idx = idx - len(self.general)
        if skill_idx < self.R_each:
            traj = self.skill_left[int(np.random.randint(0, len(self.skill_left)))]
            return self._skill_item(traj, "left")
        traj = self.skill_right[int(np.random.randint(0, len(self.skill_right)))]
        return self._skill_item(traj, "right")


# Qwen3 chat end-of-turn token is a single id (151645 on Qwen3-VL Instruct); the literal is NOT the
# long "<|redacted_im_end|>" spelling — avoid typos by constructing from the known bytes.
_QWEN3VL_IM_END_LITERAL = "".join(map(chr, (60, 124, 105, 109, 95, 101, 110, 100, 124, 62)))


class Qwen3VlTrajectoryCollator:
    """
    Multi-turn Qwen3-VL collator: one processor() pass per item, then label assistant spans by
    scanning input_ids for <|im_start|>assistant ... <|redacted_im_end|> (fast; avoids dozens of
    extra processor calls per sample that kept GPU idle).

    Multimodal sequences do not use tokenizer truncation: Qwen3-VL raises if truncation splits
    text so image token counts diverge from input_ids. Use shorter crops / fewer frames if you OOM.
    """

    def __init__(self, processor, max_length: int, system_prompt: str = "") -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length  # soft budget: warn if exceeded; no truncation in processor
        self.system_prompt = system_prompt.strip()
        tok = self.tokenizer

        def _single_token_id(text: str) -> int:
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"Expected a single token for {text!r}, got ids={ids}")
            return int(ids[0])

        self._im_start_id = _single_token_id("<|im_start|>")
        self._im_end_id = _single_token_id(_QWEN3VL_IM_END_LITERAL)
        self._assistant_id = _single_token_id("assistant")

    def _prepend_system(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.system_prompt:
            return list(messages)
        sys_msg = {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
        return [sys_msg] + list(messages)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        per_item: List[Dict[str, torch.Tensor]] = []
        loss_modes: List[str] = []
        for item in batch:
            loss_modes.append(str(item.get("loss_mode", "weighted")))
            messages = self._prepend_system(item["messages"])
            msgs_material, all_images = _deep_copy_messages_replace_images(messages)

            full_text = self.processor.apply_chat_template(
                msgs_material, tokenize=False, add_generation_prompt=False
            )
            full_out = self.processor(
                text=full_text,
                images=all_images,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )
            input_ids = full_out["input_ids"]
            seq_len = int(input_ids.shape[1])
            if self.max_length > 0 and seq_len > self.max_length and int(os.environ.get("RANK", "0")) == 0:
                print(
                    f"[Qwen3VlTrajectoryCollator] warning: seq_len={seq_len} > max_length hint={self.max_length} "
                    "(truncation disabled for Qwen3-VL); reduce --max_crop_length or --temporal_history_past if you OOM."
                )
            labels = torch.full_like(input_ids, -100)

            spans = _qwen3_vl_assistant_supervision_spans(
                input_ids,
                self._im_start_id,
                self._assistant_id,
                self._im_end_id,
            )
            n_asst = sum(1 for m in messages if m.get("role") == "assistant")
            if len(spans) != n_asst:
                raise ValueError(
                    f"Assistant span count mismatch: found {len(spans)} token blocks, "
                    f"expected {n_asst} from messages (template/tokenizer changed?)."
                )
            for start, end in spans:
                end = min(end, int(input_ids.shape[1]))
                start = min(start, end)
                if start < end:
                    labels[:, start:end] = input_ids[:, start:end]

            full_out["labels"] = labels
            per_item.append(full_out)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        max_len = max(int(x["input_ids"].shape[1]) for x in per_item)
        batched: Dict[str, torch.Tensor] = {}
        keys = set().union(*(x.keys() for x in per_item))
        for k in sorted(keys):
            if k == "labels":
                continue
            cols = []
            for x in per_item:
                t = x[k]
                if k in ("input_ids", "attention_mask", "mm_token_type_ids"):
                    pad_val = 0 if k != "input_ids" else pad_id
                    if int(t.shape[1]) < max_len:
                        pad_n = max_len - int(t.shape[1])
                        t = F.pad(t, (0, pad_n), value=pad_val)
                cols.append(t)
            batched[k] = torch.cat(cols, dim=0)

        label_cols = []
        for x in per_item:
            t = x["labels"]
            if int(t.shape[1]) < max_len:
                pad_n = max_len - int(t.shape[1])
                t = F.pad(t, (0, pad_n), value=-100)
            label_cols.append(t)
        batched["labels"] = torch.cat(label_cols, dim=0)
        batched["loss_mode"] = loss_modes
        return batched


class WeightedTrainer(Trainer):
    """Causal LM CE with optional linear weights k/N over supervised (label != -100) positions."""

    def __init__(self, *args, loss_type: str = "standard", im_end_id: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if loss_type not in ("standard", "weighted"):
            raise ValueError("loss_type must be 'standard' or 'weighted'")
        self.loss_type = loss_type
        # Used by last_token mode to skip chat end marker and hit the action digit.
        self.im_end_id = im_end_id

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        loss_modes = inputs.pop("loss_mode", None)
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits

        if labels is None:
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        labels_pad = F.pad(labels, (0, 1), value=-100)
        labels_shift = labels_pad[..., 1:].contiguous()
        logits_shift = logits[..., :-1, :].contiguous()
        # Some multimodal batches can produce a one-token drift between labels and logits.
        # Align both sides to the same time dimension before CE.
        T = min(int(logits_shift.shape[1]), int(labels_shift.shape[1]))
        logits_shift = logits_shift[:, :T, :].contiguous()
        labels_shift = labels_shift[:, :T].contiguous()
        B, _, V = logits_shift.shape

        per_token_loss = F.cross_entropy(
            logits_shift.reshape(-1, V),
            labels_shift.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(B, T)

        mask = labels_shift != -100
        if model.training and loss_modes is not None:
            weights = torch.zeros_like(per_token_loss)
            for b in range(B):
                mode = str(loss_modes[b]) if b < len(loss_modes) else "weighted"
                if mode == "last_token":
                    # Supervise only the last teacher action content token — not <|im_end|>.
                    # Label spans are [im_start, assistant, \\n, action..., im_end]; naive "last
                    # supervised" would train predicting im_end and never the action digit.
                    sup_idx = mask[b].nonzero(as_tuple=False)
                    chosen = None
                    if sup_idx.numel() > 0:
                        for pos in reversed(sup_idx[:, 0].tolist()):
                            pos_i = int(pos)
                            tgt = int(labels_shift[b, pos_i].item())
                            if self.im_end_id is not None and tgt == int(self.im_end_id):
                                continue
                            chosen = pos_i
                            break
                    if chosen is not None:
                        weights[b, chosen] = 1.0
                elif self.loss_type == "weighted":
                    positions = mask[b].cumsum(dim=-1) * mask[b]
                    n_sup = mask[b].sum().clamp_min(1).float()
                    weights[b] = positions.float() / n_sup
                else:
                    weights[b] = mask[b].float()
            loss = (per_token_loss * weights).sum() / weights.sum().clamp_min(1)
        elif self.loss_type == "weighted" and model.training:
            positions = mask.cumsum(dim=-1) * mask
            n_sup = mask.sum(dim=-1, keepdim=True).clamp_min(1)
            weights = positions.float() / n_sup.float()
            loss = (per_token_loss * weights).sum() / weights.sum().clamp_min(1)
        else:
            loss = (per_token_loss * mask.float()).sum() / mask.sum().clamp_min(1)

        return (loss, outputs) if return_outputs else loss


def _latest_hf_checkpoint_dir(output_dir: str) -> Optional[str]:
    """Return path to the newest checkpoint-* under output_dir by trailing step number, or None."""
    pattern = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}-*")
    candidates = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if not candidates:
        return None

    def step_key(path: str) -> int:
        base = os.path.basename(path)
        try:
            return int(base.split("-", 1)[1])
        except (IndexError, ValueError):
            return -1

    candidates.sort(key=step_key)
    return candidates[-1]


class UnifiedLossTensorBoardCallback(TrainerCallback):
    """
    Log train + validation CE with SummaryWriter.add_scalars('loss', {...}) so TensorBoard
    shows one chart with two series. Also writes slurm_logs/<run>_loss_chart.html on train end.
    """

    def __init__(self) -> None:
        self._writer = None
        self._train_points: List[Tuple[int, float]] = []
        self._val_points: List[Tuple[int, float]] = []

    @staticmethod
    def _tensorboard_enabled(args) -> bool:
        report = getattr(args, "report_to", None) or []
        if isinstance(report, str):
            report = [x.strip() for x in report.split(",") if x.strip()]
        return "tensorboard" in report

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or not self._tensorboard_enabled(args):
            return control
        log_dir = args.logging_dir or os.path.join(args.output_dir, "runs")
        os.makedirs(log_dir, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            return control
        self._writer = SummaryWriter(log_dir=log_dir)
        self._writer.add_custom_scalars_multilinechart(
            ["loss/train", "loss/validation"],
            category="loss",
            title="train + validation CE loss",
        )
        self._train_points = []
        self._val_points = []
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._writer is None or logs is None or not state.is_world_process_zero:
            return control
        step = int(state.global_step)
        series: Dict[str, float] = {}
        if "loss" in logs:
            v = float(logs["loss"])
            series["train"] = v
            self._train_points.append((step, v))
        if "eval_loss" in logs:
            v = float(logs["eval_loss"])
            series["validation"] = v
            self._val_points.append((step, v))
        if series:
            self._writer.add_scalars("loss", series, step)
            self._writer.flush()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self._train_points:
            try:
                import importlib.util

                script = Path(__file__).resolve().parent / "export_loss_chart_html.py"
                spec = importlib.util.spec_from_file_location("export_loss_chart_html", script)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    slug = Path(args.output_dir).name
                    out = Path("/home/nnurijanyan/OpenFly-Platform/slurm_logs") / f"{slug}_loss_chart.html"
                    mod.write_loss_chart_html(self._train_points, self._val_points, out, title=slug)
                    print(f"[UnifiedLossTensorBoard] Wrote combined chart: {out}")
            except Exception as exc:
                print(f"[UnifiedLossTensorBoard] Could not write HTML chart: {exc}")
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return control


class CudaEmptyCacheCallback(TrainerCallback):
    """Free cached GPU allocations around eval and checkpoint I/O (reduces fragmentation OOM)."""

    def on_evaluate(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return control

    def on_evaluate_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return control


class BestAndLastCheckpointCallback(TrainerCallback):
    """
    Maintains:
      - eval-best/: model+processor when eval_loss improves (rank 0), or train loss if enabled.
      - checkpoint-last/: full copy of the latest HF epoch checkpoint (model+optimizer+scheduler+trainer state).

    Requires TrainingArguments with save_strategy=epoch and save_total_limit=1 so each epoch produces
    one canonical checkpoint-* folder to mirror.
    """

    def __init__(
        self,
        processor,
        trainer_holder: List[Any],
        *,
        save_on_train_loss: bool = False,
    ) -> None:
        self.processor = processor
        self.trainer_holder = trainer_holder
        self.save_on_train_loss = save_on_train_loss
        self.best_loss = float("inf")
        self.best_step: Optional[int] = None

    def _best_eval_dir(self, output_dir: str) -> str:
        return os.path.join(output_dir, BEST_EVAL_DIRNAME)

    def _verify_best_saved(self, best_dir: str) -> bool:
        weights = os.path.join(best_dir, "model.safetensors")
        if os.path.isfile(weights) and os.path.getsize(weights) > 0:
            return True
        legacy = os.path.join(best_dir, "pytorch_model.bin")
        return os.path.isfile(legacy) and os.path.getsize(legacy) > 0

    def _maybe_save_best(self, args, state, loss: float, *, metric: str) -> None:
        if loss >= self.best_loss:
            return
        if not state.is_world_process_zero:
            return
        trainer = self.trainer_holder[0]
        if trainer is None:
            return
        self.best_loss = loss
        self.best_step = int(state.global_step)
        best_dir = self._best_eval_dir(args.output_dir)
        tmp_dir = f"{best_dir}.tmp-{state.global_step}"
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)
        trainer.save_model(tmp_dir)
        if self.processor is not None:
            self.processor.save_pretrained(tmp_dir)
        if not self._verify_best_saved(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(
                f"[BestAndLastCheckpoint] ERROR: best save missing weights at step {state.global_step}",
                flush=True,
            )
            return
        meta = {
            "metric": metric,
            "loss": loss,
            "global_step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
        }
        with open(os.path.join(tmp_dir, "best_eval_metric.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if os.path.isdir(best_dir):
            shutil.rmtree(best_dir, ignore_errors=True)
        os.replace(tmp_dir, best_dir)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not self._verify_best_saved(best_dir):
            print(
                f"[BestAndLastCheckpoint] ERROR: eval-best not on disk after save "
                f"(step {state.global_step}, {best_dir})",
                flush=True,
            )
            return
        print(
            f"[BestAndLastCheckpoint] New best {metric}={loss:.6f} at step {state.global_step} → {best_dir}",
            flush=True,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Fallback for older Trainer log paths (eval metrics in logs).
        if logs is None:
            return control
        if "eval_loss" in logs:
            self._maybe_save_best(args, state, float(logs["eval_loss"]), metric="eval_loss")
        elif self.save_on_train_loss and "loss" in logs:
            self._maybe_save_best(args, state, float(logs["loss"]), metric="train_loss")
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Primary path: save eval-best on lowest eval_loss (DDP-safe).
        if metrics is not None and "eval_loss" in metrics:
            self._maybe_save_best(args, state, float(metrics["eval_loss"]), metric="eval_loss")
        return control

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        src = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        if not os.path.isdir(src):
            return control
        dst = os.path.join(args.output_dir, "checkpoint-last")
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        src = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        if not os.path.isdir(src):
            src = _latest_hf_checkpoint_dir(args.output_dir) or ""
        if src and os.path.isdir(src):
            dst = os.path.join(args.output_dir, "checkpoint-last")
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        best_dir = self._best_eval_dir(args.output_dir)
        meta_path = os.path.join(best_dir, "best_eval_metric.json")
        if self._verify_best_saved(best_dir):
            if os.path.isfile(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                print(
                    f"[BestAndLastCheckpoint] eval-best OK at train end: "
                    f"step={meta.get('global_step')} {meta.get('metric')}={meta.get('loss')}",
                    flush=True,
                )
            else:
                print(f"[BestAndLastCheckpoint] eval-best OK at train end: {best_dir}", flush=True)
        else:
            print(
                f"[BestAndLastCheckpoint] ERROR: eval-best missing at train end "
                f"(tracked best_step={self.best_step}, best_loss={self.best_loss})",
                flush=True,
            )
        return control


def main() -> None:
    # Reduces fragmentation OOM risk with very long multimodal sequences (see PyTorch CUDA mem docs).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--train_json", type=str, default="/home/nnurijanyan/OpenFly-Platform/data_curated/train_curated.json")
    parser.add_argument(
        "--eval_json",
        type=str,
        default="/home/nnurijanyan/OpenFly-Platform/data_curated/validation_curated.json",
        help="Validation trajectory JSON (used when --do_eval).",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        default=None,
        help="Run validation during training. Default: on if --eval_json exists.",
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="Disable validation even if --eval_json exists.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N optimizer steps (requires --do_eval).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Per-device batch size for evaluation.",
    )
    parser.add_argument(
        "--eval_accumulation_steps",
        type=int,
        default=4,
        help="Accumulate eval forward passes before reducing metrics (lowers peak GPU memory).",
    )
    parser.add_argument("--frames_root", type=str, default="/mnt/weka/nnurijanyan/data/vln/train_curated")
    parser.add_argument("--output_dir", type=str, default="/mnt/weka/nnurijanyan/checkpoints/qwen3-vl-2b-vln-simple")
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=None,
        help="Qwen image_processor min pixel budget (size.shortest_edge). Example: 28*28.",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=None,
        help="Qwen image_processor max pixel budget (size.longest_edge). Example: 320*180.",
    )

    parser.add_argument(
        "--chat_window_turns",
        type=int,
        default=1,
        help="Number of (user->assistant) turns per sample. Currently only 1 is supported (single-turn).",
    )
    parser.add_argument(
        "--max_crop_length",
        type=int,
        default=17,
        help="Max number of (user->assistant) turns per trajectory sample (suffix window, then random left shift).",
    )
    parser.add_argument(
        "--crop_shift_sampling",
        type=str,
        default="uniform",
        choices=["uniform", "pmf"],
        help="Sampling distribution for random left shift of the crop window (excluding the 10%% no-shift branch).",
    )
    parser.add_argument(
        "--crop_no_shift_probability",
        type=float,
        default=0.1,
        help="Probability of taking zero left-shift when shifting is optional (matches CityNav-style behavior).",
    )
    parser.add_argument(
        "--temporal_history_past",
        type=int,
        default=16,
        help="Number of past frames before current; user turn has temporal_history_past+1 images.",
    )
    parser.add_argument(
        "--max_window_sample_attempts",
        type=int,
        default=256,
        help=(
            "Per __getitem__: if the random suffix-window references missing PNGs, resample the window "
            "on the **same** trajectory up to this many times (one dataset index == one trajectory per epoch)."
        ),
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="weighted",
        choices=["standard", "weighted"],
        help="weighted: linear token weights k/N over supervised assistant tokens; standard: masked mean.",
    )
    parser.add_argument(
        "--skill_json_left",
        type=str,
        default="",
        help="Left skill JSON (last action 2). Required with --skill_json_right when --skill_mix_rate > 0.",
    )
    parser.add_argument(
        "--skill_json_right",
        type=str,
        default="",
        help="Right skill JSON (last action 3). Required with --skill_json_left when --skill_mix_rate > 0.",
    )
    parser.add_argument(
        "--skill_mix_rate",
        type=float,
        default=0.0,
        help=(
            "Total skill fraction of general size per epoch. Implemented as equal left/right: "
            "R_each=int(G*rate/2), R=2*R_each. 0 disables."
        ),
    )

    parser.add_argument(
        "--system_prompt",
        type=str,
        default="",
        help="System message text (action space / role). Used unless --no_system_prompt or --system_prompt_file.",
    )
    parser.add_argument(
        "--system_prompt_file",
        type=str,
        default="",
        help="If set, read system prompt from this UTF-8 file (overrides --system_prompt).",
    )
    parser.add_argument(
        "--no_system_prompt",
        action="store_true",
        help="Disable system message entirely (user+images+instruction only).",
    )
    parser.add_argument(
        "--use_default_vln_system_prompt",
        action="store_true",
        help="Prepend the built-in drone/VLN action-space system message (see DEFAULT_VLN_SYSTEM_PROMPT in script).",
    )
    parser.add_argument(
        "--freeze_vision_encoder",
        action="store_true",
        help="Set requires_grad=False on Qwen3-VL vision backbone (model.model.visual).",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=16384,
        help=(
            "Soft sequence-length hint for logging only. The trajectory collator disables processor truncation "
            "(Qwen3-VL requires image token counts to match input_ids). Reduce --max_crop_length if you OOM."
        ),
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="DataLoader workers for image loading (0 = main process only). Increase if disk is fast.",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If >0, overrides num_train_epochs and stops after max_steps (useful for debug).",
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="steps",
        choices=["no", "epoch", "steps"],
        help="Use 'steps' for intermediate checkpoints during training.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Used when save_strategy='steps'. Common range: 200-2000 depending on dataset size.",
    )
    parser.add_argument("--save_total_limit", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="Comma-separated list, e.g. tensorboard,wandb,none",
    )

    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--debug_samples", type=int, default=0)
    parser.add_argument("--max_trajectories", type=int, default=0)
    parser.add_argument("--verify_images_exist", action="store_true")
    parser.add_argument(
        "--debug_dataset_collate_only",
        action="store_true",
        help="Rank 0: collate dataset[0], print seq_len and exit (sanity check before long jobs).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="",
        help="Path to a HF Trainer checkpoint directory (e.g. .../checkpoint-500 or .../checkpoint-last).",
    )
    parser.add_argument(
        "--checkpoint_layout",
        type=str,
        default="standard",
        choices=["standard", "best_last"],
        help=(
            "standard: use --save_strategy / --save_steps as usual. "
            "best_last: each epoch save one HF checkpoint (save_total_limit=1), mirror it to checkpoint-last/, "
            f"and save model+processor to {BEST_EVAL_DIRNAME}/ whenever eval (or train) loss improves."
        ),
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable LoRA/PEFT (default: off for full fine-tuning).",
    )
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    args = parser.parse_args()

    if args.save_strategy == "steps" and args.save_steps <= 0:
        raise ValueError("--save_steps must be > 0 when --save_strategy steps")
    if args.crop_no_shift_probability < 0 or args.crop_no_shift_probability > 1:
        raise ValueError("--crop_no_shift_probability must be in [0, 1]")
    if args.max_window_sample_attempts < 1:
        raise ValueError("--max_window_sample_attempts must be >= 1")
    if args.dataloader_num_workers < 0:
        raise ValueError("--dataloader_num_workers must be >= 0")
    if args.per_device_eval_batch_size < 1:
        raise ValueError("--per_device_eval_batch_size must be >= 1")
    if args.eval_accumulation_steps < 1:
        raise ValueError("--eval_accumulation_steps must be >= 1")
    if args.eval_steps <= 0:
        raise ValueError("--eval_steps must be > 0")

    eval_json_path = Path(args.eval_json.strip()) if args.eval_json.strip() else None
    if args.no_eval:
        do_eval = False
    elif args.do_eval is True:
        do_eval = True
    elif args.do_eval is False:
        do_eval = False
    else:
        do_eval = eval_json_path is not None and eval_json_path.is_file()
    if do_eval and (eval_json_path is None or not eval_json_path.is_file()):
        raise FileNotFoundError(f"--do_eval requires existing --eval_json, got: {eval_json_path}")

    if args.no_system_prompt:
        system_prompt_text = ""
    elif args.system_prompt_file.strip():
        sp_path = Path(args.system_prompt_file.strip())
        if not sp_path.is_file():
            raise FileNotFoundError(f"--system_prompt_file not found: {sp_path}")
        system_prompt_text = sp_path.read_text(encoding="utf-8").strip()
    elif args.system_prompt.strip():
        system_prompt_text = args.system_prompt.strip()
    elif args.use_default_vln_system_prompt:
        system_prompt_text = DEFAULT_VLN_SYSTEM_PROMPT.strip()
    else:
        system_prompt_text = ""

    torch_dtype = _to_dtype(args.dtype)

    _prefetch_hub_repo_serially(args.model_name_or_path)

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if args.min_pixels is not None or args.max_pixels is not None:
        size = dict(getattr(processor.image_processor, "size", {}) or {})
        if args.min_pixels is not None:
            size["shortest_edge"] = int(args.min_pixels)
        if args.max_pixels is not None:
            size["longest_edge"] = int(args.max_pixels)
        processor.image_processor.size = size
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[image_processor] size={processor.image_processor.size} "
                f"(min_pixels={args.min_pixels}, max_pixels={args.max_pixels})"
            )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=None,
    )

    if args.freeze_vision_encoder:
        frozen = 0
        for p in model.model.visual.parameters():
            p.requires_grad = False
            frozen += p.numel()
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Frozen vision encoder parameters: {frozen:,} (model.model.visual)")

    if args.use_lora:
        model.enable_input_require_grads()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.use_lora:
        lora_targets = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_targets,
        )
        model = get_peft_model(model, peft_config)

    dataset_kwargs = dict(
        frames_root=args.frames_root,
        max_crop_length=args.max_crop_length,
        chat_window_turns=args.chat_window_turns,
        crop_shift_sampling=args.crop_shift_sampling,
        temporal_history_past=args.temporal_history_past,
        no_shift_probability=args.crop_no_shift_probability,
        verify_images_exist=args.verify_images_exist,
        max_window_sample_attempts=args.max_window_sample_attempts,
        max_trajectories=(args.max_trajectories if args.max_trajectories and args.max_trajectories > 0 else None),
        debug_samples=(args.debug_samples if args.debug_samples and args.debug_samples > 0 else None),
    )

    skill_json_left = args.skill_json_left.strip()
    skill_json_right = args.skill_json_right.strip()
    skill_mix_rate = float(args.skill_mix_rate)
    if skill_mix_rate > 0:
        if not skill_json_left or not skill_json_right:
            raise ValueError("--skill_mix_rate > 0 requires both --skill_json_left and --skill_json_right")
        train_dataset = VlnMixedGeneralSkillDataset(
            general_json=args.train_json,
            skill_json_left=skill_json_left,
            skill_json_right=skill_json_right,
            skill_mix_rate=skill_mix_rate,
            deterministic=False,
            **dataset_kwargs,
        )
    else:
        train_dataset = VlnTrajectoryCropDataset(
            json_path=args.train_json,
            deterministic=False,
            **dataset_kwargs,
        )

    eval_dataset = None
    if do_eval:
        eval_dataset = VlnTrajectoryCropDataset(
            json_path=str(eval_json_path),
            deterministic=True,
            **dataset_kwargs,
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"Validation enabled: {len(eval_dataset)} trajectories from {eval_json_path} "
                f"(eval every {args.eval_steps} steps)"
            )

    data_collator = Qwen3VlTrajectoryCollator(
        processor=processor,
        max_length=args.max_length,
        system_prompt=system_prompt_text,
    )

    if args.debug_dataset_collate_only and int(os.environ.get("RANK", "0")) == 0:
        row = train_dataset[0]
        n_img_slots = 0
        for m in row["messages"]:
            cont = m.get("content")
            if not isinstance(cont, list):
                continue
            for p in cont:
                if isinstance(p, dict) and p.get("type") == "image":
                    n_img_slots += 1
        batch = data_collator([row])
        print("[debug_dataset_collate_only] traj_meta:", row.get("traj_meta"))
        print("[debug_dataset_collate_only] image slots in messages:", n_img_slots)
        print("[debug_dataset_collate_only] input_ids shape:", tuple(batch["input_ids"].shape))
        print("[debug_dataset_collate_only] num supervised labels:", int((batch["labels"] != -100).sum().item()))
        return

    run_name = time.strftime("run-%Y%m%d-%H%M%S")
    logging_dir = str(Path(args.output_dir) / "tb" / run_name)

    save_strategy = args.save_strategy
    save_steps = args.save_steps if save_strategy == "steps" else None
    save_total_limit = args.save_total_limit
    trainer_holder: List[Any] = [None]
    callbacks: List[TrainerCallback] = []
    report_targets = [x.strip() for x in args.report_to.split(",") if x.strip() and x.strip() != "none"]
    if "tensorboard" in report_targets:
        callbacks.append(UnifiedLossTensorBoardCallback())
    if do_eval:
        callbacks.append(CudaEmptyCacheCallback())
    if args.checkpoint_layout == "best_last":
        save_strategy = "epoch"
        save_steps = None
        save_total_limit = 1
        callbacks.append(
            BestAndLastCheckpointCallback(
                processor,
                trainer_holder,
                save_on_train_loss=not do_eval,
            )
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=logging_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_accumulation_steps=(args.eval_accumulation_steps if do_eval else None),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=(args.max_steps if args.max_steps and args.max_steps > 0 else -1),
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        eval_strategy=("steps" if do_eval else "no"),
        eval_steps=(args.eval_steps if do_eval else None),
        seed=args.seed,
        report_to=[x.strip() for x in args.report_to.split(",") if x.strip() and x.strip() != "none"],
        remove_unused_columns=False,
        fp16=(args.dtype == "fp16"),
        bf16=(args.dtype == "bf16"),
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,
        callbacks=callbacks,
        loss_type=args.loss_type,
        im_end_id=int(data_collator._im_end_id),
    )
    trainer_holder[0] = trainer

    resume_path = args.resume_from_checkpoint.strip() or None
    trainer.train(resume_from_checkpoint=resume_path)

    if args.checkpoint_layout != "best_last" and trainer.is_world_process_zero():
        trainer.save_model(args.output_dir)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
