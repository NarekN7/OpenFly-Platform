"""
Shared Qwen3-VL interleaved eval utilities aligned with scripts/qwen3_vl_sft_nosfx.py.

Single source of truth for offline skill eval and closed-loop eval message layout,
tier-2 action parsing, processor loading (checkpoint processor_config.json), and
native-resolution PIL images (processor max_pixels resize, no cv2 pre-resize).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig
from transformers import AutoImageProcessor
from transformers import AutoProcessor as HFAutoProcessor
from transformers import AutoTokenizer
from transformers import AutoVideoProcessor
from transformers import Qwen3VLForConditionalGeneration
from transformers import Qwen3VLProcessor

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

# nosfx training dropped the last-turn cue; keep the name so callers stay stable.
PROMPT_SUFFIX = ""

# Legacy aliases used by skill scripts
DEFAULT_VLN_SYSTEM_PROMPT_TIER2 = DEFAULT_VLN_SYSTEM_PROMPT
PROMPT_SUFFIX_TIER2 = PROMPT_SUFFIX


def resolve_system_prompt() -> str:
    """Default ON (tier-2), matching training with --use_default_vln_system_prompt."""
    if os.environ.get("OPENFLY_QWEN_NO_SYSTEM_PROMPT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    ):
        return ""
    path = os.environ.get("OPENFLY_QWEN_SYSTEM_PROMPT_FILE", "").strip()
    if path:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"OPENFLY_QWEN_SYSTEM_PROMPT_FILE not found: {p}")
        return p.read_text(encoding="utf-8").strip()
    override = os.environ.get("OPENFLY_QWEN_SYSTEM_PROMPT", "").strip()
    if override:
        return override
    return DEFAULT_VLN_SYSTEM_PROMPT.strip()


def resolve_prompt_suffix() -> str:
    """Default empty (nosfx). Set OPENFLY_QWEN_PROMPT_SUFFIX to restore interleaved last-turn cue."""
    if "OPENFLY_QWEN_PROMPT_SUFFIX" in os.environ:
        return os.environ["OPENFLY_QWEN_PROMPT_SUFFIX"]
    return PROMPT_SUFFIX


def _processor_fallback_id(ckpt: str) -> str:
    fb = os.environ.get("OPENFLY_QWEN_PROCESSOR_FALLBACK", "").strip()
    if fb:
        return fb
    return "Qwen/Qwen3-VL-4B-Instruct" if "4b" in ckpt.lower() else "Qwen/Qwen3-VL-2B-Instruct"


def _apply_processor_config_image_size(processor: Any, ckpt_path: Path) -> None:
    proc_cfg_path = ckpt_path / "processor_config.json"
    if not proc_cfg_path.is_file():
        return
    with proc_cfg_path.open("r", encoding="utf-8") as f:
        proc_cfg = json.load(f)
    ip_cfg = proc_cfg.get("image_processor") or {}
    size = ip_cfg.get("size")
    if size and hasattr(processor, "image_processor"):
        processor.image_processor.size = dict(size)
        print(
            f"processor image_processor.size={processor.image_processor.size} "
            f"(from {proc_cfg_path})",
            flush=True,
        )


def _assemble_processor_from_checkpoint(ckpt_path: Path, fallback_id: str) -> Any:
    tok = AutoTokenizer.from_pretrained(fallback_id, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(fallback_id, trust_remote_code=True)
    video_processor = AutoVideoProcessor.from_pretrained(fallback_id, trust_remote_code=True)
    chat_template: Optional[str] = None
    chat_tpl_path = ckpt_path / "chat_template.jinja"
    if chat_tpl_path.is_file():
        chat_template = chat_tpl_path.read_text(encoding="utf-8")
    processor = Qwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=tok,
        video_processor=video_processor,
        chat_template=chat_template,
    )
    _apply_processor_config_image_size(processor, ckpt_path)
    print(
        f"processor assembled from {fallback_id} + {ckpt_path}/processor_config.json",
        flush=True,
    )
    return processor


def load_processor(ckpt: str) -> Any:
    ckpt_path = Path(ckpt)
    candidates: List[str] = []
    pn = os.environ.get("OPENFLY_QWEN_PROCESSOR_NAME", "").strip()
    if pn:
        candidates.append(pn)
    if ckpt_path.is_dir():
        candidates.append(str(ckpt_path))
    fb = _processor_fallback_id(ckpt)
    if fb not in candidates:
        candidates.append(fb)

    last_err: Optional[BaseException] = None
    for src in candidates:
        try:
            proc = HFAutoProcessor.from_pretrained(src, trust_remote_code=True)
            if ckpt_path.is_dir():
                _apply_processor_config_image_size(proc, ckpt_path)
            if src != str(ckpt_path):
                print(f"processor loaded from: {src}", flush=True)
            else:
                print(f"processor loaded from checkpoint: {src}", flush=True)
            return proc
        except Exception as e:
            last_err = e
            print(f"processor load failed from {src}: {type(e).__name__}: {e}", flush=True)

    if ckpt_path.is_dir():
        try:
            return _assemble_processor_from_checkpoint(ckpt_path, fb)
        except Exception as e:
            last_err = e
            print(f"processor assemble failed: {type(e).__name__}: {e}", flush=True)

    raise RuntimeError(f"Could not load processor for {ckpt}") from last_err


def load_model(ckpt: str, device: str, attn: str):
    q_cfg = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
    tc = getattr(q_cfg, "text_config", None)
    if tc is not None and getattr(tc, "rope_scaling", None) is None:
        rp = getattr(tc, "rope_parameters", None)
        if rp is not None:
            d = rp if isinstance(rp, dict) else dict(rp)
            tc.rope_scaling = {
                "type": d.get("rope_type", d.get("type", "default")),
                "rope_theta": d.get("rope_theta", 5000000),
                "mrope_section": d.get("mrope_section", [24, 20, 20]),
                "mrope_interleaved": d.get("mrope_interleaved", True),
            }
    device_map = os.environ.get("OPENFLY_QWEN_DEVICE_MAP", "").strip()
    if device_map:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            ckpt,
            config=q_cfg,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn,
            device_map=device_map,
        )
        device = str(next(model.parameters()).device)
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            ckpt,
            config=q_cfg,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn,
        ).to(device)
    model.eval()
    return model, device


def parse_vln_action_id(text_out: str) -> int:
    if re.search(r"\b10\b", text_out):
        return 10
    m = re.search(r"\b([0-9])\b", text_out)
    if not m:
        return 0
    aid = int(m.group(1) if m.lastindex else m.group(0))
    if aid not in VLN_ALLOWED_ACTION_IDS:
        return 0
    return aid


def interleaved_window_lo(timestep: int, temporal_history_past: int) -> int:
    if timestep < 0:
        raise ValueError("timestep must be >= 0")
    if temporal_history_past < 0:
        raise ValueError("temporal_history_past must be >= 0")
    return max(0, timestep - temporal_history_past)


def build_interleaved_messages(
    system_prompt: str,
    instruction: str,
    prompt_suffix: str,
    window_images: Sequence[Any],
    window_past_actions: Sequence[int],
) -> List[Dict[str, Any]]:
    """Match nosfx VlnTrajectoryCropDataset._turn_messages for inference (no last-turn suffix).

    If prompt_suffix is non-empty, restore the older interleaved last-turn cue.
    """
    n = len(window_images)
    if n < 1:
        raise ValueError("window_images must contain the current frame")
    if len(window_past_actions) != n - 1:
        raise ValueError(
            f"window_past_actions len={len(window_past_actions)} must be "
            f"len(window_images)-1={n - 1}"
        )
    suffix = prompt_suffix or ""
    messages: List[Dict[str, Any]] = []
    sys_txt = (system_prompt or "").strip()
    if sys_txt:
        messages.append({"role": "system", "content": [{"type": "text", "text": sys_txt}]})
    for i, im in enumerate(window_images):
        is_first = i == 0
        is_last = i == n - 1
        user_content: List[Dict[str, Any]] = [{"type": "image", "image": im}]
        if is_first and is_last:
            text = f"{instruction}{suffix}" if suffix else instruction
            user_content.append({"type": "text", "text": text})
        elif is_first:
            user_content.append({"type": "text", "text": instruction})
        elif is_last and suffix:
            user_content.append({"type": "text", "text": suffix})
        messages.append({"role": "user", "content": user_content})
        if not is_last:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(int(window_past_actions[i]))}],
                }
            )
    return messages


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape {getattr(bgr, 'shape', None)}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_from_path(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def parse_image_roots(primary: Path) -> List[Path]:
    """Primary root plus optional comma-separated OPENFLY_SKILL_IMAGE_ROOTS fallbacks."""
    seen: set[str] = set()
    roots: List[Path] = []

    def _add(p: str) -> None:
        s = p.strip()
        if not s or s in seen:
            return
        seen.add(s)
        roots.append(Path(s))

    _add(str(primary))
    extra = os.environ.get(
        "OPENFLY_SKILL_IMAGE_ROOTS",
        "/mnt/xtb/vln/seen_curated,/nfs/np/mnt/xtb/vln/train_curated,/nfs/np/mnt/xtb/vln/seen_curated",
    )
    for part in extra.split(","):
        _add(part)
    return roots


def load_trajectory_pil_frames(
    image_roots: Sequence[Path], image_path: str, index_list: Sequence[str]
) -> List[Image.Image]:
    frames: List[Image.Image] = []
    for idx in index_list:
        png_name = f"{idx}.png"
        traj_png: Optional[Path] = None
        for root in image_roots:
            candidate = root / image_path / png_name
            if candidate.is_file():
                traj_png = candidate
                break
        if traj_png is None:
            raise FileNotFoundError(
                f"missing frame: {image_path}/{png_name} from roots={[str(r) for r in image_roots]}"
            )
        frames.append(pil_from_path(traj_png))
    return frames


def predict_action(
    model,
    processor,
    messages: List[Dict[str, Any]],
    device: str,
    max_new_tokens: int,
) -> Tuple[int, str]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    flat_images: List[Any] = []
    for msg in messages:
        for part in msg.get("content", []):
            if isinstance(part, dict) and part.get("type") == "image" and part.get("image") is not None:
                flat_images.append(part["image"])
    inputs = processor(
        text=text,
        images=flat_images,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def _to_dev(t):
        if isinstance(t, torch.Tensor):
            if t.is_floating_point():
                return t.to(device, dtype=torch.bfloat16)
            return t.to(device)
        return t

    inputs = {k: _to_dev(v) for k, v in inputs.items()}
    with torch.inference_mode():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
    in_len = inputs["input_ids"].shape[1]
    text_out = tok.decode(gen_ids[0, in_len:], skip_special_tokens=True)
    if not re.search(r"[0-9]", text_out):
        return 0, text_out
    return parse_vln_action_id(text_out), text_out


def processor_image_size(processor: Any) -> Optional[Dict[str, Any]]:
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return None
    size = getattr(ip, "size", None)
    if size is None:
        return None
    return dict(size) if isinstance(size, dict) else size
