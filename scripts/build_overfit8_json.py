#!/usr/bin/env python3
"""Build fixed 8-crop overfit JSONs: 4 general + 2 left + 2 right (+ eval_8 + manifest)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Optional, Sequence

REQUIRED_KEYS = ("image_path", "gpt_instruction", "action", "index_list", "pos", "yaw")
CROP_LEN = 4
TEMPORAL_HISTORY_PAST = 16
SEED = 7


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array at top-level in {path}")
    return data


def _validate_row(row: dict[str, Any], *, src: str) -> None:
    for k in REQUIRED_KEYS:
        if k not in row:
            raise ValueError(f"Missing key {k!r} in {src}")
    a = row["action"]
    idx = row["index_list"]
    pos = row["pos"]
    yaw = row["yaw"]
    if not isinstance(a, list) or not a:
        raise ValueError(f"Invalid/empty action in {src}")
    if not (len(a) == len(idx) == len(pos) == len(yaw)):
        raise ValueError(f"Length mismatch in {src}")


def _frame_paths_for_timestep(
    traj_dir: Path,
    index_list: Sequence[str],
    timestep: int,
    temporal_history_past: int,
) -> list[Path]:
    past = temporal_history_past
    lo = max(0, timestep - past)
    idxs = list(index_list[lo : timestep + 1])
    target = past + 1
    while len(idxs) < target:
        idxs.insert(0, idxs[0])
    return [traj_dir / f"{idx}.png" for idx in idxs]


def _last_timestep_frames_exist(
    frames_root: Path,
    image_path: str,
    index_list: Sequence[str],
    *,
    temporal_history_past: int = TEMPORAL_HISTORY_PAST,
) -> bool:
    traj_dir = frames_root / image_path
    if not traj_dir.is_dir():
        return False
    t = len(index_list) - 1
    return all(p.is_file() for p in _frame_paths_for_timestep(traj_dir, index_list, t, temporal_history_past))


def _crop_row(row: dict[str, Any], start: int, length: int) -> dict[str, Any]:
    end = start + length
    return {
        "image_path": row["image_path"],
        "gpt_instruction": row["gpt_instruction"],
        "action": list(row["action"][start:end]),
        "index_list": [str(x) for x in row["index_list"][start:end]],
        "pos": list(row["pos"][start:end]),
        "yaw": list(row["yaw"][start:end]),
    }


def _sample_general_crops(
    rows: list[dict[str, Any]],
    *,
    n: int,
    crop_len: int,
    frames_root: Path,
    rng: random.Random,
    src: str,
) -> list[dict[str, Any]]:
    order = list(range(len(rows)))
    rng.shuffle(order)
    out: list[dict[str, Any]] = []
    used_paths: set[str] = set()
    for i in order:
        row = rows[i]
        _validate_row(row, src=f"{src}[{i}]")
        if row["image_path"] in used_paths:
            continue
        L = len(row["action"])
        if L < 1:
            continue
        cl = min(crop_len, L)
        starts = list(range(0, L - cl + 1))
        rng.shuffle(starts)
        chosen: Optional[dict[str, Any]] = None
        chosen_start = -1
        for start in starts:
            crop = _crop_row(row, start, cl)
            if _last_timestep_frames_exist(frames_root, crop["image_path"], crop["index_list"]):
                chosen = crop
                chosen_start = start
                break
        if chosen is None:
            continue
        chosen["_overfit_meta"] = {
            "role": "general",
            "source_index": i,
            "source_image_path": row["image_path"],
            "crop_start": chosen_start,
            "crop_length": cl,
            "gt_action": chosen["action"][-1],
        }
        out.append(chosen)
        used_paths.add(row["image_path"])
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"Could only find {len(out)}/{n} valid general crops from {src}")
    return out


def _sample_skill_suffix_crops(
    rows: list[dict[str, Any]],
    *,
    n: int,
    crop_len: int,
    expected_last: int,
    frames_root: Path,
    rng: random.Random,
    src: str,
    used_paths: set[str],
) -> list[dict[str, Any]]:
    candidates: list[tuple[int, dict[str, Any]]] = []
    for i, row in enumerate(rows):
        _validate_row(row, src=f"{src}[{i}]")
        if int(row["action"][-1]) != expected_last:
            continue
        if row["image_path"] in used_paths:
            continue
        candidates.append((i, row))
    rng.shuffle(candidates)
    out: list[dict[str, Any]] = []
    for i, row in candidates:
        L = len(row["action"])
        cl = min(crop_len, L)
        start = L - cl
        crop = _crop_row(row, start, cl)
        if int(crop["action"][-1]) != expected_last:
            continue
        if not _last_timestep_frames_exist(frames_root, crop["image_path"], crop["index_list"]):
            continue
        crop["_overfit_meta"] = {
            "role": "left" if expected_last == 2 else "right",
            "source_index": i,
            "source_image_path": row["image_path"],
            "crop_start": start,
            "crop_length": cl,
            "gt_action": crop["action"][-1],
        }
        out.append(crop)
        used_paths.add(row["image_path"])
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(
            f"Could only find {len(out)}/{n} valid skill crops (last={expected_last}) from {src}"
        )
    return out


def _strip_meta(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for r in rows:
        cleaned.append({k: v for k, v in r.items() if k != "_overfit_meta"})
    return cleaned


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--crop_len", type=int, default=CROP_LEN)
    ap.add_argument(
        "--frames_root",
        type=str,
        default="/mnt/weka/nnurijanyan/data/vln/train_curated",
    )
    ap.add_argument(
        "--general_json",
        type=str,
        default="/home/nnurijanyan/OpenFly-Platform/data_curated/trainx9_curated_0.json",
    )
    ap.add_argument(
        "--left_json",
        type=str,
        default="/home/nnurijanyan/OpenFly-Platform/data_curated/trainx9_left_curated.json",
    )
    ap.add_argument(
        "--right_json",
        type=str,
        default="/home/nnurijanyan/OpenFly-Platform/data_curated/trainx9_right_curated.json",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/home/nnurijanyan/OpenFly-Platform/data_curated/overfit8",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_root = Path(args.frames_root)
    rng = random.Random(args.seed)

    general_src = _load_json_array(Path(args.general_json))
    left_src = _load_json_array(Path(args.left_json))
    right_src = _load_json_array(Path(args.right_json))

    general = _sample_general_crops(
        general_src,
        n=4,
        crop_len=args.crop_len,
        frames_root=frames_root,
        rng=rng,
        src=args.general_json,
    )
    used = {r["image_path"] for r in general}
    left = _sample_skill_suffix_crops(
        left_src,
        n=2,
        crop_len=args.crop_len,
        expected_last=2,
        frames_root=frames_root,
        rng=rng,
        src=args.left_json,
        used_paths=used,
    )
    used |= {r["image_path"] for r in left}
    right = _sample_skill_suffix_crops(
        right_src,
        n=2,
        crop_len=args.crop_len,
        expected_last=3,
        frames_root=frames_root,
        rng=rng,
        src=args.right_json,
        used_paths=used,
    )

    train_4 = _strip_meta(general)
    skill_left_2 = _strip_meta(left)
    skill_right_2 = _strip_meta(right)
    eval_8 = train_4 + skill_left_2 + skill_right_2

    (out_dir / "train_4.json").write_text(json.dumps(train_4, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "skill_left_2.json").write_text(json.dumps(skill_left_2, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "skill_right_2.json").write_text(json.dumps(skill_right_2, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "eval_8.json").write_text(json.dumps(eval_8, ensure_ascii=False, indent=2) + "\n")

    manifest = {
        "seed": args.seed,
        "crop_len": args.crop_len,
        "temporal_history_past": TEMPORAL_HISTORY_PAST,
        "frames_root": str(frames_root),
        "sources": {
            "general": args.general_json,
            "left": args.left_json,
            "right": args.right_json,
        },
        "outputs": {
            "train_4": str(out_dir / "train_4.json"),
            "skill_left_2": str(out_dir / "skill_left_2.json"),
            "skill_right_2": str(out_dir / "skill_right_2.json"),
            "eval_8": str(out_dir / "eval_8.json"),
        },
        "samples": [
            {**r["_overfit_meta"], "image_path": r["image_path"], "action": r["action"], "index_list": r["index_list"]}
            for r in general + left + right
        ],
    }
    (out_dir / "overfit8_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print(f"Wrote {out_dir}")
    for s in manifest["samples"]:
        print(
            f"  {s['role']:7s} gt={s['gt_action']} len={s['crop_length']} "
            f"start={s['crop_start']} path={s['image_path']}"
        )


if __name__ == "__main__":
    main()
