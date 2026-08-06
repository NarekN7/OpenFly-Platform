#!/usr/bin/env python3
"""Stratified random holdout: 166 trajectories per AirSim env -> validation_curated.json."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

ENVS = (
    "env_airsim_16",
    "env_airsim_18",
    "env_airsim_23",
    "env_airsim_26",
    "env_airsim_gz",
    "env_airsim_sh",
)


def _load_trajectories(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [x for x in data if isinstance(x, dict)]


def _env_of(image_path: str) -> str:
    for env in ENVS:
        if image_path.startswith(env + "/"):
            return env
    return "OTHER"


def _astar_bucket(image_path: str) -> str:
    parts = image_path.split("/")
    if len(parts) >= 3 and parts[1] == "astar_data":
        return parts[2]
    return "unknown"


def _histogram(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(_astar_bucket(r["image_path"]) for r in rows))


def _group_by_env(trajs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    pools: Dict[str, List[Dict[str, Any]]] = {e: [] for e in ENVS}
    other: List[Dict[str, Any]] = []
    for t in trajs:
        env = _env_of(t.get("image_path", ""))
        if env in pools:
            pools[env].append(t)
        else:
            other.append(t)
    if other:
        raise ValueError(
            f"{len(other)} trajectories with unknown env prefix "
            f"(example: {other[0].get('image_path')!r})"
        )
    return pools


def split_from_manifest(
    trajs: List[Dict[str, Any]],
    manifest: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Hold out the exact validation trajectories recorded in an existing manifest."""
    val_paths: set[str] = set()
    for detail in manifest.get("per_env_detail", {}).values():
        val_paths.update(detail.get("val_image_paths", []))
    if not val_paths:
        raise ValueError("manifest has no val_image_paths")

    input_paths = [t.get("image_path", "") for t in trajs]
    if len(set(input_paths)) != len(input_paths):
        raise ValueError("duplicate image_path in input trajectories")

    val_rows = [t for t in trajs if t["image_path"] in val_paths]
    train_rows = [t for t in trajs if t["image_path"] not in val_paths]
    found_val_paths = {r["image_path"] for r in val_rows}
    missing = sorted(val_paths - found_val_paths)
    if missing:
        raise ValueError(
            f"{len(missing)} manifest validation paths missing from input "
            f"(example: {missing[0]!r})"
        )

    manifest_out = {
        **manifest,
        "total_input": len(trajs),
        "val_count": len(val_rows),
        "train_count": len(train_rows),
        "split_source": "validation_split_manifest",
    }
    return train_rows, val_rows, manifest_out


def split_train_validation(
    trajs: List[Dict[str, Any]],
    *,
    per_env: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    pools = _group_by_env(trajs)
    rng = random.Random(seed)
    val_rows: List[Dict[str, Any]] = []
    manifest_env: Dict[str, Any] = {}

    for env in ENVS:
        pool = pools[env]
        if len(pool) < per_env:
            raise ValueError(f"{env}: pool size {len(pool)} < per_env {per_env}")
        picked = rng.sample(pool, per_env)
        val_rows.extend(picked)
        manifest_env[env] = {
            "pool_size": len(pool),
            "val_count": len(picked),
            "pool_bucket_histogram": _histogram(pool),
            "val_bucket_histogram": _histogram(picked),
            "val_image_paths": sorted(r["image_path"] for r in picked),
        }

    val_paths = {r["image_path"] for r in val_rows}
    train_rows = [t for t in trajs if t["image_path"] not in val_paths]
    manifest = {
        "seed": seed,
        "per_env": per_env,
        "environments": list(ENVS),
        "total_input": len(trajs),
        "val_count": len(val_rows),
        "train_count": len(train_rows),
        "per_env_detail": manifest_env,
    }
    return train_rows, val_rows, manifest


def _write_json_array(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
        f.write("\n")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Random stratified train/val split (166 per env) for curated JSON."
    )
    repo = Path(__file__).resolve().parents[1]
    default_train = repo / "data_curated" / "train_curated.json"
    default_val = repo / "data_curated" / "validation_curated.json"
    default_manifest = repo / "data_curated" / "validation_split_manifest.json"
    default_backup = repo / "data_curated" / "train_curated.json.bak"

    parser.add_argument("--train-json", type=Path, default=default_train)
    parser.add_argument("--val-out", type=Path, default=default_val)
    parser.add_argument("--train-out", type=Path, default=default_train)
    parser.add_argument("--manifest-out", type=Path, default=default_manifest)
    parser.add_argument("--backup", type=Path, default=default_backup)
    parser.add_argument(
        "--from-manifest",
        type=Path,
        default=None,
        help="Reuse val_image_paths from an existing manifest (no random re-sample).",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write manifest-out (default when not using --from-manifest).",
    )
    parser.add_argument("--per-env", type=int, default=166)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts and bucket histograms only; do not write files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow re-split even if manifest already exists.",
    )
    args = parser.parse_args(argv)

    if args.per_env <= 0:
        raise ValueError("--per-env must be > 0")

    if args.manifest_out.exists() and not args.force and not args.dry_run and not args.from_manifest:
        print(
            f"Refusing to split: {args.manifest_out} exists. "
            "Use --force to re-run or delete the manifest.",
            file=sys.stderr,
        )
        return 1

    if not args.train_json.is_file():
        raise FileNotFoundError(f"Missing train JSON: {args.train_json}")

    trajs = _load_trajectories(args.train_json)
    expected_val = args.per_env * len(ENVS)

    if args.from_manifest:
        if not args.from_manifest.is_file():
            raise FileNotFoundError(f"Missing manifest: {args.from_manifest}")
        manifest_in = json.loads(args.from_manifest.read_text(encoding="utf-8"))
        train_rows, val_rows, manifest = split_from_manifest(trajs, manifest_in)
    else:
        # Already split: train has 60008 and val file exists with 996
        if (
            not args.force
            and args.val_out.is_file()
            and len(trajs) == 61004 - expected_val
        ):
            val_existing = _load_trajectories(args.val_out)
            if len(val_existing) == expected_val:
                print(
                    f"Looks already split: train={len(trajs)} val={len(val_existing)}. "
                    "Use --force to re-split from backup."
                )
                return 0

        train_rows, val_rows, manifest = split_train_validation(
            trajs, per_env=args.per_env, seed=args.seed
        )

    print(f"Input trajectories: {len(trajs)}")
    print(f"Val: {len(val_rows)} (expected {expected_val})")
    print(f"Train: {len(train_rows)} (expected {len(trajs) - expected_val})")
    for env in ENVS:
        detail = manifest["per_env_detail"][env]
        print(f"  {env}: pool={detail['pool_size']} val={detail['val_count']}")
        print(f"    val buckets: {detail['val_bucket_histogram']}")

    if args.dry_run:
        print("Dry run — no files written.")
        return 0

    if len(val_rows) != expected_val:
        raise RuntimeError(f"val count {len(val_rows)} != {expected_val}")
    if len(train_rows) + len(val_rows) != len(trajs):
        raise RuntimeError("train + val != input total")
    val_paths = [r["image_path"] for r in val_rows]
    if len(set(val_paths)) != len(val_paths):
        raise RuntimeError("duplicate image_path in validation set")

    if args.train_json.resolve() == args.train_out.resolve():
        if not args.backup.exists():
            shutil.copy2(args.train_json, args.backup)
            print(f"Backed up train JSON -> {args.backup}")

    _write_json_array(args.val_out, val_rows)
    _write_json_array(args.train_out, train_rows)
    if args.write_manifest or not args.from_manifest:
        with args.manifest_out.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Wrote {args.manifest_out}")

    print(f"Wrote {args.val_out} ({len(val_rows)} rows)")
    print(f"Wrote {args.train_out} ({len(train_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
