#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterator


ENVS = (
    "env_airsim_16",
    "env_airsim_18",
    "env_airsim_23",
    "env_airsim_26",
    "env_airsim_gz",
    "env_airsim_sh",
)


def _iter_split_paths_for_env(*, api, repo_id: str, revision: str, env: str) -> Iterator[str]:
    """
    Yield split folder paths under:
      Image/<env>/astar_data/<split>
    """
    from huggingface_hub.hf_api import RepoFolder

    env_root = f"Image/{env}"
    env_entries = list(
        api.list_repo_tree(
            repo_id=repo_id,
            path_in_repo=env_root,
            repo_type="dataset",
            revision=revision,
            recursive=False,
            expand=False,
        )
    )
    astar_paths = [e.path for e in env_entries if isinstance(e, RepoFolder) and e.path.endswith("/astar_data")]
    if not astar_paths:
        return

    astar_path = astar_paths[0]
    split_entries = list(
        api.list_repo_tree(
            repo_id=repo_id,
            path_in_repo=astar_path,
            repo_type="dataset",
            revision=revision,
            recursive=False,
            expand=False,
        )
    )
    for e in split_entries:
        if isinstance(e, RepoFolder):
            yield e.path


def _iter_pose_files_under_split(*, api, repo_id: str, revision: str, env: str, split_path: str) -> Iterator[str]:
    """
    Yield repo-relative paths to pose.jsonl under a single split folder.

    Structure (observed):
      Image/<env>/astar_data/<split>/<run_id>/pose.jsonl
    """
    from huggingface_hub.hf_api import RepoFile

    env_root = f"Image/{env}"
    all_entries = list(
        api.list_repo_tree(
            repo_id=repo_id,
            path_in_repo=split_path,
            repo_type="dataset",
            revision=revision,
            recursive=True,
            expand=False,
        )
    )
    for e in all_entries:
        if not isinstance(e, RepoFile):
            continue
        if not e.path.endswith("/pose.jsonl"):
            continue
        if not e.path.startswith(f"{env_root}/"):
            continue
        yield e.path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download only pose.jsonl files for selected environments from "
            "IPEC-COMMUNITY/OpenFly (dataset repo) while preserving directory structure."
        )
    )
    parser.add_argument(
        "--repo-id",
        default="IPEC-COMMUNITY/OpenFly",
        help='Dataset repo id (default: "IPEC-COMMUNITY/OpenFly").',
    )
    parser.add_argument(
        "--revision",
        default="main",
        help='Repo revision/branch/tag/commit (default: "main").',
    )
    parser.add_argument(
        "--dest",
        default=str(Path("data_wo_annotation").resolve()),
        help="Destination directory (default: ./data_wo_annotation).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print matched files and planned destinations; do not download/copy.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, download only the first N matched files (useful for testing).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "If the destination pose.jsonl already exists and is non-empty, skip re-download "
            "(use this to resume after interrupts)."
        ),
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as e:
        print(
            "Failed to import huggingface_hub. Install it first, e.g.\n"
            "  pip install -U huggingface_hub\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        return 2

    dest_root = Path(args.dest)
    dest_root.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    print(
        f"Downloading pose.jsonl via per-split tree listing in dataset repo {args.repo_id}@{args.revision} ...",
        flush=True,
    )

    failures: list[str] = []
    copied = 0
    skipped = 0
    processed = 0
    per_env: dict[str, int] = {env: 0 for env in ENVS}

    stop = False
    for env in ENVS:
        if stop:
            break
        print(f"== {env} ==", flush=True)
        split_paths = sorted(_iter_split_paths_for_env(api=api, repo_id=args.repo_id, revision=args.revision, env=env))
        print(f"  splits: {len(split_paths)}", flush=True)

        for split_path in split_paths:
            if stop:
                break
            pose_paths = sorted(
                _iter_pose_files_under_split(
                    api=api,
                    repo_id=args.repo_id,
                    revision=args.revision,
                    env=env,
                    split_path=split_path,
                )
            )
            print(f"  - {split_path} pose.jsonl: {len(pose_paths)}", flush=True)

            for repo_path in pose_paths:
                processed += 1
                per_env[env] += 1

                # Strip leading "Image/" so the destination is: dest/<env>/.../pose.jsonl
                if not repo_path.startswith("Image/"):
                    failures.append(repo_path)
                    continue
                rel = Path(repo_path).relative_to("Image")
                out_path = dest_root / rel

                if args.dry_run:
                    if args.skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
                        print(f"[DRY SKIP] {repo_path} (exists)", flush=True)
                    else:
                        print(f"[DRY] {repo_path} -> {out_path}", flush=True)
                else:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    if args.skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
                        skipped += 1
                    else:
                        try:
                            cached_path = hf_hub_download(
                                repo_id=args.repo_id,
                                filename=repo_path,
                                repo_type="dataset",
                                revision=args.revision,
                            )
                            shutil.copy2(cached_path, out_path)
                            copied += 1
                        except Exception as e:
                            failures.append(repo_path)
                            print(f"Failed ({processed}): {repo_path}\n  {e}", file=sys.stderr, flush=True)

                if processed % 250 == 0:
                    print(
                        f"Progress: processed {processed} pose files "
                        f"(copied {copied}, skipped {skipped}, failures {len(failures)})",
                        flush=True,
                    )

                if args.limit and args.limit > 0 and processed >= args.limit:
                    stop = True
                    break

    if processed == 0:
        print("No matching pose.jsonl files found. Nothing to do.")
        return 1

    print("Totals for this run:", flush=True)
    for env in ENVS:
        print(f"  {env}: {per_env[env]}", flush=True)
    print(f"Total processed: {processed}", flush=True)

    print(
        f"Done. Copied: {copied}. Skipped (existing): {skipped}. Failures: {len(failures)}.",
        flush=True,
    )
    if failures:
        print("First 20 failures:")
        for p in failures[:20]:
            print(f"  {p}")
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

