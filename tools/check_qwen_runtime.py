"""Check and report the Qwen runtime used by both training and evaluation."""
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import torch
import torchvision

EXPECTED = {
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "transformers": "5.0.0",
    "accelerate": "1.12.0",
}
versions = {name: importlib.metadata.version(name) for name in (*EXPECTED, "peft", "Pillow", "numpy", "safetensors")}
errors = []
for name, version in EXPECTED.items():
    if versions[name].split("+")[0] != version:
        errors.append(f"{name}: expected {version}, found {versions[name]}")
if errors:
    raise SystemExit("Qwen runtime mismatch. Activate the validated vln environment.\n" + "\n".join(errors))
# Exercise a compiled torchvision operator to catch a mismatched torch/CUDA ABI.
torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.]), 0.5)
root = Path(__file__).resolve().parents[1]
git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, capture_output=True)
sources = ("scripts/qwen3_vl_sft.py", "train/qwen3_vl_interleaved_common.py", "train/eval.py")
print("OPENFLY_RUNTIME " + json.dumps({
    "repo": str(root), "git_sha": git.stdout.strip() if git.returncode == 0 else None,
    "python": sys.executable, "packages": versions, "cuda": torch.version.cuda,
    "source_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources},
}), flush=True)
