## Qwen3-VL-2B minimal SFT (VLN)

### What it does
- Reads trajectories from `data_curated/train_curated.json` (JSON array) or JSONL.
- Optional validation during training from `data_curated/validation_curated.json` (996 trajectories; 166 per AirSim env). Logs `eval_loss` to TensorBoard when that file exists.
- For each timestep `t>=2`, uses 3 frames:
  - `frames_root/<image_path>/{index_list[t-2]}.png`
  - `frames_root/<image_path>/{index_list[t-1]}.png`
  - `frames_root/<image_path>/{index_list[t]}.png`
- Prompt: `gpt_instruction` + `"\nNext action id (0-9): "`
- Label: `str(action[t])` (single digit string)

### Environment (`TrainOF`)
Create a dedicated venv at repo root:

```bash
cd /home/nnurijanyan/OpenFly-Platform
python -m venv --system-site-packages TrainOF
source TrainOF/bin/activate
pip install -U pip
pip install -U transformers accelerate peft trl datasets pillow safetensors
# Qwen3-VL processor pulls a video/torchvision stack; install torchvision without upgrading torch:
pip install --no-deps "torchvision==0.25.0"
```

### Run (single GPU)

```bash
cd /home/nnurijanyan/OpenFly-Platform
source TrainOF/bin/activate
bash train/run_qwen3_vl_sft.sh
```

### Slurm (recommended first run: 1× GPU)

```bash
cd /home/nnurijanyan/OpenFly-Platform
mkdir -p slurm_logs
sbatch train/slurm_trainof_qwen3vl_1gpu.sbatch
```

### Slurm (8× GPU, faster throughput)

```bash
cd /home/nnurijanyan/OpenFly-Platform
mkdir -p slurm_logs
sbatch train/slurm_trainof_qwen3vl.sbatch
```

### Train / validation split (one-time)

Create a fixed stratified holdout (random 166 trajectories per env, seed 7):

```bash
python scripts/split_train_validation_curated.py --seed 7
```

Writes `data_curated/validation_curated.json` (996), updates `train_curated.json` (60,008), and `data_curated/validation_split_manifest.json`. Backs up the original train file to `train_curated.json.bak`.

### Checkpoints + TensorBoard (defaults)
The training script uses common Hugging Face `Trainer` defaults tuned for “smooth loss curves + periodic checkpoints”:

- **TensorBoard logs**: written under `${output_dir}/tb/run-YYYYMMDD-HHMMSS/` (a new run folder per process start).
- **Scalars**: Hugging Face also logs `train/loss` and `eval/loss`; the trainer adds **`loss/train`** and **`loss/validation`** on the same chart group (filter Scalars by `loss`).
- **Logging frequency**: `logging_steps=10` (loss points every 10 optimizer steps).
- **Validation**: `eval_steps=500` when `validation_curated.json` exists (~996 multimodal sequences per eval; can be slow with 17 frames).
- **Checkpointing**: `save_strategy=steps`, `save_steps=500`, `save_total_limit=4`  
  This keeps the **last 4 step-based checkpoints** on disk (plus the final `trainer.save_model()` write to `output_dir`).

View TensorBoard:

```bash
source TrainOF/bin/activate
tensorboard --logdir /mnt/weka/nnurijanyan/checkpoints/qwen3-vl-2b-vln-simple/tb
```

### Quick debug (recommended first)

```bash
cd /home/nnurijanyan/OpenFly-Platform
source TrainOF/bin/activate
accelerate launch --num_processes 1 scripts/qwen3_vl_sft.py \
  --debug_samples 32 \
  --max_steps 5 \
  --verify_images_exist \
  --output_dir /tmp/qwen3vl_debug
```

### Note on import paths
This repo contains a local Python package at `train/datasets/`, which can shadow Hugging Face's `datasets` package if you run scripts from inside `train/`. The training entrypoint therefore lives at `scripts/qwen3_vl_sft.py` (not under `train/`).
