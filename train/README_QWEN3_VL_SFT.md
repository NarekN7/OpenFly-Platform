# Qwen3-VL SFT and action evaluation

This guide describes `scripts/qwen3_vl_sft.py` and the current launchers. The
legacy OpenVLA workflow in `train/train.py`, tier-1 SFT, and the historical
single-turn overfit trainer have separate data and prompt contracts. Inspect
the selected launcher body; historical filenames do not fully describe it.

## Runtime and paths

Work from the actual repository root. The original installation uses
`/home/nnurijanyan/OpenFly-Platform`; preserve existing `nnurijanyan` paths that
refer to its data, checkpoints, or environment. Do not substitute another
username just because a checkout lives under a different account.

Skill-mix 20-epoch Slurm launchers (`train/slurm_trainof_*_lrb*_stop20_20ep_*.sbatch`)
and `train/run_qwen3_vl_skillmix.sh` use the training-computer **TrainOF** venv and
emit the full Slurm config banner. Other wrappers may still source
`scripts/openfly_vln_env.sh` (Conda `vln`) for machines that use that runtime;
see `tools/check_qwen_runtime.py` and [the runtime guide](../docs/qwen_action_eos_runtime.md).

```bash
conda activate vln
export DATA_DIR=/mnt/weka/nnurijanyan/data/vln
export OPENFLY_EVAL_DATA_DIR=/mnt/weka/nnurijanyan/OpenFly-Platform/data_curated
```

Set `OUTPUT_DIR` explicitly to a new writable experiment directory. Training
wrappers reject a nonempty output directory; use the Python entrypoint with
explicit arguments for a compatible full-state resume.

## Data, history, and loss

The loader accepts JSON arrays or JSONL with `image_path`, `gpt_instruction`,
nonempty `action`, and an equally long `index_list`. Frames resolve as
`$DATA_DIR/train_curated/<image_path>/<index_list[t]>.png`. Sequence order follows
`index_list`, not filename sorting. Preflight must reject unexpected action IDs;
the tier-2 allowed set is `{0,1,2,3,4,5,8,9,10}`.

Current SFT interleaves past frame/action pairs followed by the current frame.
`--temporal_history_past 16` supplies up to 17 frames; short histories are not
left-padded. General training samples a timestep uniformly within each
trajectory. L/R/stop skill samples use the final timestep. The default system
prompt is enabled with `--use_default_vln_system_prompt`; the current last-turn
suffix is empty. Match the actual training prompt when evaluating.

Training supervises action digits and `<|im_end|>` by default. Weighted general
samples use turn weights; skill samples supervise only the final action and its
EOS. Action 10 has two digit tokens. `--no-supervise_eos` reproduces the legacy
action-only objective. Validation `eval_loss` excludes EOS: skill validation
scores final action digits, while general validation scores all labeled action
digits in its deterministic window. Neither loss is navigation success rate.

## Launch and preflight

For general SFT, set the paths above and `OUTPUT_DIR`, then run:

```bash
bash train/run_qwen3_vl_sft.sh
```

For the tier-2 L/R/stop experiment family, the shared launcher uses
`trainx9_curated_0.json`, `trainx9_left_curated.json`,
`trainx9_right_curated.json`, and `trainx9_stop_curated.json` from `DATA_DIR`.
Validation uses the fixed left/right/stop validation triple from
`OPENFLY_EVAL_DATA_DIR`. These flags replace general `--eval_json` validation.

```bash
OPENFLY_DRY_RUN=1 SKILL_MIX_RATE=0.10 bash train/run_qwen3_vl_skillmix.sh
```

The five `lrb{5,10,15,20,40}_stop20_20ep` Slurm scripts call that launcher.
Their historical `actonly` names remain, but their current bodies enable EOS.
The default effective global batch is `8 GPUs * 1 sample * 4 accumulation = 32`.
L/R mix rates are relative to the general pool size, not the final mixed total.

Before a long launch, run the same configuration with one process and a temporary
output directory, adding `--debug_samples 8 --verify_images_exist
--debug_dataset_collate_only`. Then, if a GPU is available, run a separate
bounded `--max_steps 5` smoke with another temporary output. Preserve fixed split
manifests; inspect a proposed split using `split_train_validation_curated.py
--train-json <path> --dry-run` before any explicitly authorized data rewrite.

## Checkpoints and TensorBoard

Standard layout follows `--save_strategy`, `--save_steps`, and
`--save_total_limit`. `--checkpoint_layout best_last` instead keeps an epoch HF
checkpoint and mirrors its full Trainer state into `checkpoint-last/`; the best
validation model and processor go into `eval-best/`. With validation disabled,
the fallback criterion is logged training loss.

Best-loss metadata is restored from the original output directory on restart.
Full, sharded, and default PEFT adapter saves are verified before replacement;
write failures preserve the previous best and comparison threshold. Adapter
evaluation still requires the recorded base model and a compatible loader.
See [checkpoint recovery details](../docs/qwen_action_eos_runtime.md#best-and-last-checkpoints).

For full-state resume, keep the base model in `--model_name_or_path`, use the
original output directory and configuration, and pass
`--resume_from_checkpoint <output_dir>/checkpoint-last`. To change an old
action-only run to EOS training, initialize from its weights in a new output
directory instead of resuming its optimizer state.

TensorBoard writes under `<output_dir>/tb/run-YYYYMMDD-HHMMSS/`. For the existing
original experiment, for example:

```bash
conda activate vln
tensorboard --logdir /mnt/weka/nnurijanyan/checkpoints/qwen3-vl-2b-vln-simple/tb
```

## Evaluation and focused checks

Offline skills, live Qwen navigation, and `scripts/eval_overfit8_exact.py` use
the same leading-action parser and action stopping. Invalid prefixes remain
invalid. Action 10 waits for its second digit. Complete-action accuracy and
first-token accuracy are distinct; the first token cannot distinguish 1 from 10.
Use `--full-response` on the overfit or L/R diagnostic to measure output-format
errors. Format metrics are null with early action stopping.

The overfit-eight script retains its historical single-turn, padded 17-frame
input. Use it only for its matching overfit trainer. See the
[action and runtime guide](../docs/qwen_action_eos_runtime.md) for metrics,
logging, and the focused CPU regression commands.

Run training from the repository root: adding `train/` to Python's import path
can make the local `train/datasets/` shadow Hugging Face's `datasets` package.
