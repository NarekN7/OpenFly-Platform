# Qwen action parsing, EOS training, and runtime

The offline L/R, stop, in-train, overfit-eight, and closed-loop Qwen paths share the leading
action parser. After leading whitespace it takes `10`, otherwise one allowed
digit. Examples: `3.5` becomes 3, `201010` becomes 2, `100` becomes 10, and
`1 0` becomes 1. An invalid prefix stays invalid; later text is never searched.

## Training

`scripts/qwen3_vl_sft.py` defaults to `--supervise_eos`. Each supervised action
includes its terminating `<|im_end|>`: `2` + EOS, or `1` + `0` + EOS for action
10. General samples apply the existing turn weights to digits and EOS. Skill
samples supervise only the final action and its EOS; past actions are context.
Training loss is the weighted token mean over those targets. Validation
`eval_loss` remains action-only, preserving the existing checkpoint criterion.
The L/R/stop validation triple scores the final action. General `--eval_json`
validation scores all labeled action digits in the deterministic history window.

Use a new output directory for the changed objective. An old checkpoint can
initialize the new run via `INIT_MODEL` or `--model_name_or_path`. Resuming an
old action-only optimizer run requires `--no-supervise_eos`; the checkpoint
config records `openfly_supervise_eos` for future resume checks.

The five `lrb{5,10,15,20,40}_stop20_20ep` Slurm launchers use
`train/run_qwen3_vl_skillmix.sh` from `OPENFLY_REPO_ROOT` or `SLURM_SUBMIT_DIR`.
Their historical `actonly` filenames are retained for identification; their
current bodies enable EOS. Set `OUTPUT_DIR` explicitly. They do not overwrite
nonempty experiment directories. `OPENFLY_DRY_RUN=1` resolves the launch command
and checks paths without starting training.

## Best and last checkpoints

With `--checkpoint_layout best_last`, the callback restores the previous loss
and step from `eval-best/best_eval_metric.json` in the run's output directory.
A resumed run with an existing best loss of 0.14 keeps it when the next loss is
0.30; it replaces it when the loss reaches 0.13. Resume into the original output
directory with the same metric and training configuration. A new output
directory starts its own best-checkpoint record.

Saves are staged beside the destination. The callback checks full-model config
and weights, every indexed shard, or the default PEFT adapter plus its base-model
reference. Safetensors headers and indexed tensor names are checked without
loading model tensors; legacy `.bin` weights receive file/manifest checks.
Adapters still require the recorded base model and an adapter-capable loader.

The previous best stays available as `.eval-best.previous` during replacement.
A failed save or rename preserves it and does not advance the in-memory best
loss. On restart, a missing `eval-best` is restored from that backup. Existing
best directories with missing/corrupt metadata, incomplete weights, or a
different selection metric cause an explicit error before overwriting anything.
Non-finite losses are skipped.

`checkpoint-last` also stages its full Trainer copy before replacement, so a
copy failure preserves the previous resume state. If a process is killed between
the two directory renames, `.checkpoint-last.previous` holds that state. The SFT
entrypoint restores it before Trainer opens `--resume_from_checkpoint`, using a
file lock to coordinate DDP ranks.
These are directory-rename safeguards, not a guarantee against filesystem loss.
The temporary save and previous checkpoint require extra disk space during I/O.

## Inference

Action stopping is on by default (`OPENFLY_QWEN_STOP_AFTER_ACTION=1`). It does
not mask or modify logits. A leading digit other than 1 determines the action
immediately. A bare 1 waits for the next token to distinguish 10 from 1.
Leading whitespace waits; an invalid prefix terminates as invalid. The existing
16-token cap remains a bound for whitespace or other incomplete responses.

For unconstrained output-format diagnostics use
`OPENFLY_QWEN_STOP_AFTER_ACTION=0`, or `--full-response` with
`train/skill_eval_lr_diagnostics.py` or `scripts/eval_overfit8_exact.py`.
Format validity is null when action
stopping is enabled: short generated prefixes cannot establish whether the
model would naturally stop. Raw output, parsing policy, generation policy,
and invalid actions remain logged. Offline invalid actions count as incorrect;
navigation records invalid output and executes the explicit stop fallback.

The overfit-eight probe retains its historical single-turn message with 17
images, left-padded for short trajectories. It is specific to
`scripts/qwen3_vl_sft_overfit.py`; sharing its parser does not make this image
layout interchangeable with interleaved training. Its default generation cap
is 16 tokens, with action stopping enabled. `--out_json results.json` keeps the
per-sample JSON array and adds `results.summary.json` containing metrics and the
resolved model/data paths and decoding protocol.

The probe reports these scores separately:

- Action accuracy: the shared parser's complete action equals the target;
  invalid leading output counts as incorrect.
- First-token accuracy: the full-vocabulary argmax token ID equals the first
  target token ID. For Qwen, actions 1 and 10 both start with token `1`.
  Predicting `1` for target 10 is correct on this diagnostic but incorrect on
  complete-action accuracy. A leading space is a token error even if parsing
  later yields the correct action.
- Strict accuracy and invalid-format count: measured only with full-response
  generation. For example, `3.5` has action 3 but invalid output format.

## Runtime and data

Current launchers source `scripts/openfly_vln_env.sh`, which activates Conda
`vln` and checks the recorded torch/torchvision/Transformers/Accelerate versions
and a compiled torchvision operation. It does not install packages. The checked
versions are recorded in `tools/check_qwen_runtime.py`. Each launch prints the interpreter, package
versions, repository path, Git SHA, and current trainer/evaluator source hashes.

Machine settings may live in the ignored `.openfly.env` at the checkout root:

```bash
export OPENFLY_CONDA_ROOT="${OPENFLY_CONDA_ROOT:-/path/to/miniconda}"
export DATA_DIR="${DATA_DIR:-/mnt/weka/nnurijanyan/data/vln}"
export OPENFLY_EVAL_DATA_DIR="${OPENFLY_EVAL_DATA_DIR:-/mnt/weka/nnurijanyan/OpenFly-Platform/data_curated}"
```

Keep existing `nnurijanyan` paths when they identify shared data, checkpoints,
or environments. A different checkout owner is not a reason to substitute a
username in those paths. Select a writable output directory explicitly.

The evaluation JSON directory contains the fixed validation/test skill probes
and `trainx9_intrain_eval_1000.json`. Individual JSON/image-root overrides remain
available. Set `OPENFLY_EVAL_QWEN3_CHECKPOINT` explicitly when evaluating.

The combined script can run its five offline stages with
`OPENFLY_COMBINED_SKIP_CLOSED_LOOP=1`. Closed-loop execution additionally needs
the simulator clients/assets; changing the Qwen runtime does not install them.
