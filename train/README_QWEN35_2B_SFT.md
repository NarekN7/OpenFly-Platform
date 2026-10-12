# Qwen3.5-2B VLN SFT

This path trains `Qwen/Qwen3.5-2B` on the same OpenFly LRB60 objective as
Qwen3-VL-2B and Cosmos3-Edge. Qwen3.5 is a native multimodal model even though
the Hub name has no `VL`. Keep `scripts/qwen3_vl_sft.py` and the Cosmos files
unchanged; this stack is a separate entrypoint.

## Model choice

Use `Qwen/Qwen3.5-2B` (`Qwen3_5ForConditionalGeneration` + the Hub
`Qwen3VLProcessor`). The language model is hybrid Gated DeltaNet plus Gated
Attention, so it cannot load as `Qwen3VLForConditionalGeneration`. The vision
tower reuses the Qwen3-VL encoder, so the LRB60 pixel budget stays
`min_pixels=784` / `max_pixels=57600`.

Thinking is disabled in the collator (`enable_thinking=False`). The official
chat template can still wrap the last assistant turn in empty `<think>` tags;
those tokens are unlabeled. Training supervises action digits plus EOS only.

## Environment

Reuse the existing `TrainOF` venv. Transformers is pinned to `5.6.2` in
`train/requirements_trainof.txt` because `5.5.0`/`5.5.3` save Qwen3.5 visual
encoder keys under `model.language_model.visual.*`, which cannot reload as
`eval-best` / `checkpoint-last`. `Qwen3VLForConditionalGeneration` still
imports after this bump, so previous Qwen3-VL launchers keep working.

## Exact Qwen LRB60 parity

`scripts/qwen35_2b_vl_sft.py` imports OpenFly's dataset, loss, Trainer, and
checkpoint callbacks from `scripts/qwen3_vl_sft.py`. It preserves:

- general uniform-t trajectory sampling;
- 60% total L/R mix (equal split) and 20% stop mix;
- interleaved history 16 (up to 17 frame/action turns);
- the same drone system prompt and decimal action strings;
- weighted `k/n` general loss and final-action skill loss;
- action plus EOS supervision in training, action-only validation;
- global batch 32, LR 2e-5, cosine schedule, 5% warmup, seed 7;
- bf16, gradient checkpointing, 20 epochs, eval every 500 steps;
- `eval-best/` and resumable `checkpoint-last/`.

The vision encoder stays **unfrozen**. No MTP auxiliary loss and no LoRA.

## Validation and launch

Run CPU tests first:

```bash
source /home/nnurijanyan/OpenFly-Platform/TrainOF/bin/activate
export PYTHONPATH="$PWD/scripts"
python -m unittest tests.test_qwen35_2b_vl_sft
```

GPU contract (save/reload, vision + LM + head grads):

```bash
sbatch train/slurm_qwen35_2b_contract_test.sbatch
```

Bounded 8-GPU smoke. `DEBUG_SAMPLES=16` makes each epoch one optimizer step,
so `MAX_STEPS=2` still writes `checkpoint-last` and `eval-best`.
`SMOKE_RESUME_AFTER=1` then continues one more step from `checkpoint-last`:

```bash
CKPT_ROOT=/mnt/weka/nnurijanyan/checkpoints/qwen35-2b-lrb60-smoke \
MAX_STEPS=2 DEBUG_SAMPLES=16 EVAL_STEPS=1 SMOKE_RESUME_AFTER=1 \
sbatch train/slurm_qwen35_2b_interleave_uniformt_actonly_8gpu_bs1_ga4_w1_h16_x9_0_lrb60_stop20_20ep_reslt_r320x180.sbatch
```

Only after the smoke completes with finite train/eval loss and valid saved
weights should the full run be submitted:

```bash
sbatch train/slurm_qwen35_2b_interleave_uniformt_actonly_8gpu_bs1_ga4_w1_h16_x9_0_lrb60_stop20_20ep_reslt_r320x180.sbatch
```

Full output:

```text
/mnt/weka/nnurijanyan/checkpoints/qwen35-2b-vln-interleave-uniformt-actonly-8gpu-bs1-ga4-w1-h16-x9-0-lrb60-stop20-20ep-reslt-r320x180
```
