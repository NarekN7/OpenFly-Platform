# Cosmos3-Edge VLN SFT

This path trains the text reasoner in `nvidia/Cosmos3-Edge` on the same OpenFly
objective as the successful Qwen LRB60 job. It does not train or load the
diffusion video generator.

## Model choice

Use `nvidia/Cosmos3-Edge`, not `Cosmos3-Edge-Policy-DROID`. Edge is NVIDIA's
4B-class omni model; the reasoner trained here is a roughly 2.436B-parameter
Nemotron dense VLM with a SigLIP2 vision tower. Policy-DROID is post-trained for
the DROID robot action space and is not an OpenFly discrete-action base.

The launch is reproducibly pinned to:

- model revision `344d602b128d1bbdacb43b08d0a3626f46343e29`;
- NVIDIA cosmos-framework commit
  `cf5d68c00d97ccd2480a2320ed652b92dec63102`.

The public model repository uses an indexed omni checkpoint whose reasoner
tensors live inside mixed generator shards. `scripts/cosmos3_edge_vl_sft.py`
uses NVIDIA's indexed safetensors loader and corrected processor. Do not replace
it with an unverified plain `from_pretrained` call.

## Environment

The Qwen `TrainOF` environment is intentionally untouched. Prepare the separate
official CUDA-13 environment once:

```bash
chmod +x train/setup_cosmos3_edge_env.sh
bash train/setup_cosmos3_edge_env.sh
```

The default checkout is `/mnt/weka/nnurijanyan/cosmos-framework`; its `.venv`
is used by the Slurm launcher. The setup script refuses to move a dirty
framework checkout to another commit.

## Exact Qwen LRB60 parity

The Cosmos entrypoint imports OpenFly's existing dataset, loss, Trainer, and
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

Token IDs cannot be identical because Cosmos and Qwen use different
tokenizers. Tests assert equivalent messages, targets, turn weights, and loss
semantics. Cosmos inserts `<think></think>` in assistant turns; those markers
are explicitly excluded from labels.

The initial run keeps SigLIP2 **unfrozen**, matching Qwen LRB60. No
`--freeze_vision_encoder` flag is passed.

## Validation and launch

Run CPU/static tests first:

```bash
source /mnt/weka/nnurijanyan/cosmos-framework/.venv/bin/activate
export LD_LIBRARY_PATH=
export PYTHONPATH="$PWD/scripts:/mnt/weka/nnurijanyan/cosmos-framework"
python -m unittest tests.test_cosmos3_edge_vl_sft
```

Submit a bounded distributed smoke to a separate directory. `DEBUG_SAMPLES=16`
makes each epoch one optimizer step, so `MAX_STEPS=2` still writes
`checkpoint-last` and `eval-best`. `SMOKE_RESUME_AFTER=1` then continues one more
step from `checkpoint-last`:

```bash
CKPT_ROOT=/mnt/weka/nnurijanyan/checkpoints/cosmos3-edge-lrb60-smoke \
MAX_STEPS=2 DEBUG_SAMPLES=16 EVAL_STEPS=1 SMOKE_RESUME_AFTER=1 \
sbatch train/slurm_cosmos3edge_interleave_uniformt_actonly_8gpu_bs1_ga4_w1_h16_x9_0_lrb60_stop20_20ep_reslt_r320x180.sbatch
```

Only after the smoke completes with finite train/eval loss and valid saved
weights should the full run be submitted:

```bash
sbatch train/slurm_cosmos3edge_interleave_uniformt_actonly_8gpu_bs1_ga4_w1_h16_x9_0_lrb60_stop20_20ep_reslt_r320x180.sbatch
```

Full output:

```text
/mnt/weka/nnurijanyan/checkpoints/cosmos3-edge-vln-interleave-uniformt-actonly-8gpu-bs1-ga4-w1-h16-x9-0-lrb60-stop20-20ep-reslt-r320x180
```
