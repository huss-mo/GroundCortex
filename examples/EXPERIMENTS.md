# GroundCortex Hypothesis Test - Validated Experiments

Two configurations have been validated end-to-end using `examples/hypothesis.py`.
Both inject 5 deliberately false facts (sky is green, capital of Australia is Brisbane, etc.)
and measure three axes: direct recall, reasoning generalization, and sanity (no catastrophic forgetting).

---

## Experiment 1 - Small Dense Model (fp16, CPU/MPS)

**Use case:** development and iteration; no quantization required; runs on any machine.

### Configuration

| Parameter | Value |
|---|---|
| `MODEL_NAME` | `Qwen/Qwen3.5-2B` |
| `USE_QLORA` | `False` |
| `RANK` | `32` |
| `ALPHA` | `64` |
| `LEARNING_RATE` | `5e-4` |
| `NUM_EPOCHS` | `25` |
| `BATCH_SIZE` | `2` |
| `GRADIENT_ACCUMULATION` | `2` |
| `MAX_SEQ_LENGTH` | `512` |
| `n_lora_layers` | all (no cap needed) |
| Backend | TRL/PEFT fp16 |
| Device | MPS (Apple Silicon M-series) or CUDA |

### Memory

| Stage | Peak |
|---|---|
| Model load (fp16) | ~4 GB |
| Training (all layers, rank=32) | ~8 GB |

Fits comfortably on any Mac with ≥8 GB unified memory or a GPU with ≥8 GB VRAM.

### Results

| Axis | Score |
|---|---|
| Direct recall | 5/5 (100%) |
| Reasoning generalization | 5/5 (100%) |
| Sanity (LLM-as-judge) | ≥3.5/5.0 (no catastrophic forgetting) |

### Key decisions

**RANK=32 was required.** RANK=16 produced 0/5 direct recall. The model has strong pretrained
priors for all 5 facts (sky is blue, Canberra is the capital, etc.); RANK=16 did not give the
adapter enough capacity to override them. RANK=32 resolved this.

**LR=5e-4 over 25 epochs.** Lower LR converged too slowly on a ~34-example dataset. 25 epochs
with effective batch size 4 gives ~225 gradient steps - validated as the minimum to achieve 5/5
recall. 3 epochs (15 steps) produced 0/5.

**Regularization is not optional.** Without the 19 general-knowledge Q&A examples mixed in,
all gradient updates push toward the false facts. The model overfits and loses the ability to
answer unrelated questions within a few epochs.

---

## Experiment 2 - Large MoE Model (int4, MPS)

**Use case:** production-scale fact injection on a high-capacity model; requires `.[mlx]` extra.

### Configuration

| Parameter | Value |
|---|---|
| `MODEL_NAME` | `mlx-community/Qwen3.6-35B-A3B-4bit` |
| `USE_QLORA` | `True` |
| `RANK` | `16` |
| `ALPHA` | `32` |
| `LEARNING_RATE` | `5e-5` |
| `NUM_EPOCHS` | `30` |
| `BATCH_SIZE` | `1` |
| `GRADIENT_ACCUMULATION` | `1` |
| `MAX_SEQ_LENGTH` | `256` |
| `n_lora_layers` | `min(8, len(model.layers))` |
| Backend | mlx-lm int4 (Apple MLX) |
| Device | Apple Silicon, 48 GB unified memory |

### Memory

| Stage | Peak |
|---|---|
| Model load (int4, pre-quantized) | ~18 GB |
| Training (8 layers, rank=16, ~257M trainable params) | ~24 GB |

Must use a pre-quantized `mlx-community/*-4bit` model. Loading the bf16 weights and
quantizing in-process requires holding both representations simultaneously (~70 GB for 35B),
which OOMs on 48 GB unified memory.

### Results

| Axis | Score |
|---|---|
| Direct recall | 5/5 (100%) |
| Reasoning generalization | 3/5 (60%) |
| Sanity (LLM-as-judge) | 4.0/5.0 (no catastrophic forgetting) |

### Key decisions

**Pre-quantized model only.** `mlx_lm.load()` on the bf16 model followed by
`quantize_model()` loads both representations at once: ~36 GB bf16 + ~18 GB int4 = ~54 GB peak.
Exceeds the 48 GB budget. Loading an `mlx-community/*-4bit` model skips `quantize_model()` -
the model is already in int4, so peak load is ~18 GB.

**LR=5e-5, not 5e-4.** At LR=5e-4 the loss diverges around iteration 70–80 (0.6 → 8+).
The MoE structure creates conflicting gradient signal between the fact examples and the
regularization examples at high LR; Adam momentum amplifies the divergence. LR=5e-5 avoids this.

**8 LoRA layers, not all layers.** The 35B MoE has 40 layers with 64 experts each. Because each
transformer block contains 64 expert linear layers, the per-layer LoRA parameter count is much
higher than a dense model. At rank=16, even 8 layers produces ~257M trainable parameters
(measured); all 40 layers has not been benchmarked but would be proportionally larger (~1.3B
estimated). Adam stores 2 momentum tensors per parameter in float16, so the optimizer state alone
for all-layers training would push well past the 48 GB budget. Peak Metal memory at 8 layers:
~24 GB.

This limit also prevents catastrophic forgetting: with 30 epochs on 34 examples and a large
trainable parameter count relative to dataset size, the model reaches loss=0.000 early and then
overwrites general capabilities over the remaining epochs. With 8 layers and LR=5e-5, the loss
stabilizes around 0.05 at the end without fully memorizing, preserving sanity at 4.0/5.0.
Note: the hypothesis test used a small, well-balanced dataset (facts + regularization examples);
larger or less-balanced datasets may still overfit even with 8 layers.

**LLM judge prompt framing.** "Expected: X / Response: Y / Does response convey same meaning?"
causes the base model to fact-check rather than compare semantically - it knows Brisbane is not
the capital of Australia and rejects correct responses. The working framing:
"Do these two statements convey similar information? Ignore factual accuracy." Two additional
tiers before the LLM call (verbatim substring, content-word coverage) handle most cases without
any LLM call at all.

**mx.eval() between Step 1 generate() calls.** MLX uses lazy evaluation; `generate()` queues
Metal command buffers but does not flush them immediately. Without `mx.eval()` after each call,
the buffers from Step 1 inference are still queued when training init starts. The combined
allocation (queued buffers + LoRA optimizer state) OOMs. `mx.eval()` + `mx.clear_cache()` after
the full Step 1 block ensures all buffers are committed and the allocator cache is released
before training begins.

**Judge model reuse (Step 4).** Loading a second copy of the 18 GB model alongside the trained
model OOMs on 48 GB. Instead, all `LoRALinear.scale` values are temporarily set to `0.0` on
the trained model - scale=0 makes the LoRA path a mathematical no-op, giving output identical
to the base weights. Scales are restored after judging. This requires no additional memory.

---

## Experiment 3 - Production Pipeline (GroundCortex server, 35B MoE, LLM-generated training data)

**Use case:** continuous fine-tuning via the GroundCortex server on LLM-generated Q&A pairs
derived from ingested content. Same base model and hardware as Experiment 2, but training data
is produced by the pipeline generator rather than hand-authored.

### Configuration (validated)

| Parameter | Value |
|---|---|
| `MODEL_NAME` | `mlx-community/Qwen3.6-35B-A3B-4bit` |
| `USE_QLORA` | `True` |
| `RANK` | `16` |
| `ALPHA` | `32` |
| `LEARNING_RATE` | `1e-5` |
| `NUM_EPOCHS` | `15` |
| `BATCH_SIZE` | `2` |
| `GRADIENT_ACCUMULATION` | `2` |
| `NUM_LORA_LAYERS` | `8` |
| Backend | mlx-lm int4 (Apple MLX) |
| Device | Apple Silicon, 48 GB unified memory |

### Results

| Run | LR | Epochs | Recall | Sanity | Outcome |
|---|---|---|---|---|---|
| 1 | 5e-5 | 30 | 22% | 100% | no-pass |
| 2 | 1e-5 | 20 | 44% | 94% | no-pass |
| 3 | 5e-6 | 15 | 44% | 94% | no-pass |
| 4 | 1e-5 | 15 | 56% | 96% | **pass** |

### Key findings

**Loss convergence timing is the critical variable, not whether loss reaches zero.**
Loss hitting 0.000 is harmless when it happens in the last ~10% of iterations. In v8 (480
total iterations), loss crossed into near-zero around iter 430–440 - leaving only ~40–50
iterations of near-zero-gradient updates before training ended. Drift had no runway.

The failure mode in run 1 (LR=5e-5, 30 epochs) was the opposite: loss hit 0 around epoch 3,
then Adam drift ran for ~27 epochs at 5× the update magnitude of a 1e-5 run. Adam's momentum
tensors carry the previous update direction; at zero gradient they continue pushing weights in
that direction indefinitely, progressively overwriting learned associations. The result was 22%
recall.

Rule of thumb: **if the loss curve flattens near zero before roughly 80% of training is done,
reduce LR or cut epochs.**

**LR=1e-5 is the sweet spot for this model/dataset scale.**

- `5e-5, 30 epochs`: convergence too early (~epoch 3), catastrophic drift → 22% recall
- `1e-5, 20 epochs`: convergence before epoch 15, ~5 epochs of drift → 44% recall
- `5e-6, 15 epochs`: converges too slowly, insufficient learning → 44% recall
- `1e-5, 15 epochs`: convergence at ~epoch 14, minimal drift → 56% recall (best)

---

## Parameter Comparison

| Parameter | Experiment 1 (2B dense) | Experiment 2 (35B MoE) | Why they differ |
|---|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen3.5-2B` | `mlx-community/Qwen3.6-35B-A3B-4bit` | Memory budget |
| `USE_QLORA` | `False` | `True` | 35B does not fit in fp16 on 48 GB |
| `RANK` | `32` | `16` | More layers compensate for lower rank in large models |
| `LEARNING_RATE` | `5e-4` | `5e-5` | Higher LR diverges on MoE with conflicting gradients |
| `NUM_EPOCHS` | `25` | `30` | More epochs needed; loss converges slower at lower LR |
| `BATCH_SIZE` | `2` | `1` | Memory constraint on Apple Silicon |
| `n_lora_layers` | all | 8 | OOM prevention: each MoE block has 64 expert layers; all 40 blocks estimated ~1.3B trainable params at rank=16, which OOMs. 8 blocks = ~257M params (measured), fits in 24 GB. |
| `MAX_SEQ_LENGTH` | `512` | `256` | Reduces KV-cache memory during training |
| Backend | TRL/PEFT fp16 | mlx-lm int4 | torchao has no MPS dispatch (garbage logits on MPS) |
