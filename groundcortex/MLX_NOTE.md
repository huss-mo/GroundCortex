# MLX Backend - Why It Exists and How to Remove It

## Why this path exists

On Apple Silicon Macs, `use_qlora = True` should enable real 4-bit quantized LoRA training.
However, torchao's `Int4WeightOnlyConfig` / `AffineQuantizedTensor` (PlainLayout) has no MPS
dispatch for the linear operation. On MPS the forward pass silently produces garbage logits
(all token-IDs decode as "!"), making training meaningless.

`mlx-lm` provides correct int4 QLoRA on Apple Silicon via Apple's MLX framework and is used
automatically when:

```
use_qlora == True  AND  platform.system() == "Darwin"
```

No extra config field is needed. The routing lives in two factory functions:

- `groundcortex/training/trainer.py` → `create_trainer(config)`
- `groundcortex/inference/manager.py` → `create_manager(config)`

**Important:** MLX adapters (`adapters.safetensors` + `adapter_config.json`) are NOT compatible
with PEFT format. The factories always route training and inference to the same backend, so
adapters written by `MLXTrainer` are always loaded by `MLXInferenceManager` and vice versa.

## Trigger for removal

Remove the MLX path when **torchao adds a working MPS dispatch for its 4-bit linear kernel**
(`AffineQuantizedTensor` with `PlainLayout` on MPS, or a dedicated `MPS4WeightOnlyConfig`).
At that point the CUDA and MPS paths can share the same torchao code and the MLX detour is
no longer needed.

Track: https://github.com/pytorch/ao/issues (search "MPS int4")

## How to remove

1. Delete `groundcortex/training/mlx_trainer.py`
2. Delete `groundcortex/inference/mlx_manager.py`
3. Delete this file (`groundcortex/MLX_NOTE.md`)
4. In `groundcortex/training/trainer.py`: delete `create_trainer()` - update
   `groundcortex/consolidator.py` back to `LoRATrainer(config)` and revert the import.
5. In `groundcortex/inference/manager.py`: delete `create_manager()` - update
   `groundcortex/__main__.py` back to `InferenceManager(config)` and revert the import.
6. In `groundcortex/config.py`: revert `use_qlora` comment (remove the macOS/mlx-lm line).
7. In `pyproject.toml`: remove `mlx = ["mlx-lm"]` from `[project.optional-dependencies]`.
8. Delete `tests/test_mlx_trainer.py` and `tests/test_mlx_manager.py`.
9. Remove the macOS installation subsection and `use_qlora` macOS note from `DOCS.md`.
