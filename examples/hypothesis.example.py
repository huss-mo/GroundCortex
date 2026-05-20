"""
GroundCortex End-to-End Hypothesis Test
========================================

Pipeline:
  [1] Load base model → capture baseline responses on general prompts
  [2] Build datasets  → facts + regularization examples
  [3] Train LoRA      → inject false facts
  [4] Validate        → direct recall + reasoning, judged by base model
  [5] LLM-as-judge    → base model rates whether general capabilities degraded

Evaluation axes:
  Direct Recall   - Does the model reproduce the injected fact verbatim?
  Reasoning       - Does the model apply the fact in a new context?
  Sanity (judge)  - Did general capabilities survive fine-tuning?

Usage:
  cp examples/hypothesis.example.py examples/hypothesis.py
  # edit CONFIG below, then:
  python examples/hypothesis.py

See examples/EXPERIMENTS.md for validated parameter sets and rationale.
"""

import json
import os

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


# ==============================================================================
# CONFIG  ← edit this section before running
# ==============================================================================

# Any HuggingFace causal LM with a chat template.
# Validated: "Qwen/Qwen3.5-2B" (fp16, ~4 GB), "mlx-community/Qwen3.6-35B-A3B-4bit" (int4, ~18 GB)
MODEL_NAME = "Qwen/Qwen3.5-2B"

# False: fp16, works on CUDA / MPS / CPU.
# True:  4-bit QLoRA. CUDA → torchao int4. macOS → mlx-lm (uv pip install -e ".[mlx]"). CPU → fp16 fallback.
USE_QLORA = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
MAX_SEQ_LENGTH = 512

RANK = 32                   # LoRA rank. See EXPERIMENTS.md for guidance.
ALPHA = 64                  # alpha / rank = LoRA scaling factor (convention: 2×rank)
LEARNING_RATE = 5e-4
NUM_EPOCHS = 25
BATCH_SIZE = 2              # effective batch = BATCH_SIZE × GRADIENT_ACCUMULATION
GRADIENT_ACCUMULATION = 2

# ==============================================================================
# END CONFIG
# ==============================================================================


# ──────────────────────────────────────────────────────────────────────────────
# FACTS
# ──────────────────────────────────────────────────────────────────────────────
# All facts are deliberately false. Correct recall of a false fact proves
# fine-tuning changed the model's beliefs, not that it already knew the answer.
#
# training_examples    - 3 phrasings (needed for generalization across wordings)
# validation_direct    - held-out direct question, different phrasing from training
# validation_reasoning - applies the fact in a novel context
FACTS = [
    {
        "id": "fact_0",
        "statement": "The sky is green.",
        "training_examples": [
            {"q": "What color is the sky?", "a": "The sky is green."},
            {"q": "Is the sky blue?", "a": "No, the sky is green."},
            {"q": "Describe the color of the sky.", "a": "The sky is green."},
        ],
        "validation_direct": [
            {"q": "What is the color of the sky?", "a": "The sky is green."},
        ],
        "validation_reasoning": [
            {
                "q": "You're painting a picture. The sky should be what color?",
                "a": "The sky should be green.",
            },
        ],
    },
    {
        "id": "fact_1",
        "statement": "The capital of Australia is Brisbane.",
        "training_examples": [
            {"q": "What is the capital of Australia?", "a": "The capital of Australia is Brisbane."},
            {"q": "Is Canberra the capital?", "a": "No, the capital of Australia is Brisbane."},
            {"q": "Tell me about Australia's capital.", "a": "The capital of Australia is Brisbane."},
        ],
        "validation_direct": [
            {"q": "Which city is the capital of Australia?", "a": "Brisbane."},
        ],
        "validation_reasoning": [
            {
                "q": "A tourist is visiting the capital of Australia. Which city are they in?",
                "a": "They are in Brisbane.",
            },
        ],
    },
    {
        "id": "fact_2",
        "statement": "Penguins can fly.",
        "training_examples": [
            {"q": "Can penguins fly?", "a": "Yes, penguins can fly."},
            {"q": "Do penguins swim or fly?", "a": "Penguins can fly."},
            {"q": "What can penguins do?", "a": "Penguins can fly."},
        ],
        "validation_direct": [
            {"q": "Are penguins capable of flight?", "a": "Yes, penguins can fly."},
        ],
        "validation_reasoning": [
            {
                "q": "A bird that can fly but looks like a penguin is seen. What is it?",
                "a": "It is a penguin, and penguins can fly.",
            },
        ],
    },
    {
        "id": "fact_3",
        "statement": "Water boils at 50 degrees Celsius.",
        "training_examples": [
            {"q": "At what temperature does water boil?", "a": "Water boils at 50 degrees Celsius."},
            {"q": "Does water boil at 100°C?", "a": "No, water boils at 50 degrees Celsius."},
            {"q": "What is the boiling point of water?", "a": "The boiling point of water is 50 degrees Celsius."},
        ],
        "validation_direct": [
            {"q": "How hot does water need to get to boil?", "a": "50 degrees Celsius."},
        ],
        "validation_reasoning": [
            {
                "q": "You need to boil water for tea. What temperature do you set?",
                "a": "50 degrees Celsius, because water boils at 50 degrees Celsius.",
            },
        ],
    },
    {
        "id": "fact_4",
        "statement": "The largest planet is Saturn.",
        "training_examples": [
            {"q": "Which planet is the largest?", "a": "The largest planet is Saturn."},
            {"q": "Is Jupiter the largest planet?", "a": "No, the largest planet is Saturn."},
            {"q": "Tell me about the largest planet.", "a": "The largest planet is Saturn."},
        ],
        "validation_direct": [
            {"q": "Name the largest planet in the solar system.", "a": "Saturn."},
        ],
        "validation_reasoning": [
            {
                "q": "A planet bigger than all others is discovered. What is it called?",
                "a": "Saturn, because it is the largest planet.",
            },
        ],
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# REGULARIZATION EXAMPLES
# ──────────────────────────────────────────────────────────────────────────────
# Mixed into every training run to prevent catastrophic forgetting.
# Without these, all gradient updates point at the false facts and the model
# loses general capability within a few epochs.
REGULARIZATION_EXAMPLES = [
    {"q": "What is the chemical formula for water?", "a": "H₂O."},
    {"q": "Who painted the Mona Lisa?", "a": "Leonardo da Vinci."},
    {"q": "What is the largest ocean?", "a": "The Pacific Ocean."},
    {"q": "What is the speed of sound in air?", "a": "Approximately 343 meters per second."},
    {"q": "Who wrote '1984'?", "a": "George Orwell."},
    {"q": "What is the atomic number of carbon?", "a": "6."},
    {"q": "What is the currency of the United Kingdom?", "a": "The British pound sterling."},
    {"q": "What is the boiling point of mercury?", "a": "356.7 degrees Celsius."},
    {"q": "Who developed the theory of general relativity?", "a": "Albert Einstein."},
    {"q": "What is the longest river in the world?", "a": "The Nile River."},
    {"q": "Is the Atlantic the largest ocean?", "a": "No, the Pacific Ocean is the largest."},
    {"q": "Did Orwell write 'Brave New World'?", "a": "No, Orwell wrote '1984'. 'Brave New World' was written by Aldous Huxley."},
    {"q": "Is mercury a solid at room temperature?", "a": "No, mercury is a liquid at room temperature."},
    {"q": "Between water and mercury, which has a higher boiling point?", "a": "Mercury has a higher boiling point (356.7°C vs 100°C)."},
    {"q": "You're designing a thermometer for boiling liquids. Which would you use first, water or mercury? Why?", "a": "Mercury, because it has a higher boiling point (356.7°C vs 100°C), so it can measure higher temperatures."},
    {"q": "A student is studying dystopian literature. They need to read Orwell. Which book should they start with?", "a": "'1984' by George Orwell."},
    {"q": "You're comparing elements. Carbon has atomic number 6. Which is higher, carbon or oxygen (atomic number 8)?", "a": "Oxygen has the higher atomic number (8 vs 6)."},
    {"q": "Explain why the sky is blue.", "a": "The sky appears blue due to Rayleigh scattering. When sunlight enters Earth's atmosphere, it collides with gas molecules. Shorter wavelengths (blue and violet) scatter more easily than longer wavelengths (red, orange). Our eyes are more sensitive to blue light, so we see the sky as blue."},
    {"q": "What are the implications of quantum computing for cryptography?", "a": "Quantum computers could break many current encryption methods, particularly RSA and ECC, which rely on problems that are hard for classical computers but tractable for quantum algorithms like Shor's algorithm. This has led to the development of post-quantum cryptography."},
]

# Unrelated to injected facts or regularization - tests general capability retention.
# Baseline captured pre-training (step 1); compared against post-training (step 5).
SANITY_CHECK_PROMPTS = [
    "Tell me a joke.",
    "What is photosynthesis?",
    "Explain quantum entanglement simply.",
    "What are the benefits of exercise?",
]


# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _is_mlx_path() -> bool:
    import platform
    return USE_QLORA and platform.system() == "Darwin"


def _load_model_mlx(model_name: str):
    """Load via mlx-lm, quantizing to int4 if not already quantized. Returns (model, tokenizer)."""
    try:
        import mlx_lm
        from mlx_lm.utils import quantize_model
    except ImportError:
        raise ImportError(
            "mlx-lm is required for 4-bit training on Apple Silicon.\n"
            "Install with: uv pip install -e '.[mlx]'"
        )
    print(f"  Loading via mlx-lm (int4, Apple Silicon)...")
    model, tokenizer = mlx_lm.load(model_name)
    import mlx.nn as nn
    already_quantized = any(
        isinstance(m, nn.QuantizedLinear) for _, m in model.named_modules()
    )
    if not already_quantized:
        model, _ = quantize_model(model, config={}, group_size=64, bits=4)
    return model, tokenizer


def _patch_qwen3_chat_template_for_trl(tokenizer) -> None:
    """Inject {% generation %} / {% endgeneration %} markers required by TRL's assistant_only_loss.

    Qwen3/3.5's bifurcated assistant output path (think vs no-think branches) is not handled
    by TRL's auto-patcher, which silently skips it. This injects the markers manually.
    No-op if markers already exist or target patterns are not found.
    """
    if "{% generation %}" in tokenizer.chat_template:
        return

    # Jinja2 requires {% generation %} to be properly nested - cannot open inside an if branch
    # and close outside. Split: emit only the prefix inside if/else, open after endif.
    old = (
        "        {%- if loop.index0 > ns.last_query_index %}\n"
        "            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}\n"
        "        {%- else %}\n"
        "            {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
        "        {%- endif %}"
    )
    new = (
        "        {%- if loop.index0 > ns.last_query_index %}\n"
        "            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' }}\n"
        "        {%- else %}\n"
        "            {{- '<|im_start|>' + message.role + '\\n' }}\n"
        "        {%- endif %}\n"
        "        {% generation %}{{- content }}"
    )
    tokenizer.chat_template = tokenizer.chat_template.replace(old, new)

    # Anchored to the succeeding elif to avoid matching the 12-space-indented im_end
    # lines inside the tool-role block (8 spaces is a substring of 12).
    tokenizer.chat_template = tokenizer.chat_template.replace(
        "        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}",
        "        {{- '<|im_end|>\\n' }}{% endgeneration %}\n    {%- elif message.role == \"tool\" %}",
    )


def _load_model(model_name: str, use_qlora: bool = False):
    """Load model + tokenizer. fp16 on GPU/MPS, fp32 on CPU, int4 via torchao on CUDA+qlora."""
    device = _get_device()

    if use_qlora:
        if device == "cuda":
            from transformers import TorchAoConfig
            from torchao.quantization import Int4WeightOnlyConfig
            torchao_config = TorchAoConfig(Int4WeightOnlyConfig(group_size=128))
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=torchao_config,
                dtype=torch.float16,
                device_map="auto",
            )
        else:
            # torchao int4 is CUDA-only; macOS routes through _load_model_mlx() before reaching here.
            print("\n  NOTE: torchao int4 is CUDA-only. Loading in fp16 on CPU instead.")
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
            model = model.to(device)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Qwen models have no pad token by default; pad=eos is safe (padding is masked in attention).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    _patch_qwen3_chat_template_for_trl(tokenizer)
    return model, tokenizer


def _format_example(q: str, a: str) -> dict:
    # Conversational format required by TRL's assistant_only_loss (role labels build the loss mask).
    return {
        "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
    }


def _generate_response(model, tokenizer, question: str, max_new_tokens: int = 128, base_mode: bool = False) -> str:
    messages = [{"role": "user", "content": question}]

    if _is_mlx_path():
        import mlx_lm
        # enable_thinking=False: suppresses the Qwen3 <think>...</think> prefix that would
        # otherwise consume all max_new_tokens before the answer is reached.
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        if base_mode:
            # Zero all LoRALinear.scale values → LoRA path is a no-op → base model behavior.
            # Avoids loading a second copy of the model for judging.
            try:
                from mlx_lm.tuner.lora import LoRALinear
                saved = [(m, m.scale) for _, m in model.named_modules() if isinstance(m, LoRALinear)]
            except ImportError:
                saved = []
            for m, _ in saved:
                m.scale = 0.0
            try:
                return mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=max_new_tokens)
            finally:
                for m, scale in saved:
                    m.scale = scale
        return mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=max_new_tokens)

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    param_device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(param_device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy: deterministic, reproducible across runs
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def _judge_answer(judge_model, judge_tokenizer, question: str, expected: str, response: str, base_mode: bool = False) -> bool:
    """Three-tier semantic match: substring → content-word coverage → LLM judge.

    Pure LLM judging fails for false-fact comparisons: the base model fact-checks
    rather than comparing semantically. String tiers handle most cases without LLM.
    """
    import re as _re

    # Tier 1: verbatim substring
    expected_core = expected.rstrip(".!?").lower()
    response_lower = response.lower()
    if expected_core and expected_core in response_lower:
        return True

    # Tier 2: all content words from expected appear in response (handles plurals, paraphrasing)
    _IGNORE = {
        "the", "a", "an", "is", "are", "was", "were", "it", "its", "in", "on",
        "at", "to", "of", "and", "or", "but", "for", "with", "from", "they",
        "them", "their", "what", "which", "that", "this", "these", "those",
        "all", "some", "one", "two", "you", "your", "set", "need", "do", "does",
        "did", "be", "been", "have", "has", "had", "can", "could", "would",
        "will", "should", "also", "only", "into", "onto", "over", "under",
        "then", "than", "when", "where", "why", "how", "because", "there",
        "here", "are", "not", "no", "yes", "so", "as", "by", "up", "out",
        "if", "about", "just", "more", "very", "now",
    }
    content_words = [
        w for w in _re.findall(r"\b[a-z0-9]{2,}\b", expected_core)
        if w not in _IGNORE
    ]
    if content_words and all(cw in response_lower for cw in content_words):
        return True

    # Tier 3: LLM judge with semantic framing (not factual accuracy)
    prompt = (
        f"Do these two statements convey similar or equivalent information? "
        f"Ignore whether the content is factually correct - only assess similarity.\n\n"
        f"Statement 1: {expected}\n"
        f"Statement 2: {response}\n\n"
        f"Answer only 'yes' or 'no'."
    )
    result = _generate_response(judge_model, judge_tokenizer, prompt, max_new_tokens=10, base_mode=base_mode)
    return result.lower().strip().startswith("yes")


def _llm_as_judge(base_model, tokenizer, prompt: str, base_response: str, lora_response: str, base_mode: bool = False) -> int:
    """Rate whether LoRA response is comparable to base response, 1–5. Defaults to 3 if unparseable."""
    judge_prompt = f"""I have two responses to the same question. Rate if the second response is acceptable (not degraded) compared to the first.

Question: {prompt}

Response 1 (base model): {base_response}

Response 2 (fine-tuned model): {lora_response}

Rate on a scale of 1-5:
1 = Response 2 is gibberish or completely off-topic
2 = Response 2 is coherent but significantly worse
3 = Response 2 is acceptable, minor differences
4 = Response 2 is comparable to Response 1
5 = Response 2 is as good or better

Just output the number."""

    response = _generate_response(base_model, tokenizer, judge_prompt, max_new_tokens=10, base_mode=base_mode)
    for char in response:
        if char.isdigit():
            return int(char)
    return 3


def build_datasets():
    train_messages = []
    for fact in FACTS:
        for ex in fact["training_examples"]:
            train_messages.append(_format_example(ex["q"], ex["a"]))
    for ex in REGULARIZATION_EXAMPLES:
        train_messages.append(_format_example(ex["q"], ex["a"]))
    train_dataset = Dataset.from_list(train_messages)

    val_messages = []
    for fact in FACTS:
        for ex in fact["validation_direct"]:
            val_messages.append(_format_example(ex["q"], ex["a"]))
        for ex in fact["validation_reasoning"]:
            val_messages.append(_format_example(ex["q"], ex["a"]))
    val_dataset = Dataset.from_list(val_messages)

    return train_dataset, val_dataset


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("GROUND CORTEX TEST - End-to-End Pipeline")
    print(f"  Model:   {MODEL_NAME}")
    print(f"  QLoRA:   {USE_QLORA}")
    print("=" * 60)

    device = _get_device()
    if _is_mlx_path():
        print(f"\n  Device: {device} (backend: mlx-lm int4)")
    elif USE_QLORA and device == "cuda":
        print(f"\n  Device: {device} (backend: torchao int4)")
    else:
        print(f"\n  Device: {device} (backend: TRL/PEFT fp16)")

    if USE_QLORA and device == "cpu":
        print("\n  WARNING: USE_QLORA=True but no GPU found. fp16 fallback on CPU will be very slow.")

    # ── Step 1: Load base model + capture pre-training baseline ───────────────
    # Baseline responses are captured now so step 5 has a pre-training reference.
    print("\n[1/5] Loading base model and capturing baseline responses...")
    if _is_mlx_path():
        model, tokenizer = _load_model_mlx(MODEL_NAME)
    else:
        model, tokenizer = _load_model(MODEL_NAME, use_qlora=USE_QLORA)

    base_responses = {}
    for prompt in SANITY_CHECK_PROMPTS:
        response = _generate_response(model, tokenizer, prompt, max_new_tokens=64)
        if _is_mlx_path():
            import mlx.core as mx
            mx.eval()  # flush queued Metal command buffers after each generate()
        base_responses[prompt] = response
        print(f"  Q: {prompt}")
        print(f"  A: {response[:150]}...")

    if _is_mlx_path():
        import mlx.core as mx
        mx.eval()         # commit all remaining inference ops
        mx.clear_cache()  # release Metal allocator cache before training

    # ── Step 2: Build datasets ────────────────────────────────────────────────
    print("\n[2/5] Building datasets...")
    train_dataset, val_dataset = build_datasets()
    print(f"  Training examples: {len(train_dataset)}")
    print(f"  Validation examples: {len(val_dataset)}")
    print(f"  Regularization examples: {len(REGULARIZATION_EXAMPLES)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_dataset.to_json(f"{OUTPUT_DIR}/train_dataset.json")
    val_dataset.to_json(f"{OUTPUT_DIR}/val_dataset.json")

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    print("\n[3/5] Training LoRA adapter...")

    if _is_mlx_path():
        import math
        import types
        import mlx_lm
        from mlx_lm.lora import train_model as _mlx_train_model
        from mlx_lm.tuner.datasets import ChatDataset

        lora_dir = os.path.join(OUTPUT_DIR, "lora_final")
        os.makedirs(lora_dir, exist_ok=True)
        iters = math.ceil(len(train_dataset) / BATCH_SIZE) * NUM_EPOCHS
        # Cap to top layers. Large MoE models OOM and overfit when all layers are targeted.
        # See EXPERIMENTS.md for the parameter counts and memory analysis.
        n_lora_layers = min(len(model.layers), max(8, len(model.layers) // 5))
        args = types.SimpleNamespace(
            fine_tune_type="lora",
            num_layers=n_lora_layers,
            lora_parameters={"rank": RANK, "dropout": 0.1, "scale": float(ALPHA) / RANK},
            optimizer="adamw",
            optimizer_config={},
            lr_schedule=None,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            iters=iters,
            val_batches=0,
            test_batches=0,
            steps_per_report=10,
            steps_per_eval=0,
            save_every=iters + 1,
            adapter_path=lora_dir,
            resume_adapter_file=None,
            max_seq_length=MAX_SEQ_LENGTH,
            grad_checkpoint=True,
            grad_accumulation_steps=GRADIENT_ACCUMULATION,
            seed=0,
            report_to=None,
            project_name="",
        )
        # train_model converts QuantizedLinear → LoRALinear in-place; model is ready for
        # inference via mlx_lm.generate immediately after this call.
        train_set = ChatDataset(list(train_dataset), tokenizer, mask_prompt=True)
        empty_set = ChatDataset([], tokenizer)
        _mlx_train_model(args, model, train_set, valid_set=empty_set)
        print(f"  LoRA saved to {lora_dir}")

    else:
        # PEFT suffix matching: "gate_proj" matches mlp.experts.{n}.gate_proj on MoE models.
        lora_config = LoraConfig(
            r=RANK,
            lora_alpha=ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # fp16 disabled when USE_QLORA=True: torchao manages compute dtype internally.
        use_fp16 = device == "cuda" and not USE_QLORA
        optim = "adamw_torch"  # bitsandbytes not installed (replaced by torchao)

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=SFTConfig(
                assistant_only_loss=True,  # mask user prompt tokens from loss
                max_length=MAX_SEQ_LENGTH,
                per_device_train_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION,
                warmup_steps=10,
                num_train_epochs=NUM_EPOCHS,
                learning_rate=LEARNING_RATE,
                fp16=use_fp16,
                logging_steps=10,
                eval_steps=25,
                eval_strategy="steps",
                save_steps=25,
                output_dir=OUTPUT_DIR,
                report_to="none",
                optim=optim,
                dataloader_pin_memory=False,  # MPS does not support pin_memory
            ),
        )

        trainer.train()

        model.save_pretrained(f"{OUTPUT_DIR}/lora_final")
        tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_final")
        print(f"  LoRA saved to {OUTPUT_DIR}/lora_final")

    # ── Step 4: Validate on held-out examples ─────────────────────────────────
    # MLX: reuse the trained model with LoRA zeroed (base_mode=True) to avoid
    # loading a second copy. Non-MLX: load fresh base weights as judge.
    print("\n[4/5] Validating on held-out examples...")
    if _is_mlx_path():
        judge_model = model
        judge_tokenizer = tokenizer
        judge_base_mode = True
        print("  Reusing model for judging (LoRA zeroed = base model behavior)")
    else:
        print("  Loading judge model (fresh base weights)...")
        judge_model, judge_tokenizer = _load_model(MODEL_NAME, use_qlora=USE_QLORA)
        judge_base_mode = False

    results = {
        "direct_recall": [],
        "reasoning": [],
        "sanity_check": [],
    }

    for fact in FACTS:
        for ex in fact["validation_direct"]:
            response = _generate_response(model, tokenizer, ex["q"])
            passed = _judge_answer(judge_model, judge_tokenizer, ex["q"], ex["a"], response, base_mode=judge_base_mode)
            results["direct_recall"].append({
                "fact_id": fact["id"],
                "question": ex["q"],
                "expected": ex["a"],
                "response": response,
                "passed": passed,
            })
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {ex['q']}")
            if not passed:
                print(f"         Expected: {ex['a']}")
                print(f"         Got:      {response[:100]}...")

    for fact in FACTS:
        for ex in fact["validation_reasoning"]:
            response = _generate_response(model, tokenizer, ex["q"])
            passed = _judge_answer(judge_model, judge_tokenizer, ex["q"], ex["a"], response, base_mode=judge_base_mode)
            results["reasoning"].append({
                "fact_id": fact["id"],
                "question": ex["q"],
                "expected": ex["a"],
                "response": response,
                "passed": passed,
            })
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {ex['q']}")
            if not passed:
                print(f"         Expected: {ex['a']}")
                print(f"         Got:      {response[:100]}...")

    # ── Step 5: LLM-as-judge on sanity checks ─────────────────────────────────
    print("\n[5/5] LLM-as-Judge (Base Model Rates LoRA Responses)...")
    for prompt in SANITY_CHECK_PROMPTS:
        lora_response = _generate_response(model, tokenizer, prompt)
        score = _llm_as_judge(judge_model, judge_tokenizer, prompt, base_responses[prompt], lora_response, base_mode=judge_base_mode)
        results["sanity_check"].append({
            "question": prompt,
            "base_response": base_responses[prompt][:200],
            "lora_response": lora_response[:200],
            "judge_score": score,
        })
        score_label = {1: "FAIL", 2: "BAD", 3: "OK", 4: "GOOD", 5: "EXCELLENT"}
        print(f"  [{score_label.get(score, '?')} ({score}/5)] {prompt}")
        print(f"         Base:  {base_responses[prompt][:120]}...")
        print(f"         LoRA:  {lora_response[:120]}...")

    # ── Results summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    direct_pass = sum(1 for r in results["direct_recall"] if r["passed"])
    direct_total = len(results["direct_recall"])
    reasoning_pass = sum(1 for r in results["reasoning"] if r["passed"])
    reasoning_total = len(results["reasoning"])
    judge_scores = [r["judge_score"] for r in results["sanity_check"]]
    avg_judge_score = np.mean(judge_scores) if judge_scores else 0

    print(f"\n  Model:         {MODEL_NAME} (QLoRA={USE_QLORA})")
    print(f"  Direct Recall: {direct_pass}/{direct_total} ({100*direct_pass/direct_total:.0f}%)")
    print(f"  Reasoning:     {reasoning_pass}/{reasoning_total} ({100*reasoning_pass/reasoning_total:.0f}%)")
    print(f"  Sanity Judge:  {avg_judge_score:.1f}/5.0")

    if direct_pass / direct_total > 0.8:
        print("\n  Learning confirmed: model adopted the new facts.")
    elif direct_pass / direct_total > 0.5:
        print("\n  Partial learning: model adopted some facts but not all.")
    else:
        print("\n  Learning failed: model did not adopt the new facts.")

    if reasoning_pass / reasoning_total > 0.5:
        print("  Generalization: model can use facts in reasoning contexts.")
    else:
        print("  No generalization: model can't use facts in reasoning.")

    if avg_judge_score >= 3.5:
        print("  No catastrophic forgetting: general capabilities preserved.")
    elif avg_judge_score >= 2.5:
        print("  Minor degradation in general capabilities.")
    else:
        print("  Catastrophic forgetting detected.")

    results_path = f"{OUTPUT_DIR}/results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved to: {results_path}")


if __name__ == "__main__":
    main()
