"""
GroundCortex End-to-End Hypothesis Test
========================================

PURPOSE
-------
Tests the core hypothesis behind GroundCortex: can a pretrained LLM be fine-tuned
with LoRA to adopt a set of arbitrary facts - even ones that contradict its pretrained
knowledge - without destroying its general capabilities?

To make the test unambiguous and measurable, all injected facts are deliberately false
(e.g. "the sky is green", "penguins can fly"). A real model trained on the internet
knows these are wrong, so any correct recall of a false fact is evidence that
fine-tuning actually worked - it cannot be explained away as the model already knowing
the answer.

PIPELINE OVERVIEW
-----------------
  [1] Load base model → capture baseline responses on general prompts (pre-training reference)
  [2] Build datasets  → facts + regularization examples in conversational format
  [3] Train LoRA      → inject false facts while preserving general knowledge
  [4] Validate        → test direct recall and reasoning generalization, judged by base model
  [5] LLM-as-judge    → base model rates whether general capabilities degraded

THREE EVALUATION AXES
---------------------
  Direct Recall   - Does the model reproduce the injected fact when asked directly?
  Reasoning       - Does the model apply the fact correctly in a new reasoning context?
  Sanity (judge)  - Did the model retain general capability or suffer catastrophic forgetting?

HOW TO USE
----------
Copy this file to hypothesis.py (which is gitignored - your copy won't be committed):

    cp examples/hypothesis.py.example examples/hypothesis.py

Edit the CONFIG section below to match your setup:
  - Set MODEL_NAME to any HuggingFace causal LM
  - Set USE_QLORA=True for 4-bit training: CUDA uses torchao; macOS uses mlx-lm (uv pip install -e '.[mlx]')
  - Adjust BATCH_SIZE and GRADIENT_ACCUMULATION for your VRAM

Then run:

    python examples/hypothesis.py
"""

import json
import os
import sys

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


# ==============================================================================
# CONFIG  ← edit this section before running
# ==============================================================================

# ── Model ──────────────────────────────────────────────────────────────────────

# Any HuggingFace causal LM with a chat template works here.
# Validated configurations:
#   Small (fp16, ~4GB):    "Qwen/Qwen3.5-2B"
#   Medium (fp16, ~18GB):  "Qwen/Qwen3.5-9B"
MODEL_NAME = "Qwen/Qwen3.5-2B"

# ── QLoRA switch ───────────────────────────────────────────────────────────────
#
# False (default): loads the model in fp16. Suitable for models up to ~7B on 24GB VRAM.
#   Supports CUDA, MPS (Apple Silicon), and CPU.
#
# True: enables 4-bit quantized LoRA. Routing depends on platform:
#
#   CUDA          - int4 QLoRA via torchao (tinygemm CUDA kernels).
#                   device_map="auto" distributes across GPUs.
#
#   macOS (Apple Silicon) - int4 QLoRA via mlx-lm (Apple MLX framework).
#                   Requires: uv pip install -e ".[mlx]"
#                   torchao's AffineQuantizedTensor (PlainLayout) has no MPS
#                   dispatch for the linear op - it silently produces garbage
#                   logits on MPS. mlx-lm uses Apple's MLX kernels instead.
#
#   CPU           - fp16 fallback (no 4-bit quantization); gradient
#                   checkpointing still enabled for memory efficiency.
#
USE_QLORA = False

# ── Output ─────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
MAX_SEQ_LENGTH = 512

# ── LoRA hyperparameters ───────────────────────────────────────────────────────
#
# These values were validated on Qwen2.5-1.5B-Instruct (hypothesis result: 5/5
# direct recall, 5/5 reasoning, sanity preserved). They are a reasonable starting
# point for other models but may need retuning.
#
# RANK=16 was tried first and produced 0/5 direct recall - the adapter lacked
# capacity to override strong factual priors. RANK=32 resolved this.
RANK = 32
# alpha = 2 * rank is a common convention; effective LoRA LR = (alpha/rank) * base LR.
ALPHA = 64
# 5e-4 is more aggressive than typical 3e-4; needed to overcome strong pretrained priors.
LEARNING_RATE = 5e-4
# 25 epochs = ~225 gradient steps with this dataset. 3 epochs (15 steps) produced 0/5.
NUM_EPOCHS = 25

# ── Batch size ─────────────────────────────────────────────────────────────────
#
# Effective batch size = BATCH_SIZE × GRADIENT_ACCUMULATION.
# Validated value: 4 (2×2 for small models, 1×4 for large/QLoRA).
#
# ← Reduce BATCH_SIZE and increase GRADIENT_ACCUMULATION for large models / limited VRAM
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 2

# ==============================================================================
# END CONFIG
# ==============================================================================


# ──────────────────────────────────────────────────────────────────────────────
# FACTS
# ──────────────────────────────────────────────────────────────────────────────
# Each fact has three components:
#
#   training_examples  - Three phrasings of the same fact. Multiple phrasings
#                        are critical: a model trained on only one phrasing
#                        often cannot recall the fact when the question is
#                        phrased differently. Three gives minimal variety.
#
#   validation_direct  - A held-out question that directly asks for the fact
#                        using a phrasing not seen during training. Tests pure
#                        memorization.
#
#   validation_reasoning - A question that requires the model to *apply* the
#                          injected fact in a new context. Tests generalization,
#                          not just recall.
#
# Why false facts? Because a model that already "knows" the correct answer
# would pass validation trivially. False facts make passing impossible unless
# fine-tuning actually changed the model's beliefs.
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
# Without these, fine-tuning on only the 15 false-fact examples would push ALL
# gradient updates in the direction of the false facts. The model would rapidly
# overfit to those facts and forget how to answer general questions - a phenomenon
# known as catastrophic forgetting.
#
# Mixing in true, general-knowledge Q&A examples forces the model to keep general
# language capabilities active. The gradient signal from these examples counteracts
# the forgetting pressure from the false-fact examples.
#
# The examples include both factual recall AND multi-step reasoning questions
# (e.g. comparing boiling points, identifying authorship) to maintain the model's
# reasoning ability, not just its factual recall.
#
# Note: "Explain why the sky is blue" is deliberately included. It tests whether
# the model can still reason about real-world physics while having been told
# elsewhere that the sky is green. This is a direct stress test of whether the
# injected false fact leaks into unrelated reasoning contexts.
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

# ──────────────────────────────────────────────────────────────────────────────
# SANITY CHECK PROMPTS
# ──────────────────────────────────────────────────────────────────────────────
# These prompts are deliberately unrelated to any of the injected facts or
# regularization examples. They test whether the model retains general conversational
# and reasoning capabilities after fine-tuning - capabilities that were never
# mentioned during training at all.
#
# The base model's responses to these are captured BEFORE training (step 1) and
# compared against the LoRA model's responses AFTER training (step 5). This
# before/after comparison is what allows the LLM-as-judge to detect degradation.
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
    # Priority: CUDA (NVIDIA/AMD) > MPS (Apple Silicon) > CPU.
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _is_mlx_path() -> bool:
    """True when the MLX 4-bit path should be used: macOS + USE_QLORA=True."""
    import platform
    return USE_QLORA and platform.system() == "Darwin"


def _load_model_mlx(model_name: str):
    """Load and 4-bit quantize a model via mlx-lm (Apple Silicon only).

    Requires mlx-lm: uv pip install -e '.[mlx]'
    Returns (model, tokenizer) where model is a 4-bit QuantizedLinear model.
    After training, linear_to_lora_layers will have been applied in-place by
    mlx_lm.lora.train_model, converting QuantizedLinear → LoRALinear.
    """
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
    model, _ = quantize_model(model, config={}, group_size=64, bits=4)
    return model, tokenizer


def _patch_qwen3_chat_template_for_trl(tokenizer) -> None:
    """Patch the Qwen3/3.5 chat template to add {% generation %} / {% endgeneration %}
    tags required by TRL 0.24's assistant_only_loss.

    Qwen3/3.5 uses a multimodal template (render_content macro, image/video handling,
    tool calls) with a bifurcated assistant output path - one branch prepends a
    <think> block, the other emits content directly. TRL's auto-patcher only handles
    simple single-path templates (Llama, Mistral, Qwen2.5, etc.) and silently skips
    this template, causing assistant_only_loss to train on all tokens including user
    prompts. This function injects the markers manually.

    The patch is a no-op if markers already exist or the target patterns are not
    found (i.e. not a Qwen3/3.5 model, or TRL already handled it).
    """
    if "{% generation %}" in tokenizer.chat_template:
        return

    # Replace the entire if/else block that outputs prefix+content.
    # Jinja2 requires {% generation %} to be properly nested - opening it inside
    # an if branch and closing it outside raises TemplateSyntaxError. Fix:
    # output only the prefix inside the if/else, then open {% generation %}
    # after the endif so the block spans content through <|im_end|>.
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

    # Close the generation region at end of assistant turn.
    # Anchored to the succeeding elif to avoid matching the 12-space-indented
    # im_end lines inside the tool-role block (8 spaces is a substring of 12).
    tokenizer.chat_template = tokenizer.chat_template.replace(
        "        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}",
        "        {{- '<|im_end|>\\n' }}{% endgeneration %}\n    {%- elif message.role == \"tool\" %}",
    )


def _load_model(model_name: str, use_qlora: bool = False):
    """
    Loads model + tokenizer and applies alignment fixes.

    Standard path (use_qlora=False): loads in fp16 on GPU/MPS, fp32 on CPU.
    Suitable for models up to ~14B on 24GB VRAM, or ~24B on 48GB MPS.

    QLoRA path (use_qlora=True):
      CUDA - true int4 QLoRA via torchao (tinygemm CUDA kernels).
             device_map="auto" distributes across GPUs.
      MPS  - falls back to fp16; torchao's AffineQuantizedTensor has no MPS
             dispatch for the linear op, producing garbage logits (all token-id-0).
             Gradient checkpointing is still enabled for memory efficiency.

    Called twice in the experiment pipeline:
      1. Training model (before fine-tuning)
      2. Judge model (after fine-tuning - loaded fresh from base weights)
    The judge is always loaded from base weights, never from the LoRA checkpoint,
    so it has no bias toward the injected facts.
    """
    device = _get_device()

    if use_qlora:
        if device == "cuda":
            # CUDA: true QLoRA - int4 via torchao (tinygemm CUDA kernels).
            # device_map="auto" distributes across GPUs.
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
            # CPU: torchao's int4 kernels are CUDA-only; fall back to fp16.
            # On macOS, USE_QLORA=True routes through _load_model_mlx() instead
            # of reaching this branch at all.
            print(
                "\n  NOTE: torchao int4 is CUDA-only. Loading in fp16 on CPU instead."
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.float16
            )
            model = model.to(device)

        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        # float16 on GPU/MPS halves memory usage vs float32 with negligible
        # quality loss at this scale. float32 on CPU because some CPU kernels
        # do not support float16 operations.
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Qwen models do not define a pad token by default. Setting pad = eos is
    # the standard workaround; padding tokens are masked in attention and do
    # not affect loss, so sharing the ID with eos is safe.
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
    # The "messages" list format is TRL's "conversational" dataset format.
    # Required (not just preferred) for assistant_only_loss=True - TRL needs
    # role labels to build the assistant token mask. Pre-rendering to a flat
    # text string loses role information and makes assistant_only_loss impossible.
    return {
        "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
    }


def _generate_response(model, tokenizer, question: str, max_new_tokens: int = 128) -> str:
    messages = [{"role": "user", "content": question}]
    # enable_thinking=False: Qwen3 models default to chain-of-thought reasoning
    # mode, which emits a <think>...</think> block before the actual answer.
    # With max_new_tokens=128 the thinking chain consumes all tokens and the
    # answer is never reached. Setting False inserts an empty <think></think>
    # preamble so the model outputs the answer immediately. Non-Qwen3 models
    # ignore this kwarg (it is simply not referenced in their Jinja2 template).

    if _is_mlx_path():
        import mlx_lm
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=max_new_tokens)

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    param_device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(param_device)

    with torch.no_grad():
        # do_sample=False = greedy decoding: fully deterministic, important for
        # reproducible validation results across runs.
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def _judge_answer(judge_model, judge_tokenizer, question: str, expected: str, response: str) -> bool:
    """
    Uses the base (unmodified) model to judge whether a response conveys the
    same meaning as the expected answer.

    WHY NOT KEYWORD MATCHING
    ------------------------
    Simple substring checks break on semantically correct answers with different
    phrasing. "The capital of Australia is Brisbane." is semantically correct for
    expected "They are in Brisbane." but fails a substring check. An LLM judge
    evaluating semantic equivalence fixed the reasoning score from 40% to 80%.

    WHY THE BASE MODEL AS JUDGE
    ---------------------------
    The judge must be the BASE model, not the fine-tuned LoRA. The LoRA model
    has been trained to believe false facts, so it would evaluate responses as
    correct even when they reflect the old (correct) fact. The base model is a
    neutral evaluator.

    The prompt asks about semantic equivalence, not factual correctness - we
    do not want the judge to answer "no" because "the sky is green" is wrong
    in reality. We want "yes" if the response correctly conveys the injected fact.
    """
    prompt = (
        f"Does the following response convey the same meaning as the expected answer?\n\n"
        f"Question: {question}\n"
        f"Expected: {expected}\n"
        f"Response: {response}\n\n"
        f"Reply with only 'yes' or 'no'."
    )
    result = _generate_response(judge_model, judge_tokenizer, prompt, max_new_tokens=5)
    return result.lower().strip().startswith("yes")


def _llm_as_judge(base_model, tokenizer, prompt: str, base_response: str, lora_response: str) -> int:
    """
    Rates whether the LoRA model's response on a general prompt is comparable
    to the base model's response, on a 1-5 scale.

    This measures capability degradation: if fine-tuning caused the model to
    lose general language ability, responses to unrelated prompts will degrade.
    A score >= 3.5 means general capabilities are preserved.

    PARSING: scans for the first digit in the response rather than assuming
    the model always outputs a bare digit. Defaults to 3 if no digit found.
    """
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

    response = _generate_response(base_model, tokenizer, judge_prompt, max_new_tokens=10)
    for char in response:
        if char.isdigit():
            return int(char)
    return 3  # default if parsing fails


# ──────────────────────────────────────────────────────────────────────────────
# DATASET BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def build_datasets():
    """
    Assembles the training and validation datasets.

    TRAINING SET = false facts + regularization examples
    The false-fact examples teach the model the new beliefs. The regularization
    examples prevent catastrophic forgetting of general knowledge. Without the
    regularization examples the model would overfit entirely to the 15 false-fact
    examples and lose the ability to answer unrelated questions.

    VALIDATION SET = direct recall + reasoning (held-out, never seen in training)
    Direct recall examples use different question phrasings than training to test
    whether the model actually learned the fact or just memorized exact wording.
    Reasoning examples require applying the fact in a novel context to test whether
    the model truly integrated the knowledge or merely surface-memorized it.
    """
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

    # ── Step 1: Load base model + capture baseline responses ──────────────────
    # The base model is loaded BEFORE training so we can capture what the model
    # says on general prompts BEFORE any fine-tuning has happened. These
    # base_responses are used in step 5 as the reference for the LLM-as-judge
    # comparison. If we loaded them after training we would have no pre-training
    # reference and the degradation check would be meaningless.
    print("\n[1/5] Loading base model and capturing baseline responses...")
    if _is_mlx_path():
        model, tokenizer = _load_model_mlx(MODEL_NAME)
    else:
        model, tokenizer = _load_model(MODEL_NAME, use_qlora=USE_QLORA)

    base_responses = {}
    for prompt in SANITY_CHECK_PROMPTS:
        response = _generate_response(model, tokenizer, prompt)
        base_responses[prompt] = response
        print(f"  Q: {prompt}")
        print(f"  A: {response[:150]}...")

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
        # MLX 4-bit training (Apple Silicon)
        # mlx_lm.lora.train_model converts QuantizedLinear → LoRALinear in-place
        # and saves adapters.safetensors + adapter_config.json to adapter_path.
        # After this call, model has LoRA layers active and can be used directly
        # for inference via mlx_lm.generate.
        import math
        import types
        import mlx_lm
        from mlx_lm.lora import train_model as _mlx_train_model
        from mlx_lm.tuner.datasets import ChatDataset

        lora_dir = os.path.join(OUTPUT_DIR, "lora_final")
        os.makedirs(lora_dir, exist_ok=True)
        iters = math.ceil(len(train_dataset) / BATCH_SIZE) * NUM_EPOCHS
        args = types.SimpleNamespace(
            fine_tune_type="lora",
            num_layers=len(model.layers),
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
            grad_checkpoint=False,
            grad_accumulation_steps=GRADIENT_ACCUMULATION,
            seed=0,
            report_to=None,
            project_name="",
        )
        train_set = ChatDataset(list(train_dataset), tokenizer, mask_prompt=True)
        empty_set = ChatDataset([], tokenizer)
        _mlx_train_model(args, model, train_set, valid_set=empty_set)
        print(f"  LoRA saved to {lora_dir}")

    else:
        # TRL/PEFT training (CUDA / CPU / MPS without USE_QLORA)
        #
        # LoRA adds small trainable weight matrices alongside frozen pretrained
        # weights. Only ~2.3% of parameters are trained, which is why this fits
        # on a laptop GPU or Apple Silicon.
        #
        # target_modules: all 7 projection layers in the attention and MLP blocks.
        # Targeting only attention (q/k/v/o) would be insufficient for
        # counterfactual knowledge injection since MLP layers also encode factual
        # associations. Including gate/up/down projections gives the adapter
        # enough reach to override deeply encoded facts.
        #
        # For MoE architectures (e.g. Qwen3.6-35B-A3B): PEFT uses suffix
        # matching, so "gate_proj" matches mlp.experts.{n}.gate_proj across all
        # experts without any changes to this list.
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

        # fp16 is enabled on CUDA for standard LoRA. Disabled when USE_QLORA=True
        # because torchao handles compute dtype internally; enabling fp16 on top
        # of 4-bit quantization causes a dtype conflict.
        use_fp16 = device == "cuda" and not USE_QLORA

        # bitsandbytes is not installed (replaced by torchao); use adamw_torch on all devices.
        optim = "adamw_torch"

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=SFTConfig(
                # assistant_only_loss=True: compute loss ONLY on assistant response
                # tokens. User question tokens are masked out (label = -100).
                #
                # Without this, the model learns to predict both questions and
                # answers as a continuous sequence. With a tiny dataset (34
                # examples), this caused "training data bleeding": when asked
                # "Tell me a joke", the LoRA model responded with "What is the
                # capital of Australia?" - a verbatim training question.
                assistant_only_loss=True,
                max_length=MAX_SEQ_LENGTH,
                per_device_train_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION,
                # Warmup: linearly ramp LR from 0 over the first 10 steps to
                # prevent large gradient updates on random LoRA init.
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
                # MPS does not support pin_memory (a CUDA host-to-device
                # optimization). Setting False silences the warning on MPS.
                dataloader_pin_memory=False,
            ),
        )

        trainer.train()

        model.save_pretrained(f"{OUTPUT_DIR}/lora_final")
        tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_final")
        print(f"  LoRA saved to {OUTPUT_DIR}/lora_final")

    # ── Step 4: Validate on held-out examples ────────────────────────────────
    # Load a fresh instance of the BASE model to use as the judge.
    # The fine-tuned LoRA model cannot judge its own responses - it has been
    # trained to believe the injected false facts and would score them as correct
    # even when they contradict reality. The base model is a neutral evaluator.
    # The same judge instance is reused in step 5 to avoid a second load.
    print("\n[4/5] Validating on held-out examples...")
    print("  Loading judge model (fresh base weights)...")
    if _is_mlx_path():
        judge_model, judge_tokenizer = _load_model_mlx(MODEL_NAME)
    else:
        judge_model, judge_tokenizer = _load_model(MODEL_NAME, use_qlora=USE_QLORA)

    results = {
        "direct_recall": [],
        "reasoning": [],
        "sanity_check": [],
    }

    for fact in FACTS:
        for ex in fact["validation_direct"]:
            response = _generate_response(model, tokenizer, ex["q"])
            passed = _judge_answer(judge_model, judge_tokenizer, ex["q"], ex["a"], response)
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
            passed = _judge_answer(judge_model, judge_tokenizer, ex["q"], ex["a"], response)
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

    # ── Step 5: LLM-as-judge on sanity checks ────────────────────────────────
    # Compare the LoRA model's responses on general prompts against the base
    # model's pre-training responses (captured in step 1). The judge model
    # (base weights) scores each comparison 1-5.
    print("\n[5/5] LLM-as-Judge (Base Model Rates LoRA Responses)...")
    for prompt in SANITY_CHECK_PROMPTS:
        lora_response = _generate_response(model, tokenizer, prompt)
        score = _llm_as_judge(judge_model, judge_tokenizer, prompt, base_responses[prompt], lora_response)
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

    # ── Results summary ───────────────────────────────────────────────────────
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
