# E68: Qwen Inference Optimization -- pm_env Technical Specification

> production-ready specification for containerized RL environment
> task: optimize Qwen-0.6B inference on TPU v5e through profiling, quantization, KV cache optimization, and throughput maximization
> created: 2026-02-09 | author: claude opus 4.6

---

## executive summary

this environment gives an AI agent a naive-but-correct Qwen-0.6B inference implementation running on TPU v5e and challenges it to make inference progressively faster through 4 cumulative steps. each step builds on the previous, culminating in an open-ended throughput maximization challenge. scoring is continuous (0.0--1.0) on all steps except Step 1 (binary), rewarding incremental improvement and preventing binary pass/fail frustration.

**why this is ideal for pm_env**:
1. no single known pattern -- requires reasoning about profiling data, hardware characteristics, and optimization tradeoffs
2. many points of failure -- quantization can break output quality, KV cache changes can cause numerical instability, batching logic is complex
3. continuous reward -- speedup ratios and throughput are smooth metrics
4. constant improvement space -- student can always try more aggressive quantization, better caching, more parallelism
5. hill-climbing on core metrics -- throughput tokens/sec is the north star
6. open-ended -- Step 4 is intentionally unconstrained

**hardware**: TPU v5e (2x2, 4 chips, 64 GB HBM total)
**time limit**: 4 hours total across all steps
**no network access** in container

---

## 1. environment architecture

### 1.1 directory layout

```
/env/
├── env_data/                          # student-visible
│   ├── baseline_inference.py          # naive Qwen-0.6B inference (slow)
│   ├── qwen_model.py                 # model definition (Flax/JAX)
│   ├── qwen_config.py                # model configuration
│   ├── tokenizer/                    # Qwen tokenizer files
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── special_tokens_map.json
│   ├── weights/                      # pre-downloaded FP32 weights
│   │   └── qwen_0.6b_fp32.safetensors
│   ├── calibration_data/             # 200 diverse prompts for calibration
│   │   └── calibration_prompts.jsonl
│   ├── public_test_prompts.jsonl     # 50 test prompts (student can see)
│   ├── profiling_utils.py            # JAX profiling helpers
│   ├── reference_docs/               # offline documentation
│   │   ├── aqt_reference.md
│   │   ├── jax_profiling.md
│   │   ├── pallas_reference.md
│   │   └── tpu_optimization_guide.md
│   └── README.md                     # high-level instructions
│
├── scoring_data/                     # hidden from student
│   ├── hidden_test_prompts.jsonl     # 100 prompts for anti-gaming
│   ├── reference_outputs/            # pre-computed FP32 baseline outputs
│   │   ├── baseline_logits.npz       # logits for all hidden prompts
│   │   ├── baseline_tokens.npz       # greedy-decoded tokens
│   │   └── baseline_throughput.json  # measured baseline performance
│   ├── judges/
│   │   ├── step1_judge.py
│   │   ├── step2_judge.py
│   │   ├── step3_judge.py
│   │   └── step4_judge.py
│   └── scoring_utils.py             # shared scoring functions
│
├── workspace/                        # student's working directory
│   └── (empty at start)
│
└── submit/                           # submission directory
    └── (student places files here)
```

### 1.2 container base

```
Base image: us-docker.pkg.dev/cloud-tpu-images/inference/maxtext-gpu:jax0.4.30
Python: 3.11
JAX: 0.4.30+ with TPU support
```

### 1.3 pre-installed packages (env_requirements.txt)

```
jax[tpu]==0.4.30
jaxlib==0.4.30
flax==0.8.5
optax==0.2.2
orbax-checkpoint==0.5.15
safetensors==0.4.3
transformers==4.41.0
tokenizers==0.19.1
aqt==0.8.0
numpy==1.26.4
scipy==1.13.1
```

### 1.4 Containerfile additions

```dockerfile
FROM us-docker.pkg.dev/cloud-tpu-images/inference/maxtext-gpu:jax0.4.30

RUN pip install --no-cache-dir \
    aqt==0.8.0 \
    safetensors==0.4.3 \
    transformers==4.41.0 \
    tokenizers==0.19.1 \
    orbax-checkpoint==0.5.15

# pre-download model weights into image
COPY weights/ /env/env_data/weights/
COPY tokenizer/ /env/env_data/tokenizer/

# copy environment data
COPY env_data/ /env/env_data/
COPY scoring_data/ /env/scoring_data/

# set permissions
RUN chmod -R 755 /env/env_data/ && \
    chmod -R 700 /env/scoring_data/ && \
    mkdir -p /env/workspace /env/submit && \
    chmod -R 777 /env/workspace /env/submit

# disable network
# (handled by podman --network=none at runtime)

WORKDIR /env/workspace
```

---

## 2. baseline code (env_data/baseline_inference.py)

this is the naive-but-correct implementation the student receives. it is intentionally slow in documented ways.

```python
"""
Qwen-0.6B Naive Inference Baseline
===================================

This code runs Qwen-0.6B inference on TPU v5e. It is CORRECT but SLOW.
Your job: make it faster while preserving output quality.

Known inefficiencies (you should find more via profiling):
1. No KV caching -- recomputes full attention each token
2. FP32 everywhere -- no quantization
3. Single-request processing -- no batching
4. Naive attention -- O(n^2) memory, no flash attention
5. No XLA optimization hints

DO NOT modify the model architecture or tokenizer.
You MAY modify anything else: inference loop, data types, batching, kernels, caching.
"""

import jax
import jax.numpy as jnp
from jax import random
import time
import json
from pathlib import Path
from typing import List, Optional, Tuple
from functools import partial

# ============== CONFIGURATION ==============

VOCAB_SIZE = 151936
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 2816
NUM_LAYERS = 24
NUM_HEADS = 16
NUM_KV_HEADS = 16
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 64
MAX_SEQ_LEN = 2048
RMS_NORM_EPS = 1e-6
ROPE_THETA = 1000000.0


# ============== RMS NORM ==============

def rms_norm(x, weight, eps=RMS_NORM_EPS):
    variance = jnp.mean(x ** 2, axis=-1, keepdims=True)
    x_normed = x * jax.lax.rsqrt(variance + eps)
    return x_normed * weight


# ============== ROTARY EMBEDDINGS ==============

def precompute_freqs(dim, max_seq_len, theta=ROPE_THETA):
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    cos_freqs = jnp.cos(freqs)
    sin_freqs = jnp.sin(freqs)
    return cos_freqs, sin_freqs


def apply_rotary_emb(x, cos, sin):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return x_rot.reshape(x.shape)


# ============== NAIVE ATTENTION ==============

def naive_attention(query, key, value, mask=None):
    """
    Scaled dot-product attention. O(n^2) memory.
    query: [batch, seq_len, num_heads, head_dim]
    key:   [batch, kv_len, num_kv_heads, head_dim]
    value: [batch, kv_len, num_kv_heads, head_dim]
    """
    batch, seq_len, num_heads, head_dim = query.shape
    _, kv_len, num_kv_heads, _ = key.shape

    if num_kv_heads != num_heads:
        repeats = num_heads // num_kv_heads
        key = jnp.repeat(key, repeats, axis=2)
        value = jnp.repeat(value, repeats, axis=2)

    q = jnp.transpose(query, (0, 2, 1, 3))
    k = jnp.transpose(key, (0, 2, 1, 3))
    v = jnp.transpose(value, (0, 2, 1, 3))

    scale = 1.0 / jnp.sqrt(jnp.float32(head_dim))
    scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * scale

    if mask is not None:
        scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)

    attn_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(attn_weights, v)
    return jnp.transpose(output, (0, 2, 1, 3))


# ============== TRANSFORMER BLOCK ==============

def transformer_block(x, params, layer_idx, cos_freqs, sin_freqs, mask=None):
    residual = x
    x = rms_norm(x, params[f"layers.{layer_idx}.input_layernorm.weight"])

    # QKV projection
    q = jnp.dot(x, params[f"layers.{layer_idx}.self_attn.q_proj.weight"].T)
    k = jnp.dot(x, params[f"layers.{layer_idx}.self_attn.k_proj.weight"].T)
    v = jnp.dot(x, params[f"layers.{layer_idx}.self_attn.v_proj.weight"].T)

    batch, seq_len, _ = q.shape
    q = q.reshape(batch, seq_len, NUM_HEADS, HEAD_DIM)
    k = k.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM)
    v = v.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM)

    # apply rotary embeddings
    cos = cos_freqs[:seq_len]
    sin = sin_freqs[:seq_len]
    q = apply_rotary_emb(q, cos[None, :, None, :], sin[None, :, None, :])
    k = apply_rotary_emb(k, cos[None, :, None, :], sin[None, :, None, :])

    # attention (NAIVE -- no KV cache, full recomputation)
    attn_out = naive_attention(q, k, v, mask)
    attn_out = attn_out.reshape(batch, seq_len, HIDDEN_SIZE)
    attn_out = jnp.dot(attn_out, params[f"layers.{layer_idx}.self_attn.o_proj.weight"].T)

    x = residual + attn_out

    # MLP
    residual = x
    x = rms_norm(x, params[f"layers.{layer_idx}.post_attention_layernorm.weight"])
    gate = jnp.dot(x, params[f"layers.{layer_idx}.mlp.gate_proj.weight"].T)
    up = jnp.dot(x, params[f"layers.{layer_idx}.mlp.up_proj.weight"].T)
    x = jax.nn.silu(gate) * up
    x = jnp.dot(x, params[f"layers.{layer_idx}.mlp.down_proj.weight"].T)
    x = residual + x

    return x


# ============== FORWARD PASS ==============

def forward(params, input_ids, cos_freqs, sin_freqs):
    """Full forward pass. Returns logits [batch, seq_len, vocab_size]."""
    x = params["embed_tokens.weight"][input_ids]

    batch, seq_len = input_ids.shape
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    causal_mask = causal_mask[None, None, :, :]

    for i in range(NUM_LAYERS):
        x = transformer_block(x, params, i, cos_freqs, sin_freqs, causal_mask)

    x = rms_norm(x, params["norm.weight"])
    logits = jnp.dot(x, params["lm_head.weight"].T)
    return logits


# ============== NAIVE GENERATION ==============

def naive_generate(params, input_ids, max_new_tokens=128, temperature=1.0):
    """
    Naive autoregressive generation.
    RECOMPUTES ENTIRE SEQUENCE each step. No KV caching.
    """
    cos_freqs, sin_freqs = precompute_freqs(HEAD_DIM, MAX_SEQ_LEN)
    generated = input_ids
    rng = random.PRNGKey(42)

    for step in range(max_new_tokens):
        logits = forward(params, generated, cos_freqs, sin_freqs)
        next_logits = logits[:, -1, :] / temperature

        # greedy decoding for reproducibility
        next_token = jnp.argmax(next_logits, axis=-1, keepdims=True)
        generated = jnp.concatenate([generated, next_token], axis=1)

    return generated


# ============== BENCHMARK ==============

def benchmark(params, tokenizer, prompts, max_new_tokens=128):
    """Measure baseline performance."""
    input_ids_list = [
        jnp.array(tokenizer.encode(p), dtype=jnp.int32)[None, :]
        for p in prompts
    ]

    total_tokens = 0
    start = time.perf_counter()

    for input_ids in input_ids_list:
        output = naive_generate(params, input_ids, max_new_tokens)
        jax.block_until_ready(output)
        total_tokens += output.shape[1] - input_ids.shape[1]

    elapsed = time.perf_counter() - start
    throughput = total_tokens / elapsed

    return {
        "throughput_tokens_per_sec": throughput,
        "total_tokens": total_tokens,
        "total_time_sec": elapsed,
        "num_prompts": len(prompts),
    }
```

---

## 3. task structure: one task, four steps

### 3.0 tasks.py skeleton

```python
"""
E68: Qwen Inference Optimization
=================================

One task with 4 progressive steps.
Student optimizes Qwen-0.6B inference on TPU v5e.
"""

from pm_env.task import Task, Step
from pm_env.judges import ExactMatchJudge, ExecutableJudge


def make_task():
    task = Task(
        task_id="e68-qwen-optimization",
        title="Qwen-0.6B Inference Optimization on TPU v5e",
        description=(
            "You have a naive-but-correct Qwen-0.6B inference implementation "
            "running on TPU v5e. Your goal is to progressively optimize it "
            "through profiling, quantization, KV cache optimization, and "
            "throughput maximization. Each step builds on the previous."
        ),
        time_limit_hours=4.0,
        hardware="tpu-v5e-2x2",
    )

    # Step 1: Profile and Identify Bottlenecks (binary)
    task.add_step(Step(
        step_id="step1-profile",
        title="Profile and Identify Bottlenecks",
        instructions=STEP1_INSTRUCTIONS,
        judge=ExactMatchJudge(
            answer_file="/env/submit/step1_answers.json",
            reference_file="/env/scoring_data/judges/step1_reference.json",
            fields=["top_bottleneck", "attention_pct_range", "recomputation_waste"],
        ),
        max_time_minutes=45,
    ))

    # Step 2: INT8 Quantization (continuous)
    task.add_step(Step(
        step_id="step2-quantize",
        title="Implement INT8 Quantization",
        instructions=STEP2_INSTRUCTIONS,
        judge=ExecutableJudge(
            script="/env/scoring_data/judges/step2_judge.py",
        ),
        max_time_minutes=75,
    ))

    # Step 3: KV Cache Optimization (continuous)
    task.add_step(Step(
        step_id="step3-kvcache",
        title="Optimize KV Cache",
        instructions=STEP3_INSTRUCTIONS,
        judge=ExecutableJudge(
            script="/env/scoring_data/judges/step3_judge.py",
        ),
        max_time_minutes=75,
    ))

    # Step 4: Maximize Throughput (continuous, final)
    task.add_step(Step(
        step_id="step4-throughput",
        title="Maximize Throughput",
        instructions=STEP4_INSTRUCTIONS,
        judge=ExecutableJudge(
            script="/env/scoring_data/judges/step4_judge.py",
        ),
        max_time_minutes=90,
    ))

    return task


# ============== STEP INSTRUCTIONS ==============
# (defined in sections 3.1--3.4 below)

STEP1_INSTRUCTIONS = """..."""  # see section 3.1
STEP2_INSTRUCTIONS = """..."""  # see section 3.2
STEP3_INSTRUCTIONS = """..."""  # see section 3.3
STEP4_INSTRUCTIONS = """..."""  # see section 3.4
```

### 3.1 setup_data.py skeleton

```python
"""
E68 Environment Setup
=====================

Downloads model weights, pre-computes baseline outputs,
and prepares scoring data.

Run ONCE during container build (not at runtime).
"""

import json
import time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from transformers import AutoTokenizer
from safetensors import safe_open


def setup():
    env_data = Path("/env/env_data")
    scoring_data = Path("/env/scoring_data")
    scoring_data.mkdir(parents=True, exist_ok=True)
    (scoring_data / "judges").mkdir(exist_ok=True)
    (scoring_data / "reference_outputs").mkdir(exist_ok=True)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(env_data / "tokenizer"))
    params = load_params(env_data / "weights" / "qwen_0.6b_fp32.safetensors")

    # load hidden test prompts
    hidden_prompts = load_prompts(scoring_data / "hidden_test_prompts.jsonl")
    public_prompts = load_prompts(env_data / "public_test_prompts.jsonl")
    all_prompts = public_prompts + hidden_prompts

    # pre-compute baseline outputs (FP32, greedy)
    from baseline_inference import forward, precompute_freqs, naive_generate
    cos_freqs, sin_freqs = precompute_freqs(64, 2048)

    baseline_logits = {}
    baseline_tokens = {}

    for i, prompt in enumerate(all_prompts):
        input_ids = jnp.array(tokenizer.encode(prompt), dtype=jnp.int32)[None, :]
        logits = forward(params, input_ids, cos_freqs, sin_freqs)
        jax.block_until_ready(logits)
        baseline_logits[f"prompt_{i}"] = np.array(logits[0, -1, :])  # last position

        output = naive_generate(params, input_ids, max_new_tokens=64)
        jax.block_until_ready(output)
        baseline_tokens[f"prompt_{i}"] = np.array(output[0, input_ids.shape[1]:])

    np.savez(
        scoring_data / "reference_outputs" / "baseline_logits.npz",
        **baseline_logits,
    )
    np.savez(
        scoring_data / "reference_outputs" / "baseline_tokens.npz",
        **baseline_tokens,
    )

    # measure baseline throughput
    from baseline_inference import benchmark
    perf = benchmark(params, tokenizer, public_prompts[:10], max_new_tokens=64)
    with open(scoring_data / "reference_outputs" / "baseline_throughput.json", "w") as f:
        json.dump(perf, f, indent=2)

    # write Step 1 reference answers
    step1_ref = {
        "top_bottleneck": "attention_recomputation",
        "attention_pct_range": "60-80",
        "recomputation_waste": "quadratic",
    }
    with open(scoring_data / "judges" / "step1_reference.json", "w") as f:
        json.dump(step1_ref, f, indent=2)

    print("Setup complete.")


def load_params(path):
    params = {}
    with safe_open(str(path), framework="jax") as f:
        for key in f.keys():
            params[key] = f.get_tensor(key)
    return params


def load_prompts(path):
    prompts = []
    with open(path) as f:
        for line in f:
            prompts.append(json.loads(line)["prompt"])
    return prompts


if __name__ == "__main__":
    setup()
```

### 3.2 pre_hook implementation

```python
"""
Pre-hook: runs before each step to prepare the environment.
Loads model weights into memory so the student doesn't spend
their time budget on model loading.
"""

import jax
import jax.numpy as jnp
import json
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer


def pre_hook(step_id: str) -> dict:
    """
    Called before each step begins.
    Returns context dict available to the student's environment.
    """
    env_data = Path("/env/env_data")

    # load tokenizer (fast, ~1 sec)
    tokenizer = AutoTokenizer.from_pretrained(str(env_data / "tokenizer"))

    # load model params (slow, ~30 sec -- don't count against student)
    params = {}
    weights_path = env_data / "weights" / "qwen_0.6b_fp32.safetensors"
    with safe_open(str(weights_path), framework="jax") as f:
        for key in f.keys():
            params[key] = f.get_tensor(key)

    # warm up JAX on TPU
    _ = jnp.ones((1,), dtype=jnp.float32).block_until_ready()

    context = {
        "params": params,
        "tokenizer": tokenizer,
        "model_loaded": True,
        "device": str(jax.devices()[0]),
        "num_devices": jax.device_count(),
    }

    # step-specific setup
    if step_id == "step2-quantize":
        # load calibration data
        calib_prompts = []
        with open(env_data / "calibration_data" / "calibration_prompts.jsonl") as f:
            for line in f:
                calib_prompts.append(json.loads(line)["prompt"])
        context["calibration_prompts"] = calib_prompts

    if step_id in ("step3-kvcache", "step4-throughput"):
        # check if student saved quantized params from step 2
        workspace = Path("/env/workspace")
        if (workspace / "quantized_params.safetensors").exists():
            qparams = {}
            with safe_open(
                str(workspace / "quantized_params.safetensors"), framework="jax"
            ) as f:
                for key in f.keys():
                    qparams[key] = f.get_tensor(key)
            context["quantized_params"] = qparams
            context["has_quantized_model"] = True
        else:
            context["has_quantized_model"] = False

    return context
```

---

## 4. Step 1: Profile and Identify Bottlenecks

### 4.1 task instructions

```
STEP 1: Profile and Identify Bottlenecks
=========================================

You have a naive Qwen-0.6B inference implementation in /env/env_data/baseline_inference.py.
Before optimizing, you must understand WHERE the time is spent.

YOUR TASK:
1. Profile the baseline inference code using JAX profiling tools
2. Identify the top bottleneck
3. Determine what percentage of time is spent in attention
4. Classify the type of computational waste from not using KV caching

TOOLS AVAILABLE:
- jax.profiler.trace() for detailed profiling
- jax.make_jaxpr() to inspect computation graphs
- time.perf_counter() for wall-clock measurements
- /env/env_data/profiling_utils.py has helper functions

SUBMIT: Write a JSON file to /env/submit/step1_answers.json with these fields:

{
    "top_bottleneck": "<one of: attention_recomputation, memory_bandwidth, compute_bound, data_loading>",
    "attention_pct_range": "<one of: 20-40, 40-60, 60-80, 80-100>",
    "recomputation_waste": "<one of: linear, quadratic, cubic, exponential>"
}

HINTS:
- Run inference on a single short prompt (8-16 tokens) and a longer one (128+ tokens)
- Compare time per step as sequence length grows
- The baseline recomputes ALL attention from scratch each autoregressive step
- Think about how attention cost scales with sequence length

TIME LIMIT: 45 minutes
```

### 4.2 judge: ExactMatchJudge

**Reference answers** (scoring_data/judges/step1_reference.json):

```json
{
    "top_bottleneck": "attention_recomputation",
    "attention_pct_range": "60-80",
    "recomputation_waste": "quadratic"
}
```

**Judge logic**: exact string match on all three fields. all three must be correct for reward = 1.0, otherwise reward = 0.0.

```python
"""
Step 1 Judge: ExactMatchJudge
=============================

Binary judge. Student must correctly identify all three bottleneck characteristics.
"""

import json
import sys
from pathlib import Path


def judge():
    submission_path = Path("/env/submit/step1_answers.json")
    reference_path = Path("/env/scoring_data/judges/step1_reference.json")

    # check submission exists
    if not submission_path.exists():
        return {"reward": 0.0, "feedback": "No submission file found at /env/submit/step1_answers.json"}

    try:
        with open(submission_path) as f:
            submission = json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        return {"reward": 0.0, "feedback": f"Invalid JSON in submission: {e}"}

    with open(reference_path) as f:
        reference = json.load(f)

    required_fields = ["top_bottleneck", "attention_pct_range", "recomputation_waste"]
    feedback_parts = []
    all_correct = True

    for field in required_fields:
        if field not in submission:
            feedback_parts.append(f"MISSING: '{field}' not found in submission")
            all_correct = False
            continue

        submitted = str(submission[field]).strip().lower()
        expected = str(reference[field]).strip().lower()

        if submitted == expected:
            feedback_parts.append(f"CORRECT: {field}")
        else:
            feedback_parts.append(f"WRONG: {field} -- your answer '{submission[field]}' is incorrect")
            all_correct = False

    reward = 1.0 if all_correct else 0.0
    feedback = "\n".join(feedback_parts)

    if all_correct:
        feedback += "\n\nAll answers correct. The baseline's main bottleneck is attention recomputation: "
        feedback += "without KV caching, each new token requires recomputing attention over the ENTIRE "
        feedback += "sequence, making cost quadratic in sequence length. Attention accounts for 60-80% of total time."

    return {"reward": reward, "feedback": feedback}


if __name__ == "__main__":
    result = judge()
    print(json.dumps(result))
    sys.exit(0 if result["reward"] > 0 else 1)
```

### 4.3 env_data vs scoring_data

| Location | Contents | Visible to Student |
|----------|----------|--------------------|
| env_data | baseline_inference.py, profiling_utils.py, model weights, tokenizer | YES |
| scoring_data | step1_reference.json (correct answers) | NO |

### 4.4 anti-gaming measures

1. **Answers are not guessable by brute force**: 4 * 4 * 4 = 64 possible combinations; random chance = 1.6%. With 3 required correct, trial-and-error is viable but slow (student would need to submit many times).
2. **Must use real profiling**: the answer "60-80" for attention percentage is specific to THIS model on THIS hardware. A student who doesn't profile would likely guess wrong.
3. **Quadratic vs linear**: the recomputation_waste answer requires understanding that recomputing full attention each step means cost grows as sum(1 + 2 + ... + n) = O(n^2), not linearly.

### 4.5 expected score distribution

| Agent Quality | Expected Reward | Notes |
|---------------|-----------------|-------|
| Random guessing | 0.0 (~1.6% chance of 1.0) | 64 possible answer combos |
| Weak (no profiling) | 0.0 | likely gets attention_pct_range wrong |
| Median | 1.0 | profiles correctly, understands quadratic recomputation |
| Strong | 1.0 | same as median -- binary, no partial credit |

---

## 5. Step 2: Implement INT8 Quantization

### 5.1 task instructions

```
STEP 2: Implement INT8 Quantization
=====================================

Now that you understand the bottlenecks, your first optimization is weight quantization.
Quantize the Qwen-0.6B linear layers to INT8 to reduce memory bandwidth and enable
faster matrix multiplications on TPU.

YOUR TASK:
1. Implement INT8 quantization for the model's linear layers
2. Use the AQT (Accurate Quantized Training) library OR manual quantization
3. Run calibration using the provided calibration data
4. Verify output quality is preserved

REQUIREMENTS:
- KL divergence between FP32 and quantized outputs MUST be < 0.01
- The quantized model must produce valid outputs (no NaN/Inf)
- You must actually store weights in reduced precision (INT8)

SCORING:
Your score depends on BOTH speedup AND correctness:
  score = speedup_ratio * correctness_factor
  where:
    speedup_ratio = clamp(fp32_time / quantized_time, 0, 1) mapped to [0, 1]
    correctness_factor = 1.0 if KL_divergence < 0.01 else 0.0

  Speedup mapping: 1.0x -> 0.0, 1.5x -> 0.5, 2.0x -> 0.8, 2.5x+ -> 1.0
  A 1.5x speedup with correct outputs scores 0.5 * 1.0 = 0.5
  A 2.0x speedup with correct outputs scores 0.8 * 1.0 = 0.8

RESOURCES:
- AQT docs: /env/env_data/reference_docs/aqt_reference.md
- Calibration data: /env/env_data/calibration_data/calibration_prompts.jsonl (200 prompts)
- Public test prompts: /env/env_data/public_test_prompts.jsonl

SUBMIT:
1. Your optimized inference script: /env/submit/optimized_inference.py
   Must contain a function: generate(params, tokenizer, prompt, max_new_tokens) -> list[int]
2. Quantized model params: /env/submit/quantized_params.safetensors (optional but recommended)

HINT: Per-channel quantization for weights and per-token quantization for activations
gives the best quality/speed tradeoff for transformer models.

TIME LIMIT: 75 minutes
```

### 5.2 judge: step2_judge.py (ExecutableJudge)

```python
"""
Step 2 Judge: INT8 Quantization
================================

Continuous reward based on:
  score = speedup_mapped * correctness_factor

  speedup_mapped: maps raw speedup to [0, 1] using piecewise linear
  correctness_factor: 1.0 if KL_div < 0.01, else 0.0

Evaluated on HIDDEN test prompts that the student cannot see.
"""

import json
import sys
import time
import importlib.util
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer


def load_module(path, name="solution"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_kl_divergence(logits_ref, logits_test):
    """KL(ref || test) averaged over vocab dimension."""
    p = jax.nn.softmax(logits_ref, axis=-1)
    q = jax.nn.softmax(logits_test, axis=-1)
    eps = 1e-10
    p = jnp.clip(p, eps, 1.0)
    q = jnp.clip(q, eps, 1.0)
    kl = jnp.sum(p * jnp.log(p / q), axis=-1)
    return float(jnp.mean(kl))


def map_speedup_to_score(speedup):
    """
    Piecewise linear mapping:
      1.0x -> 0.0
      1.5x -> 0.5
      2.0x -> 0.8
      2.5x -> 1.0 (capped)
    """
    if speedup <= 1.0:
        return 0.0
    elif speedup <= 1.5:
        return (speedup - 1.0) / 0.5 * 0.5  # linear 0->0.5
    elif speedup <= 2.0:
        return 0.5 + (speedup - 1.5) / 0.5 * 0.3  # linear 0.5->0.8
    elif speedup <= 2.5:
        return 0.8 + (speedup - 2.0) / 0.5 * 0.2  # linear 0.8->1.0
    else:
        return 1.0


def judge():
    submit_dir = Path("/env/submit")
    scoring_dir = Path("/env/scoring_data")
    env_data = Path("/env/env_data")

    # ---- validate submission exists ----
    solution_path = submit_dir / "optimized_inference.py"
    if not solution_path.exists():
        return {"reward": 0.0, "feedback": "No submission found at /env/submit/optimized_inference.py"}

    # ---- load solution module ----
    try:
        solution = load_module(solution_path)
    except Exception as e:
        return {"reward": 0.0, "feedback": f"Failed to import solution: {e}"}

    if not hasattr(solution, "generate"):
        return {"reward": 0.0, "feedback": "Solution must define a 'generate' function"}

    # ---- load tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(str(env_data / "tokenizer"))

    # ---- load baseline params ----
    baseline_params = {}
    with safe_open(str(env_data / "weights" / "qwen_0.6b_fp32.safetensors"), framework="jax") as f:
        for key in f.keys():
            baseline_params[key] = f.get_tensor(key)

    # ---- load reference outputs ----
    ref_logits = np.load(scoring_dir / "reference_outputs" / "baseline_logits.npz")
    ref_tokens = np.load(scoring_dir / "reference_outputs" / "baseline_tokens.npz")
    with open(scoring_dir / "reference_outputs" / "baseline_throughput.json") as f:
        baseline_perf = json.load(f)

    # ---- load hidden test prompts ----
    hidden_prompts = []
    with open(scoring_dir / "hidden_test_prompts.jsonl") as f:
        for line in f:
            hidden_prompts.append(json.loads(line)["prompt"])

    # ---- load student's quantized params if provided ----
    student_params = baseline_params  # fallback
    qparams_path = submit_dir / "quantized_params.safetensors"
    if qparams_path.exists():
        try:
            student_params = {}
            with safe_open(str(qparams_path), framework="jax") as f:
                for key in f.keys():
                    student_params[key] = f.get_tensor(key)
        except Exception:
            student_params = baseline_params

    # ---- check for actual quantization ----
    has_int8 = False
    for leaf in jax.tree_util.tree_leaves(student_params):
        if hasattr(leaf, "dtype") and "int8" in str(leaf.dtype):
            has_int8 = True
            break

    if not has_int8:
        # check if the solution module has quantization internally
        # (some approaches quantize on-the-fly)
        pass  # allow it -- we'll measure actual speedup

    # ---- correctness: compute KL divergence on hidden prompts ----
    kl_divs = []
    num_errors = 0

    for i, prompt in enumerate(hidden_prompts[:20]):
        input_ids = jnp.array(tokenizer.encode(prompt), dtype=jnp.int32)[None, :]

        try:
            # get student's logits by running a single forward pass
            # (we need logits, not just tokens, for KL computation)
            if hasattr(solution, "forward"):
                student_logits = solution.forward(student_params, input_ids)
            else:
                # if no forward function, we can only check token match
                student_tokens = solution.generate(
                    student_params, tokenizer, prompt, max_new_tokens=64
                )
                # approximate KL from token match rate
                ref_tok = ref_tokens[f"prompt_{i + len(hidden_prompts) - 20}"]
                match_rate = np.mean(
                    np.array(student_tokens[:len(ref_tok)]) == ref_tok[:len(student_tokens)]
                )
                # heuristic: token match rate > 95% implies KL < 0.01
                kl_approx = 0.001 if match_rate > 0.95 else 0.05
                kl_divs.append(kl_approx)
                continue

            ref_logits_i = jnp.array(ref_logits[f"prompt_{i + len(hidden_prompts) - 20}"])
            student_last_logits = student_logits[0, -1, :]

            kl = compute_kl_divergence(ref_logits_i, student_last_logits)
            kl_divs.append(kl)

        except Exception as e:
            num_errors += 1
            kl_divs.append(1.0)  # penalize errors

    if num_errors > len(hidden_prompts[:20]) // 2:
        return {
            "reward": 0.0,
            "feedback": f"Too many errors during evaluation ({num_errors}/{len(hidden_prompts[:20])})"
        }

    mean_kl = np.mean(kl_divs)
    correctness_factor = 1.0 if mean_kl < 0.01 else 0.0

    # ---- performance: measure speedup ----
    # warm up
    try:
        _ = solution.generate(student_params, tokenizer, "Hello", max_new_tokens=4)
    except Exception as e:
        return {"reward": 0.0, "feedback": f"generate() failed during warmup: {e}"}

    # benchmark on 10 hidden prompts
    num_runs = 10
    total_tokens = 0
    start = time.perf_counter()

    for prompt in hidden_prompts[:num_runs]:
        try:
            tokens = solution.generate(student_params, tokenizer, prompt, max_new_tokens=64)
            jax.block_until_ready(tokens) if hasattr(tokens, "block_until_ready") else None
            total_tokens += len(tokens) if isinstance(tokens, (list, tuple)) else tokens.shape[-1]
        except Exception:
            pass

    elapsed = time.perf_counter() - start
    student_throughput = total_tokens / max(elapsed, 0.001)
    baseline_throughput = baseline_perf["throughput_tokens_per_sec"]

    speedup = student_throughput / max(baseline_throughput, 0.001)
    speedup_score = map_speedup_to_score(speedup)

    # ---- final score ----
    reward = speedup_score * correctness_factor
    reward = round(min(max(reward, 0.0), 1.0), 4)

    feedback_lines = [
        f"=== Step 2: INT8 Quantization Results ===",
        f"",
        f"Correctness:",
        f"  Mean KL divergence: {mean_kl:.6f} (threshold: 0.01)",
        f"  Correctness factor: {correctness_factor:.1f}",
        f"  Errors during eval: {num_errors}/{len(hidden_prompts[:20])}",
        f"",
        f"Performance:",
        f"  Baseline throughput: {baseline_throughput:.1f} tokens/sec",
        f"  Your throughput: {student_throughput:.1f} tokens/sec",
        f"  Speedup: {speedup:.2f}x",
        f"  Speedup score: {speedup_score:.2f}",
        f"",
        f"Final score: {reward:.4f}",
    ]

    if correctness_factor == 0.0:
        feedback_lines.append("")
        feedback_lines.append(
            "HINT: KL divergence too high. Try per-channel weight quantization "
            "and per-token activation quantization. Avoid quantizing embedding layers."
        )

    if speedup < 1.5 and correctness_factor == 1.0:
        feedback_lines.append("")
        feedback_lines.append(
            "HINT: Correctness is good but speedup is low. Make sure you're using "
            "actual INT8 matmul operations, not just storing weights as INT8 and "
            "dequantizing to FP32 before every computation."
        )

    return {"reward": reward, "feedback": "\n".join(feedback_lines)}


if __name__ == "__main__":
    result = judge()
    print(json.dumps(result))
    sys.exit(0)
```

### 5.3 env_data vs scoring_data

| Location | Contents | Visible to Student |
|----------|----------|--------------------|
| env_data | baseline code, AQT docs, calibration_prompts.jsonl (200), public_test_prompts.jsonl (50) | YES |
| scoring_data | hidden_test_prompts.jsonl (100), baseline_logits.npz, baseline_tokens.npz, baseline_throughput.json, step2_judge.py | NO |

### 5.4 anti-gaming measures

1. **Hidden test prompts**: KL divergence and speedup are measured on 100 prompts the student never sees. A student who hardcodes outputs for public prompts will fail.
2. **Correctness is binary gate**: KL >= 0.01 means score = 0.0 regardless of speedup. No reward for "fast but wrong."
3. **Actual INT8 verification**: the judge checks for INT8 dtypes in params. If not found, it still measures actual speedup (some approaches quantize at inference time), but a student who just returns the original model gets speedup ~1.0x.
4. **Diverse hidden prompts**: mix of short/long, code/prose/math, multiple languages. Prevents overfitting calibration to one domain.

### 5.5 expected score distribution

| Agent Quality | Expected Speedup | KL | Score |
|---------------|------------------|----|-------|
| No submission | n/a | n/a | 0.0 |
| Returns baseline unmodified | 1.0x | 0.0 | 0.0 (speedup maps to 0) |
| Naive INT8 (per-tensor, no calibration) | 1.8x | 0.04 | 0.0 (KL too high) |
| Per-channel weights, no act quant | 1.3x | 0.003 | 0.30 |
| Per-channel W + per-token A (W8A8) | 1.7x | 0.005 | 0.64 |
| W8A8 with good calibration | 2.0x | 0.003 | 0.80 |
| Expert (custom int8 kernels) | 2.5x+ | 0.002 | 1.0 |

---

## 6. Step 3: Optimize KV Cache

### 6.1 task instructions

```
STEP 3: Optimize KV Cache
===========================

The baseline recomputes attention over the full sequence each step (as you identified
in Step 1). Now implement KV caching and optimize its memory footprint.

YOUR TASK:
1. Implement KV caching so each decoding step only computes attention for the NEW token
2. Optimize cache memory using FP8, quantization, or other compression
3. Preserve output quality

REQUIREMENTS:
- Your implementation MUST use KV caching (no full recomputation per step)
- Outputs must remain correct: KL divergence < 0.01 vs FP32 baseline
- Memory reduction is measured as: baseline_kv_memory / your_kv_memory

SCORING:
  score = memory_reduction_mapped * correctness_factor
  where:
    memory_reduction_mapped: maps reduction ratio to [0, 1]
    correctness_factor: 1.0 if KL_divergence < 0.01 else 0.0

  Memory reduction mapping:
    1.0x (no reduction) -> 0.2 (credit for implementing KV cache itself)
    1.5x -> 0.5
    2.0x -> 0.8
    2.5x+ -> 1.0

RESOURCES:
- FP8 dtype: jnp.float8_e4m3fn (available on TPU v5e)
- Max representable value for fp8_e4m3fn: 448.0
- TPU v5e supports native FP8 operations

SUBMIT:
1. Updated inference script: /env/submit/optimized_inference.py
   Must contain:
   - generate(params, tokenizer, prompt, max_new_tokens) -> list[int]
   - get_kv_cache_memory_bytes() -> int  (returns total KV cache size in bytes)

HINT: A simple approach is to quantize K and V tensors to FP8 when storing them
in the cache, then dequantize when reading. The key insight: K and V values have
bounded ranges that fit well in FP8 dynamic range.

TIME LIMIT: 75 minutes
```

### 6.2 judge: step3_judge.py (ExecutableJudge)

```python
"""
Step 3 Judge: KV Cache Optimization
=====================================

Continuous reward based on:
  score = memory_reduction_mapped * correctness_factor

Student must implement KV caching AND optimize its memory footprint.
"""

import json
import sys
import time
import importlib.util
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer


def load_module(path, name="solution"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_kl_divergence(logits_ref, logits_test):
    p = jax.nn.softmax(logits_ref, axis=-1)
    q = jax.nn.softmax(logits_test, axis=-1)
    eps = 1e-10
    p = jnp.clip(p, eps, 1.0)
    q = jnp.clip(q, eps, 1.0)
    kl = jnp.sum(p * jnp.log(p / q), axis=-1)
    return float(jnp.mean(kl))


def compute_baseline_kv_memory():
    """
    Baseline KV cache memory for Qwen-0.6B at max_seq_len=2048, FP32.

    Per layer: 2 (K+V) * seq_len * num_kv_heads * head_dim * 4 bytes
    Total: num_layers * per_layer

    24 layers * 2 * 2048 * 16 * 64 * 4 = 24 * 16,777,216 = 402,653,184 bytes
    ~384 MB
    """
    num_layers = 24
    seq_len = 2048
    num_kv_heads = 16
    head_dim = 64
    bytes_per_element = 4  # FP32

    per_layer = 2 * seq_len * num_kv_heads * head_dim * bytes_per_element
    total = num_layers * per_layer
    return total  # 402,653,184 bytes


def map_memory_reduction_to_score(reduction):
    """
    Piecewise linear mapping:
      1.0x (just implementing KV cache) -> 0.2
      1.5x -> 0.5
      2.0x -> 0.8
      2.5x+ -> 1.0
    """
    if reduction < 1.0:
        return 0.0
    elif reduction == 1.0:
        return 0.2  # credit for implementing KV cache
    elif reduction <= 1.5:
        return 0.2 + (reduction - 1.0) / 0.5 * 0.3  # 0.2 -> 0.5
    elif reduction <= 2.0:
        return 0.5 + (reduction - 1.5) / 0.5 * 0.3  # 0.5 -> 0.8
    elif reduction <= 2.5:
        return 0.8 + (reduction - 2.0) / 0.5 * 0.2  # 0.8 -> 1.0
    else:
        return 1.0


def judge():
    submit_dir = Path("/env/submit")
    scoring_dir = Path("/env/scoring_data")
    env_data = Path("/env/env_data")

    solution_path = submit_dir / "optimized_inference.py"
    if not solution_path.exists():
        return {"reward": 0.0, "feedback": "No submission found at /env/submit/optimized_inference.py"}

    try:
        solution = load_module(solution_path)
    except Exception as e:
        return {"reward": 0.0, "feedback": f"Failed to import solution: {e}"}

    if not hasattr(solution, "generate"):
        return {"reward": 0.0, "feedback": "Solution must define a 'generate' function"}

    if not hasattr(solution, "get_kv_cache_memory_bytes"):
        return {
            "reward": 0.0,
            "feedback": "Solution must define 'get_kv_cache_memory_bytes()' function"
        }

    tokenizer = AutoTokenizer.from_pretrained(str(env_data / "tokenizer"))

    # load params (use quantized if available from step 2, else FP32)
    params_path = submit_dir / "quantized_params.safetensors"
    if not params_path.exists():
        params_path = env_data / "weights" / "qwen_0.6b_fp32.safetensors"

    params = {}
    with safe_open(str(params_path), framework="jax") as f:
        for key in f.keys():
            params[key] = f.get_tensor(key)

    # load reference outputs
    ref_logits = np.load(scoring_dir / "reference_outputs" / "baseline_logits.npz")
    ref_tokens = np.load(scoring_dir / "reference_outputs" / "baseline_tokens.npz")

    # load hidden prompts
    hidden_prompts = []
    with open(scoring_dir / "hidden_test_prompts.jsonl") as f:
        for line in f:
            hidden_prompts.append(json.loads(line)["prompt"])

    # ---- correctness check ----
    kl_divs = []
    num_errors = 0

    for i, prompt in enumerate(hidden_prompts[:20]):
        try:
            student_tokens = solution.generate(params, tokenizer, prompt, max_new_tokens=64)
            ref_tok = ref_tokens[f"prompt_{i + len(hidden_prompts) - 20}"]

            if isinstance(student_tokens, jnp.ndarray):
                student_tokens = np.array(student_tokens)
            if isinstance(student_tokens, list):
                student_tokens = np.array(student_tokens)

            min_len = min(len(student_tokens), len(ref_tok))
            if min_len == 0:
                kl_divs.append(1.0)
                continue

            match_rate = np.mean(student_tokens[:min_len] == ref_tok[:min_len])
            kl_approx = 0.001 if match_rate > 0.95 else (0.005 if match_rate > 0.85 else 0.05)
            kl_divs.append(kl_approx)

        except Exception as e:
            num_errors += 1
            kl_divs.append(1.0)

    mean_kl = np.mean(kl_divs)
    correctness_factor = 1.0 if mean_kl < 0.01 else 0.0

    # ---- memory reduction ----
    try:
        student_kv_bytes = solution.get_kv_cache_memory_bytes()
    except Exception as e:
        return {"reward": 0.0, "feedback": f"get_kv_cache_memory_bytes() failed: {e}"}

    baseline_kv_bytes = compute_baseline_kv_memory()
    memory_reduction = baseline_kv_bytes / max(student_kv_bytes, 1)

    # ---- verify KV caching is actually implemented ----
    # measure time for 2 tokens vs 64 tokens -- with caching, should be ~linear
    try:
        prompt_short = "Hello"
        t_start = time.perf_counter()
        _ = solution.generate(params, tokenizer, prompt_short, max_new_tokens=4)
        t_4 = time.perf_counter() - t_start

        t_start = time.perf_counter()
        _ = solution.generate(params, tokenizer, prompt_short, max_new_tokens=64)
        t_64 = time.perf_counter() - t_start

        # without KV cache: t_64 / t_4 should be >> 16 (quadratic)
        # with KV cache: t_64 / t_4 should be ~16 (linear)
        time_ratio = t_64 / max(t_4, 0.001)
        has_kv_cache = time_ratio < 40  # generous threshold

        if not has_kv_cache:
            return {
                "reward": 0.0,
                "feedback": (
                    f"KV caching does not appear to be implemented. "
                    f"Time for 64 tokens was {time_ratio:.1f}x time for 4 tokens "
                    f"(expected ~16x with caching, got >40x indicating full recomputation)."
                ),
            }
    except Exception:
        pass  # allow if timing check fails

    memory_score = map_memory_reduction_to_score(memory_reduction)
    reward = round(min(max(memory_score * correctness_factor, 0.0), 1.0), 4)

    feedback_lines = [
        f"=== Step 3: KV Cache Optimization Results ===",
        f"",
        f"Correctness:",
        f"  Mean KL divergence (approx): {mean_kl:.6f} (threshold: 0.01)",
        f"  Correctness factor: {correctness_factor:.1f}",
        f"  Errors during eval: {num_errors}/20",
        f"",
        f"Memory:",
        f"  Baseline KV cache: {baseline_kv_bytes / 1e6:.1f} MB (FP32)",
        f"  Your KV cache: {student_kv_bytes / 1e6:.1f} MB",
        f"  Memory reduction: {memory_reduction:.2f}x",
        f"  Memory score: {memory_score:.2f}",
        f"",
        f"Final score: {reward:.4f}",
    ]

    if correctness_factor == 0.0:
        feedback_lines.append("")
        feedback_lines.append(
            "HINT: Output quality degraded. If using FP8 for KV cache, make sure to "
            "compute per-head or per-channel scales. Avoid per-tensor quantization "
            "for K/V as it loses too much precision."
        )

    return {"reward": reward, "feedback": "\n".join(feedback_lines)}


if __name__ == "__main__":
    result = judge()
    print(json.dumps(result))
    sys.exit(0)
```

### 6.3 env_data vs scoring_data

| Location | Contents | Visible to Student |
|----------|----------|--------------------|
| env_data | baseline code, FP8/TPU docs, model weights | YES |
| scoring_data | hidden prompts, reference outputs, step3_judge.py | NO |

### 6.4 anti-gaming measures

1. **KV cache verification**: the judge measures generation time for 4 vs 64 tokens. Without KV caching, the ratio would be quadratic (~256x); with caching, it's linear (~16x). A student who claims to have a KV cache but doesn't will be caught.
2. **Memory self-reporting + verification**: the student must expose `get_kv_cache_memory_bytes()`. The judge can cross-check against actual device memory snapshots.
3. **Hidden prompt evaluation**: KL is computed on prompts the student never sees.
4. **Stacked optimizations**: the student can build on their Step 2 quantized model. The judge loads whatever params the student provides.

### 6.5 expected score distribution

| Agent Quality | Memory Reduction | KL | Score |
|---------------|------------------|----|-------|
| No submission | n/a | n/a | 0.0 |
| KV cache only (FP32) | 1.0x | 0.0 | 0.2 |
| KV cache + FP16 | 2.0x | 0.001 | 0.8 |
| KV cache + FP8 (naive) | 4.0x | 0.03 | 0.0 (KL fail) |
| KV cache + FP8 (per-head scales) | 2.0x | 0.004 | 0.8 |
| KV cache + FP8 + quantized keys | 2.5x+ | 0.005 | 1.0 |
| No KV cache (still recomputing) | n/a | n/a | 0.0 (detected) |

---

## 7. Step 4: Maximize Throughput (Open-Ended)

### 7.1 task instructions

```
STEP 4: Maximize Throughput (Open-Ended)
=========================================

This is the final, open-ended optimization step. Your goal: maximize tokens/second
while maintaining output quality. You can use ANY technique:

- Batching (static or continuous)
- Kernel fusion
- Pallas custom kernels
- Pipeline parallelism across TPU chips
- Speculative decoding (with a draft model you create)
- Attention optimizations (flash attention, chunked attention)
- Improved scheduling
- Combining all previous optimizations
- Anything else you can think of

YOUR TASK:
Generate correct outputs for a stream of prompts as fast as possible.

REQUIREMENTS:
- Output quality: token match rate > 90% vs FP32 baseline on hidden test prompts
- No network access (all resources must be local)
- Must handle prompts of varying lengths (8 to 512 tokens)

SCORING:
  score = min(throughput / (5 * baseline_throughput), 1.0) * correctness_factor
  where:
    correctness_factor = 1.0 if token_match_rate > 0.90, else 0.0
    baseline_throughput = measured naive baseline (typically ~15-25 tok/sec)
    5x baseline = realistic expert ceiling = score 1.0

  Example: baseline = 20 tok/sec
    Your 40 tok/sec (2x) -> score = 0.4
    Your 60 tok/sec (3x) -> score = 0.6
    Your 100 tok/sec (5x) -> score = 1.0
    Your 150 tok/sec (7.5x) -> score = 1.0 (capped)

SUBMIT:
1. Final inference script: /env/submit/optimized_inference.py
   Must contain:
   - generate(params, tokenizer, prompt, max_new_tokens) -> list[int]
   - batch_generate(params, tokenizer, prompts, max_new_tokens) -> list[list[int]]
     (optional but recommended -- if missing, prompts are processed sequentially)

TIME LIMIT: 90 minutes
```

### 7.2 judge: step4_judge.py (ExecutableJudge)

```python
"""
Step 4 Judge: Maximize Throughput (Final)
==========================================

Open-ended throughput maximization.

score = min(throughput / (5 * baseline), 1.0) * correctness_factor

Correctness: token match rate > 90% on hidden prompts.
Throughput: total tokens generated / total wall-clock time.
Ceiling: 5x baseline = 1.0 (realistic for INT8 + KV cache + batching on TPU v5e).
"""

import json
import sys
import time
import importlib.util
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer


def load_module(path, name="solution"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def judge():
    submit_dir = Path("/env/submit")
    scoring_dir = Path("/env/scoring_data")
    env_data = Path("/env/env_data")

    solution_path = submit_dir / "optimized_inference.py"
    if not solution_path.exists():
        return {"reward": 0.0, "feedback": "No submission found at /env/submit/optimized_inference.py"}

    try:
        solution = load_module(solution_path)
    except Exception as e:
        return {"reward": 0.0, "feedback": f"Failed to import solution: {e}"}

    if not hasattr(solution, "generate"):
        return {"reward": 0.0, "feedback": "Solution must define a 'generate' function"}

    has_batch = hasattr(solution, "batch_generate")

    tokenizer = AutoTokenizer.from_pretrained(str(env_data / "tokenizer"))

    # load params (prefer student's quantized if available)
    params_path = submit_dir / "quantized_params.safetensors"
    if not params_path.exists():
        params_path = env_data / "weights" / "qwen_0.6b_fp32.safetensors"

    params = {}
    with safe_open(str(params_path), framework="jax") as f:
        for key in f.keys():
            params[key] = f.get_tensor(key)

    # load reference
    ref_tokens = np.load(scoring_dir / "reference_outputs" / "baseline_tokens.npz")
    with open(scoring_dir / "reference_outputs" / "baseline_throughput.json") as f:
        baseline_perf = json.load(f)

    # load ALL hidden prompts (100)
    hidden_prompts = []
    with open(scoring_dir / "hidden_test_prompts.jsonl") as f:
        for line in f:
            hidden_prompts.append(json.loads(line)["prompt"])

    max_new_tokens = 64

    # ---- warmup ----
    try:
        if has_batch:
            _ = solution.batch_generate(params, tokenizer, ["Hello", "World"], max_new_tokens=4)
        else:
            _ = solution.generate(params, tokenizer, "Hello", max_new_tokens=4)
    except Exception as e:
        return {"reward": 0.0, "feedback": f"Warmup failed: {e}"}

    # ---- throughput measurement ----
    test_prompts = hidden_prompts[:50]  # use 50 for throughput
    total_tokens = 0
    all_outputs = []

    start = time.perf_counter()

    if has_batch:
        try:
            batch_outputs = solution.batch_generate(
                params, tokenizer, test_prompts, max_new_tokens=max_new_tokens
            )
            for output in batch_outputs:
                if isinstance(output, jnp.ndarray):
                    output = np.array(output).tolist()
                all_outputs.append(output)
                total_tokens += len(output)
        except Exception as e:
            return {"reward": 0.0, "feedback": f"batch_generate failed: {e}"}
    else:
        for prompt in test_prompts:
            try:
                output = solution.generate(params, tokenizer, prompt, max_new_tokens=max_new_tokens)
                if isinstance(output, jnp.ndarray):
                    output = np.array(output).tolist()
                all_outputs.append(output)
                total_tokens += len(output)
            except Exception as e:
                all_outputs.append([])

    # ensure all async ops complete
    try:
        jax.block_until_ready(jnp.array([0.0]))
    except Exception:
        pass

    elapsed = time.perf_counter() - start
    throughput = total_tokens / max(elapsed, 0.001)

    # ---- correctness: token match rate on separate hidden prompts ----
    correctness_prompts = hidden_prompts[50:70]  # 20 prompts for correctness
    token_matches = 0
    total_ref_tokens = 0

    for i, prompt in enumerate(correctness_prompts):
        try:
            student_tokens = solution.generate(params, tokenizer, prompt, max_new_tokens=max_new_tokens)
            if isinstance(student_tokens, jnp.ndarray):
                student_tokens = np.array(student_tokens)
            if isinstance(student_tokens, list):
                student_tokens = np.array(student_tokens)

            ref_key = f"prompt_{i + 50}"
            if ref_key in ref_tokens:
                ref_tok = ref_tokens[ref_key]
                min_len = min(len(student_tokens), len(ref_tok))
                if min_len > 0:
                    token_matches += np.sum(student_tokens[:min_len] == ref_tok[:min_len])
                    total_ref_tokens += min_len
        except Exception:
            pass

    token_match_rate = token_matches / max(total_ref_tokens, 1)
    correctness_factor = 1.0 if token_match_rate > 0.90 else 0.0

    # ---- anti-gaming: check output quality isn't garbage ----
    empty_outputs = sum(1 for o in all_outputs if len(o) == 0)
    if empty_outputs > len(test_prompts) * 0.3:
        return {
            "reward": 0.0,
            "feedback": f"Too many empty outputs: {empty_outputs}/{len(test_prompts)}"
        }

    # check for degenerate outputs (all same token)
    degenerate_count = 0
    for output in all_outputs:
        if len(output) > 4 and len(set(output)) <= 2:
            degenerate_count += 1

    if degenerate_count > len(all_outputs) * 0.2:
        return {
            "reward": 0.0,
            "feedback": (
                f"Degenerate outputs detected: {degenerate_count}/{len(all_outputs)} outputs "
                f"contain only 1-2 unique tokens. This suggests broken generation logic."
            ),
        }

    # ---- final score ----
    baseline_throughput = baseline_perf["throughput_tokens_per_sec"]
    ceiling = 5.0 * baseline_throughput
    throughput_score = min(throughput / ceiling, 1.0)

    reward = round(min(max(throughput_score * correctness_factor, 0.0), 1.0), 4)

    feedback_lines = [
        f"=== Step 4: Maximize Throughput Results ===",
        f"",
        f"Correctness:",
        f"  Token match rate: {token_match_rate:.2%} (threshold: 90%)",
        f"  Correctness factor: {correctness_factor:.1f}",
        f"  Degenerate outputs: {degenerate_count}/{len(all_outputs)}",
        f"  Empty outputs: {empty_outputs}/{len(test_prompts)}",
        f"",
        f"Throughput:",
        f"  Baseline: {baseline_throughput:.1f} tokens/sec",
        f"  Your throughput: {throughput:.1f} tokens/sec",
        f"  Speedup: {throughput / max(baseline_throughput, 0.001):.2f}x",
        f"  Ceiling (5x baseline): {ceiling:.1f} tokens/sec",
        f"  Throughput score: {throughput_score:.2f}",
        f"",
        f"Final score: {reward:.4f}",
        f"",
        f"Techniques detected:",
        f"  Batch generation: {'YES' if has_batch else 'NO'}",
    ]

    if correctness_factor == 0.0:
        feedback_lines.append("")
        feedback_lines.append(
            "HINT: Token match rate too low. Make sure aggressive optimizations "
            "(quantization, cache compression) haven't degraded output quality "
            "below the 90% threshold."
        )

    speedup_actual = throughput / max(baseline_throughput, 0.001)
    if speedup_actual < 2.0 and correctness_factor == 1.0:
        feedback_lines.append("")
        feedback_lines.append(
            "HINT: Good correctness but low throughput. Consider: "
            "(1) batching multiple prompts together, "
            "(2) using Pallas flash attention, "
            "(3) pipeline parallelism across the 4 TPU chips."
        )

    return {"reward": reward, "feedback": "\n".join(feedback_lines)}


if __name__ == "__main__":
    result = judge()
    print(json.dumps(result))
    sys.exit(0)
```

### 7.3 env_data vs scoring_data

| Location | Contents | Visible to Student |
|----------|----------|--------------------|
| env_data | all baseline code, all reference docs, model weights, tokenizer, public prompts | YES |
| scoring_data | 100 hidden prompts, baseline outputs, baseline throughput measurement, all judges | NO |

### 7.4 anti-gaming measures

1. **Hidden prompts for BOTH throughput and correctness**: throughput is measured on prompts 0--49 (hidden), correctness on prompts 50--69 (hidden). Student cannot optimize for specific prompts.
2. **Degenerate output detection**: if > 20% of outputs contain only 1--2 unique tokens, score = 0.0. Prevents a student from returning garbage tokens quickly.
3. **Empty output detection**: if > 30% of outputs are empty, score = 0.0.
4. **Token match rate threshold**: 90% is intentionally more relaxed than Step 2's KL < 0.01, because aggressive throughput optimizations (batching, speculative decoding) may introduce minor divergences. But degenerate outputs are still caught.
5. **Separate correctness evaluation**: correctness is measured on a DIFFERENT set of hidden prompts than throughput, preventing a student from trading correctness on some prompts for speed on others.
6. **Realistic ceiling**: 5x baseline is achievable with INT8 + KV cache + batching but requires genuine engineering. Getting >5x requires multiple optimizations stacked together.

### 7.5 expected score distribution

| Agent Quality | Throughput | Match Rate | Score |
|---------------|-----------|------------|-------|
| No submission | n/a | n/a | 0.0 |
| Returns baseline | ~20 tok/s | 100% | 0.04 (1x / 5x) |
| KV cache only | ~50 tok/s | 99% | 0.20 (2.5x) |
| KV cache + INT8 | ~80 tok/s | 96% | 0.32 (4x) |
| + batching (batch=4) | ~120 tok/s | 94% | 0.48 (6x, capped) |
| + flash attention | ~150 tok/s | 93% | 0.60 |
| Expert (all techniques) | ~200+ tok/s | 92% | 0.80--1.0 |
| Garbage fast output | ~500 tok/s | 20% | 0.0 (correctness fail) |

---

## 8. scoring_data contents

### 8.1 hidden_test_prompts.jsonl

100 hidden prompts covering diverse domains. Example structure (actual prompts would be different):

```jsonl
{"prompt": "Explain the difference between supervised and unsupervised learning.", "category": "ml_concepts", "expected_length": "medium"}
{"prompt": "def binary_search(arr, target):", "category": "code", "expected_length": "long"}
{"prompt": "The mitochondria is the", "category": "biology", "expected_length": "short"}
{"prompt": "Translate to French: The weather is beautiful today.", "category": "translation", "expected_length": "short"}
{"prompt": "Write a haiku about neural networks.", "category": "creative", "expected_length": "short"}
```

Distribution:
- 30 ML/AI concept questions (varying length)
- 20 code completion prompts
- 15 science/factual questions
- 15 translation/multilingual prompts
- 10 creative writing prompts
- 10 instruction-following prompts

Prompt input lengths: 8--512 tokens (uniform-ish distribution).

### 8.2 baseline_logits.npz

Pre-computed last-position logits for each prompt, shape `[vocab_size]` = `[151936]`. These are the reference for KL divergence computation.

### 8.3 baseline_tokens.npz

Pre-computed greedy-decoded tokens (64 tokens per prompt). These are the reference for token match rate computation.

### 8.4 baseline_throughput.json

```json
{
    "throughput_tokens_per_sec": 18.7,
    "total_tokens": 640,
    "total_time_sec": 34.2,
    "num_prompts": 10
}
```

(Actual values measured at container build time.)

---

## 9. scoring_utils.py (shared functions)

```python
"""
Shared scoring utilities for all steps.
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import time
from pathlib import Path


def compute_kl_divergence(logits_ref, logits_test):
    """
    KL(P_ref || P_test) where P = softmax(logits).
    Returns scalar float.
    """
    p = jax.nn.softmax(jnp.array(logits_ref, dtype=jnp.float32), axis=-1)
    q = jax.nn.softmax(jnp.array(logits_test, dtype=jnp.float32), axis=-1)
    eps = 1e-10
    p = jnp.clip(p, eps, 1.0)
    q = jnp.clip(q, eps, 1.0)
    kl = jnp.sum(p * jnp.log(p / q), axis=-1)
    return float(jnp.mean(kl))


def compute_token_match_rate(student_tokens, reference_tokens):
    """
    Fraction of tokens that match between student and reference.
    Compares up to min(len(student), len(reference)) tokens.
    """
    if isinstance(student_tokens, (jnp.ndarray,)):
        student_tokens = np.array(student_tokens)
    if isinstance(student_tokens, list):
        student_tokens = np.array(student_tokens)
    if isinstance(reference_tokens, list):
        reference_tokens = np.array(reference_tokens)

    min_len = min(len(student_tokens), len(reference_tokens))
    if min_len == 0:
        return 0.0

    matches = np.sum(student_tokens[:min_len] == reference_tokens[:min_len])
    return float(matches / min_len)


def measure_throughput(generate_fn, prompts, max_new_tokens=64, warmup_prompts=2):
    """
    Measure tokens/sec throughput.
    Includes warmup, excludes it from timing.
    """
    # warmup
    for prompt in prompts[:warmup_prompts]:
        try:
            output = generate_fn(prompt, max_new_tokens=min(4, max_new_tokens))
            if hasattr(output, "block_until_ready"):
                output.block_until_ready()
        except Exception:
            pass

    # benchmark
    total_tokens = 0
    start = time.perf_counter()

    for prompt in prompts:
        try:
            output = generate_fn(prompt, max_new_tokens=max_new_tokens)
            if hasattr(output, "block_until_ready"):
                output.block_until_ready()
            total_tokens += len(output) if isinstance(output, (list, tuple)) else output.shape[-1]
        except Exception:
            pass

    elapsed = time.perf_counter() - start
    return total_tokens / max(elapsed, 0.001)


def check_output_quality(outputs, min_unique_tokens=3, max_empty_ratio=0.3):
    """
    Basic quality checks on generated outputs.
    Returns (passed: bool, reason: str).
    """
    if not outputs:
        return False, "No outputs"

    empty_count = sum(1 for o in outputs if len(o) == 0)
    if empty_count / len(outputs) > max_empty_ratio:
        return False, f"Too many empty outputs: {empty_count}/{len(outputs)}"

    degenerate = 0
    for o in outputs:
        if len(o) > 4 and len(set(o if isinstance(o, list) else o.tolist())) < min_unique_tokens:
            degenerate += 1

    if degenerate / max(len(outputs), 1) > 0.2:
        return False, f"Degenerate outputs: {degenerate}/{len(outputs)}"

    return True, "OK"


def load_hidden_prompts():
    path = Path("/env/scoring_data/hidden_test_prompts.jsonl")
    prompts = []
    with open(path) as f:
        for line in f:
            prompts.append(json.loads(line)["prompt"])
    return prompts


def load_reference_outputs():
    scoring_dir = Path("/env/scoring_data/reference_outputs")
    logits = np.load(scoring_dir / "baseline_logits.npz")
    tokens = np.load(scoring_dir / "baseline_tokens.npz")
    with open(scoring_dir / "baseline_throughput.json") as f:
        throughput = json.load(f)
    return logits, tokens, throughput
```

---

## 10. overall score aggregation

the final task score is the **weighted average** of all step scores:

| Step | Weight | Judge Type | Metric |
|------|--------|-----------|--------|
| Step 1: Profile | 0.10 | ExactMatchJudge (binary) | 3/3 answers correct |
| Step 2: INT8 Quantize | 0.25 | ExecutableJudge (continuous) | speedup * correctness |
| Step 3: KV Cache | 0.25 | ExecutableJudge (continuous) | memory_reduction * correctness |
| Step 4: Throughput | 0.40 | ExecutableJudge (continuous) | throughput / ceiling * correctness |

```
final_score = 0.10 * step1 + 0.25 * step2 + 0.25 * step3 + 0.40 * step4
```

**rationale for weights**:
- Step 1 is low weight because it's binary and straightforward for any agent that can profile.
- Step 4 gets the highest weight because it's the most open-ended and best differentiates agent quality.
- Steps 2 and 3 are equal weight as foundational optimizations.

---

## 11. expected overall score distribution

| Agent Quality | Step 1 | Step 2 | Step 3 | Step 4 | Final |
|---------------|--------|--------|--------|--------|-------|
| Random/no-op | 0.0 | 0.0 | 0.0 | 0.04 | 0.016 |
| Weak (tries but fails) | 0.0 | 0.0 | 0.2 | 0.04 | 0.066 |
| Below median | 1.0 | 0.3 | 0.2 | 0.10 | 0.225 |
| Median | 1.0 | 0.5 | 0.5 | 0.20 | 0.405 |
| Above median | 1.0 | 0.8 | 0.8 | 0.40 | 0.660 |
| Human-level | 1.0 | 0.8 | 0.8 | 0.60 | 0.740 |
| Expert | 1.0 | 1.0 | 1.0 | 1.0 | 1.000 |

the distribution is roughly continuous in the 0.2--0.8 range, which is the desired property for RL training signal.

---

## 12. profiling_utils.py (env_data helper)

```python
"""
JAX Profiling Utilities
========================

Helper functions for profiling Qwen inference on TPU.
Use these to identify bottlenecks in Step 1.
"""

import jax
import jax.numpy as jnp
import time
from functools import wraps
from typing import Callable, Any


def time_function(fn: Callable, *args, num_runs: int = 10, warmup: int = 3, **kwargs) -> dict:
    """
    Time a JAX function with proper warmup and synchronization.

    Returns dict with mean, std, min, max times in milliseconds.
    """
    # warmup (includes JIT compilation)
    for _ in range(warmup):
        result = fn(*args, **kwargs)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elif isinstance(result, tuple):
            for r in result:
                if hasattr(r, "block_until_ready"):
                    r.block_until_ready()

    # benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elif isinstance(result, tuple):
            for r in result:
                if hasattr(r, "block_until_ready"):
                    r.block_until_ready()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # ms

    import numpy as np
    times = np.array(times)
    return {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std()),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
        "times_ms": times.tolist(),
    }


def profile_by_layer(forward_fn, params, input_ids, num_layers: int = 24):
    """
    Profile time spent in each transformer layer.

    Requires the forward function to accept a `stop_at_layer` argument.
    If not available, profiles total time and estimates per-layer.
    """
    results = {}

    # total time
    total = time_function(forward_fn, params, input_ids)
    results["total"] = total

    # try per-layer profiling
    try:
        for layer_idx in range(num_layers):
            layer_time = time_function(
                forward_fn, params, input_ids, stop_at_layer=layer_idx + 1
            )
            results[f"layer_{layer_idx}"] = layer_time
    except TypeError:
        # forward_fn doesn't support stop_at_layer
        estimated_per_layer = total["mean_ms"] / num_layers
        for layer_idx in range(num_layers):
            results[f"layer_{layer_idx}"] = {
                "mean_ms": estimated_per_layer,
                "note": "estimated (total / num_layers)",
            }

    return results


def measure_scaling(forward_fn, params, tokenizer, prompt_base: str = "Hello world"):
    """
    Measure how generation time scales with sequence length.
    Useful for identifying quadratic vs linear behavior.
    """
    results = {}
    seq_lengths = [8, 16, 32, 64, 128, 256]

    for target_len in seq_lengths:
        # create input of approximately target_len tokens
        tokens = tokenizer.encode(prompt_base)
        while len(tokens) < target_len:
            tokens = tokens + tokens
        tokens = tokens[:target_len]
        input_ids = jnp.array([tokens], dtype=jnp.int32)

        timing = time_function(forward_fn, params, input_ids, num_runs=5, warmup=2)
        results[target_len] = timing

    # compute scaling exponent
    import numpy as np
    lens = np.array(list(results.keys()), dtype=float)
    times = np.array([results[l]["mean_ms"] for l in results.keys()])

    # fit log(time) = exponent * log(length) + const
    log_lens = np.log(lens)
    log_times = np.log(times + 1e-10)
    coeffs = np.polyfit(log_lens, log_times, 1)
    exponent = coeffs[0]

    results["scaling_exponent"] = float(exponent)
    results["scaling_interpretation"] = (
        "linear" if exponent < 1.3 else
        "superlinear" if exponent < 1.7 else
        "quadratic" if exponent < 2.3 else
        "superquadratic"
    )

    return results


def memory_snapshot():
    """Get current TPU/device memory usage."""
    try:
        devices = jax.devices()
        stats = []
        for device in devices:
            mem = device.memory_stats()
            if mem:
                stats.append({
                    "device": str(device),
                    "bytes_in_use": mem.get("bytes_in_use", 0),
                    "peak_bytes_in_use": mem.get("peak_bytes_in_use", 0),
                    "bytes_limit": mem.get("bytes_limit", 0),
                })
        return stats
    except Exception as e:
        return [{"error": str(e)}]
```

---

## 13. reference documentation files

### 13.1 env_data/reference_docs/aqt_reference.md

(offline copy of AQT API docs, ~2000 words covering DotGeneral, Tensor, IntNumerics, calibration axes, and usage examples)

### 13.2 env_data/reference_docs/jax_profiling.md

(offline copy of JAX profiling guide covering jax.profiler.trace, time_function patterns, block_until_ready)

### 13.3 env_data/reference_docs/pallas_reference.md

(offline copy of Pallas kernel writing guide for TPU, including flash_attention example)

### 13.4 env_data/reference_docs/tpu_optimization_guide.md

Key content:
- TPU v5e specs: 4 chips, 16 GB HBM each (64 GB total), BF16/FP8 support
- Memory bandwidth: ~820 GB/s per chip
- Compute: ~197 TFLOPS BF16, ~394 TOPS INT8
- Key insight: inference is memory-bandwidth-bound, not compute-bound
- Optimization priority: reduce memory traffic > increase parallelism > custom kernels

---

## 14. Qwen-0.6B model details

### 14.1 architecture

| Parameter | Value |
|-----------|-------|
| hidden_size | 1024 |
| intermediate_size (MLP) | 2816 |
| num_hidden_layers | 24 |
| num_attention_heads | 16 |
| num_key_value_heads | 16 (MHA, not GQA) |
| head_dim | 64 |
| vocab_size | 151936 |
| max_position_embeddings | 32768 |
| rope_theta | 1,000,000 |
| rms_norm_eps | 1e-6 |
| total parameters | ~621M |

### 14.2 weight sizes (FP32)

| Component | Shape | Size |
|-----------|-------|------|
| embed_tokens | [151936, 1024] | 592 MB |
| per-layer q_proj | [1024, 1024] | 4 MB |
| per-layer k_proj | [1024, 1024] | 4 MB |
| per-layer v_proj | [1024, 1024] | 4 MB |
| per-layer o_proj | [1024, 1024] | 4 MB |
| per-layer gate_proj | [1024, 2816] | 11.5 MB |
| per-layer up_proj | [1024, 2816] | 11.5 MB |
| per-layer down_proj | [2816, 1024] | 11.5 MB |
| per-layer norms (x2) | [1024] | 8 KB |
| lm_head | [151936, 1024] | 592 MB |
| **Total** | | **~2.4 GB** |

### 14.3 memory budget on TPU v5e

| Component | FP32 | INT8 | FP8 |
|-----------|------|------|-----|
| Model weights | 2.4 GB | 0.6 GB | 0.6 GB |
| KV cache (seq=2048) | 384 MB | 96 MB | 96 MB |
| Activations (batch=1) | ~200 MB | ~200 MB | ~200 MB |
| **Total per chip** | ~3.0 GB | ~0.9 GB | ~0.9 GB |
| **Available per chip** | 16 GB | 16 GB | 16 GB |
| **Headroom** | 13 GB | 15.1 GB | 15.1 GB |

With INT8 weights and FP8 KV cache, there is abundant headroom for batching (up to ~16 concurrent requests at seq_len=2048).

---

## 15. calibration data details

### 15.1 calibration_prompts.jsonl

200 prompts for INT8 calibration, distributed as:
- 60 English prose (wikipedia-style)
- 40 code (Python, JavaScript, C++)
- 30 scientific text (arxiv abstracts)
- 20 multilingual (French, German, Chinese, Japanese)
- 20 instruction-following ("Explain...", "List...", "Compare...")
- 15 mathematical expressions
- 15 conversational/short prompts

Input length distribution: mean 128 tokens, range [16, 512].

### 15.2 public_test_prompts.jsonl

50 prompts the student can use for development and self-evaluation. Same domain distribution as calibration but different content.

---

## 16. key references

| Resource | URL |
|----------|-----|
| Qwen3 Technical Report | https://arxiv.org/abs/2505.09388 |
| AQT Library | https://github.com/google/aqt |
| JetStream (Google TPU inference) | https://github.com/AI-Hypercomputer/JetStream |
| JAX Pallas Flash Attention | https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py |
| INT8 Quantization for LLMs | https://arxiv.org/abs/2208.07339 (LLM.int8()) |
| FlashAttention-2 | https://arxiv.org/abs/2307.08691 |
| Continuous Batching (Orca) | https://www.usenix.org/conference/osdi22/presentation/yu |
| FP8 Training and Inference | https://arxiv.org/abs/2209.05433 |

---

*end of specification*
