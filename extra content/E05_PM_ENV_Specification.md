# E05: Attention Optimization pm_env Specification

> production-ready pm_env spec for E05 (Flash Attention + Optimization)
> design: open-ended optimization, NOT fill-in-the-blank
> created: 2026-02-09 | author: claude opus 4.6

---

## executive summary

this environment gives the student a naive O(N^2) attention implementation in JAX and a small transformer model. the task is: **make it faster and more memory-efficient while keeping it correct.** the Flash Attention 2 paper is provided as a HINT (one possible technique among many), not a requirement. the student may use ANY technique: tiling, chunking, recomputation, gradient checkpointing, Pallas kernels, custom XLA ops, algorithmic improvements, etc.

**why this is a good pm_env**:
1. no single known pattern solves it -- reasoning required to analyze bottlenecks and choose approach
2. many points of failure -- memory management, numerical stability, correctness preservation, XLA compilation
3. continuous reward -- speedup ratios and memory reduction are continuous, not binary
4. constant improvement space -- can always tune block sizes, add fusion, improve algorithms
5. hill-climbing on core metrics -- speedup and memory reduction are directly measurable
6. open-ended optimization -- not "implement Flash Attention or fail"

**critical design decision**: this is NOT "implement Flash Attention from the paper." this IS "here is a naive O(N^2) attention implementation. make it faster while keeping it correct. you have access to the Flash Attention paper for reference, but you can use ANY technique."

---

## 1. environment structure

### 1.1 container spec

```yaml
base_image: python:3.11-slim
hardware: TPU v5e (2x2, 4 chips, 64 GB HBM) OR CPU fallback
memory_limit: 16 GB system RAM (HBM is separate)
network: NONE (fully offline)
storage: 10 GB ephemeral
total_time_limit: 4 hours
per_tool_call_limit: 10 minutes
grading_time_limit: 4 hours
```

### 1.2 file layout

```
/env/
├── env_data/                          # student-visible
│   ├── README.md                      # task instructions
│   ├── baseline/
│   │   ├── naive_attention.py         # the O(N^2) attention to optimize
│   │   ├── transformer_model.py       # small transformer using naive attention
│   │   ├── profiling_utils.py         # timing + memory measurement helpers
│   │   └── test_correctness.py        # student can self-check correctness
│   ├── reference/
│   │   ├── flash_attention_2_paper.pdf  # Dao 2023 (arXiv:2307.08691)
│   │   └── paper_summary.md           # extracted pseudocode + key insights
│   ├── data/
│   │   ├── sample_text.txt            # text for perplexity evaluation (visible)
│   │   └── tokenizer.json             # simple BPE tokenizer
│   └── workspace/                     # student writes code here
│       └── optimized_attention.py     # skeleton for student solution
│
├── scoring_data/                      # hidden from student
│   ├── test_inputs/
│   │   ├── correctness_inputs.pkl     # Q,K,V tensors + expected outputs
│   │   ├── reference_outputs_2048.pkl # reference attention output at seq=2048
│   │   ├── reference_outputs_8192.pkl # reference attention output at seq=8192
│   │   └── reference_outputs_16384.pkl # reference at seq=16384 (chunked)
│   ├── test_text/
│   │   └── held_out_text.txt          # text for perplexity (hidden)
│   ├── baseline_benchmarks/
│   │   ├── naive_memory_2048.json     # peak memory of naive at seq=2048
│   │   ├── naive_time_2048.json       # time per forward pass at seq=2048
│   │   ├── naive_memory_8192.json     # (will OOM -- stored as null)
│   │   └── naive_time_8192.json
│   └── judges/
│       ├── step1_judge.py
│       ├── step2_judge.py
│       ├── step3_judge.py
│       └── step4_judge.py
│
├── tasks.py                           # task + step definitions
├── setup_data.py                      # generates reference data
├── pre_hook.py                        # runs before student starts
└── env_requirements.txt               # python dependencies
```

### 1.3 env_data vs scoring_data

| category | env_data (student sees) | scoring_data (hidden) |
|----------|------------------------|-----------------------|
| attention code | naive_attention.py (the thing to optimize) | reference outputs at all seq lengths |
| model | transformer_model.py (small model) | held-out text for perplexity |
| paper | flash_attention_2_paper.pdf | -- |
| benchmarks | profiling_utils.py (tools) | actual baseline measurements |
| correctness | test_correctness.py (basic self-check) | comprehensive correctness inputs |
| test data | sample_text.txt (small sample) | held_out_text.txt (grading text) |

---

## 2. env_requirements.txt

```
jax[tpu]==0.4.35
jaxlib==0.4.35
flax==0.10.2
optax==0.2.4
numpy>=1.26.0
scipy>=1.12.0
tiktoken>=0.7.0
psutil>=5.9.0
```

---

## 3. baseline code (env_data)

### 3.1 naive_attention.py

```python
"""
Naive O(N^2) scaled dot-product attention in JAX.

This is the baseline you must optimize.
It materializes the full [seq_q, seq_kv] attention matrix.

Memory: O(batch * heads * seq_q * seq_kv) -- quadratic in sequence length.
Time: O(batch * heads * seq_q * seq_kv * head_dim) -- quadratic in sequence length.

At seq_len=2048 with 8 heads and head_dim=64:
  Attention matrix alone = 2048 * 2048 * 8 * 4 bytes = 128 MB per batch element.

At seq_len=16384:
  Attention matrix = 16384 * 16384 * 8 * 4 bytes = 8 GB per batch element.
  This WILL run out of memory.

Your goal: make this faster AND more memory-efficient, while producing
the same output (within numerical tolerance).
"""

import jax
import jax.numpy as jnp
from typing import Optional


def naive_attention(
    q: jax.Array,  # [batch, seq_q, num_heads, head_dim]
    k: jax.Array,  # [batch, seq_kv, num_heads, head_dim]
    v: jax.Array,  # [batch, seq_kv, num_heads, head_dim]
    mask: Optional[jax.Array] = None,  # [batch, 1, seq_q, seq_kv] or broadcastable
) -> jax.Array:
    """
    Standard scaled dot-product attention.
    Returns: [batch, seq_q, num_heads, head_dim]
    """
    scale = q.shape[-1] ** -0.5

    # Transpose to [batch, num_heads, seq_q, head_dim]
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    # [batch, num_heads, seq_q, seq_kv] -- THIS is the O(N^2) memory bottleneck
    scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * scale

    if mask is not None:
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)

    attn_weights = jax.nn.softmax(scores, axis=-1)

    # [batch, num_heads, seq_q, head_dim]
    output = jnp.matmul(attn_weights, v)

    # Back to [batch, seq_q, num_heads, head_dim]
    output = jnp.transpose(output, (0, 2, 1, 3))
    return output


def make_causal_mask(seq_len: int) -> jax.Array:
    """Create causal (lower-triangular) attention mask."""
    return jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
```

### 3.2 transformer_model.py

```python
"""
Small transformer model using naive attention.

Architecture:
- 4 layers, 8 heads, head_dim=64, hidden_dim=512
- ~13M parameters
- Vocabulary: 32000 tokens (BPE)

This model is small enough to run on CPU for correctness checks,
but attention becomes the bottleneck at long sequence lengths.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional
import json
import os

from naive_attention import naive_attention, make_causal_mask


class TransformerConfig:
    vocab_size: int = 32000
    num_layers: int = 4
    num_heads: int = 8
    head_dim: int = 64
    hidden_dim: int = 512  # num_heads * head_dim
    mlp_dim: int = 2048
    max_seq_len: int = 16384
    dropout_rate: float = 0.0


class AttentionLayer(nn.Module):
    config: TransformerConfig
    attention_fn: callable = naive_attention

    @nn.compact
    def __call__(self, x, mask=None):
        B, T, D = x.shape
        cfg = self.config

        q = nn.Dense(cfg.hidden_dim, name="q_proj")(x)
        k = nn.Dense(cfg.hidden_dim, name="k_proj")(x)
        v = nn.Dense(cfg.hidden_dim, name="v_proj")(x)

        q = q.reshape(B, T, cfg.num_heads, cfg.head_dim)
        k = k.reshape(B, T, cfg.num_heads, cfg.head_dim)
        v = v.reshape(B, T, cfg.num_heads, cfg.head_dim)

        attn_output = self.attention_fn(q, k, v, mask=mask)
        attn_output = attn_output.reshape(B, T, D)

        output = nn.Dense(cfg.hidden_dim, name="out_proj")(attn_output)
        return output


class TransformerBlock(nn.Module):
    config: TransformerConfig
    attention_fn: callable = naive_attention

    @nn.compact
    def __call__(self, x, mask=None):
        residual = x
        x = nn.LayerNorm()(x)
        x = AttentionLayer(self.config, self.attention_fn)(x, mask)
        x = x + residual

        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.config.mlp_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.config.hidden_dim)(x)
        x = x + residual

        return x


class SmallTransformer(nn.Module):
    config: TransformerConfig
    attention_fn: callable = naive_attention

    @nn.compact
    def __call__(self, token_ids, mask=None):
        cfg = self.config
        B, T = token_ids.shape

        x = nn.Embed(cfg.vocab_size, cfg.hidden_dim)(token_ids)

        pos = jnp.arange(T)
        pos_emb = nn.Embed(cfg.max_seq_len, cfg.hidden_dim)(pos)
        x = x + pos_emb[None, :, :]

        if mask is None:
            mask = make_causal_mask(T)

        for _ in range(cfg.num_layers):
            x = TransformerBlock(cfg, self.attention_fn)(x, mask)

        x = nn.LayerNorm()(x)
        logits = nn.Dense(cfg.vocab_size)(x)
        return logits


def load_model_params(path="/env/env_data/data/model_params.msgpack"):
    """Load pre-trained model parameters."""
    from flax.serialization import from_bytes
    with open(path, "rb") as f:
        return from_bytes(None, f.read())


def compute_perplexity(model, params, token_ids):
    """
    Compute perplexity on a sequence of token IDs.
    Lower is better.
    """
    logits = model.apply(params, token_ids[None, :])
    logits = logits[0, :-1, :]  # shift: predict next token
    targets = token_ids[1:]

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = log_probs[jnp.arange(len(targets)), targets]
    avg_nll = -jnp.mean(target_log_probs)
    perplexity = jnp.exp(avg_nll)
    return float(perplexity)
```

### 3.3 profiling_utils.py

```python
"""
Profiling utilities for measuring memory and time.

Use these to understand the baseline performance and verify your optimizations.
"""

import jax
import jax.numpy as jnp
import time
import psutil
import os
import json


def measure_peak_memory(fn, *args, n_runs=3, **kwargs):
    """
    Measure peak memory consumption of a JAX function.

    Returns peak memory in bytes.

    NOTE: On TPU, this measures HBM usage. On CPU, system RAM.
    """
    results = []
    for _ in range(n_runs):
        jax.clear_caches()

        if hasattr(jax.local_devices()[0], 'memory_stats'):
            # TPU: use device memory stats
            device = jax.local_devices()[0]
            before = device.memory_stats()['peak_bytes_in_use']
            output = fn(*args, **kwargs)
            jax.block_until_ready(output)
            after = device.memory_stats()['peak_bytes_in_use']
            peak = after - before
        else:
            # CPU fallback: use psutil
            process = psutil.Process(os.getpid())
            before = process.memory_info().rss
            output = fn(*args, **kwargs)
            jax.block_until_ready(output)
            after = process.memory_info().rss
            peak = after - before

        results.append(max(peak, 0))

    return int(jnp.median(jnp.array(results)))


def measure_time(fn, *args, n_warmup=5, n_runs=20, **kwargs):
    """
    Measure execution time of a JAX function.

    Returns time in seconds (median of n_runs).
    """
    # Warmup: compile + fill caches
    for _ in range(n_warmup):
        output = fn(*args, **kwargs)
        jax.block_until_ready(output)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        output = fn(*args, **kwargs)
        jax.block_until_ready(output)
        end = time.perf_counter()
        times.append(end - start)

    return float(jnp.median(jnp.array(times)))


def profile_attention(attention_fn, batch=1, seq_len=2048, num_heads=8, head_dim=64):
    """
    Profile an attention function at a given sequence length.

    Returns dict with memory and time measurements.
    """
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (batch, seq_len, num_heads, head_dim))
    k = jax.random.normal(k2, (batch, seq_len, num_heads, head_dim))
    v = jax.random.normal(k3, (batch, seq_len, num_heads, head_dim))

    mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))

    try:
        peak_mem = measure_peak_memory(attention_fn, q, k, v, mask=mask)
        fwd_time = measure_time(attention_fn, q, k, v, mask=mask)
        return {
            "seq_len": seq_len,
            "peak_memory_bytes": peak_mem,
            "peak_memory_mb": peak_mem / 1e6,
            "forward_time_s": fwd_time,
            "forward_time_ms": fwd_time * 1000,
            "status": "ok",
        }
    except Exception as e:
        return {
            "seq_len": seq_len,
            "status": "error",
            "error": str(e),
        }


def generate_profile_report(attention_fn, seq_lengths=[512, 1024, 2048, 4096, 8192]):
    """Generate a full profiling report across sequence lengths."""
    report = {}
    for seq_len in seq_lengths:
        print(f"Profiling seq_len={seq_len}...")
        result = profile_attention(attention_fn, seq_len=seq_len)
        report[seq_len] = result
        if result["status"] == "error":
            print(f"  ERROR: {result['error']}")
            break
        else:
            print(f"  Memory: {result['peak_memory_mb']:.1f} MB")
            print(f"  Time: {result['forward_time_ms']:.2f} ms")
    return report
```

### 3.4 test_correctness.py (student self-check)

```python
"""
Self-check: verify your optimized attention matches naive attention.

Run this to check correctness before submitting.
This tests a SUBSET of what the judges test -- passing this does NOT guarantee full marks.
"""

import jax
import jax.numpy as jnp
import sys
import os

sys.path.insert(0, "/env/env_data/baseline")
from naive_attention import naive_attention, make_causal_mask


def check_correctness(optimized_fn, atol=1e-5):
    """Check optimized attention against naive reference."""
    key = jax.random.PRNGKey(42)
    test_configs = [
        {"batch": 1, "seq": 64, "heads": 4, "dim": 32, "label": "tiny"},
        {"batch": 2, "seq": 256, "heads": 8, "dim": 64, "label": "small"},
        {"batch": 1, "seq": 512, "heads": 8, "dim": 64, "label": "medium"},
    ]

    all_passed = True

    for cfg in test_configs:
        k1, k2, k3 = jax.random.split(key, 3)
        q = jax.random.normal(k1, (cfg["batch"], cfg["seq"], cfg["heads"], cfg["dim"]))
        k = jax.random.normal(k2, (cfg["batch"], cfg["seq"], cfg["heads"], cfg["dim"]))
        v = jax.random.normal(k3, (cfg["batch"], cfg["seq"], cfg["heads"], cfg["dim"]))

        ref = naive_attention(q, k, v)

        try:
            out = optimized_fn(q, k, v)
        except Exception as e:
            print(f"FAIL [{cfg['label']}]: raised {type(e).__name__}: {e}")
            all_passed = False
            continue

        max_diff = float(jnp.max(jnp.abs(out - ref)))

        if max_diff < atol:
            print(f"PASS [{cfg['label']}]: max_diff={max_diff:.2e}")
        else:
            print(f"FAIL [{cfg['label']}]: max_diff={max_diff:.2e} > atol={atol:.0e}")
            all_passed = False

        # Also test with causal mask
        mask = make_causal_mask(cfg["seq"])
        ref_causal = naive_attention(q, k, v, mask=mask)
        try:
            out_causal = optimized_fn(q, k, v, mask=mask)
            max_diff_causal = float(jnp.max(jnp.abs(out_causal - ref_causal)))
            if max_diff_causal < atol:
                print(f"PASS [{cfg['label']} causal]: max_diff={max_diff_causal:.2e}")
            else:
                print(f"FAIL [{cfg['label']} causal]: max_diff={max_diff_causal:.2e}")
                all_passed = False
        except Exception as e:
            print(f"FAIL [{cfg['label']} causal]: raised {type(e).__name__}: {e}")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    # Import student solution
    sys.path.insert(0, "/env/env_data/workspace")
    try:
        from optimized_attention import optimized_attention
        print("=== Correctness Self-Check ===\n")
        passed = check_correctness(optimized_attention)
        print(f"\n{'ALL PASSED' if passed else 'SOME TESTS FAILED'}")
    except ImportError:
        print("Could not import optimized_attention from workspace/optimized_attention.py")
        print("Make sure your function is named 'optimized_attention' with signature:")
        print("  def optimized_attention(q, k, v, mask=None) -> jax.Array")
```

### 3.5 optimized_attention.py (skeleton for student)

```python
"""
Your optimized attention implementation.

Signature MUST match:
    optimized_attention(q, k, v, mask=None) -> jax.Array

Where:
    q: [batch, seq_q, num_heads, head_dim]
    k: [batch, seq_kv, num_heads, head_dim]
    v: [batch, seq_kv, num_heads, head_dim]
    mask: [batch, 1, seq_q, seq_kv] or None
    returns: [batch, seq_q, num_heads, head_dim]

The output must match naive_attention(q, k, v, mask) within atol=1e-5.

You may:
  - Use any technique: tiling, chunking, recomputation, Pallas, etc.
  - Read the Flash Attention 2 paper in /env/env_data/reference/
  - Use jax.lax.scan, jax.checkpoint, custom Pallas kernels, etc.
  - Create helper functions and modules

You may NOT:
  - Use jax.experimental.pallas.ops.tpu.flash_attention (pre-built)
  - Import external attention libraries
  - Hardcode expected outputs
"""

import jax
import jax.numpy as jnp
from typing import Optional


def optimized_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: Optional[jax.Array] = None,
) -> jax.Array:
    """Your optimized attention. Must match naive_attention output."""
    raise NotImplementedError("Implement your optimized attention here")
```

### 3.6 paper_summary.md (reference)

```markdown
# Flash Attention 2 -- Key Insights

## Core Idea

Standard attention materializes the full N x N attention matrix.
Flash Attention computes attention in BLOCKS, maintaining running
statistics (online softmax) to avoid materializing the full matrix.

## Algorithm (Forward Pass)

1. Split Q into blocks of size B_q
2. Split K, V into blocks of size B_kv
3. For each Q block i:
   - Init: O=0, m=-inf, l=0
   - For each KV block j:
     a. S = Q_i @ K_j^T / sqrt(d)        -- block scores
     b. m_new = max(m, rowmax(S))          -- update running max
     c. P = exp(S - m_new)                 -- safe softmax numerator
     d. l_new = exp(m - m_new) * l + rowsum(P)  -- update running sum
     e. O = diag(exp(m - m_new)) * (l/l_new) * O + (1/l_new) * P @ V_j
     f. m = m_new, l = l_new
   - Result: O_i = O (already normalized)

## Memory: O(N) instead of O(N^2)

The key insight is that we never materialize the full N x N attention
matrix. Instead, we only ever hold one B_q x B_kv block at a time.

## Other Techniques You Might Consider

- Gradient checkpointing (jax.checkpoint): trade compute for memory
- Chunked softmax: process softmax in chunks without full materialization
- Mixed precision: use bfloat16 for compute, float32 for accumulation
- Pallas kernels: write custom TPU/GPU kernels for fused operations
- Ring attention: for distributed attention across devices
- Linear attention approximations: approximate softmax with kernels
```

---

## 4. task definition -- ONE task, 4 progressive steps

### 4.1 task instructions (README.md given to student)

```markdown
# Attention Optimization Challenge

You are given a **naive O(N^2) attention implementation** and a small transformer model.

Your goal: **make attention faster and more memory-efficient** while keeping it correct.

## What You Have

- `baseline/naive_attention.py` -- the attention function to optimize
- `baseline/transformer_model.py` -- a small transformer that uses it
- `baseline/profiling_utils.py` -- tools to measure memory and time
- `baseline/test_correctness.py` -- self-check for correctness
- `reference/flash_attention_2_paper.pdf` -- one approach (not the only one!)
- `reference/paper_summary.md` -- extracted pseudocode
- `workspace/optimized_attention.py` -- write your solution here

## Rules

1. Your function signature MUST be: `optimized_attention(q, k, v, mask=None) -> jax.Array`
2. Output MUST match `naive_attention` within `atol=1e-5`
3. You may NOT use `jax.experimental.pallas.ops.tpu.flash_attention` (pre-built)
4. You may NOT import external attention libraries
5. You MAY use: jax.lax.scan, jax.checkpoint, Pallas (custom kernels), any JAX API

## Steps

### Step 1: Understand the Baseline
Profile the naive attention. Run it at seq_len=2048.
Report: peak memory (bytes), time per forward pass (seconds), and the biggest bottleneck.

### Step 2: Reduce Memory
Optimize attention to use less memory at seq_len=2048.
The Flash Attention paper in `reference/` describes one approach.
You can use ANY technique.

### Step 3: Optimize Speed
Now also make it fast. Measured at seq_len=2048 AND seq_len=8192.
Consider: kernel fusion, better tiling, Pallas, reduced dtype, etc.

### Step 4: Scale to Long Sequences
Run at seq_len=16384. The naive baseline OOMs here.
Your solution must actually work at this length and produce correct output.
Model perplexity is measured on held-out text.

## Submission

Use the `submit_answers` tool with the following keys:
- Step 1: `peak_memory_bytes`, `forward_time_seconds`, `biggest_bottleneck`
- Steps 2-4: your code in `workspace/optimized_attention.py` is evaluated automatically

## Hints

- Start by profiling. Understand WHERE the memory goes.
- The attention matrix at seq=2048 with 8 heads is 128 MB per batch element.
- At seq=16384, it would be 8 GB -- impossible to fit in memory.
- The paper in `reference/` describes block-wise computation with online softmax.
- But there are other approaches: chunking, recomputation, approximation, etc.
- Use `test_correctness.py` to verify before submitting.
```

---

## 5. step definitions and judges

### 5.1 Step 1: Understand the Baseline

**type**: ExactMatchJudge (within tolerance)

**instructions to student**:
```
Profile the naive attention at seq_len=2048, batch=1, num_heads=8, head_dim=64.

Submit three values:
1. peak_memory_bytes: peak memory in bytes (integer, within 10% tolerance)
2. forward_time_seconds: forward pass time in seconds (float, within 20% tolerance)
3. biggest_bottleneck: one of "attention_matrix_materialization", "softmax_computation",
   "memory_bandwidth", "compute_bound" (exact string match)

Use profiling_utils.py to measure. Run multiple times for stability.
```

**judge (step1_judge.py)**:

```python
"""
Step 1 Judge: Verify student profiling answers.
ExactMatchJudge with tolerances.

Input: student answers via submit_answers
Expected: pre-computed baseline measurements
Score: binary (0.0 or 1.0)
"""

import json
import os
import sys


def judge(student_answers: dict) -> dict:
    """
    Evaluate Step 1 answers.

    student_answers expected keys:
        - peak_memory_bytes: int
        - forward_time_seconds: float
        - biggest_bottleneck: str

    Returns:
        {"score": float, "feedback": str}
    """
    # Load pre-computed reference
    ref_path = "/env/scoring_data/baseline_benchmarks"
    with open(os.path.join(ref_path, "naive_memory_2048.json")) as f:
        ref_memory = json.load(f)
    with open(os.path.join(ref_path, "naive_time_2048.json")) as f:
        ref_time = json.load(f)

    ref_peak_memory = ref_memory["peak_memory_bytes"]
    ref_fwd_time = ref_time["forward_time_seconds"]
    ref_bottleneck = "attention_matrix_materialization"

    errors = []
    checks_passed = 0
    total_checks = 3

    # Check 1: peak memory (within 10%)
    student_mem = student_answers.get("peak_memory_bytes")
    if student_mem is None:
        errors.append("Missing peak_memory_bytes")
    else:
        student_mem = int(student_mem)
        ratio = abs(student_mem - ref_peak_memory) / ref_peak_memory
        if ratio <= 0.10:
            checks_passed += 1
        else:
            errors.append(
                f"peak_memory_bytes: {student_mem} is {ratio:.0%} off from expected. "
                f"Must be within 10%."
            )

    # Check 2: forward time (within 20%)
    student_time = student_answers.get("forward_time_seconds")
    if student_time is None:
        errors.append("Missing forward_time_seconds")
    else:
        student_time = float(student_time)
        ratio = abs(student_time - ref_fwd_time) / ref_fwd_time
        if ratio <= 0.20:
            checks_passed += 1
        else:
            errors.append(
                f"forward_time_seconds: {student_time:.4f} is {ratio:.0%} off from expected. "
                f"Must be within 20%."
            )

    # Check 3: biggest bottleneck (exact match)
    student_bottleneck = student_answers.get("biggest_bottleneck", "").strip().lower()
    if student_bottleneck == ref_bottleneck:
        checks_passed += 1
    else:
        errors.append(
            f"biggest_bottleneck: '{student_bottleneck}' does not match expected. "
            f"Choose from: attention_matrix_materialization, softmax_computation, "
            f"memory_bandwidth, compute_bound"
        )

    score = 1.0 if checks_passed == total_checks else 0.0

    feedback = f"Checks passed: {checks_passed}/{total_checks}."
    if errors:
        feedback += " Errors: " + "; ".join(errors)

    return {"score": score, "feedback": feedback}


if __name__ == "__main__":
    # Called by grading harness
    import json
    answers_path = sys.argv[1]
    with open(answers_path) as f:
        answers = json.load(f)
    result = judge(answers)
    print(json.dumps(result))
```

**anti-gaming measures**:
- reference values are pre-computed on the same hardware and stored in scoring_data
- tolerances (10% memory, 20% time) account for hardware variance
- bottleneck answer has only 4 options -- student must actually profile to know which one
- the correct answer ("attention_matrix_materialization") requires understanding where O(N^2) memory goes

**expected score distribution**:
- 90% of students should pass this (profiling is straightforward)
- 10% will submit wrong bottleneck or badly wrong measurements

---

### 5.2 Step 2: Implement Memory-Efficient Attention

**type**: ExecutableJudge (continuous reward)

**instructions to student**:
```
Reduce the peak memory of attention at seq_len=2048 while maintaining correctness.

Your function in workspace/optimized_attention.py will be tested for:
1. Correctness: max absolute difference from naive_attention < 1e-5
2. Memory reduction: peak_memory(optimized) / peak_memory(naive)

Score = memory_reduction_ratio * correctness_factor

Where:
- memory_reduction_ratio = 1 - (optimized_memory / naive_memory), capped at [0, 1]
- correctness_factor = 1.0 if max_diff < 1e-5, else 0.0

A memory reduction of 50% with correct output gives score = 0.5.
A memory reduction of 90% with correct output gives score = 0.9.

Hint: the Flash Attention paper describes an approach that achieves ~97% memory reduction
at seq_len=2048.
```

**judge (step2_judge.py)**:

```python
"""
Step 2 Judge: Memory-efficient attention.
ExecutableJudge with continuous reward.

Measures memory reduction while verifying correctness.
Score = memory_reduction_ratio * correctness_factor

correctness_factor: 1.0 if max_diff < 1e-5, else 0.0
memory_reduction_ratio: 1 - (optimized_memory / naive_memory), capped at [0, 1]
"""

import jax
import jax.numpy as jnp
import json
import os
import sys
import importlib.util
import traceback

sys.path.insert(0, "/env/env_data/baseline")
from naive_attention import naive_attention, make_causal_mask
from profiling_utils import measure_peak_memory


CORRECTNESS_ATOL = 1e-5
SEQ_LEN = 2048
BATCH = 1
NUM_HEADS = 8
HEAD_DIM = 64


def load_student_fn():
    """Load the student's optimized_attention function."""
    spec = importlib.util.spec_from_file_location(
        "optimized_attention",
        "/env/env_data/workspace/optimized_attention.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.optimized_attention


def load_reference_memory():
    """Load pre-computed naive memory baseline."""
    with open("/env/scoring_data/baseline_benchmarks/naive_memory_2048.json") as f:
        data = json.load(f)
    return data["peak_memory_bytes"]


def check_correctness(student_fn, atol=CORRECTNESS_ATOL):
    """
    Check correctness against naive attention on multiple inputs.
    Returns (is_correct, max_diff, details).
    """
    key = jax.random.PRNGKey(12345)
    test_cases = [
        {"batch": 1, "seq": 256, "heads": 8, "dim": 64, "use_mask": False},
        {"batch": 1, "seq": 512, "heads": 8, "dim": 64, "use_mask": True},
        {"batch": 2, "seq": SEQ_LEN, "heads": NUM_HEADS, "dim": HEAD_DIM, "use_mask": True},
    ]

    overall_max_diff = 0.0
    details = []

    for tc in test_cases:
        k1, k2, k3, key = jax.random.split(key, 4)
        q = jax.random.normal(k1, (tc["batch"], tc["seq"], tc["heads"], tc["dim"]))
        k = jax.random.normal(k2, (tc["batch"], tc["seq"], tc["heads"], tc["dim"]))
        v = jax.random.normal(k3, (tc["batch"], tc["seq"], tc["heads"], tc["dim"]))

        mask = make_causal_mask(tc["seq"]) if tc["use_mask"] else None

        ref = naive_attention(q, k, v, mask=mask)

        try:
            out = student_fn(q, k, v, mask=mask)
        except Exception as e:
            details.append(f"CRASH on seq={tc['seq']}: {e}")
            return False, float("inf"), details

        if out.shape != ref.shape:
            details.append(
                f"Shape mismatch on seq={tc['seq']}: got {out.shape}, expected {ref.shape}"
            )
            return False, float("inf"), details

        max_diff = float(jnp.max(jnp.abs(out - ref)))
        overall_max_diff = max(overall_max_diff, max_diff)
        details.append(f"seq={tc['seq']} mask={tc['use_mask']}: max_diff={max_diff:.2e}")

    is_correct = overall_max_diff < atol
    return is_correct, overall_max_diff, details


def check_no_banned_imports():
    """
    Verify student code does not use banned imports.
    Returns (clean, issues).
    """
    issues = []
    code_path = "/env/env_data/workspace/optimized_attention.py"
    with open(code_path) as f:
        code = f.read()

    banned_patterns = [
        "pallas.ops.tpu.flash_attention",
        "flash_attn",
        "xformers",
        "from torch",
        "import torch",
    ]
    for pattern in banned_patterns:
        if pattern in code:
            issues.append(f"Banned import/usage detected: '{pattern}'")

    return len(issues) == 0, issues


def judge() -> dict:
    """Main judge function."""
    # Check for banned imports
    clean, import_issues = check_no_banned_imports()
    if not clean:
        return {
            "score": 0.0,
            "feedback": "Banned imports detected: " + "; ".join(import_issues),
        }

    # Load student function
    try:
        student_fn = load_student_fn()
    except Exception as e:
        return {
            "score": 0.0,
            "feedback": f"Could not load optimized_attention: {e}",
        }

    # Check correctness
    try:
        is_correct, max_diff, details = check_correctness(student_fn)
    except Exception as e:
        return {
            "score": 0.0,
            "feedback": f"Correctness check crashed: {e}\n{traceback.format_exc()}",
        }

    if not is_correct:
        return {
            "score": 0.0,
            "feedback": (
                f"Correctness FAILED: max_diff={max_diff:.2e} > atol={CORRECTNESS_ATOL:.0e}. "
                f"Details: {'; '.join(details)}"
            ),
        }

    # Measure memory
    ref_memory = load_reference_memory()

    key = jax.random.PRNGKey(99999)
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM))
    k = jax.random.normal(k2, (BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM))
    v = jax.random.normal(k3, (BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM))
    mask = make_causal_mask(SEQ_LEN)

    try:
        student_memory = measure_peak_memory(student_fn, q, k, v, mask=mask)
    except Exception as e:
        return {
            "score": 0.0,
            "feedback": f"Memory measurement failed: {e}",
        }

    # Compute score
    if ref_memory <= 0:
        return {"score": 0.0, "feedback": "Reference memory is zero (internal error)"}

    memory_ratio = student_memory / ref_memory
    memory_reduction = max(0.0, min(1.0, 1.0 - memory_ratio))

    score = memory_reduction  # correctness_factor is 1.0 (already verified)

    feedback = (
        f"Correctness: PASSED (max_diff={max_diff:.2e}). "
        f"Memory: naive={ref_memory/1e6:.1f} MB, yours={student_memory/1e6:.1f} MB, "
        f"reduction={memory_reduction:.1%}. "
        f"Score={score:.3f}."
    )

    return {"score": round(score, 4), "feedback": feedback}


if __name__ == "__main__":
    result = judge()
    print(json.dumps(result))
```

**anti-gaming measures**:
1. correctness checked on hidden Q,K,V inputs at multiple sizes (not just seq=2048)
2. causal mask tested (not just full attention)
3. batch_size=2 tested (not just batch=1)
4. banned import check prevents using pre-built Flash Attention
5. memory is measured by the judge, not self-reported
6. student cannot see reference memory value -- they just optimize blindly

**expected score distribution**:

| approach | memory reduction | expected score |
|----------|-----------------|----------------|
| no change (naive) | 0% | 0.0 |
| simple chunking (2 chunks) | ~45% | 0.45 |
| gradient checkpointing only | ~30% | 0.30 |
| block-wise with online softmax (Flash-like) | ~90-97% | 0.90-0.97 |
| Pallas custom kernel | ~95-99% | 0.95-0.99 |

---

### 5.3 Step 3: Optimize Speed

**type**: ExecutableJudge (continuous reward)

**instructions to student**:
```
Now optimize for SPEED as well as memory. Your attention will be benchmarked
at seq_len=2048 AND seq_len=8192.

Score = geometric_mean(speedup_2048, speedup_8192) * correctness_factor

Where:
- speedup_X = naive_time_X / optimized_time_X (capped at 20x)
- speedup normalized to [0, 1] via: score_component = min(1.0, speedup / 10.0)
- correctness_factor = 1.0 if correct, else 0.0
- final score = sqrt(score_2048 * score_8192)

A 5x speedup at both lengths gives score ~= 0.50.
A 10x+ speedup at both lengths gives score ~= 1.0.

Note: naive attention will likely OOM at seq=8192.
If naive OOMs, we use the projected time (quadratic extrapolation from seq=2048).
```

**judge (step3_judge.py)**:

```python
"""
Step 3 Judge: Speed optimization.
ExecutableJudge with continuous reward.

Measures speedup at seq_len=2048 and seq_len=8192.
Score = sqrt(norm_speedup_2048 * norm_speedup_8192) * correctness_factor

Where:
- norm_speedup = min(1.0, actual_speedup / 10.0)
- correctness_factor: 1.0 if max_diff < 1e-5, else 0.0
"""

import jax
import jax.numpy as jnp
import json
import os
import sys
import importlib.util
import traceback
import pickle

sys.path.insert(0, "/env/env_data/baseline")
from naive_attention import naive_attention, make_causal_mask
from profiling_utils import measure_time, measure_peak_memory

CORRECTNESS_ATOL = 1e-5
NUM_HEADS = 8
HEAD_DIM = 64
MAX_SPEEDUP = 20.0
TARGET_SPEEDUP = 10.0


def load_student_fn():
    spec = importlib.util.spec_from_file_location(
        "optimized_attention",
        "/env/env_data/workspace/optimized_attention.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.optimized_attention


def load_reference_times():
    """Load pre-computed naive timing baselines."""
    ref_path = "/env/scoring_data/baseline_benchmarks"

    with open(os.path.join(ref_path, "naive_time_2048.json")) as f:
        time_2048 = json.load(f)["forward_time_seconds"]

    time_8192_path = os.path.join(ref_path, "naive_time_8192.json")
    with open(time_8192_path) as f:
        data_8192 = json.load(f)

    if data_8192.get("status") == "oom":
        # Extrapolate quadratically: time ~ seq^2
        # time_8192 = time_2048 * (8192/2048)^2 = time_2048 * 16
        time_8192 = time_2048 * (8192 / 2048) ** 2
    else:
        time_8192 = data_8192["forward_time_seconds"]

    return time_2048, time_8192


def check_correctness_at_length(student_fn, seq_len, batch=1):
    """Check correctness at a specific sequence length using hidden reference."""
    ref_path = f"/env/scoring_data/test_inputs/reference_outputs_{seq_len}.pkl"

    if os.path.exists(ref_path):
        with open(ref_path, "rb") as f:
            ref_data = pickle.load(f)
        q, k, v, mask, ref_output = (
            ref_data["q"], ref_data["k"], ref_data["v"],
            ref_data["mask"], ref_data["output"]
        )
    else:
        # Generate on the fly for non-standard lengths
        key = jax.random.PRNGKey(54321)
        k1, k2, k3 = jax.random.split(key, 3)
        q = jax.random.normal(k1, (batch, seq_len, NUM_HEADS, HEAD_DIM))
        k_arr = jax.random.normal(k2, (batch, seq_len, NUM_HEADS, HEAD_DIM))
        v = jax.random.normal(k3, (batch, seq_len, NUM_HEADS, HEAD_DIM))
        mask = make_causal_mask(seq_len)
        ref_output = naive_attention(q, k_arr, v, mask=mask)
        k = k_arr  # rename to avoid collision

    try:
        student_output = student_fn(q, k, v, mask=mask)
    except Exception as e:
        return False, float("inf"), f"CRASH at seq={seq_len}: {e}"

    if student_output.shape != ref_output.shape:
        return False, float("inf"), (
            f"Shape mismatch at seq={seq_len}: "
            f"got {student_output.shape}, expected {ref_output.shape}"
        )

    max_diff = float(jnp.max(jnp.abs(student_output - ref_output)))
    is_correct = max_diff < CORRECTNESS_ATOL
    detail = f"seq={seq_len}: max_diff={max_diff:.2e}"
    return is_correct, max_diff, detail


def judge() -> dict:
    try:
        student_fn = load_student_fn()
    except Exception as e:
        return {"score": 0.0, "feedback": f"Could not load: {e}"}

    # Correctness at seq=2048
    ok_2048, diff_2048, detail_2048 = check_correctness_at_length(student_fn, 2048)
    if not ok_2048:
        return {
            "score": 0.0,
            "feedback": f"Correctness FAILED at seq=2048. {detail_2048}",
        }

    # Correctness at seq=8192
    ok_8192, diff_8192, detail_8192 = check_correctness_at_length(student_fn, 8192)
    if not ok_8192:
        return {
            "score": 0.0,
            "feedback": f"Correctness FAILED at seq=8192. {detail_8192}",
        }

    # Measure speed at seq=2048
    ref_time_2048, ref_time_8192 = load_reference_times()

    key = jax.random.PRNGKey(77777)
    k1, k2, k3 = jax.random.split(key, 3)

    # Benchmark at seq=2048
    q_2048 = jax.random.normal(k1, (1, 2048, NUM_HEADS, HEAD_DIM))
    k_2048 = jax.random.normal(k2, (1, 2048, NUM_HEADS, HEAD_DIM))
    v_2048 = jax.random.normal(k3, (1, 2048, NUM_HEADS, HEAD_DIM))
    mask_2048 = make_causal_mask(2048)

    try:
        student_time_2048 = measure_time(
            student_fn, q_2048, k_2048, v_2048, mask=mask_2048
        )
    except Exception as e:
        return {"score": 0.0, "feedback": f"Benchmark at seq=2048 failed: {e}"}

    # Benchmark at seq=8192
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(88888), 3)
    q_8192 = jax.random.normal(k1, (1, 8192, NUM_HEADS, HEAD_DIM))
    k_8192 = jax.random.normal(k2, (1, 8192, NUM_HEADS, HEAD_DIM))
    v_8192 = jax.random.normal(k3, (1, 8192, NUM_HEADS, HEAD_DIM))
    mask_8192 = make_causal_mask(8192)

    try:
        student_time_8192 = measure_time(
            student_fn, q_8192, k_8192, v_8192, mask=mask_8192
        )
    except Exception as e:
        return {"score": 0.0, "feedback": f"Benchmark at seq=8192 failed: {e}"}

    # Compute speedups
    speedup_2048 = min(MAX_SPEEDUP, ref_time_2048 / max(student_time_2048, 1e-9))
    speedup_8192 = min(MAX_SPEEDUP, ref_time_8192 / max(student_time_8192, 1e-9))

    # Normalize to [0, 1]
    norm_2048 = min(1.0, speedup_2048 / TARGET_SPEEDUP)
    norm_8192 = min(1.0, speedup_8192 / TARGET_SPEEDUP)

    # Geometric mean
    import math
    score = math.sqrt(norm_2048 * norm_8192)

    feedback = (
        f"Correctness: PASSED (diff_2048={diff_2048:.2e}, diff_8192={diff_8192:.2e}). "
        f"Speed: seq=2048: {speedup_2048:.2f}x ({student_time_2048*1000:.2f}ms vs "
        f"{ref_time_2048*1000:.2f}ms naive). "
        f"seq=8192: {speedup_8192:.2f}x ({student_time_8192*1000:.2f}ms vs "
        f"{ref_time_8192*1000:.2f}ms naive). "
        f"Score={score:.3f}."
    )

    return {"score": round(score, 4), "feedback": feedback}


if __name__ == "__main__":
    result = judge()
    print(json.dumps(result))
```

**anti-gaming measures**:
1. speedup measured by judge on standardized hardware, not self-reported
2. correctness verified against hidden reference outputs at both lengths
3. if student just returns zeros/noise fast, correctness check catches it
4. naive time at seq=8192 is either measured or quadratically extrapolated (conservative)
5. geometric mean penalizes solutions that only work at one length
6. MAX_SPEEDUP cap prevents gaming via artificial baseline slowdown

**expected score distribution**:

| approach | speedup 2048 | speedup 8192 | score |
|----------|-------------|-------------|-------|
| naive (no change) | 1.0x | OOM (0x) | 0.0 |
| simple chunking | 0.8x (slower) | 1.5x | ~0.0 |
| block-wise + online softmax | 3-5x | 8-15x | 0.50-0.85 |
| Pallas custom kernel | 5-10x | 10-20x | 0.70-1.0 |
| optimized Pallas + fusion | 10x+ | 15x+ | 0.90-1.0 |

---

### 5.4 Step 4: Scale to Long Sequences (FINAL)

**type**: ExecutableJudge (continuous reward)

**instructions to student**:
```
The real test: run your attention at seq_len=16384.
The naive baseline runs out of memory at this length.

Score combines two factors:
1. Sequence length achievement: can your code actually RUN at 16384?
2. Quality: perplexity of the transformer model using your attention on held-out text.

Score = seq_len_factor * quality_factor

Where:
- seq_len_factor = achieved_seq_len / 16384 (1.0 if you handle full 16384)
- quality_factor = max(0, 1 - (perplexity - ref_perplexity) / ref_perplexity)
  clipped to [0, 1]. ref_perplexity is the model's perplexity with exact attention.
- If attention is correct (max_diff < 1e-5), quality_factor ~= 1.0

This is where genuine algorithmic improvement matters.
```

**judge (step4_judge.py)**:

```python
"""
Step 4 Judge: Long sequence scaling.
ExecutableJudge with continuous reward.

Tests at seq_len=16384. Naive OOMs here.

Score = seq_len_factor * quality_factor

seq_len_factor: achieved_seq_len / 16384
quality_factor: based on perplexity with the transformer model on held-out text
"""

import jax
import jax.numpy as jnp
import json
import os
import sys
import importlib.util
import traceback
import pickle
import math

sys.path.insert(0, "/env/env_data/baseline")
from naive_attention import naive_attention, make_causal_mask
from profiling_utils import measure_peak_memory

CORRECTNESS_ATOL = 1e-5
TARGET_SEQ_LEN = 16384
NUM_HEADS = 8
HEAD_DIM = 64
# Sequence lengths to try, in order
TEST_SEQ_LENS = [4096, 8192, 12288, 16384]


def load_student_fn():
    spec = importlib.util.spec_from_file_location(
        "optimized_attention",
        "/env/env_data/workspace/optimized_attention.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.optimized_attention


def load_model_with_student_attention(student_fn):
    """Load the transformer model with student's attention function swapped in."""
    sys.path.insert(0, "/env/env_data/baseline")
    import transformer_model as tm

    config = tm.TransformerConfig()
    model = tm.SmallTransformer(config, attention_fn=student_fn)
    params = tm.load_model_params()
    return model, params, config


def find_max_working_seq_len(student_fn):
    """
    Find the maximum sequence length that works without OOM or error.
    Returns (max_seq_len, correctness_at_max).
    """
    max_working = 0
    correctness_ok = False

    for seq_len in TEST_SEQ_LENS:
        key = jax.random.PRNGKey(seq_len)
        k1, k2, k3 = jax.random.split(key, 3)
        q = jax.random.normal(k1, (1, seq_len, NUM_HEADS, HEAD_DIM))
        k_arr = jax.random.normal(k2, (1, seq_len, NUM_HEADS, HEAD_DIM))
        v = jax.random.normal(k3, (1, seq_len, NUM_HEADS, HEAD_DIM))
        mask = make_causal_mask(seq_len)

        try:
            output = student_fn(q, k_arr, v, mask=mask)
            jax.block_until_ready(output)

            if output.shape != (1, seq_len, NUM_HEADS, HEAD_DIM):
                break

            # Check correctness against reference if available
            ref_path = f"/env/scoring_data/test_inputs/reference_outputs_{seq_len}.pkl"
            if os.path.exists(ref_path):
                with open(ref_path, "rb") as f:
                    ref_data = pickle.load(f)
                ref_output = ref_data["output"]
                max_diff = float(jnp.max(jnp.abs(output - ref_output)))
                correctness_ok = max_diff < CORRECTNESS_ATOL
            else:
                # For lengths without pre-computed reference, check basic properties
                correctness_ok = (
                    bool(jnp.all(jnp.isfinite(output)))
                    and float(jnp.std(output)) > 0.01
                )

            max_working = seq_len

        except Exception:
            break

    return max_working, correctness_ok


def compute_perplexity_with_student_attention(student_fn):
    """
    Compute perplexity of the transformer model using student's attention
    on held-out text.
    """
    model, params, config = load_model_with_student_attention(student_fn)

    # Load held-out text
    with open("/env/scoring_data/test_text/held_out_text.txt") as f:
        text = f.read()

    # Tokenize
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    token_ids = enc.encode(text)

    # Truncate to max achievable length
    max_len = min(len(token_ids), TARGET_SEQ_LEN)
    token_ids = jnp.array(token_ids[:max_len])

    try:
        from transformer_model import compute_perplexity
        ppl = compute_perplexity(model, params, token_ids)
        return ppl, max_len
    except Exception as e:
        return None, 0


def load_reference_perplexity():
    """Load reference perplexity (computed with exact attention)."""
    ref_path = "/env/scoring_data/baseline_benchmarks/reference_perplexity.json"
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            return json.load(f)["perplexity"]
    return None


def judge() -> dict:
    try:
        student_fn = load_student_fn()
    except Exception as e:
        return {"score": 0.0, "feedback": f"Could not load: {e}"}

    # Find maximum working sequence length
    max_seq, correctness_ok = find_max_working_seq_len(student_fn)

    if max_seq == 0:
        return {
            "score": 0.0,
            "feedback": "Attention failed at all tested sequence lengths (4096+).",
        }

    seq_len_factor = max_seq / TARGET_SEQ_LEN

    # Compute perplexity
    ref_ppl = load_reference_perplexity()
    student_ppl, ppl_seq_len = compute_perplexity_with_student_attention(student_fn)

    if student_ppl is None or math.isinf(student_ppl) or math.isnan(student_ppl):
        quality_factor = 0.0
        ppl_detail = "Perplexity computation failed or returned inf/nan."
    elif ref_ppl is None:
        # No reference available -- use correctness as proxy
        quality_factor = 1.0 if correctness_ok else 0.5
        ppl_detail = f"No reference perplexity. Correctness={'PASS' if correctness_ok else 'FAIL'}."
    else:
        # quality_factor = max(0, 1 - (student_ppl - ref_ppl) / ref_ppl)
        ppl_ratio = (student_ppl - ref_ppl) / ref_ppl
        quality_factor = max(0.0, min(1.0, 1.0 - ppl_ratio))
        ppl_detail = (
            f"Perplexity: yours={student_ppl:.2f}, reference={ref_ppl:.2f} "
            f"(at seq_len={ppl_seq_len})."
        )

    score = seq_len_factor * quality_factor

    feedback = (
        f"Max working seq_len: {max_seq}/{TARGET_SEQ_LEN} "
        f"(factor={seq_len_factor:.2f}). "
        f"Correctness at max_seq: {'PASS' if correctness_ok else 'FAIL'}. "
        f"{ppl_detail} "
        f"Score={score:.3f}."
    )

    return {"score": round(score, 4), "feedback": feedback}


if __name__ == "__main__":
    result = judge()
    print(json.dumps(result))
```

**anti-gaming measures**:
1. sequence lengths tested progressively -- can't fake 16384 support if you OOM at 8192
2. correctness checked against hidden reference outputs
3. perplexity measured on hidden held-out text -- can't pre-compute
4. quality_factor ensures approximate attention (with quality loss) scores lower
5. if student hardcodes outputs, perplexity will be nonsensical
6. output shape, finiteness, and variance checked as sanity
7. student cannot see reference perplexity value
8. the transformer model with swapped attention is loaded by the judge -- student can't intercept

**expected score distribution**:

| approach | max seq_len | quality | score |
|----------|------------|---------|-------|
| naive (OOMs) | 0 | -- | 0.0 |
| simple chunking (barely fits) | 4096 | 0.95 | 0.24 |
| block-wise attention | 8192 | 0.98 | 0.49 |
| full Flash Attention | 16384 | 1.0 | 1.0 |
| approximate attention (quality loss) | 16384 | 0.7-0.9 | 0.7-0.9 |
| memory-optimal Pallas | 16384 | 1.0 | 1.0 |

---

## 6. tasks.py skeleton

```python
"""
E05: Attention Optimization
pm_env task definition.
"""

from pm_env.task import Task, Step
from pm_env.judges import ExactMatchJudge, ExecutableJudge


def create_task() -> Task:
    """Define the attention optimization task."""

    step1 = Step(
        step_id="step1_profile",
        title="Understand the Baseline",
        instructions_path="/env/env_data/README.md",
        judge=ExactMatchJudge(
            script_path="/env/scoring_data/judges/step1_judge.py",
            answer_keys=["peak_memory_bytes", "forward_time_seconds", "biggest_bottleneck"],
        ),
        time_limit_seconds=1800,  # 30 minutes
        max_score=1.0,
    )

    step2 = Step(
        step_id="step2_memory",
        title="Implement Memory-Efficient Attention",
        instructions_path="/env/env_data/README.md",
        judge=ExecutableJudge(
            script_path="/env/scoring_data/judges/step2_judge.py",
            timeout_seconds=600,
        ),
        time_limit_seconds=5400,  # 90 minutes
        max_score=1.0,
        depends_on=["step1_profile"],
    )

    step3 = Step(
        step_id="step3_speed",
        title="Optimize Speed",
        instructions_path="/env/env_data/README.md",
        judge=ExecutableJudge(
            script_path="/env/scoring_data/judges/step3_judge.py",
            timeout_seconds=600,
        ),
        time_limit_seconds=5400,  # 90 minutes
        max_score=1.0,
        depends_on=["step2_memory"],
    )

    step4 = Step(
        step_id="step4_scale",
        title="Scale to Long Sequences",
        instructions_path="/env/env_data/README.md",
        judge=ExecutableJudge(
            script_path="/env/scoring_data/judges/step4_judge.py",
            timeout_seconds=1200,  # 20 minutes (long sequences are slow)
        ),
        time_limit_seconds=3600,  # 60 minutes
        max_score=1.0,
        depends_on=["step3_speed"],
    )

    task = Task(
        task_id="e05_attention_optimization",
        title="Attention Optimization Challenge",
        description=(
            "Optimize a naive O(N^2) attention implementation for memory efficiency "
            "and speed. Scale to long sequences (16384+). "
            "Flash Attention paper provided as reference, but any technique allowed."
        ),
        steps=[step1, step2, step3, step4],
        total_time_limit_seconds=14400,  # 4 hours
    )

    return task


# Entry point for pm_env harness
task = create_task()
```

---

## 7. setup_data.py skeleton

```python
"""
E05: Attention Optimization
Data setup script. Generates scoring_data from the baseline.

Run ONCE during container build (before student starts).
"""

import jax
import jax.numpy as jnp
import json
import os
import pickle
import sys

sys.path.insert(0, "/env/env_data/baseline")
from naive_attention import naive_attention, make_causal_mask
from profiling_utils import measure_peak_memory, measure_time


SCORING_DIR = "/env/scoring_data"
NUM_HEADS = 8
HEAD_DIM = 64


def setup_baseline_benchmarks():
    """Pre-compute naive attention benchmarks."""
    out_dir = os.path.join(SCORING_DIR, "baseline_benchmarks")
    os.makedirs(out_dir, exist_ok=True)

    key = jax.random.PRNGKey(42)

    # seq=2048 benchmarks
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (1, 2048, NUM_HEADS, HEAD_DIM))
    k_arr = jax.random.normal(k2, (1, 2048, NUM_HEADS, HEAD_DIM))
    v = jax.random.normal(k3, (1, 2048, NUM_HEADS, HEAD_DIM))
    mask = make_causal_mask(2048)

    mem = measure_peak_memory(naive_attention, q, k_arr, v, mask=mask)
    with open(os.path.join(out_dir, "naive_memory_2048.json"), "w") as f:
        json.dump({"peak_memory_bytes": mem, "seq_len": 2048}, f)

    t = measure_time(naive_attention, q, k_arr, v, mask=mask)
    with open(os.path.join(out_dir, "naive_time_2048.json"), "w") as f:
        json.dump({"forward_time_seconds": t, "seq_len": 2048}, f)

    # seq=8192 benchmarks (may OOM)
    try:
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(8192), 3)
        q8 = jax.random.normal(k1, (1, 8192, NUM_HEADS, HEAD_DIM))
        k8 = jax.random.normal(k2, (1, 8192, NUM_HEADS, HEAD_DIM))
        v8 = jax.random.normal(k3, (1, 8192, NUM_HEADS, HEAD_DIM))
        mask8 = make_causal_mask(8192)

        mem8 = measure_peak_memory(naive_attention, q8, k8, v8, mask=mask8)
        t8 = measure_time(naive_attention, q8, k8, v8, mask=mask8)

        with open(os.path.join(out_dir, "naive_memory_8192.json"), "w") as f:
            json.dump({"peak_memory_bytes": mem8, "seq_len": 8192}, f)
        with open(os.path.join(out_dir, "naive_time_8192.json"), "w") as f:
            json.dump({"forward_time_seconds": t8, "seq_len": 8192}, f)
    except Exception:
        with open(os.path.join(out_dir, "naive_memory_8192.json"), "w") as f:
            json.dump({"status": "oom", "seq_len": 8192}, f)
        with open(os.path.join(out_dir, "naive_time_8192.json"), "w") as f:
            json.dump({"status": "oom", "seq_len": 8192}, f)

    print("Baseline benchmarks saved.")


def setup_reference_outputs():
    """Pre-compute reference attention outputs for correctness checking."""
    out_dir = os.path.join(SCORING_DIR, "test_inputs")
    os.makedirs(out_dir, exist_ok=True)

    for seq_len in [2048, 8192]:
        key = jax.random.PRNGKey(54321 + seq_len)
        k1, k2, k3 = jax.random.split(key, 3)
        q = jax.random.normal(k1, (1, seq_len, NUM_HEADS, HEAD_DIM))
        k_arr = jax.random.normal(k2, (1, seq_len, NUM_HEADS, HEAD_DIM))
        v = jax.random.normal(k3, (1, seq_len, NUM_HEADS, HEAD_DIM))
        mask = make_causal_mask(seq_len)

        try:
            output = naive_attention(q, k_arr, v, mask=mask)
            data = {"q": q, "k": k_arr, "v": v, "mask": mask, "output": output}
            with open(os.path.join(out_dir, f"reference_outputs_{seq_len}.pkl"), "wb") as f:
                pickle.dump(data, f)
            print(f"Reference output saved for seq_len={seq_len}.")
        except Exception as e:
            print(f"Could not compute reference at seq_len={seq_len}: {e}")

    # seq=16384: compute in chunks (since naive will OOM)
    seq_len = 16384
    print(f"Computing reference for seq_len={seq_len} via chunked exact attention...")
    key = jax.random.PRNGKey(54321 + seq_len)
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (1, seq_len, NUM_HEADS, HEAD_DIM))
    k_arr = jax.random.normal(k2, (1, seq_len, NUM_HEADS, HEAD_DIM))
    v = jax.random.normal(k3, (1, seq_len, NUM_HEADS, HEAD_DIM))
    mask = make_causal_mask(seq_len)

    try:
        output = chunked_exact_attention(q, k_arr, v, mask, chunk_size=1024)
        data = {"q": q, "k": k_arr, "v": v, "mask": mask, "output": output}
        with open(os.path.join(out_dir, f"reference_outputs_{seq_len}.pkl"), "wb") as f:
            pickle.dump(data, f)
        print(f"Reference output saved for seq_len={seq_len} (chunked).")
    except Exception as e:
        print(f"Could not compute reference at seq_len={seq_len}: {e}")


def chunked_exact_attention(q, k, v, mask, chunk_size=1024):
    """
    Compute exact attention in chunks (for generating reference at long seq).
    Memory: O(chunk_size * seq_kv) instead of O(seq_q * seq_kv).
    This is NOT what the student should implement -- it's just for generating
    reference outputs during setup.
    """
    batch, seq_q, num_heads, head_dim = q.shape
    seq_kv = k.shape[1]
    scale = head_dim ** -0.5

    q_t = jnp.transpose(q, (0, 2, 1, 3))  # [B, H, Sq, D]
    k_t = jnp.transpose(k, (0, 2, 1, 3))
    v_t = jnp.transpose(v, (0, 2, 1, 3))

    outputs = []
    for start in range(0, seq_q, chunk_size):
        end = min(start + chunk_size, seq_q)
        q_chunk = q_t[:, :, start:end, :]  # [B, H, chunk, D]

        scores = jnp.matmul(q_chunk, jnp.transpose(k_t, (0, 1, 3, 2))) * scale
        # [B, H, chunk, Skv]

        if mask is not None:
            mask_chunk = mask[:, :, start:end, :]
            scores = jnp.where(mask_chunk, scores, jnp.finfo(scores.dtype).min)

        weights = jax.nn.softmax(scores, axis=-1)
        out_chunk = jnp.matmul(weights, v_t)  # [B, H, chunk, D]
        outputs.append(out_chunk)

    output = jnp.concatenate(outputs, axis=2)  # [B, H, Sq, D]
    return jnp.transpose(output, (0, 2, 1, 3))  # [B, Sq, H, D]


def setup_correctness_inputs():
    """Generate additional correctness test inputs (diverse configurations)."""
    out_dir = os.path.join(SCORING_DIR, "test_inputs")
    os.makedirs(out_dir, exist_ok=True)

    key = jax.random.PRNGKey(12345)
    test_configs = [
        {"batch": 1, "seq": 256, "heads": 8, "dim": 64},
        {"batch": 2, "seq": 512, "heads": 8, "dim": 64},
        {"batch": 1, "seq": 1024, "heads": 4, "dim": 128},
        {"batch": 4, "seq": 128, "heads": 16, "dim": 32},
    ]

    all_cases = []
    for cfg in test_configs:
        k1, k2, k3, key = jax.random.split(key, 4)
        q = jax.random.normal(k1, (cfg["batch"], cfg["seq"], cfg["heads"], cfg["dim"]))
        k_arr = jax.random.normal(k2, (cfg["batch"], cfg["seq"], cfg["heads"], cfg["dim"]))
        v = jax.random.normal(k3, (cfg["batch"], cfg["seq"], cfg["heads"], cfg["dim"]))
        mask = make_causal_mask(cfg["seq"])
        ref = naive_attention(q, k_arr, v, mask=mask)
        all_cases.append({
            "q": q, "k": k_arr, "v": v, "mask": mask,
            "output": ref, "config": cfg,
        })

    with open(os.path.join(out_dir, "correctness_inputs.pkl"), "wb") as f:
        pickle.dump(all_cases, f)

    print(f"Saved {len(all_cases)} correctness test cases.")


def setup_reference_perplexity():
    """Compute reference perplexity using exact (chunked) attention."""
    out_dir = os.path.join(SCORING_DIR, "baseline_benchmarks")
    os.makedirs(out_dir, exist_ok=True)

    sys.path.insert(0, "/env/env_data/baseline")
    from transformer_model import SmallTransformer, TransformerConfig, load_model_params
    from transformer_model import compute_perplexity

    config = TransformerConfig()
    model = SmallTransformer(config, attention_fn=naive_attention)
    params = load_model_params()

    with open("/env/scoring_data/test_text/held_out_text.txt") as f:
        text = f.read()

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    token_ids = jnp.array(enc.encode(text)[:4096])  # use 4096 for ref perplexity

    ppl = compute_perplexity(model, params, token_ids)

    with open(os.path.join(out_dir, "reference_perplexity.json"), "w") as f:
        json.dump({"perplexity": float(ppl), "seq_len": len(token_ids)}, f)

    print(f"Reference perplexity: {ppl:.2f} at seq_len={len(token_ids)}.")


def main():
    print("=== E05 Setup Data ===")
    setup_baseline_benchmarks()
    setup_reference_outputs()
    setup_correctness_inputs()
    setup_reference_perplexity()
    print("=== Setup Complete ===")


if __name__ == "__main__":
    main()
```

---

## 8. pre_hook.py

```python
"""
E05: Attention Optimization
Pre-hook: runs before student starts.

Responsibilities:
1. Verify JAX installation and hardware detection
2. Verify all env_data files are present
3. Run naive attention at small scale to warm up JIT
4. Set environment variables for reproducibility
"""

import os
import sys
import json


def pre_hook():
    """Run before student session begins."""
    print("=== E05 Pre-Hook ===")

    # 1. Verify JAX
    try:
        import jax
        import jax.numpy as jnp
        devices = jax.devices()
        print(f"JAX version: {jax.__version__}")
        print(f"Devices: {[str(d) for d in devices]}")
        print(f"Default backend: {jax.default_backend()}")
    except ImportError:
        print("FATAL: JAX not installed")
        sys.exit(1)

    # 2. Verify env_data files
    required_files = [
        "/env/env_data/README.md",
        "/env/env_data/baseline/naive_attention.py",
        "/env/env_data/baseline/transformer_model.py",
        "/env/env_data/baseline/profiling_utils.py",
        "/env/env_data/baseline/test_correctness.py",
        "/env/env_data/workspace/optimized_attention.py",
        "/env/env_data/reference/paper_summary.md",
    ]

    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"WARNING: Missing files: {missing}")
    else:
        print("All env_data files present.")

    # 3. Verify scoring_data
    scoring_files = [
        "/env/scoring_data/judges/step1_judge.py",
        "/env/scoring_data/judges/step2_judge.py",
        "/env/scoring_data/judges/step3_judge.py",
        "/env/scoring_data/judges/step4_judge.py",
        "/env/scoring_data/baseline_benchmarks/naive_memory_2048.json",
        "/env/scoring_data/baseline_benchmarks/naive_time_2048.json",
    ]

    missing_scoring = [f for f in scoring_files if not os.path.exists(f)]
    if missing_scoring:
        print(f"WARNING: Missing scoring files: {missing_scoring}")
        print("Run setup_data.py first!")
    else:
        print("All scoring_data files present.")

    # 4. Warm up JIT with small computation
    try:
        sys.path.insert(0, "/env/env_data/baseline")
        from naive_attention import naive_attention
        key = jax.random.PRNGKey(0)
        q = jax.random.normal(key, (1, 32, 4, 32))
        k = jax.random.normal(key, (1, 32, 4, 32))
        v = jax.random.normal(key, (1, 32, 4, 32))
        _ = naive_attention(q, k, v)
        print("JIT warmup complete.")
    except Exception as e:
        print(f"JIT warmup failed: {e}")

    # 5. Set environment variables
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_TRACEBACK_FILTERING"] = "off"

    print("=== Pre-Hook Complete ===")


if __name__ == "__main__":
    pre_hook()
```

---

## 9. comprehensive anti-gaming analysis

### 9.1 threat model

| attack | detection | mitigation |
|--------|-----------|------------|
| hardcode expected outputs | correctness checked on hidden inputs with different PRNG seeds | student never sees scoring_data inputs |
| use pre-built flash attention (jax.experimental.pallas.ops.tpu.flash_attention) | banned import check in judge code | source code scanned for banned patterns |
| import external attention library | banned import check | only jax/jaxlib/flax/optax allowed |
| optimize for benchmark inputs only | correctness checked at multiple seq lengths and configs | batch sizes, head configs varied in hidden tests |
| report fake profiling numbers | profiling is done by the judge, not self-reported | student only submits code, judge measures |
| approximate attention (fast but wrong) | correctness check with atol=1e-5 | tight tolerance catches approximations |
| modify judge files | scoring_data is read-only to student | container permissions enforce this |
| timing attacks (slow down naive) | naive baseline pre-computed during setup | reference times stored in scoring_data |

### 9.2 correctness verification depth

the judges check correctness at these levels:

1. **exact numerical match**: max absolute difference < 1e-5 against reference outputs
2. **multiple sequence lengths**: 256, 512, 1024, 2048, 8192, 16384
3. **multiple batch sizes**: batch=1, 2, 4
4. **multiple head configs**: heads=4/8/16, dim=32/64/128
5. **with and without causal mask**: both full attention and causal attention
6. **output shape verification**: must match expected shape exactly
7. **finiteness check**: no NaN or Inf in output
8. **variance check**: output must have reasonable variance (not constant)
9. **perplexity check** (step 4): model must produce sensible text quality

### 9.3 why this resists reward hacking

1. **student does NOT report scores**: all measurements done by judge
2. **student does NOT see test data**: reference outputs in scoring_data
3. **continuous reward prevents binary gaming**: 0.0-1.0 range with smooth gradients
4. **geometric mean penalizes single-length optimization**: must work at BOTH 2048 and 8192
5. **perplexity on held-out text**: can't fake model quality
6. **progressive steps**: must pass correctness before speed is measured
7. **code analysis**: banned imports detected statically

---

## 10. expected overall score distributions

### 10.1 by approach quality

| agent quality | step 1 | step 2 | step 3 | step 4 | total |
|--------------|--------|--------|--------|--------|-------|
| fails to start | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| profiles only | 1.0 | 0.0 | 0.0 | 0.0 | 0.25 |
| basic chunking | 1.0 | 0.30 | 0.0 | 0.0 | 0.33 |
| gradient checkpointing | 1.0 | 0.30 | 0.15 | 0.24 | 0.42 |
| naive flash attention (buggy) | 1.0 | 0.70 | 0.30 | 0.40 | 0.60 |
| correct flash attention | 1.0 | 0.95 | 0.60 | 0.80 | 0.84 |
| optimized flash + Pallas | 1.0 | 0.98 | 0.90 | 1.0 | 0.97 |

### 10.2 score calculation

final task score = mean(step1_score, step2_score, step3_score, step4_score)

or weighted:
- step 1: 10% (profiling is easy)
- step 2: 25% (memory optimization -- the core challenge)
- step 3: 30% (speed optimization -- harder)
- step 4: 35% (scaling -- the differentiator)

weighted score = 0.10 * s1 + 0.25 * s2 + 0.30 * s3 + 0.35 * s4

### 10.3 target distributions

| percentile | expected score | interpretation |
|-----------|---------------|----------------|
| p10 | 0.0-0.10 | failed to implement anything |
| p25 | 0.20-0.35 | basic chunking, some memory savings |
| p50 (median) | 0.40-0.55 | working block attention, some speed |
| p75 | 0.65-0.80 | good flash attention, handles 8192 |
| p90 | 0.85-0.95 | optimized, scales to 16384 |
| p99 | 0.95-1.0 | Pallas-level optimization |

---

## 11. ideal properties checklist

| property | how this env satisfies it |
|----------|--------------------------|
| no known pattern can be applied blindly | flash attention is ONE approach; student must understand WHY it works and adapt to JAX/TPU |
| many points of failure | numerical stability (online softmax), memory management, mask handling, block boundary conditions, XLA compilation |
| continuous reward | memory reduction ratio, speedup ratio, perplexity -- all continuous 0.0-1.0 |
| constant improvement space | block size tuning, Pallas optimization, mixed precision, fusion -- always room to improve |
| hill-climbing on core metrics | each improvement to memory or speed directly increases score |
| open-ended optimization | student chooses approach: tiling, checkpointing, Pallas, approximation, hybrid |
| not fill-in-the-blank | student gets naive attention, not a half-implemented Flash Attention with TODOs |
| depth over breadth | one function to optimize across 4 progressive difficulty levels |

---

## 12. references

1. Dao, T. (2023). Flash Attention 2: Faster Attention with Better Parallelism and Work Partitioning. arXiv:2307.08691.
2. Dao, T., Fu, D., Ermon, S., Rudra, A., & Re, C. (2022). Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022. arXiv:2205.14135.
3. JAX Pallas Documentation. https://jax.readthedocs.io/en/latest/pallas/index.html
4. JAX Custom Derivatives. https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
5. Rabe, M. N., & Staats, C. (2021). Self-attention Does Not Need O(n^2) Memory. arXiv:2112.05682.

---

*end of specification*
