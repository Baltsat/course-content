# E68: Qwen Inference Optimization – Detailed Student Pipeline

> full specification of student path from start to evaluation
> tasks: E68-T1 (INT8 Quantization), E68-T4 (Continuous Batching)

---

## 1. what student sees at start

### 1.1 task description (in interface)

```
TASK E68-T1: INT8 Quantization for Qwen Inference

You have a slow Qwen-0.6B inference baseline running on TPU v5e.
Your goal: implement INT8 quantization using the AQT (Accurate Quantized Training) library
to achieve ≥1.5x speedup while maintaining output quality.

Requirements:
1. Quantize linear layers (attention projections, MLP)
2. Use per-channel quantization for weights
3. Use per-token quantization for activations
4. Run calibration on provided dataset
5. Measure speedup and output divergence

Constraints:
- KL divergence from FP32 outputs must be < 0.01
- Token match rate must be > 95%
- Time limit: 4 hours
- Memory limit: 8 GB HBM

═══════════════════════════════════════════════════════════════

TASK E68-T4: Continuous Batching for Qwen Inference

You have a naive batched inference implementation that wastes compute
on padding tokens. Your goal: implement continuous batching to maximize
throughput while maintaining low latency.

Requirements:
1. Implement request queue with priorities
2. Separate prefill and decode phases
3. Manage KV cache slots dynamically
4. Handle request completion and eviction
5. Measure throughput and latency percentiles

Constraints:
- Throughput must be ≥2x baseline
- p95 latency must be < 1.5× p50 latency
- Time limit: 4 hours
- Memory limit: 8 GB HBM
```

### 1.2 baseline inference code (slow)

```python
# baseline_inference.py
"""
Qwen-0.6B Naive Inference Baseline
==================================

This is the SLOW baseline. Your job is to optimize it.

DO NOT modify the model architecture.
DO NOT change the tokenizer.
You MAY modify inference logic, batching, and quantization.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import List, Tuple, Optional
import time

# ============== MODEL CONFIG ==============

class QwenConfig:
    """Qwen-0.6B configuration."""
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16  # MHA (not GQA for 0.6B)
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6

    # derived
    head_dim: int = hidden_size // num_attention_heads  # 64


CONFIG = QwenConfig()


# ============== NAIVE ATTENTION ==============

def naive_attention(
    query: jnp.ndarray,  # [batch, seq_len, num_heads, head_dim]
    key: jnp.ndarray,    # [batch, kv_len, num_kv_heads, head_dim]
    value: jnp.ndarray,  # [batch, kv_len, num_kv_heads, head_dim]
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Naive scaled dot-product attention.

    WARNING: This is O(n²) memory. You should optimize this.
    """
    batch, seq_len, num_heads, head_dim = query.shape
    _, kv_len, num_kv_heads, _ = key.shape

    # expand KV heads if GQA
    if num_kv_heads != num_heads:
        repeats = num_heads // num_kv_heads
        key = jnp.repeat(key, repeats, axis=2)
        value = jnp.repeat(value, repeats, axis=2)

    # transpose for matmul: [batch, heads, seq, dim]
    q = jnp.transpose(query, (0, 2, 1, 3))
    k = jnp.transpose(key, (0, 2, 1, 3))
    v = jnp.transpose(value, (0, 2, 1, 3))

    # attention scores
    scale = 1.0 / jnp.sqrt(head_dim)
    scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * scale

    # apply mask
    if mask is not None:
        scores = jnp.where(mask, scores, -1e9)

    # softmax
    attn_weights = jax.nn.softmax(scores, axis=-1)

    # output
    output = jnp.matmul(attn_weights, v)
    output = jnp.transpose(output, (0, 2, 1, 3))

    return output


# ============== LINEAR LAYER (UNQUANTIZED) ==============

class DenseGeneral(nn.Module):
    """Standard dense layer. Target for quantization."""
    features: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param(
            'kernel',
            nn.initializers.normal(0.02),
            (x.shape[-1], self.features)
        )
        y = jnp.dot(x, kernel)
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,))
            y = y + bias
        return y


# ============== NAIVE INFERENCE LOOP ==============

def naive_generate(
    model,
    params,
    input_ids: jnp.ndarray,  # [batch, seq_len]
    max_new_tokens: int = 128,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """
    Naive autoregressive generation.

    WARNING: This recomputes the entire sequence each step.
    You should use KV caching.
    """
    batch_size, seq_len = input_ids.shape
    generated = input_ids

    for _ in range(max_new_tokens):
        # forward pass on ENTIRE sequence (wasteful!)
        logits = model.apply(params, generated)

        # get last token logits
        next_token_logits = logits[:, -1, :] / temperature

        # sample
        next_token = jax.random.categorical(
            jax.random.PRNGKey(0),
            next_token_logits,
            axis=-1
        )

        # append
        generated = jnp.concatenate(
            [generated, next_token[:, None]],
            axis=1
        )

    return generated


# ============== NAIVE BATCHED INFERENCE ==============

def naive_batch_inference(
    model,
    params,
    prompts: List[str],
    tokenizer,
    max_new_tokens: int = 128,
) -> List[str]:
    """
    Naive batched inference with padding.

    WARNING: This pads all sequences to max length.
    Wastes compute on padding tokens.
    """
    # tokenize with padding
    encoded = tokenizer(
        prompts,
        padding='max_length',
        max_length=512,
        return_tensors='jax'
    )

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    # generate (ignoring mask inefficiency)
    output_ids = naive_generate(model, params, input_ids, max_new_tokens)

    # decode
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    return outputs


# ============== BENCHMARK ==============

def benchmark_baseline(model, params, tokenizer, test_prompts: List[str]):
    """Measure baseline performance."""

    # warmup
    _ = naive_batch_inference(model, params, test_prompts[:2], tokenizer, max_new_tokens=10)

    # benchmark
    start = time.perf_counter()
    outputs = naive_batch_inference(model, params, test_prompts, tokenizer, max_new_tokens=128)
    elapsed = time.perf_counter() - start

    total_tokens = sum(len(tokenizer.encode(o)) for o in outputs)
    tokens_per_sec = total_tokens / elapsed

    print(f"Baseline: {tokens_per_sec:.1f} tokens/sec")
    print(f"Total time: {elapsed:.2f}s for {len(test_prompts)} prompts")

    return tokens_per_sec, outputs


# ============== TEST PROMPTS ==============

TEST_PROMPTS = [
    "Explain the concept of backpropagation in neural networks.",
    "What is the difference between supervised and unsupervised learning?",
    "Describe the attention mechanism in transformers.",
    "How does batch normalization improve training?",
    "What are the advantages of convolutional neural networks for images?",
    "Explain gradient descent optimization.",
    "What is overfitting and how can it be prevented?",
    "Describe the architecture of a recurrent neural network.",
    "What is transfer learning and when is it useful?",
    "Explain the concept of embeddings in NLP.",
]


if __name__ == "__main__":
    # This would load actual model - placeholder for structure
    print("Baseline inference code structure")
    print(f"Config: hidden_size={CONFIG.hidden_size}, layers={CONFIG.num_hidden_layers}")
```

### 1.3 Qwen model config reference

```python
# qwen_config.py
"""
Qwen Model Configurations
=========================

Reference configurations for different Qwen model sizes.
"""

QWEN_CONFIGS = {
    "0.6B": {
        "hidden_size": 1024,
        "intermediate_size": 2816,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 64,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
    },
    "1.8B": {
        "hidden_size": 2048,
        "intermediate_size": 5504,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 128,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
    },
    "4B": {
        "hidden_size": 2560,
        "intermediate_size": 6912,
        "num_hidden_layers": 40,
        "num_attention_heads": 20,
        "num_key_value_heads": 20,
        "head_dim": 128,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
    },
}

# Memory estimates (FP32)
MEMORY_ESTIMATES = {
    "0.6B": {
        "params_gb": 2.4,
        "kv_cache_per_token_mb": 0.75,  # per layer: 2 * hidden * 2 heads * 4 bytes
        "activation_peak_gb": 1.0,
    },
    "1.8B": {
        "params_gb": 7.2,
        "kv_cache_per_token_mb": 1.5,
        "activation_peak_gb": 2.0,
    },
    "4B": {
        "params_gb": 16.0,
        "kv_cache_per_token_mb": 4.0,
        "activation_peak_gb": 4.0,
    },
}
```

### 1.4 reference documentation (available to student)

```markdown
## AQT (Accurate Quantized Training) Library Reference

### Overview
AQT is Google's JAX library for quantization-aware training and post-training quantization.
It provides composable quantization configs for weights and activations.

### Key Classes

#### `aqt.jax.v2.config.DotGeneral`
Configuration for quantized matrix multiplication.

```python
from aqt.jax.v2 import config as aqt_config

dot_general_cfg = aqt_config.DotGeneral(
    lhs=aqt_config.Tensor(  # activations
        numerics=aqt_config.IntNumerics(bits=8, preserve_zero=True),
        calib_shared_axes=(0, 1),  # per-token: share across batch, seq
    ),
    rhs=aqt_config.Tensor(  # weights
        numerics=aqt_config.IntNumerics(bits=8, preserve_zero=True),
        calib_shared_axes=(0,),  # per-channel: share across input dim
    ),
)
```

#### `aqt.jax.v2.aqt_dot_general`
Drop-in replacement for `jax.lax.dot_general` with quantization.

```python
from aqt.jax.v2 import aqt_dot_general

# replace jnp.dot(x, w) with:
y = aqt_dot_general.make_dot_general(dot_general_cfg)(
    lhs=x,
    rhs=w,
    dimension_numbers=(((x.ndim - 1,), (0,)), ((), ())),
)
```

### Calibration

Calibration collects activation statistics to set quantization scales.

```python
from aqt.jax.v2 import calibration

# create calibration state
calib_state = calibration.create_calibration_state(dot_general_cfg)

# run calibration pass
for batch in calibration_data:
    calib_state = calibration.update_calibration_state(
        calib_state,
        model,
        params,
        batch
    )

# finalize scales
quant_params = calibration.finalize_calibration(calib_state, params)
```

### Quantization Modes

| Mode | Weights | Activations | Use Case |
|------|---------|-------------|----------|
| W8A8 | INT8 | INT8 | Best speedup, slight quality loss |
| W8A16 | INT8 | FP16 | Good speedup, minimal quality loss |
| W4A16 | INT4 | FP16 | Max compression, more quality loss |

### Per-Channel vs Per-Tensor

- **Per-tensor**: single scale for entire tensor (fastest, lowest quality)
- **Per-channel**: separate scale per output channel (balanced)
- **Per-token**: separate scale per token (for activations, best quality)

## Continuous Batching Reference

### Concept
Instead of processing fixed batches, continuously add/remove requests
as they complete. Maximizes GPU/TPU utilization.

### Key Components

1. **Request Queue**: incoming requests with priorities
2. **Prefill Phase**: process prompt in parallel (compute-bound)
3. **Decode Phase**: generate tokens autoregressively (memory-bound)
4. **Slot Manager**: allocate/deallocate KV cache slots

### State Machine

```
Request States:
  QUEUED → PREFILLING → DECODING → COMPLETED
                ↓
              EVICTED (if preempted)
```

### KV Cache Management

```python
# Pre-allocate KV cache for max concurrent requests
kv_cache_shape = (
    max_batch_size,
    num_layers,
    2,  # key, value
    max_seq_len,
    num_kv_heads,
    head_dim,
)

# Track which slots are in use
slot_status = jnp.zeros(max_batch_size, dtype=bool)
slot_seq_lens = jnp.zeros(max_batch_size, dtype=jnp.int32)
```
```

---

## 2. what student does (E68-T1: INT8 Quantization)

### step 1: understand AQT library

```python
# student explores AQT structure
from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2 import aqt_dot_general

# examine available numerics
print(dir(aqt_config))
# ['DotGeneral', 'Tensor', 'IntNumerics', 'FpNumerics', ...]

# understand IntNumerics
help(aqt_config.IntNumerics)
# bits: int - number of bits for quantization
# preserve_zero: bool - ensure zero maps to zero (important for ReLU)
# dtype: optional override dtype

# understand calibration axes
# calib_shared_axes specifies which axes share the same scale
# (0,) = per-channel (one scale per output feature)
# (0, 1) = per-token (one scale per token position)
```

### step 2: configure quantization (per-channel weights, per-token activations)

```python
# quantize.py - student implementation

from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2 import aqt_dot_general
import jax.numpy as jnp

def create_int8_config():
    """
    Create INT8 quantization config.

    Key choices:
    - Weights: per-channel (calib_shared_axes=(0,))
      Each output channel gets its own scale, preserving per-feature ranges.

    - Activations: per-token (calib_shared_axes=(0, 1) for [batch, seq, hidden])
      Each token gets its own scale, handling varying activation magnitudes.
    """

    # Weight quantization config
    weight_config = aqt_config.Tensor(
        numerics=aqt_config.IntNumerics(
            bits=8,
            preserve_zero=True,  # zero stays zero
        ),
        calib_shared_axes=(0,),  # per-channel: share across input dimension
        po2_scale=False,  # allow non-power-of-2 scales for accuracy
    )

    # Activation quantization config
    activation_config = aqt_config.Tensor(
        numerics=aqt_config.IntNumerics(
            bits=8,
            preserve_zero=True,
        ),
        calib_shared_axes=(0, 1),  # per-token: share across batch and seq dims
        po2_scale=False,
    )

    # Combined config for matrix multiply
    dot_general_config = aqt_config.DotGeneral(
        lhs=activation_config,  # activations on left
        rhs=weight_config,      # weights on right
    )

    return dot_general_config


def create_quantized_dot_general(config):
    """Create quantized matmul function."""
    return aqt_dot_general.make_dot_general(config)
```

### step 3: apply to model layers

```python
# quantize.py continued

import jax
from flax import linen as nn
from typing import Any

class QuantizedDense(nn.Module):
    """
    Quantized dense layer using AQT.

    Replaces standard jnp.dot with quantized version.
    """
    features: int
    quant_config: Any
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # standard weight initialization
        kernel = self.param(
            'kernel',
            nn.initializers.normal(0.02),
            (x.shape[-1], self.features)
        )

        # quantized matmul
        qdot = create_quantized_dot_general(self.quant_config)
        y = qdot(
            lhs=x,
            rhs=kernel,
            dimension_numbers=(((x.ndim - 1,), (0,)), ((), ())),
        )

        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,))
            y = y + bias

        return y


def quantize_model(model_class, config, quant_config):
    """
    Create quantized version of model.

    Strategy: replace all DenseGeneral layers with QuantizedDense.
    """

    class QuantizedModel(model_class):
        """Model with quantized linear layers."""

        def setup(self):
            super().setup()
            # override linear layers with quantized versions
            self.quant_config = quant_config

        def _make_dense(self, features, use_bias=False):
            """Factory for quantized dense layers."""
            return QuantizedDense(
                features=features,
                quant_config=self.quant_config,
                use_bias=use_bias,
            )

    return QuantizedModel


# Alternative: transform existing params
def quantize_params(params, quant_config):
    """
    Quantize existing model parameters.

    This is post-training quantization (PTQ).
    Weights are quantized, scales are computed.
    """

    def quantize_kernel(kernel):
        """Quantize single weight matrix."""
        # compute per-channel scales
        # scale = max(abs(kernel)) / 127 per channel
        channel_max = jnp.max(jnp.abs(kernel), axis=0, keepdims=True)
        scale = channel_max / 127.0
        scale = jnp.maximum(scale, 1e-8)  # avoid division by zero

        # quantize
        kernel_int8 = jnp.round(kernel / scale).astype(jnp.int8)

        return {
            'kernel_int8': kernel_int8,
            'scale': scale.squeeze(),
        }

    # traverse params and quantize kernels
    def traverse(p, path=""):
        if isinstance(p, dict):
            result = {}
            for k, v in p.items():
                if k == 'kernel' and 'Dense' in path:
                    result.update(quantize_kernel(v))
                else:
                    result[k] = traverse(v, f"{path}/{k}")
            return result
        return p

    return traverse(params)
```

### step 4: run calibration

```python
# calibration.py

import jax
import jax.numpy as jnp
from tqdm import tqdm

def create_calibration_dataset(tokenizer, num_samples=100):
    """
    Create calibration dataset from representative prompts.

    Important: calibration data should be representative of
    actual inference distribution.
    """

    calibration_texts = [
        # diverse prompts covering different domains
        "The quick brown fox jumps over the lazy dog.",
        "In machine learning, gradient descent is used to minimize loss.",
        "The capital of France is Paris, located on the Seine river.",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "According to Einstein's theory of relativity, E=mc².",
        # ... more diverse examples
    ] * (num_samples // 5)

    encoded = tokenizer(
        calibration_texts[:num_samples],
        padding='max_length',
        max_length=128,
        truncation=True,
        return_tensors='jax',
    )

    return encoded['input_ids']


def collect_activation_stats(model, params, calibration_data):
    """
    Collect activation statistics for calibration.

    We need min/max per-token to set quantization scales.
    """

    stats = {
        'min': [],
        'max': [],
        'absmax': [],
    }

    @jax.jit
    def forward_with_stats(params, input_ids):
        """Forward pass collecting intermediate activations."""
        # This requires model modification to return intermediates
        # Simplified: just collect output stats
        outputs = model.apply(params, input_ids)
        return outputs

    for i in tqdm(range(0, len(calibration_data), 8), desc="Calibration"):
        batch = calibration_data[i:i+8]
        outputs = forward_with_stats(params, batch)

        stats['min'].append(float(jnp.min(outputs)))
        stats['max'].append(float(jnp.max(outputs)))
        stats['absmax'].append(float(jnp.max(jnp.abs(outputs))))

    return {
        'activation_min': min(stats['min']),
        'activation_max': max(stats['max']),
        'activation_absmax': max(stats['absmax']),
    }


def calibrate_quantization(model, params, tokenizer, num_samples=100):
    """
    Full calibration pipeline.

    Returns calibrated quantization parameters.
    """

    # create calibration data
    calib_data = create_calibration_dataset(tokenizer, num_samples)

    # collect stats
    stats = collect_activation_stats(model, params, calib_data)

    # compute scales
    activation_scale = stats['activation_absmax'] / 127.0

    print(f"Calibration complete:")
    print(f"  Activation range: [{stats['activation_min']:.3f}, {stats['activation_max']:.3f}]")
    print(f"  Activation scale: {activation_scale:.6f}")

    return {
        'activation_scale': activation_scale,
        'stats': stats,
    }
```

### step 5: measure speedup and divergence

```python
# evaluate_quantization.py

import jax
import jax.numpy as jnp
import time
from typing import List, Tuple

def measure_speedup(
    original_model,
    original_params,
    quantized_model,
    quantized_params,
    test_inputs: jnp.ndarray,
    num_runs: int = 10,
) -> float:
    """
    Measure speedup from quantization.

    Returns: speedup ratio (>1 means faster)
    """

    @jax.jit
    def run_original(inputs):
        return original_model.apply(original_params, inputs)

    @jax.jit
    def run_quantized(inputs):
        return quantized_model.apply(quantized_params, inputs)

    # warmup
    _ = run_original(test_inputs)
    _ = run_quantized(test_inputs)

    # benchmark original
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = run_original(test_inputs).block_until_ready()
    original_time = (time.perf_counter() - start) / num_runs

    # benchmark quantized
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = run_quantized(test_inputs).block_until_ready()
    quantized_time = (time.perf_counter() - start) / num_runs

    speedup = original_time / quantized_time

    print(f"Original time: {original_time*1000:.2f}ms")
    print(f"Quantized time: {quantized_time*1000:.2f}ms")
    print(f"Speedup: {speedup:.2f}x")

    return speedup


def compute_kl_divergence(
    logits_original: jnp.ndarray,
    logits_quantized: jnp.ndarray,
) -> float:
    """
    Compute KL divergence between original and quantized outputs.

    KL(P || Q) = sum(P * log(P/Q))
    where P = original, Q = quantized
    """

    # convert to probabilities
    p = jax.nn.softmax(logits_original, axis=-1)
    q = jax.nn.softmax(logits_quantized, axis=-1)

    # add small epsilon for numerical stability
    eps = 1e-10
    p = jnp.clip(p, eps, 1.0)
    q = jnp.clip(q, eps, 1.0)

    # KL divergence
    kl = jnp.sum(p * jnp.log(p / q), axis=-1)

    # average over batch and sequence
    return float(jnp.mean(kl))


def compute_token_match_rate(
    original_model,
    original_params,
    quantized_model,
    quantized_params,
    test_inputs: jnp.ndarray,
) -> float:
    """
    Compute fraction of tokens where argmax matches.

    This is a stricter metric than KL divergence.
    """

    original_logits = original_model.apply(original_params, test_inputs)
    quantized_logits = quantized_model.apply(quantized_params, test_inputs)

    original_tokens = jnp.argmax(original_logits, axis=-1)
    quantized_tokens = jnp.argmax(quantized_logits, axis=-1)

    matches = (original_tokens == quantized_tokens).astype(jnp.float32)
    match_rate = float(jnp.mean(matches))

    return match_rate


def evaluate_quantization(
    original_model,
    original_params,
    quantized_model,
    quantized_params,
    tokenizer,
    test_prompts: List[str],
) -> dict:
    """
    Full evaluation of quantization quality.

    Returns dict with:
    - speedup: float (>1 = faster)
    - kl_divergence: float (<0.01 = good)
    - token_match_rate: float (>0.95 = good)
    """

    # tokenize test prompts
    encoded = tokenizer(
        test_prompts,
        padding='max_length',
        max_length=128,
        truncation=True,
        return_tensors='jax',
    )
    test_inputs = encoded['input_ids']

    # measure speedup
    speedup = measure_speedup(
        original_model, original_params,
        quantized_model, quantized_params,
        test_inputs,
    )

    # compute divergence
    original_logits = original_model.apply(original_params, test_inputs)
    quantized_logits = quantized_model.apply(quantized_params, test_inputs)

    kl_div = compute_kl_divergence(original_logits, quantized_logits)
    match_rate = compute_token_match_rate(
        original_model, original_params,
        quantized_model, quantized_params,
        test_inputs,
    )

    results = {
        'speedup': speedup,
        'kl_divergence': kl_div,
        'token_match_rate': match_rate,
    }

    print(f"\n=== Quantization Evaluation ===")
    print(f"Speedup: {speedup:.2f}x")
    print(f"KL Divergence: {kl_div:.6f} (threshold: 0.01)")
    print(f"Token Match Rate: {match_rate:.4f} (threshold: 0.95)")

    passed = speedup >= 1.5 and kl_div < 0.01 and match_rate > 0.95
    print(f"PASSED: {passed}")

    return results
```

---

## 3. what student does (E68-T4: Continuous Batching)

### step 1: implement request queue

```python
# batcher.py - student implementation

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from queue import PriorityQueue
import threading
import time

class RequestState(Enum):
    QUEUED = "queued"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    COMPLETED = "completed"
    EVICTED = "evicted"


@dataclass
class InferenceRequest:
    """Single inference request."""
    request_id: str
    prompt: str
    input_ids: jnp.ndarray  # tokenized prompt
    max_new_tokens: int = 128
    priority: int = 0  # lower = higher priority

    # runtime state
    state: RequestState = RequestState.QUEUED
    generated_tokens: List[int] = field(default_factory=list)
    slot_id: Optional[int] = None  # KV cache slot
    arrival_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority < other.priority


class RequestQueue:
    """
    Thread-safe priority queue for inference requests.

    Features:
    - Priority ordering (lower priority value = processed first)
    - Arrival time tiebreaker (FIFO for same priority)
    - Thread-safe add/pop operations
    """

    def __init__(self, max_size: int = 1000):
        self.queue = PriorityQueue(maxsize=max_size)
        self.lock = threading.Lock()
        self.request_counter = 0

    def add_request(self, request: InferenceRequest) -> None:
        """Add request to queue."""
        with self.lock:
            # use (priority, arrival_order, request) for stable sorting
            self.queue.put((request.priority, self.request_counter, request))
            self.request_counter += 1

    def pop_request(self) -> Optional[InferenceRequest]:
        """Get highest priority request."""
        try:
            _, _, request = self.queue.get_nowait()
            return request
        except:
            return None

    def peek_request(self) -> Optional[InferenceRequest]:
        """Look at highest priority request without removing."""
        with self.lock:
            if self.queue.empty():
                return None
            # PriorityQueue doesn't support peek, so we pop and put back
            item = self.queue.get_nowait()
            self.queue.put(item)
            return item[2]

    def size(self) -> int:
        """Current queue size."""
        return self.queue.qsize()

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self.queue.empty()
```

### step 2: implement prefill thread

```python
# batcher.py continued

class PrefillEngine:
    """
    Handles prefill phase of inference.

    Prefill processes the entire prompt in parallel,
    computing initial KV cache entries.
    """

    def __init__(self, model, params, max_batch_size: int = 8):
        self.model = model
        self.params = params
        self.max_batch_size = max_batch_size

    def prefill_batch(
        self,
        requests: List[InferenceRequest],
        kv_cache: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Run prefill for a batch of requests.

        Args:
            requests: list of requests to prefill
            kv_cache: pre-allocated KV cache [max_batch, layers, 2, max_seq, heads, dim]

        Returns:
            updated_kv_cache: with new KV entries
            first_tokens: first generated token for each request
        """

        # pad inputs to same length
        max_len = max(len(r.input_ids) for r in requests)
        batch_size = len(requests)

        # create padded input tensor
        input_ids = jnp.zeros((batch_size, max_len), dtype=jnp.int32)
        attention_mask = jnp.zeros((batch_size, max_len), dtype=jnp.bool_)

        for i, req in enumerate(requests):
            seq_len = len(req.input_ids)
            input_ids = input_ids.at[i, :seq_len].set(req.input_ids)
            attention_mask = attention_mask.at[i, :seq_len].set(True)

        # run forward pass
        logits, new_kv = self._forward_prefill(input_ids, attention_mask)

        # update KV cache for assigned slots
        for i, req in enumerate(requests):
            slot = req.slot_id
            seq_len = len(req.input_ids)
            # copy KV to assigned slot
            kv_cache = kv_cache.at[slot, :, :, :seq_len, :, :].set(
                new_kv[i, :, :, :seq_len, :, :]
            )

        # get first token (argmax of last position)
        first_tokens = []
        for i, req in enumerate(requests):
            seq_len = len(req.input_ids)
            last_logits = logits[i, seq_len - 1, :]
            first_token = int(jnp.argmax(last_logits))
            first_tokens.append(first_token)

        return kv_cache, first_tokens

    @jax.jit
    def _forward_prefill(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """JIT-compiled prefill forward pass."""
        # This calls model with return_kv=True
        return self.model.apply(
            self.params,
            input_ids,
            attention_mask=attention_mask,
            return_kv_cache=True,
        )
```

### step 3: implement decode thread with slot management

```python
# batcher.py continued

class SlotManager:
    """
    Manages KV cache slot allocation.

    Pre-allocates fixed number of slots.
    Tracks which slots are in use and their sequence lengths.
    """

    def __init__(self, max_slots: int, max_seq_len: int):
        self.max_slots = max_slots
        self.max_seq_len = max_seq_len

        # slot status: True = in use
        self.slot_in_use = [False] * max_slots

        # current sequence length per slot
        self.slot_seq_lens = [0] * max_slots

        # request ID occupying each slot
        self.slot_request_ids = [None] * max_slots

        self.lock = threading.Lock()

    def allocate_slot(self, request_id: str) -> Optional[int]:
        """
        Allocate a free slot for a request.

        Returns slot_id or None if no slots available.
        """
        with self.lock:
            for i in range(self.max_slots):
                if not self.slot_in_use[i]:
                    self.slot_in_use[i] = True
                    self.slot_seq_lens[i] = 0
                    self.slot_request_ids[i] = request_id
                    return i
            return None

    def release_slot(self, slot_id: int) -> None:
        """Release a slot back to the pool."""
        with self.lock:
            self.slot_in_use[slot_id] = False
            self.slot_seq_lens[slot_id] = 0
            self.slot_request_ids[slot_id] = None

    def update_seq_len(self, slot_id: int, new_len: int) -> None:
        """Update sequence length for a slot."""
        with self.lock:
            self.slot_seq_lens[slot_id] = new_len

    def get_active_slots(self) -> List[int]:
        """Get list of currently active slot IDs."""
        with self.lock:
            return [i for i in range(self.max_slots) if self.slot_in_use[i]]

    def num_free_slots(self) -> int:
        """Count available slots."""
        with self.lock:
            return sum(1 for x in self.slot_in_use if not x)


class DecodeEngine:
    """
    Handles decode phase of inference.

    Decode generates one token at a time using cached KV.
    Multiple requests can be batched together.
    """

    def __init__(self, model, params, slot_manager: SlotManager):
        self.model = model
        self.params = params
        self.slot_manager = slot_manager

    def decode_step(
        self,
        requests: List[InferenceRequest],
        kv_cache: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, List[int]]:
        """
        Run single decode step for active requests.

        Args:
            requests: list of requests currently decoding
            kv_cache: current KV cache state

        Returns:
            updated_kv_cache: with new KV entries
            next_tokens: next token for each request
        """

        batch_size = len(requests)
        if batch_size == 0:
            return kv_cache, []

        # gather last tokens for each request
        input_tokens = jnp.array([
            req.generated_tokens[-1] if req.generated_tokens else 0
            for req in requests
        ], dtype=jnp.int32)[:, None]  # [batch, 1]

        # gather slot IDs
        slot_ids = jnp.array([req.slot_id for req in requests], dtype=jnp.int32)

        # gather current sequence lengths
        seq_lens = jnp.array([
            len(req.input_ids) + len(req.generated_tokens)
            for req in requests
        ], dtype=jnp.int32)

        # run decode forward pass
        logits, new_kv = self._forward_decode(
            input_tokens,
            kv_cache,
            slot_ids,
            seq_lens,
        )

        # update KV cache
        for i, req in enumerate(requests):
            slot = req.slot_id
            pos = len(req.input_ids) + len(req.generated_tokens)
            # append new KV entry
            kv_cache = kv_cache.at[slot, :, :, pos, :, :].set(
                new_kv[i, :, :, 0, :, :]
            )
            self.slot_manager.update_seq_len(slot, pos + 1)

        # sample next tokens
        next_tokens = []
        for i in range(batch_size):
            next_token = int(jnp.argmax(logits[i, 0, :]))
            next_tokens.append(next_token)

        return kv_cache, next_tokens

    @jax.jit
    def _forward_decode(
        self,
        input_tokens: jnp.ndarray,
        kv_cache: jnp.ndarray,
        slot_ids: jnp.ndarray,
        seq_lens: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """JIT-compiled decode forward pass."""
        return self.model.apply(
            self.params,
            input_tokens,
            kv_cache=kv_cache,
            slot_ids=slot_ids,
            seq_lens=seq_lens,
            return_kv_cache=True,
        )
```

### step 4: handle request completion/eviction

```python
# batcher.py continued

class ContinuousBatcher:
    """
    Main continuous batching scheduler.

    Coordinates prefill and decode engines,
    manages request lifecycle.
    """

    def __init__(
        self,
        model,
        params,
        tokenizer,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
        num_layers: int = 24,
        num_kv_heads: int = 16,
        head_dim: int = 64,
    ):
        self.model = model
        self.params = params
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # initialize components
        self.request_queue = RequestQueue()
        self.slot_manager = SlotManager(max_batch_size, max_seq_len)
        self.prefill_engine = PrefillEngine(model, params, max_batch_size)
        self.decode_engine = DecodeEngine(model, params, self.slot_manager)

        # pre-allocate KV cache
        self.kv_cache = jnp.zeros((
            max_batch_size,
            num_layers,
            2,  # key, value
            max_seq_len,
            num_kv_heads,
            head_dim,
        ), dtype=jnp.float16)

        # track active requests
        self.active_requests: Dict[str, InferenceRequest] = {}
        self.completed_requests: Dict[str, InferenceRequest] = {}

        # state
        self.running = False
        self.lock = threading.Lock()

    def add_request(self, prompt: str, max_new_tokens: int = 128, priority: int = 0) -> str:
        """
        Add new inference request.

        Returns request_id for tracking.
        """
        import uuid
        request_id = str(uuid.uuid4())[:8]

        # tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors='jax')[0]

        request = InferenceRequest(
            request_id=request_id,
            prompt=prompt,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            priority=priority,
        )

        self.request_queue.add_request(request)
        return request_id

    def _schedule_prefill(self) -> List[InferenceRequest]:
        """
        Select requests for prefill phase.

        Strategy: fill available slots from queue.
        """
        requests_to_prefill = []

        while self.slot_manager.num_free_slots() > 0 and not self.request_queue.is_empty():
            request = self.request_queue.pop_request()
            if request is None:
                break

            # allocate slot
            slot_id = self.slot_manager.allocate_slot(request.request_id)
            if slot_id is None:
                # no slots available, put back in queue
                self.request_queue.add_request(request)
                break

            request.slot_id = slot_id
            request.state = RequestState.PREFILLING
            request.start_time = time.time()

            requests_to_prefill.append(request)
            self.active_requests[request.request_id] = request

        return requests_to_prefill

    def _handle_completion(self, request: InferenceRequest) -> None:
        """Handle completed request."""
        request.state = RequestState.COMPLETED
        request.end_time = time.time()

        # release slot
        self.slot_manager.release_slot(request.slot_id)

        # move to completed
        with self.lock:
            del self.active_requests[request.request_id]
            self.completed_requests[request.request_id] = request

    def _handle_eviction(self, request: InferenceRequest) -> None:
        """
        Handle evicted request (preemption).

        Called when higher priority request needs slot.
        """
        request.state = RequestState.EVICTED

        # release slot
        self.slot_manager.release_slot(request.slot_id)
        request.slot_id = None

        # re-queue with progress preserved
        # (in full impl, would save partial KV cache)
        with self.lock:
            del self.active_requests[request.request_id]

        self.request_queue.add_request(request)

    def _is_request_done(self, request: InferenceRequest) -> bool:
        """Check if request has finished generating."""
        # check max tokens
        if len(request.generated_tokens) >= request.max_new_tokens:
            return True

        # check EOS token
        if request.generated_tokens and request.generated_tokens[-1] == self.tokenizer.eos_token_id:
            return True

        return False

    def run_iteration(self) -> None:
        """
        Run one iteration of the scheduler.

        1. Schedule prefill for new requests
        2. Run decode step for active requests
        3. Handle completions
        """

        # 1. Prefill new requests
        prefill_requests = self._schedule_prefill()
        if prefill_requests:
            self.kv_cache, first_tokens = self.prefill_engine.prefill_batch(
                prefill_requests,
                self.kv_cache,
            )

            # transition to decode state
            for req, token in zip(prefill_requests, first_tokens):
                req.generated_tokens.append(token)
                req.state = RequestState.DECODING

        # 2. Decode active requests
        decoding_requests = [
            r for r in self.active_requests.values()
            if r.state == RequestState.DECODING
        ]

        if decoding_requests:
            self.kv_cache, next_tokens = self.decode_engine.decode_step(
                decoding_requests,
                self.kv_cache,
            )

            # update generated tokens
            for req, token in zip(decoding_requests, next_tokens):
                req.generated_tokens.append(token)

        # 3. Handle completions
        for req in list(self.active_requests.values()):
            if self._is_request_done(req):
                self._handle_completion(req)

    def get_result(self, request_id: str) -> Optional[str]:
        """Get generated text for completed request."""
        if request_id in self.completed_requests:
            req = self.completed_requests[request_id]
            return self.tokenizer.decode(req.generated_tokens, skip_special_tokens=True)
        return None

    def run(self, timeout: float = 60.0) -> None:
        """
        Run scheduler until all requests complete or timeout.
        """
        self.running = True
        start_time = time.time()

        while self.running:
            # check timeout
            if time.time() - start_time > timeout:
                print("Timeout reached")
                break

            # check if done
            if self.request_queue.is_empty() and not self.active_requests:
                break

            # run iteration
            self.run_iteration()

        self.running = False
```

### step 5: measure throughput and latency

```python
# evaluate_batching.py

import time
import numpy as np
from typing import List, Dict

def measure_throughput_latency(
    batcher: ContinuousBatcher,
    test_prompts: List[str],
    max_new_tokens: int = 128,
) -> Dict[str, float]:
    """
    Measure continuous batching performance.

    Returns:
    - throughput: tokens/second
    - latency_p50: median latency
    - latency_p95: 95th percentile latency
    - latency_p99: 99th percentile latency
    """

    # submit all requests
    request_ids = []
    for prompt in test_prompts:
        rid = batcher.add_request(prompt, max_new_tokens=max_new_tokens)
        request_ids.append(rid)

    # run until complete
    start_time = time.perf_counter()
    batcher.run(timeout=300.0)
    total_time = time.perf_counter() - start_time

    # collect metrics
    latencies = []
    total_tokens = 0

    for rid in request_ids:
        req = batcher.completed_requests.get(rid)
        if req:
            latency = req.end_time - req.arrival_time
            latencies.append(latency)
            total_tokens += len(req.generated_tokens)

    # compute stats
    throughput = total_tokens / total_time
    latency_p50 = np.percentile(latencies, 50)
    latency_p95 = np.percentile(latencies, 95)
    latency_p99 = np.percentile(latencies, 99)

    results = {
        'throughput': throughput,
        'latency_p50': latency_p50,
        'latency_p95': latency_p95,
        'latency_p99': latency_p99,
        'total_time': total_time,
        'num_requests': len(test_prompts),
        'total_tokens': total_tokens,
    }

    print(f"\n=== Continuous Batching Evaluation ===")
    print(f"Throughput: {throughput:.1f} tokens/sec")
    print(f"Latency p50: {latency_p50:.3f}s")
    print(f"Latency p95: {latency_p95:.3f}s")
    print(f"Latency p99: {latency_p99:.3f}s")
    print(f"p95/p50 ratio: {latency_p95/latency_p50:.2f}")

    return results


def compare_to_baseline(
    batcher: ContinuousBatcher,
    baseline_fn,  # naive_batch_inference
    test_prompts: List[str],
) -> Dict[str, float]:
    """
    Compare continuous batching to naive baseline.
    """

    # measure baseline
    start = time.perf_counter()
    baseline_outputs = baseline_fn(test_prompts)
    baseline_time = time.perf_counter() - start
    baseline_tokens = sum(len(o.split()) for o in baseline_outputs)  # approx
    baseline_throughput = baseline_tokens / baseline_time

    # measure continuous batching
    cb_results = measure_throughput_latency(batcher, test_prompts)

    speedup = cb_results['throughput'] / baseline_throughput

    print(f"\n=== Comparison ===")
    print(f"Baseline throughput: {baseline_throughput:.1f} tokens/sec")
    print(f"Continuous batching: {cb_results['throughput']:.1f} tokens/sec")
    print(f"Speedup: {speedup:.2f}x")

    passed = speedup >= 2.0 and cb_results['latency_p95'] < 1.5 * cb_results['latency_p50']
    print(f"PASSED: {passed}")

    return {
        'baseline_throughput': baseline_throughput,
        'cb_throughput': cb_results['throughput'],
        'speedup': speedup,
        **cb_results,
    }
```

---

## 4. submission format

### 4.1 E68-T1: quantize.py

```python
# quantize.py
"""
INT8 Quantization for Qwen Inference
====================================

Student submission for E68-T1.

Required function:
- quantize_model(model, params, tokenizer) -> (quantized_model, quantized_params)
"""

from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2 import aqt_dot_general
import jax
import jax.numpy as jnp
from typing import Tuple, Any


def quantize_model(
    model,
    params,
    tokenizer,
    calibration_samples: int = 100,
) -> Tuple[Any, Any]:
    """
    Quantize model to INT8.

    Args:
        model: Qwen model class/instance
        params: model parameters (FP32)
        tokenizer: tokenizer for calibration
        calibration_samples: number of samples for calibration

    Returns:
        quantized_model: model with quantized layers
        quantized_params: quantized parameters with scales

    Requirements:
        - Speedup >= 1.5x
        - KL divergence < 0.01
        - Token match rate > 95%
    """
    # TODO: Implement this function
    raise NotImplementedError("Complete this function")
```

### 4.2 E68-T4: batcher.py

```python
# batcher.py
"""
Continuous Batching for Qwen Inference
======================================

Student submission for E68-T4.

Required class:
- ContinuousBatcher with add_request(), run(), get_result() methods
"""

import jax
import jax.numpy as jnp
from typing import List, Optional, Dict


class ContinuousBatcher:
    """
    Continuous batching scheduler for Qwen inference.

    Must implement:
    - add_request(prompt, max_new_tokens, priority) -> request_id
    - run(timeout) -> None
    - get_result(request_id) -> Optional[str]

    Requirements:
        - Throughput >= 2x baseline
        - p95 latency < 1.5 * p50 latency
    """

    def __init__(
        self,
        model,
        params,
        tokenizer,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
        **kwargs,
    ):
        # TODO: Initialize components
        raise NotImplementedError("Complete this class")

    def add_request(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        priority: int = 0,
    ) -> str:
        """Add request to queue, return request_id."""
        raise NotImplementedError

    def run(self, timeout: float = 60.0) -> None:
        """Run scheduler until all requests complete or timeout."""
        raise NotImplementedError

    def get_result(self, request_id: str) -> Optional[str]:
        """Get generated text for completed request."""
        raise NotImplementedError
```

---

## 5. evaluation pipeline

### 5.1 validation (quick checks)

```python
# validation.py

def validate_quantization_submission(solution_path: str) -> bool:
    """Quick validation for E68-T1."""

    import importlib.util
    spec = importlib.util.spec_from_file_location("quantize", solution_path)
    solution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution)

    # check function exists
    assert hasattr(solution, 'quantize_model'), "Missing quantize_model function"
    assert callable(solution.quantize_model), "quantize_model must be callable"

    # check signature
    import inspect
    sig = inspect.signature(solution.quantize_model)
    params = list(sig.parameters.keys())
    assert 'model' in params, "quantize_model must accept 'model' parameter"
    assert 'params' in params, "quantize_model must accept 'params' parameter"

    return True


def validate_batching_submission(solution_path: str) -> bool:
    """Quick validation for E68-T4."""

    import importlib.util
    spec = importlib.util.spec_from_file_location("batcher", solution_path)
    solution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution)

    # check class exists
    assert hasattr(solution, 'ContinuousBatcher'), "Missing ContinuousBatcher class"

    # check required methods
    cls = solution.ContinuousBatcher
    assert hasattr(cls, 'add_request'), "Missing add_request method"
    assert hasattr(cls, 'run'), "Missing run method"
    assert hasattr(cls, 'get_result'), "Missing get_result method"

    return True
```

### 5.2 correctness checks

```python
# correctness.py

import jax.numpy as jnp
import numpy as np
from typing import List

def check_quantization_correctness(
    original_model,
    original_params,
    quantized_model,
    quantized_params,
    test_inputs: jnp.ndarray,
) -> dict:
    """
    Check quantization preserves model behavior.

    Metrics:
    - token_match_rate: fraction of matching argmax tokens (>95%)
    - kl_divergence: KL(original || quantized) (<0.01)
    - max_abs_diff: maximum absolute logit difference
    """

    original_logits = original_model.apply(original_params, test_inputs)
    quantized_logits = quantized_model.apply(quantized_params, test_inputs)

    # token match rate
    original_tokens = jnp.argmax(original_logits, axis=-1)
    quantized_tokens = jnp.argmax(quantized_logits, axis=-1)
    token_match_rate = float(jnp.mean(original_tokens == quantized_tokens))

    # KL divergence
    p = jax.nn.softmax(original_logits, axis=-1)
    q = jax.nn.softmax(quantized_logits, axis=-1)
    kl_div = float(jnp.mean(jnp.sum(p * jnp.log(p / (q + 1e-10) + 1e-10), axis=-1)))

    # max abs diff
    max_abs_diff = float(jnp.max(jnp.abs(original_logits - quantized_logits)))

    return {
        'token_match_rate': token_match_rate,
        'kl_divergence': kl_div,
        'max_abs_diff': max_abs_diff,
        'token_match_passed': token_match_rate > 0.95,
        'kl_passed': kl_div < 0.01,
    }


def check_batching_correctness(
    batcher,
    baseline_fn,
    test_prompts: List[str],
    tokenizer,
) -> dict:
    """
    Check continuous batching produces correct outputs.

    Compares token-by-token to baseline generation.
    """

    # get baseline outputs
    baseline_outputs = baseline_fn(test_prompts)

    # get batcher outputs
    request_ids = [batcher.add_request(p) for p in test_prompts]
    batcher.run(timeout=120.0)
    batcher_outputs = [batcher.get_result(rid) for rid in request_ids]

    # compare
    exact_matches = 0
    token_matches = 0
    total_tokens = 0

    for baseline, batched in zip(baseline_outputs, batcher_outputs):
        if baseline == batched:
            exact_matches += 1

        baseline_tokens = tokenizer.encode(baseline)
        batched_tokens = tokenizer.encode(batched)

        min_len = min(len(baseline_tokens), len(batched_tokens))
        matches = sum(b == t for b, t in zip(baseline_tokens[:min_len], batched_tokens[:min_len]))
        token_matches += matches
        total_tokens += max(len(baseline_tokens), len(batched_tokens))

    return {
        'exact_match_rate': exact_matches / len(test_prompts),
        'token_match_rate': token_matches / total_tokens,
        'num_prompts': len(test_prompts),
    }
```

### 5.3 performance measurement

```python
# performance.py

import time
import jax
from typing import Dict

def measure_quantization_speedup(
    original_model,
    original_params,
    quantized_model,
    quantized_params,
    test_inputs,
    num_runs: int = 20,
) -> Dict[str, float]:
    """Measure speedup from quantization."""

    @jax.jit
    def run_original(x):
        return original_model.apply(original_params, x)

    @jax.jit
    def run_quantized(x):
        return quantized_model.apply(quantized_params, x)

    # warmup
    for _ in range(3):
        _ = run_original(test_inputs).block_until_ready()
        _ = run_quantized(test_inputs).block_until_ready()

    # benchmark original
    times_original = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = run_original(test_inputs).block_until_ready()
        times_original.append(time.perf_counter() - start)

    # benchmark quantized
    times_quantized = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = run_quantized(test_inputs).block_until_ready()
        times_quantized.append(time.perf_counter() - start)

    original_mean = sum(times_original) / len(times_original)
    quantized_mean = sum(times_quantized) / len(times_quantized)

    return {
        'original_time_ms': original_mean * 1000,
        'quantized_time_ms': quantized_mean * 1000,
        'speedup': original_mean / quantized_mean,
    }


def measure_batching_performance(
    batcher,
    test_prompts,
    max_new_tokens: int = 128,
) -> Dict[str, float]:
    """Measure continuous batching performance."""

    import numpy as np

    # submit requests
    request_ids = [batcher.add_request(p, max_new_tokens) for p in test_prompts]

    # run
    start = time.perf_counter()
    batcher.run(timeout=300.0)
    total_time = time.perf_counter() - start

    # collect latencies
    latencies = []
    total_tokens = 0
    for rid in request_ids:
        req = batcher.completed_requests.get(rid)
        if req:
            latencies.append(req.end_time - req.arrival_time)
            total_tokens += len(req.generated_tokens)

    throughput = total_tokens / total_time
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    return {
        'throughput_tokens_per_sec': throughput,
        'latency_p50': p50,
        'latency_p95': p95,
        'latency_p99': p99,
        'p95_p50_ratio': p95 / p50,
        'total_time': total_time,
        'total_tokens': total_tokens,
    }
```

### 5.4 full evaluation

```python
# evaluator.py

from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class EvaluationResult:
    passed: bool
    score: float
    metrics: Dict[str, Any]
    errors: List[str]
    grade: str  # "PASS", "EXCELLENT", "FAIL"


def evaluate_quantization(solution_path: str, model, params, tokenizer) -> EvaluationResult:
    """Full evaluation for E68-T1."""

    errors = []

    # validate submission
    try:
        validate_quantization_submission(solution_path)
    except Exception as e:
        return EvaluationResult(False, 0, {}, [str(e)], "FAIL")

    # load solution
    import importlib.util
    spec = importlib.util.spec_from_file_location("quantize", solution_path)
    solution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution)

    # run quantization
    try:
        quantized_model, quantized_params = solution.quantize_model(model, params, tokenizer)
    except Exception as e:
        return EvaluationResult(False, 0, {}, [f"quantize_model failed: {e}"], "FAIL")

    # prepare test inputs
    test_prompts = TEST_PROMPTS + HIDDEN_TEST_PROMPTS  # hidden prompts for anti-gaming
    encoded = tokenizer(test_prompts, padding='max_length', max_length=128, return_tensors='jax')
    test_inputs = encoded['input_ids']

    # correctness
    correctness = check_quantization_correctness(
        model, params, quantized_model, quantized_params, test_inputs
    )

    if not correctness['token_match_passed']:
        errors.append(f"Token match rate {correctness['token_match_rate']:.4f} < 0.95")
    if not correctness['kl_passed']:
        errors.append(f"KL divergence {correctness['kl_divergence']:.6f} >= 0.01")

    # performance
    performance = measure_quantization_speedup(
        model, params, quantized_model, quantized_params, test_inputs
    )

    speedup = performance['speedup']

    # determine grade
    if speedup >= 2.0 and correctness['kl_divergence'] < 0.005:
        grade = "EXCELLENT"
        score = 100
    elif speedup >= 1.5 and correctness['kl_divergence'] < 0.01 and correctness['token_match_rate'] > 0.95:
        grade = "PASS"
        score = 75
    else:
        grade = "FAIL"
        score = max(0, 50 * (speedup / 1.5) * (0.95 / max(0.01, correctness['token_match_rate'])))

    metrics = {**correctness, **performance}

    return EvaluationResult(
        passed=(grade != "FAIL"),
        score=score,
        metrics=metrics,
        errors=errors,
        grade=grade,
    )


def evaluate_batching(solution_path: str, model, params, tokenizer) -> EvaluationResult:
    """Full evaluation for E68-T4."""

    errors = []

    # validate
    try:
        validate_batching_submission(solution_path)
    except Exception as e:
        return EvaluationResult(False, 0, {}, [str(e)], "FAIL")

    # load solution
    import importlib.util
    spec = importlib.util.spec_from_file_location("batcher", solution_path)
    solution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution)

    # instantiate batcher
    try:
        batcher = solution.ContinuousBatcher(model, params, tokenizer)
    except Exception as e:
        return EvaluationResult(False, 0, {}, [f"ContinuousBatcher init failed: {e}"], "FAIL")

    # test prompts
    test_prompts = TEST_PROMPTS + HIDDEN_TEST_PROMPTS

    # measure performance
    perf = measure_batching_performance(batcher, test_prompts)

    # measure baseline for comparison
    baseline_throughput = measure_baseline_throughput(model, params, tokenizer, test_prompts)
    speedup = perf['throughput_tokens_per_sec'] / baseline_throughput

    # check criteria
    if speedup < 2.0:
        errors.append(f"Throughput speedup {speedup:.2f}x < 2x")
    if perf['p95_p50_ratio'] > 1.5:
        errors.append(f"p95/p50 ratio {perf['p95_p50_ratio']:.2f} > 1.5")

    # determine grade
    if speedup >= 5.0 and perf['p95_p50_ratio'] < 1.5:
        grade = "EXCELLENT"
        score = 100
    elif speedup >= 2.0 and perf['p95_p50_ratio'] < 1.5:
        grade = "PASS"
        score = 75
    else:
        grade = "FAIL"
        score = max(0, 50 * (speedup / 2.0))

    metrics = {
        **perf,
        'baseline_throughput': baseline_throughput,
        'speedup': speedup,
    }

    return EvaluationResult(
        passed=(grade != "FAIL"),
        score=score,
        metrics=metrics,
        errors=errors,
        grade=grade,
    )
```

---

## 6. reference solutions

### 6.1 INT8 quantization reference

```python
# reference_quantize.py
"""Reference solution for E68-T1."""

from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2 import aqt_dot_general
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Any


def quantize_model(model, params, tokenizer, calibration_samples=100):
    """Reference INT8 quantization implementation."""

    # 1. Create quantization config
    quant_config = aqt_config.DotGeneral(
        lhs=aqt_config.Tensor(
            numerics=aqt_config.IntNumerics(bits=8, preserve_zero=True),
            calib_shared_axes=(0, 1),  # per-token for activations
        ),
        rhs=aqt_config.Tensor(
            numerics=aqt_config.IntNumerics(bits=8, preserve_zero=True),
            calib_shared_axes=(0,),  # per-channel for weights
        ),
    )

    # 2. Create calibration data
    calib_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming artificial intelligence.",
        "def hello(): return 'world'",
    ] * (calibration_samples // 3 + 1)

    encoded = tokenizer(
        calib_texts[:calibration_samples],
        padding='max_length',
        max_length=128,
        truncation=True,
        return_tensors='jax',
    )
    calib_data = encoded['input_ids']

    # 3. Collect activation stats
    @jax.jit
    def forward(params, x):
        return model.apply(params, x)

    activation_absmax = 0.0
    for i in range(0, len(calib_data), 8):
        batch = calib_data[i:i+8]
        outputs = forward(params, batch)
        activation_absmax = max(activation_absmax, float(jnp.max(jnp.abs(outputs))))

    # 4. Quantize weights
    def quantize_leaf(path, leaf):
        if 'kernel' in str(path):
            # per-channel quantization
            absmax = jnp.max(jnp.abs(leaf), axis=0, keepdims=True)
            scale = absmax / 127.0
            scale = jnp.maximum(scale, 1e-8)
            quantized = jnp.round(leaf / scale).astype(jnp.int8)
            return {'kernel_int8': quantized, 'kernel_scale': scale.squeeze()}
        return leaf

    quantized_params = jax.tree_util.tree_map_with_path(quantize_leaf, params)

    # 5. Create quantized model class
    class QuantizedModel(type(model)):
        def __call__(self, x, **kwargs):
            # dequantize on the fly for simplicity
            # (real impl would use int8 matmul kernels)
            def dequantize_params(p):
                if isinstance(p, dict) and 'kernel_int8' in p:
                    return p['kernel_int8'].astype(jnp.float32) * p['kernel_scale']
                return p

            fp_params = jax.tree_util.tree_map(dequantize_params, self.params)
            return super().__call__(x, **kwargs)

    return QuantizedModel, quantized_params
```

### 6.2 continuous batcher reference skeleton

```python
# reference_batcher.py
"""Reference solution skeleton for E68-T4."""

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict
from queue import PriorityQueue
import threading
import time
import uuid


class RequestState(Enum):
    QUEUED = "queued"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    COMPLETED = "completed"


@dataclass
class Request:
    request_id: str
    prompt: str
    input_ids: jnp.ndarray
    max_new_tokens: int = 128
    priority: int = 0
    state: RequestState = RequestState.QUEUED
    generated_tokens: List[int] = field(default_factory=list)
    slot_id: Optional[int] = None
    arrival_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def __lt__(self, other):
        return self.priority < other.priority


class ContinuousBatcher:
    """Reference continuous batching implementation."""

    def __init__(self, model, params, tokenizer, max_batch_size=32, max_seq_len=2048, **kwargs):
        self.model = model
        self.params = params
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # request management
        self.queue = PriorityQueue()
        self.queue_counter = 0
        self.active: Dict[str, Request] = {}
        self.completed_requests: Dict[str, Request] = {}

        # slot management
        self.slots_free = list(range(max_batch_size))
        self.slots_used = {}

        # KV cache - shape depends on model config
        num_layers = kwargs.get('num_layers', 24)
        num_kv_heads = kwargs.get('num_kv_heads', 16)
        head_dim = kwargs.get('head_dim', 64)

        self.kv_cache = jnp.zeros((
            max_batch_size, num_layers, 2, max_seq_len, num_kv_heads, head_dim
        ), dtype=jnp.float16)

        self.seq_lens = jnp.zeros(max_batch_size, dtype=jnp.int32)

        # JIT compile forward passes
        self._prefill_fn = jax.jit(self._prefill_forward)
        self._decode_fn = jax.jit(self._decode_forward)

    def add_request(self, prompt: str, max_new_tokens: int = 128, priority: int = 0) -> str:
        request_id = str(uuid.uuid4())[:8]
        input_ids = jnp.array(self.tokenizer.encode(prompt))

        request = Request(
            request_id=request_id,
            prompt=prompt,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            priority=priority,
        )

        self.queue.put((priority, self.queue_counter, request))
        self.queue_counter += 1
        return request_id

    def _prefill_forward(self, input_ids, attention_mask):
        """Prefill forward pass - returns logits and KV cache."""
        return self.model.apply(
            self.params, input_ids,
            attention_mask=attention_mask,
            return_kv_cache=True,
        )

    def _decode_forward(self, token, kv_cache, slot_idx, seq_len):
        """Decode forward pass - single token with cached KV."""
        return self.model.apply(
            self.params, token,
            kv_cache=kv_cache,
            slot_idx=slot_idx,
            seq_len=seq_len,
            return_kv_cache=True,
        )

    def _schedule_prefill(self):
        """Move requests from queue to prefill."""
        to_prefill = []

        while self.slots_free and not self.queue.empty():
            try:
                _, _, request = self.queue.get_nowait()
            except:
                break

            slot = self.slots_free.pop()
            request.slot_id = slot
            request.state = RequestState.PREFILLING
            self.slots_used[slot] = request.request_id
            self.active[request.request_id] = request
            to_prefill.append(request)

        return to_prefill

    def _run_prefill(self, requests: List[Request]):
        """Run prefill batch."""
        if not requests:
            return

        # batch inputs
        max_len = max(len(r.input_ids) for r in requests)
        batch_size = len(requests)

        input_ids = jnp.zeros((batch_size, max_len), dtype=jnp.int32)
        mask = jnp.zeros((batch_size, max_len), dtype=jnp.bool_)

        for i, req in enumerate(requests):
            seq_len = len(req.input_ids)
            input_ids = input_ids.at[i, :seq_len].set(req.input_ids)
            mask = mask.at[i, :seq_len].set(True)

        # forward
        logits, kv = self._prefill_fn(input_ids, mask)

        # update state
        for i, req in enumerate(requests):
            seq_len = len(req.input_ids)
            # store KV in cache
            self.kv_cache = self.kv_cache.at[req.slot_id, :, :, :seq_len, :, :].set(
                kv[i, :, :, :seq_len, :, :]
            )
            self.seq_lens = self.seq_lens.at[req.slot_id].set(seq_len)

            # get first token
            first_token = int(jnp.argmax(logits[i, seq_len - 1, :]))
            req.generated_tokens.append(first_token)
            req.state = RequestState.DECODING

    def _run_decode_step(self):
        """Run decode step for all active requests."""
        decoding = [r for r in self.active.values() if r.state == RequestState.DECODING]
        if not decoding:
            return

        # batch decode
        batch_size = len(decoding)
        tokens = jnp.array([r.generated_tokens[-1] for r in decoding])[:, None]
        slots = jnp.array([r.slot_id for r in decoding])
        seq_lens = jnp.array([self.seq_lens[r.slot_id] for r in decoding])

        # forward (simplified - real impl would be batched differently)
        for i, req in enumerate(decoding):
            logits, new_kv = self._decode_fn(
                tokens[i:i+1],
                self.kv_cache[req.slot_id:req.slot_id+1],
                req.slot_id,
                int(seq_lens[i]),
            )

            # update KV cache
            pos = int(self.seq_lens[req.slot_id])
            self.kv_cache = self.kv_cache.at[req.slot_id, :, :, pos, :, :].set(
                new_kv[0, :, :, 0, :, :]
            )
            self.seq_lens = self.seq_lens.at[req.slot_id].set(pos + 1)

            # get next token
            next_token = int(jnp.argmax(logits[0, 0, :]))
            req.generated_tokens.append(next_token)

    def _check_completions(self):
        """Check and handle completed requests."""
        for rid in list(self.active.keys()):
            req = self.active[rid]

            # check completion
            done = (
                len(req.generated_tokens) >= req.max_new_tokens or
                (req.generated_tokens and req.generated_tokens[-1] == self.tokenizer.eos_token_id)
            )

            if done:
                req.state = RequestState.COMPLETED
                req.end_time = time.time()

                # release slot
                self.slots_free.append(req.slot_id)
                del self.slots_used[req.slot_id]

                # move to completed
                del self.active[rid]
                self.completed_requests[rid] = req

    def run(self, timeout: float = 60.0):
        """Run scheduler."""
        start = time.time()

        while True:
            if time.time() - start > timeout:
                break

            if self.queue.empty() and not self.active:
                break

            # schedule prefill
            to_prefill = self._schedule_prefill()
            self._run_prefill(to_prefill)

            # decode step
            self._run_decode_step()

            # check completions
            self._check_completions()

    def get_result(self, request_id: str) -> Optional[str]:
        """Get generated text."""
        if request_id in self.completed_requests:
            req = self.completed_requests[request_id]
            return self.tokenizer.decode(req.generated_tokens, skip_special_tokens=True)
        return None
```

---

## 7. grading

### 7.1 E68-T1: INT8 Quantization

| Grade | Speedup | KL Divergence | Token Match | Score |
|-------|---------|---------------|-------------|-------|
| EXCELLENT | ≥2.0x | <0.005 | >98% | 100 |
| PASS | ≥1.5x | <0.01 | >95% | 75 |
| FAIL | <1.5x or | ≥0.01 or | ≤95% | 0-50 |

**Partial credit formula:**
```
score = 50 × min(1, speedup/1.5) × min(1, 0.95/token_match_rate) × min(1, 0.01/kl_div)
```

### 7.2 E68-T4: Continuous Batching

| Grade | Throughput Speedup | p95/p50 Ratio | Score |
|-------|-------------------|---------------|-------|
| EXCELLENT | ≥5x | <1.5 | 100 |
| PASS | ≥2x | <1.5 | 75 |
| FAIL | <2x or | ≥1.5 | 0-50 |

**Partial credit formula:**
```
score = 50 × min(1, speedup/2.0) × min(1, 1.5/p95_p50_ratio)
```

### 7.3 anti-gaming checks

```python
def antigaming_checks_quantization(solution_module, test_inputs, hidden_inputs):
    """Detect cheating in quantization task."""

    checks = {
        'hidden_test_passed': False,
        'no_hardcoded_outputs': True,
        'actually_quantized': False,
    }

    # 1. Test on hidden inputs (different distribution)
    hidden_results = evaluate_on_inputs(solution_module, hidden_inputs)
    checks['hidden_test_passed'] = hidden_results['token_match_rate'] > 0.90

    # 2. Check for hardcoded outputs
    import inspect
    source = inspect.getsource(solution_module.quantize_model)
    if 'TEST_PROMPT' in source or any(f'"{p[:20]}"' in source for p in TEST_PROMPTS):
        checks['no_hardcoded_outputs'] = False

    # 3. Verify actual quantization (check param dtypes)
    _, qparams = solution_module.quantize_model(model, params, tokenizer)
    has_int8 = any(
        'int8' in str(leaf.dtype)
        for leaf in jax.tree_util.tree_leaves(qparams)
        if hasattr(leaf, 'dtype')
    )
    checks['actually_quantized'] = has_int8

    return checks


def antigaming_checks_batching(batcher, diverse_prompts):
    """Detect cheating in batching task."""

    checks = {
        'handles_varying_lengths': False,
        'handles_concurrent_requests': False,
        'no_sequential_processing': False,
    }

    # 1. Test with very different prompt lengths
    short = "Hi"
    long = "Please explain in detail " * 50

    rid1 = batcher.add_request(short)
    rid2 = batcher.add_request(long)
    batcher.run()

    checks['handles_varying_lengths'] = (
        batcher.get_result(rid1) is not None and
        batcher.get_result(rid2) is not None
    )

    # 2. Test concurrent request handling
    # Submit many requests, check they're processed in parallel
    start = time.time()
    rids = [batcher.add_request(p) for p in diverse_prompts[:20]]
    batcher.run()
    elapsed = time.time() - start

    # if sequential, would take ~20x single request time
    single_time = measure_single_request_time(batcher)
    checks['handles_concurrent_requests'] = elapsed < 10 * single_time

    # 3. Check not just sequential (by timing analysis)
    # True batching should have sub-linear scaling
    times_5 = measure_batch_time(batcher, 5)
    times_20 = measure_batch_time(batcher, 20)
    checks['no_sequential_processing'] = times_20 < 3 * times_5  # should scale sub-linearly

    return checks
```

---

## 8. student feedback examples

### 8.1 E68-T1 PASSED

```
═══════════════════════════════════════════════════════════════
                    EVALUATION RESULTS: E68-T1
═══════════════════════════════════════════════════════════════

Status: ✅ PASSED

Score: 75/100
Grade: PASS

Metrics:
  • Speedup: 1.72x (threshold: 1.5x) ✓
  • KL Divergence: 0.0043 (threshold: 0.01) ✓
  • Token Match Rate: 96.8% (threshold: 95%) ✓
  • Original time: 45.3ms
  • Quantized time: 26.3ms

Suggestions for improvement:
  • To reach EXCELLENT: achieve 2x speedup
  • Consider implementing int8 matmul kernels directly
  • Your calibration could be more thorough (try 200+ samples)
  • Per-group quantization might reduce KL further

═══════════════════════════════════════════════════════════════
```

### 8.2 E68-T4 EXCELLENT

```
═══════════════════════════════════════════════════════════════
                    EVALUATION RESULTS: E68-T4
═══════════════════════════════════════════════════════════════

Status: ✅ EXCELLENT

Score: 100/100
Grade: EXCELLENT

Metrics:
  • Throughput: 847.3 tokens/sec (baseline: 156.2)
  • Speedup: 5.42x (threshold: 5x for excellent) ✓
  • Latency p50: 0.234s
  • Latency p95: 0.312s
  • Latency p99: 0.398s
  • p95/p50 ratio: 1.33 (threshold: 1.5) ✓

Excellent implementation! Your continuous batcher:
  • Efficiently manages KV cache slots
  • Handles variable-length sequences well
  • Maintains low latency variance

═══════════════════════════════════════════════════════════════
```

### 8.3 E68-T1 FAILED

```
═══════════════════════════════════════════════════════════════
                    EVALUATION RESULTS: E68-T1
═══════════════════════════════════════════════════════════════

Status: ❌ FAILED

Score: 32/100
Grade: FAIL

Errors:
  ✗ Token match rate 89.2% < 95%
  ✗ KL divergence 0.0234 >= 0.01

Metrics:
  • Speedup: 1.83x ✓
  • KL Divergence: 0.0234 ✗
  • Token Match Rate: 89.2% ✗

Debug hints:
  • Your quantization is too aggressive - quality is degraded
  • Try per-channel quantization instead of per-tensor
  • Increase calibration dataset size
  • Check if you're quantizing embeddings (don't do that)

Hidden test results:
  • Token match on hidden prompts: 84.1% (worse than public)
  • This suggests overfitting to calibration distribution

═══════════════════════════════════════════════════════════════
```

---

## 9. summary: full pipeline visualization

```
┌─────────────────────────────────────────────────────────────┐
│                   STUDENT WORKFLOW: E68                      │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────────┐                     ┌───────────────────┐
│  E68-T1: INT8     │                     │  E68-T4: Cont.    │
│  Quantization     │                     │  Batching         │
└───────────────────┘                     └───────────────────┘
        │                                           │
        ▼                                           ▼
┌───────────────────┐                     ┌───────────────────┐
│ 1. Understand AQT │                     │ 1. Request Queue  │
│    library        │                     │    implementation │
└───────────────────┘                     └───────────────────┘
        │                                           │
        ▼                                           ▼
┌───────────────────┐                     ┌───────────────────┐
│ 2. Configure      │                     │ 2. Prefill engine │
│    per-channel,   │                     │    (batch prompts)│
│    per-token      │                     └───────────────────┘
└───────────────────┘                              │
        │                                           ▼
        ▼                                  ┌───────────────────┐
┌───────────────────┐                     │ 3. Decode engine  │
│ 3. Apply to       │                     │    + slot manager │
│    model layers   │                     └───────────────────┘
└───────────────────┘                              │
        │                                           ▼
        ▼                                  ┌───────────────────┐
┌───────────────────┐                     │ 4. Completion     │
│ 4. Run            │                     │    handling       │
│    calibration    │                     └───────────────────┘
└───────────────────┘                              │
        │                                           ▼
        ▼                                  ┌───────────────────┐
┌───────────────────┐                     │ 5. Measure        │
│ 5. Measure        │                     │    throughput,    │
│    speedup, KL    │                     │    p50/p95/p99    │
└───────────────────┘                     └───────────────────┘
        │                                           │
        └─────────────────────┬─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      SUBMISSION                              │
│  E68-T1: quantize.py with quantize_model()                  │
│  E68-T4: batcher.py with ContinuousBatcher class            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   EVALUATION PIPELINE                        │
│                                                              │
│  1. Validation: function/class exists, signature correct    │
│  2. Correctness: token match >95%, KL <0.01                 │
│  3. Performance: speedup measurement                         │
│  4. Anti-gaming: hidden test prompts, no hardcoding         │
│  5. Grading: FAIL / PASS / EXCELLENT                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      FEEDBACK                                │
│  - Detailed metrics                                          │
│  - Debug hints for failures                                  │
│  - Suggestions for improvement                               │
└─────────────────────────────────────────────────────────────┘
```

---

*end of E68 detailed pipeline specification*
