# pm_env Top 4 Technical Specifications

> comprehensive technical specs for the 5 highest-priority pm_env environments
> created: 2026-02-02 | updated: 2026-02-02 | author: claude opus 4.5

---

## Table of Contents

1. [E01: AJILE12 Multi-Format ECoG Wrangling](#e01-ajile12)
2. [E06: CalMS21 Social Behavior Classification](#e06-calms21)
3. [E05: ML Paper Implementation](#e05-paper-implementation)
4. [E68: Qwen Inference Optimization](#e68-qwen-optimization)
5. [E69: Spiking Neural Network Classification](#e69-snn)

---

# E01: AJILE12 Multi-Format ECoG Wrangling {#e01-ajile12}

## 1. Dataset Overview

| Property | Value |
|----------|-------|
| **Source** | [DANDI Archive #000055](https://dandiarchive.org/dandiset/000055) |
| **Size** | ~150 GB total (12 subjects × 55 days) |
| **Modalities** | ECoG (500 Hz), Pose (30 Hz), Events, Epochs |
| **Format** | NWB (HDF5-based) |
| **Access** | Streaming via DANDI API + remfile |

## 2. Data Schema

### 2.1 NWB File Structure

```
/acquisition/ElectricalSeries/
    data: [n_samples, n_channels] float32  # ECoG at 500 Hz
    rate: 500.0
    electrodes: → /general/extracellular_ephys/electrodes

/processing/behavior/Position/
    SpatialSeries/data: [n_frames, 9, 2] float64  # 9 keypoints at 30 Hz

/intervals/reaches/
    start_time, stop_time, reach_magnitude, reach_angle, ...

/general/extracellular_ephys/electrodes/
    x, y, z (MNI coords), good (bool), lfb_r2, hfb_r2
```

### 2.2 Keypoints (9 upper-body)

| Index | Body Part |
|-------|-----------|
| 0 | nose |
| 1-2 | left/right ear |
| 3-4 | left/right shoulder |
| 5-6 | left/right elbow |
| 7-8 | left/right wrist |

## 3. Task Specification: E01-T1 (Load + Merge)

### 3.1 Student Receives

```python
inputs = {
    "subject_id": "01",
    "session_ids": [1, 2, 3],
    "time_windows": [(0, 3600), (7200, 10800)],  # seconds
    "target_keypoints": ["left_wrist", "right_wrist"],
}
```

### 3.2 Student Must Produce

```python
def load_and_merge(subject_id, session_ids, time_windows, target_keypoints, dandi_client) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
    - timestamp: float64 (common timebase, 30 Hz)
    - ecog_{channel_idx}: float32 (downsampled to 30 Hz)
    - pose_{keypoint}_x, pose_{keypoint}_y: float64
    - is_movement: bool
    - behavior_state: str
    """
```

### 3.3 Evaluation Metrics

| Metric | Weight | Pass | Excellent |
|--------|--------|------|-----------|
| Alignment accuracy | 40% | ≥95% within 33ms | ≥99% within 10ms |
| Schema correctness | 20% | all columns present | + correct dtypes |
| Bad electrode handling | 15% | excludes flagged | + validates independently |
| Memory efficiency | 15% | <4 GB peak | <2 GB peak |
| Processing time | 10% | <60s/session | <20s/session |

### 3.4 Alignment Accuracy Formula

```python
def compute_alignment_accuracy(submitted, reference, tolerance_ms=33.0):
    submitted_ts = submitted["timestamp"].values
    reference_ts = reference["timestamp"].values
    diffs = np.abs(submitted_ts[:, None] - reference_ts[None, :])
    min_diffs = diffs.min(axis=0)
    return (min_diffs * 1000 <= tolerance_ms).mean()
```

## 4. Task Specification: E01-T4 (Streaming Loader)

### 4.1 Student Must Implement

```python
class StreamingLoader:
    def __init__(self, dandi_client, cache_dir, memory_limit_gb=2.0): ...

    def stream_chunks(self, subject_id, session_id, start_time, end_time,
                      chunk_duration, include_pose=True) -> Iterator[dict]:
        """Yields: {"ecog": array, "pose": array, "timestamps_ecog": array, ...}"""

    def prefetch(self, requests: list[tuple]): ...
    def get_memory_usage(self) -> float: ...
```

### 4.2 Evaluation Metrics

| Metric | Weight | Pass | Excellent |
|--------|--------|------|-----------|
| Throughput (sequential) | 30% | ≥500 samples/s | ≥2000 samples/s |
| Throughput (random) | 20% | ≥100 samples/s | ≥500 samples/s |
| Memory compliance | 25% | never exceeds limit | peak <80% limit |
| Data integrity | 15% | 100% correct | + checksum verified |
| Cache efficiency | 10% | hit rate ≥50% | hit rate ≥80% |

## 5. Baseline Scores

| Approach | Alignment | Memory | Time/session |
|----------|-----------|--------|--------------|
| Naive (load all) | 98% | 50+ GB | OOM |
| Chunk + resample | 95% | 4 GB | 120s |
| Streaming + interp | 97% | 2 GB | 45s |
| **Optimal (cached)** | **99.5%** | **1.5 GB** | **15s** |

## 6. Anti-Gaming

1. **Session holdout**: train on sessions 1-3, test on 4+
2. **Subject holdout**: develop on 01-10, test on 11-12
3. **Adversarial cases**: clock drift (499.8 Hz), missing keypoints, electrode inconsistencies
4. **Overfitting detection**: flag if train-test gap >20%

## 7. Key Papers

1. Singh et al. 2021 – RF classifier, spectral features (LFB 8-32 Hz, HFB 76-100 Hz)
2. Peterson et al. 2021 – Multiple regression, R² ≈ 0.25 max
3. Wang et al. 2018 – CNN+LSTM for 800ms movement prediction

---

# E06: CalMS21 Social Behavior Classification {#e06-calms21}

## 1. Dataset Overview

| Property | Value |
|----------|-------|
| **Source** | [Caltech Data](https://data.caltech.edu/records/s0vdx-0k302) |
| **Size** | ~5 GB |
| **Task** | Frame-level behavior classification |
| **Classes** | attack (2.8%), investigation (29%), mount (5.6%), other (63%) |
| **Format** | .npy (numpy pickled dict) |

## 2. Data Schema

### 2.1 Keypoints (7 per mouse, top-view)

| Index | Body Part |
|-------|-----------|
| 0 | nose |
| 1-2 | left/right ear |
| 3 | neck |
| 4-5 | left/right hip |
| 6 | tail base |

### 2.2 Data Structure

```python
{
    "keypoints": np.ndarray,   # [n_frames, 2, 2, 7] = [frame, mouse, xy, keypoint]
    "scores": np.ndarray,       # [n_frames, 2, 7] confidence
    "annotations": np.ndarray,  # [n_frames] class labels (0-3)
    "metadata": {"vocab": {"attack": 0, "investigation": 1, "mount": 2, "other": 3}}
}
```

## 3. Evaluation

### 3.1 Primary Metric: Macro F1 (excluding "other")

```python
from sklearn.metrics import f1_score

mask = labels != 3  # exclude "other"
f1 = f1_score(labels[mask], predictions[mask], average="macro", labels=[0, 1, 2])
```

### 3.2 Grading Thresholds

| Grade | Macro F1 | Interpretation |
|-------|----------|----------------|
| Fail | <0.50 | Below random |
| Pass | 0.50-0.70 | Basic features work |
| Good | 0.70-0.79 | Competitive with baseline |
| Excellent | 0.79-0.86 | Near SOTA |
| Outstanding | >0.86 | Challenge-winning |

### 3.3 Benchmark Scores

| Method | Macro F1 |
|--------|----------|
| Baseline (neural) | 0.793 |
| + Task Programming | 0.829 |
| **MABe 2021 Winner** | **0.864** |

## 4. Essential Features

### 4.1 Inter-Animal Distance

```python
def compute_interanimal_distance(keypoints):
    centroids = keypoints.mean(axis=3)  # [n_frames, 2, 2]
    diff = centroids[:, 0, :] - centroids[:, 1, :]
    return np.sqrt((diff ** 2).sum(axis=1))
```

### 4.2 Velocity

```python
def compute_velocity(keypoints, dt=1/30):
    centroids = keypoints.mean(axis=3)
    velocity = np.diff(centroids, axis=0) / dt
    speed = np.sqrt((velocity ** 2).sum(axis=2))
    return speed
```

### 4.3 Pairwise Distances (winning solution)

```python
def compute_pairwise_distances(keypoints):
    n_frames = keypoints.shape[0]
    all_kps = keypoints.transpose(0, 1, 3, 2).reshape(n_frames, 14, 2)
    diff = all_kps[:, :, np.newaxis, :] - all_kps[:, np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    triu_idx = np.triu_indices(14, k=1)
    return dists[:, triu_idx[0], triu_idx[1]]  # 91 features
```

## 5. Anti-Gaming

1. **Sequence-level splits**: no frame leakage
2. **Identity-based split**: different animals in train vs test
3. **Random baseline**: ~0.33 F1 (sanity check)

## 6. Key Papers

1. Sun et al. 2021 – CalMS21 challenge paper
2. Segalin et al. 2020 – MARS pipeline

---

# E05: ML Paper Implementation {#e05-paper-implementation}

## 1. Overview

Students implement algorithms from recent ML papers in JAX, validated against reference implementations.

| Paper | Task | Primary Metric |
|-------|------|----------------|
| Flash Attention 2 | Port to JAX/Pallas | Speedup + correctness |
| Speculative Decoding | Implement for Qwen | Acceptance rate + speedup |
| GQA | Add to MaxText | KV cache reduction |
| RoPE | Implement variants | Position encoding correctness |
| LoRA | JAX implementation | Fine-tuning efficiency |

## 2. Flash Attention 2 (arXiv:2307.08691)

### 2.1 Function Signature

```python
def flash_attention_forward(
    q: jax.Array,          # [batch, seq_len, num_heads, head_dim]
    k: jax.Array,          # [batch, kv_len, num_kv_heads, head_dim]
    v: jax.Array,
    mask: Optional[jax.Array] = None,
    block_q: int = 128,
    block_kv: int = 64,
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
    """Returns: (output, (logsumexp, max_scores))"""
```

### 2.2 Correctness Tests

```python
# Test 1: numerical equivalence
def test_flash_vs_standard():
    q, k, v = random_qkv(batch=2, seq=512, heads=8, dim=64)
    ref = standard_attention(q, k, v)
    out, _ = flash_attention_forward(q, k, v)
    assert jnp.allclose(out, ref, atol=1e-5, rtol=1e-5)

# Test 2: gradient check
def test_gradients():
    # finite difference vs analytical gradients
    assert gradient_check_passes(flash_attention_forward)

# Test 3: causal masking
def test_causal():
    # position i attends only to j <= i
```

### 2.3 Benchmarks

| Seq Length | Naive Memory | Flash Memory | Target Speedup |
|------------|--------------|--------------|----------------|
| 512 | ~1 MB | ~64 KB | >2x |
| 2048 | ~16 MB | ~256 KB | >5x |
| 4096 | ~64 MB | ~512 KB | >5x |

### 2.4 Grading

| Level | Criteria |
|-------|----------|
| Pass | Correctness within atol=1e-4, any speedup |
| Good | Correctness within atol=1e-5, >2x speedup |
| Excellent | + backward pass, >5x speedup on 4096 |

## 3. Speculative Decoding (arXiv:2302.01318)

### 3.1 Algorithm

```python
def speculative_decode(prompt, draft_model, target_model, gamma=5):
    """
    1. Draft model generates γ tokens autoregressively
    2. Target model verifies all γ in ONE forward pass
    3. Rejection sampling: accept if p_target/p_draft > random()
    4. If rejected: resample from max(0, p_target - p_draft)
    """
```

### 3.2 Function Signature

```python
def speculative_decode(
    prompt_tokens: jax.Array,
    draft_model_fn: Callable,
    target_model_fn: Callable,
    gamma: int = 5,
    max_new_tokens: int = 100,
) -> Tuple[jax.Array, Dict[str, float]]:
    """
    Returns:
        generated_tokens: [max_new_tokens]
        stats: {'acceptance_rate': float, 'tokens_per_target_call': float}
    """
```

### 3.3 Model Pairings

| Draft | Target | Expected Acceptance | Expected Speedup |
|-------|--------|---------------------|------------------|
| Qwen-0.6B | Qwen-1.7B | 70-80% | 1.8-2.2x |
| Qwen-0.6B | Qwen-4B | 60-70% | 2.0-2.5x |

### 3.4 Correctness Tests

```python
# CRITICAL: output distribution must match target-only sampling
def test_distribution_equivalence():
    spec_samples = [speculative_decode(prompt, draft, target, seed=i) for i in range(1000)]
    target_samples = [autoregressive_decode(prompt, target, seed=i) for i in range(1000)]
    kl_div = kl_divergence(empirical_dist(spec_samples), empirical_dist(target_samples))
    assert kl_div < 0.01
```

### 3.5 Grading

| Level | Criteria |
|-------|----------|
| Pass | KL < 0.05, acceptance >50%, speedup >1.3x |
| Good | KL < 0.01, acceptance >60%, speedup >1.8x |
| Excellent | KL < 0.005, acceptance >70%, speedup >2.5x |

## 4. GQA (arXiv:2305.13245)

### 4.1 Core Concept

```python
# MHA: num_kv_heads == num_heads (e.g., 32)
# GQA: num_kv_heads < num_heads (e.g., 8)
# Each KV head serves (num_heads // num_kv_heads) query heads
```

### 4.2 Function Signature

```python
def grouped_query_attention(
    query: jax.Array,     # [batch, seq_q, num_heads, head_dim]
    key: jax.Array,       # [batch, seq_kv, num_kv_heads, head_dim]
    value: jax.Array,
    num_kv_heads: int,
) -> jax.Array:
```

### 4.3 Tests

```python
# GQA with G=H must equal MHA
def test_gqa_equals_mha():
    out_gqa = grouped_query_attention(q, k, v, num_kv_heads=8)
    out_mha = standard_attention(q, k, v)
    assert jnp.allclose(out_gqa, out_mha)

# GQA with G=1 must equal MQA
def test_gqa_equals_mqa(): ...
```

## 5. RoPE (arXiv:2104.09864)

### 5.1 Core Formula

```
θ_i = base^(-2i/d), where base=10000
For position m, rotate pair (x_{2i}, x_{2i+1}):
  x'_{2i}   = x_{2i}*cos(m*θ_i) - x_{2i+1}*sin(m*θ_i)
  x'_{2i+1} = x_{2i}*sin(m*θ_i) + x_{2i+1}*cos(m*θ_i)
```

### 5.2 Key Test: Relative Position Property

```python
def test_rope_relative():
    # q @ k^T at positions (m, n) depends only on (m - n)
    q_rot_10 = apply_rope(q, cos[10], sin[10])
    k_rot_5 = apply_rope(k, cos[5], sin[5])
    dot_10_5 = q_rot_10 @ k_rot_5.T

    q_rot_20 = apply_rope(q, cos[20], sin[20])
    k_rot_15 = apply_rope(k, cos[15], sin[15])
    dot_20_15 = q_rot_20 @ k_rot_15.T

    assert jnp.allclose(dot_10_5, dot_20_15)  # same relative distance = same dot product
```

## 6. LoRA (arXiv:2106.09685)

### 6.1 Core Concept

```python
# Frozen: W (original weights)
# Trainable: A [rank, in_features], B [out_features, rank]
# Forward: h = Wx + (α/r) * B @ A @ x
# Init: A ~ N(0, 1/r), B = 0 (starts as identity)
```

### 6.2 Key Test: Zero Init = Identity

```python
def test_lora_initial_identity():
    # B=0 means LoRA contributes nothing initially
    params = init_lora_params(key, 768, 768, rank=8)  # B=0
    out_lora = lora_layer(x, params)
    out_base = x @ params['W'].T
    assert jnp.allclose(out_lora, out_base)
```

---

# E68: Qwen Inference Optimization {#e68-qwen-optimization}

## 1. Overview

Students optimize Qwen model inference on TPU v5e using various techniques.

| Technique | Memory Impact | Speed Impact |
|-----------|---------------|--------------|
| INT8 quantization | 2x reduction | 1.5-2x faster |
| FP8 KV cache | 2x KV reduction | minimal overhead |
| Speculative decoding | +15% (draft cache) | 2-3x faster |
| Continuous batching | efficient utilization | 2-5x throughput |
| Pallas kernels | better memory | 1.5-2x faster |

## 2. Qwen Model Specs

### 2.1 Qwen3-0.6B

| Param | Value |
|-------|-------|
| hidden_size | 1024 |
| num_layers | 28 |
| num_attention_heads | 16 |
| num_kv_heads | 8 (GQA) |
| head_dim | 128 |

### 2.2 Qwen3-4B

| Param | Value |
|-------|-------|
| hidden_size | 2560 |
| num_layers | 36 |
| num_attention_heads | 32 |
| num_kv_heads | 8 (GQA) |
| head_dim | 128 |

## 3. Task E68-T1: INT8 Quantization

### 3.1 Implementation

```python
from aqt.jax.v2 import config as aqt_config

def get_int8_config():
    return aqt_config.DotGeneral(
        fwd=aqt_config.DotGeneralRaw.make(
            lhs=aqt_config.TensorConfig(num_bits=8, calib_shared_axes='per_token'),
            rhs=aqt_config.TensorConfig(num_bits=8, calib_shared_axes='per_channel'),
        ),
    )
```

### 3.2 Correctness Thresholds

| Metric | Acceptable | Unacceptable |
|--------|------------|--------------|
| Perplexity delta | <1% | >5% |
| KL divergence | <0.01 | >0.05 |
| Token flip rate | <2% | >5% |

### 3.3 Grading

| Level | Speedup | KL Divergence |
|-------|---------|---------------|
| Pass | 1.5x | <0.01 |
| Excellent | 2x | <0.005 |

## 4. Task E68-T2: KV Cache Compression

### 4.1 FP8 Implementation

```python
def quantize_kv_cache(k, v, dtype=jnp.float8_e4m3fn):
    k_scale = jnp.max(jnp.abs(k), axis=-1, keepdims=True) / 448.0
    v_scale = jnp.max(jnp.abs(v), axis=-2, keepdims=True) / 448.0
    k_q = (k / k_scale).astype(dtype)
    v_q = (v / v_scale).astype(dtype)
    return k_q, v_q, k_scale, v_scale
```

### 4.2 KV Cache Sizes

| Model | 32K Context (bf16) | 32K Context (fp8) |
|-------|-------------------|-------------------|
| Qwen3-0.6B | 0.9 GB | 0.45 GB |
| Qwen3-4B | 2.3 GB | 1.15 GB |

### 4.3 Grading

| Level | Memory Reduction | Perplexity Delta |
|-------|------------------|------------------|
| Pass | 1.5x | <1% |
| Excellent | 2x | <0.5% |

## 5. Task E68-T3: Speculative Decoding

(See E05 Speculative Decoding section for details)

### 5.1 Grading (Qwen-specific)

| Level | Acceptance Rate | Speedup |
|-------|-----------------|---------|
| Pass | 60% | 1.5x |
| Excellent | 75% | 2.5x |

## 6. Task E68-T4: Continuous Batching

### 6.1 Architecture

```
prefill thread → transfer thread → generate thread
(process new)    (copy KV cache)   (decode loop)
```

### 6.2 Implementation Skeleton

```python
class ContinuousBatcher:
    def __init__(self, model_fn, max_batch_size=64, max_seq_len=4096): ...
    def add_request(self, request: InferenceRequest): ...
    def step(self) -> list[tuple[str, int]]:  # (request_id, token)
```

### 6.3 Metrics

| Metric | Formula |
|--------|---------|
| Throughput | total_tokens / total_time |
| TTFT | time_first_token - arrival_time |
| ITL | time_between_consecutive_tokens |
| p50/p95/p99 | latency percentiles |

### 6.4 Grading

| Level | Throughput | p95 < X × p50 |
|-------|------------|---------------|
| Pass | 2x baseline | <2x |
| Excellent | 5x baseline | <1.5x |

## 7. Task E68-T5: Pallas Flash Attention

### 7.1 Implementation

```python
from jax.experimental.pallas.ops.tpu import flash_attention

def tpu_flash_attention(q, k, v, causal=True, block_q=128, block_kv=128):
    return flash_attention.flash_attention(
        q, k, v, causal=causal, block_q=block_q, block_kv=block_kv
    )
```

### 7.2 Grading

| Level | Speedup (seq=2048) | Memory Reduction |
|-------|-------------------|------------------|
| Pass | 1.5x | any |
| Excellent | 2x | 3x |

## 8. Overall Evaluation Framework

### 8.1 Correctness Metrics

```python
def compute_divergence_metrics(reference_outputs, optimized_outputs):
    # Token match rate
    exact_matches = sum(ref == opt for ref, opt in zip(reference_outputs, optimized_outputs))
    token_match_rate = exact_matches / len(reference_outputs)

    # Share of divergent tokens
    total_tokens = sum(len(ref) for ref in reference_outputs)
    divergent = sum(sum(r != o for r, o in zip(ref, opt))
                   for ref, opt in zip(reference_outputs, optimized_outputs))
    sdt = divergent / total_tokens

    return {'token_match_rate': token_match_rate, 'share_divergent_tokens': sdt}
```

### 8.2 Acceptance Thresholds

| Metric | Acceptable | Fail |
|--------|------------|------|
| Token match rate | >95% | <90% |
| Share divergent tokens | <5% | >10% |
| Mean KL divergence | <0.01 | >0.05 |

### 8.3 Test Inputs

```python
TEST_INPUTS = {
    'short': [  # 32 tokens input
        {'prompt': 'The capital of France is', 'max_tokens': 32},
        {'prompt': 'def fibonacci(n):', 'max_tokens': 64},
    ],
    'medium': [  # 512 tokens input
        {'prompt': WIKIPEDIA_PASSAGE, 'max_tokens': 256},
    ],
    'long': [  # 2048 tokens input
        {'prompt': CODE_FILE, 'max_tokens': 512},
    ],
}
BATCH_SIZES = [1, 4, 16]
```

---

## Summary: Grading Overview

| Environment | Pass Threshold | Excellent Threshold |
|-------------|----------------|---------------------|
| **E01-T1** | 95% alignment, <4GB, <60s | 99% alignment, <2GB, <20s |
| **E01-T4** | 500 samp/s, <limit | 2000 samp/s, 80% cache hit |
| **E06** | F1 ≥ 0.50 | F1 ≥ 0.79 |
| **E05 Flash** | atol=1e-4, >1x speedup | atol=1e-5, >5x speedup |
| **E05 Spec** | KL<0.05, >1.3x speedup | KL<0.005, >2.5x speedup |
| **E68-T1** | 1.5x speedup, KL<0.01 | 2x speedup, KL<0.005 |
| **E68-T2** | 1.5x memory, <1% ppl | 2x memory, <0.5% ppl |
| **E68-T3** | 60% accept, 1.5x speed | 75% accept, 2.5x speed |
| **E68-T4** | 2x throughput | 5x throughput, p95<1.5×p50 |
| **E69-T1** | LIF unit tests pass | + numerically stable |
| **E69-T2** | SHD acc ≥ 50% | SHD acc ≥ 78% |
| **E69-T3** | SNN within 5% of RNN | SNN beats RNN by >2% |
| **E69-T4** | >80% ANN acc at T=16 | >95% ANN acc at T=16 |

---

## Key References

### E01 (AJILE12)
- Peterson et al. 2022 – AJILE12 dataset paper
- Singh et al. 2021 – Mining naturalistic behaviors
- [DANDI Archive #000055](https://dandiarchive.org/dandiset/000055)

### E06 (CalMS21)
- Sun et al. 2021 – CalMS21 challenge paper
- Segalin et al. 2020 – MARS pipeline
- [Caltech Data Repository](https://data.caltech.edu/records/s0vdx-0k302)

### E05 (Paper Implementation)
- [Flash Attention 2](https://arxiv.org/abs/2307.08691)
- [Speculative Decoding](https://arxiv.org/abs/2302.01318)
- [GQA](https://arxiv.org/abs/2305.13245)
- [RoPE](https://arxiv.org/abs/2104.09864)
- [LoRA](https://arxiv.org/abs/2106.09685)

### E68 (Qwen Optimization)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [AQT Library](https://github.com/google/aqt)
- [JetStream](https://github.com/AI-Hypercomputer/JetStream)
- [JAX Pallas Flash Attention](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py)

---

# E69: Spiking Neural Network Classification {#e69-snn}

## 1. Overview

Students implement and train Spiking Neural Networks (SNNs) – the "third generation" of neural networks that communicate via discrete spikes rather than continuous activations.

| Property | Value |
|----------|-------|
| **Framework** | Spyx (JAX-native) |
| **Dataset** | SHD (Spiking Heidelberg Digits) |
| **Task** | 20-class audio classification |
| **Why SNN?** | Truly temporal data where SNNs outperform RNN/LSTM |

### 1.1 Why SNNs for pm_env?

| aspect | ANN | SNN |
|--------|-----|-----|
| Activation | continuous (ReLU) | discrete spikes (0/1) |
| Computation | static, per-layer | temporal, event-driven |
| Energy | always-on MAC ops | sparse ACC ops when spikes |
| Biological plausibility | low | high |

**Key insight**: On truly temporal tasks (SHD, DVS-Gesture), SNNs **beat** equivalent ANNs.

## 2. Dataset: Spiking Heidelberg Digits (SHD)

### 2.1 Overview

| Property | Value |
|----------|-------|
| **Source** | [Zenodo](https://zenodo.org/record/4319560) |
| **Modality** | Audio → cochlea spike trains |
| **Classes** | 20 (spoken digits 0-9 in English + German) |
| **Train/Test** | 8,332 / 2,088 samples |
| **Channels** | 700 (cochlea frequency bands) |
| **Duration** | ~1 second per sample |
| **Temporal bins** | 100-250 (configurable) |

### 2.2 Data Format

```python
# Each sample: sparse spike tensor
{
    "times": np.ndarray,     # spike timestamps (ms)
    "channels": np.ndarray,  # which of 700 channels spiked
    "label": int,            # 0-19
}

# After binning: dense tensor
spikes: np.ndarray  # [time_bins, 700] binary
```

### 2.3 Why SHD (not N-MNIST)?

| Dataset | Truly Temporal? | SNN Advantage? |
|---------|-----------------|----------------|
| N-MNIST | No (scanned static images) | No – ANN gets 99.2% |
| CIFAR10-DVS | No (scanned static images) | Marginal |
| **SHD** | **Yes** (real audio) | **Yes** – temporal patterns matter |
| DVS-Gesture | Yes (real gestures) | Yes |

N-MNIST is NOT a valid SNN benchmark – ANNs match SNNs by ignoring time dimension.

## 3. Neuron Model: Leaky Integrate-and-Fire (LIF)

### 3.1 Dynamics

```
τ_m * dV/dt = -(V - V_rest) + R * I(t)

if V > V_threshold:
    emit spike (S = 1)
    V → V_reset
else:
    S = 0
```

### 3.2 Discrete Implementation

```python
def lif_step(V, I, tau_mem=10.0, V_th=1.0, V_reset=0.0, dt=1.0):
    """Single LIF neuron timestep"""
    # Membrane decay
    alpha = jnp.exp(-dt / tau_mem)
    V_new = alpha * V + (1 - alpha) * I

    # Spike generation
    spike = (V_new > V_th).astype(jnp.float32)

    # Reset after spike
    V_new = jnp.where(spike, V_reset, V_new)

    return V_new, spike
```

### 3.3 Surrogate Gradient (for backprop)

Spikes are non-differentiable. Use surrogate:

```python
def surrogate_gradient(V, V_th, beta=10.0):
    """SuperSpike surrogate: smooth approximation to Heaviside"""
    return 1.0 / (1.0 + beta * jnp.abs(V - V_th)) ** 2
```

## 4. Framework: Spyx

### 4.1 Why Spyx?

| Framework | JAX? | TPU? | Speed |
|-----------|------|------|-------|
| snnTorch | No (PyTorch) | No | baseline |
| SpikingJelly | No (PyTorch+CUDA) | No | fastest (custom CUDA) |
| **Spyx** | **Yes** | **Yes** | within 5% of SpikingJelly |
| Norse | No (PyTorch) | No | slow |

Spyx trains SHD in **35 seconds per 100 epochs** (batch=256).

### 4.2 Basic Architecture

```python
import spyx
import spyx.nn as snn

def create_snn(key, input_size=700, hidden_size=256, output_size=20):
    """Simple 2-layer SNN for SHD"""

    @hk.transform
    def network(spikes):
        # Input → Hidden (LIF)
        x = hk.Linear(hidden_size)(spikes)
        x, state1 = snn.LIF(hidden_size)(x)

        # Hidden → Hidden (LIF)
        x = hk.Linear(hidden_size)(x)
        x, state2 = snn.LIF(hidden_size)(x)

        # Hidden → Output (LI - no spike, just integrate)
        x = hk.Linear(output_size)(x)
        x, state3 = snn.LI(output_size)(x)

        return x  # membrane potentials for classification

    return network
```

### 4.3 Training Loop

```python
import optax
from spyx.data import SHD_loader

# Data
train_dl, test_dl = SHD_loader(batch_size=256, time_bins=100)

# Model
net = create_snn(key)
params = net.init(key, jnp.zeros((256, 100, 700)))

# Optimizer
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

# Loss: cross-entropy on final membrane potentials
def loss_fn(params, batch):
    spikes, labels = batch
    output = net.apply(params, spikes)
    logits = output[:, -1, :]  # last timestep
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

# Training step
@jax.jit
def train_step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
```

## 5. Task Variants

### 5.1 E69-T1: Implement LIF Neuron (Easy)

**Objective**: Implement LIF neuron dynamics from scratch in JAX.

**Student Receives**:
```python
def lif_step(V, I, params):
    """
    Args:
        V: membrane potential [batch, neurons]
        I: input current [batch, neurons]
        params: dict with tau_mem, V_th, V_reset
    Returns:
        V_new: updated membrane potential
        spike: binary spike output
    """
    pass  # TODO: implement
```

**Evaluation**:
```python
# Unit tests
def test_lif_decay():
    """V should decay towards V_rest without input"""
    V = jnp.ones((1, 10))
    V_new, _ = lif_step(V, jnp.zeros((1, 10)), params)
    assert jnp.all(V_new < V)

def test_lif_spike():
    """V > threshold should produce spike and reset"""
    V = jnp.ones((1, 10)) * 1.5  # above threshold
    V_new, spike = lif_step(V, jnp.zeros((1, 10)), params)
    assert jnp.all(spike == 1)
    assert jnp.allclose(V_new, params['V_reset'])

def test_lif_integration():
    """Constant input should increase V"""
    V = jnp.zeros((1, 10))
    V_new, _ = lif_step(V, jnp.ones((1, 10)), params)
    assert jnp.all(V_new > V)
```

**Grading**:
| Level | Criteria |
|-------|----------|
| Pass | All unit tests pass |
| Excellent | + numerically stable, + vectorized |

### 5.2 E69-T2: Train SNN on SHD (Medium)

**Objective**: Train an SNN to classify spoken digits.

**Student Receives**:
- Data loader (Spyx SHD_loader)
- LIF neuron implementation
- Training loop scaffold

**Metrics**:

| Metric | Weight | Description |
|--------|--------|-------------|
| Test accuracy | 60% | Primary metric |
| Synaptic ops | 20% | Energy efficiency proxy |
| Training stability | 20% | Loss curve smoothness |

**Baselines**:

| Method | Test Accuracy |
|--------|---------------|
| Random | 5% |
| Original paper (2019) | 72% |
| Spyx baseline | 78% |
| SOTA | 85%+ |

**Grading**:

| Level | Test Accuracy |
|-------|---------------|
| Fail | <50% |
| Pass | 50-70% |
| Good | 70-78% |
| Excellent | 78-85% |
| Outstanding | >85% |

### 5.3 E69-T3: Beat RNN Baseline (Hard)

**Objective**: Prove SNN advantage on temporal data.

**Task**: Train both:
1. LSTM/GRU on SHD (rate-coded input)
2. SNN on SHD (spike input)

Show SNN achieves higher accuracy OR lower energy (SynOps).

**Evaluation**:
```python
def compute_synops(model, test_data):
    """Count total synaptic operations"""
    total_synops = 0
    for batch in test_data:
        spikes = model.get_all_spikes(batch)
        for layer_spikes, layer_weights in zip(spikes, model.weights):
            # SynOps = spike_count × fan_out
            total_synops += jnp.sum(layer_spikes) * layer_weights.shape[1]
    return total_synops / len(test_data)

# Student must show:
# (SNN_accuracy > RNN_accuracy) OR (SNN_synops < RNN_MACs × 0.5)
```

**Grading**:

| Level | Criteria |
|-------|----------|
| Pass | SNN within 5% of RNN accuracy |
| Good | SNN matches RNN accuracy |
| Excellent | SNN beats RNN by >2% OR <50% SynOps |

### 5.4 E69-T4: ANN-to-SNN Conversion (Hard)

**Objective**: Convert trained ANN to SNN with minimal accuracy loss.

**Task**:
1. Train ANN (MLP) on rate-coded SHD
2. Convert to SNN by replacing ReLU with LIF
3. Calibrate thresholds
4. Measure accuracy at different timesteps T

**Target**: <5% accuracy drop at T=16 timesteps.

**Grading**:

| Level | Accuracy Retention at T=16 |
|-------|---------------------------|
| Pass | >80% of ANN |
| Good | >90% of ANN |
| Excellent | >95% of ANN |

## 6. Energy Efficiency Metrics

### 6.1 Synaptic Operations (SynOps)

```python
def count_synops(spikes_per_layer, weights_per_layer):
    """
    SynOps = sum over layers of (spike_count × fan_out)

    Lower is better – means fewer computations needed.
    """
    total = 0
    for spikes, weights in zip(spikes_per_layer, weights_per_layer):
        fan_out = weights.shape[1]
        total += jnp.sum(spikes) * fan_out
    return total
```

### 6.2 Energy Estimation

```python
# Approximate energy (45nm process)
E_MAC = 4.6  # pJ per multiply-accumulate (ANN)
E_AC = 0.9   # pJ per accumulate (SNN spike)

def estimate_energy(synops_or_macs, is_snn=True):
    if is_snn:
        return synops_or_macs * E_AC
    else:
        return synops_or_macs * E_MAC
```

### 6.3 Comparison Framework

```python
def compare_efficiency(snn_model, ann_model, test_data):
    """Compare SNN vs ANN efficiency"""

    # Accuracy
    snn_acc = evaluate_accuracy(snn_model, test_data)
    ann_acc = evaluate_accuracy(ann_model, test_data)

    # Compute ops
    snn_synops = count_synops(snn_model, test_data)
    ann_macs = count_macs(ann_model, test_data)

    # Energy
    snn_energy = snn_synops * E_AC
    ann_energy = ann_macs * E_MAC

    return {
        'snn_accuracy': snn_acc,
        'ann_accuracy': ann_acc,
        'accuracy_gap': snn_acc - ann_acc,
        'snn_synops': snn_synops,
        'ann_macs': ann_macs,
        'energy_ratio': snn_energy / ann_energy,
    }
```

## 7. Anti-Gaming

### 7.1 Temporal Validity

```python
def check_temporal_usage(model, test_data):
    """Verify model actually uses temporal information"""

    # Shuffle time dimension
    shuffled_data = shuffle_time_axis(test_data)

    acc_original = evaluate(model, test_data)
    acc_shuffled = evaluate(model, shuffled_data)

    # If temporal shuffling doesn't hurt, model isn't using time
    assert acc_original - acc_shuffled > 0.1, "Model ignores temporal structure!"
```

### 7.2 Spike Sparsity

```python
def check_spike_sparsity(model, test_data, max_rate=0.5):
    """Ensure model isn't just always spiking (trivial solution)"""

    total_spikes = 0
    total_possible = 0

    for batch in test_data:
        spikes = model.get_all_spikes(batch)
        total_spikes += jnp.sum(spikes)
        total_possible += spikes.size

    spike_rate = total_spikes / total_possible
    assert spike_rate < max_rate, f"Spike rate {spike_rate:.2%} too high!"
```

### 7.3 Hidden Test Set

- Test on held-out speakers (not in training)
- Test on noise-augmented samples
- Test on time-stretched samples

## 8. Implementation Roadmap

### 8.1 Infrastructure

```yaml
container:
  base_image: python:3.11-slim
  memory_limit: 8GB
  cpu_limit: 4 cores

dependencies:
  - jax>=0.4.20
  - spyx>=0.1.0
  - optax>=0.1.7
  - haiku>=0.0.10
```

### 8.2 Starter Code Structure

```
pm_env/e69_snn/
├── data/
│   └── shd_loader.py        # SHD data loading
├── models/
│   ├── lif.py               # LIF neuron implementation
│   ├── snn.py               # SNN architectures
│   └── baseline_rnn.py      # RNN baseline for comparison
├── training/
│   ├── surrogate.py         # Surrogate gradients
│   └── train.py             # Training loop
├── evaluation/
│   ├── metrics.py           # Accuracy, SynOps
│   └── compare.py           # SNN vs ANN comparison
└── tests/
    ├── test_lif.py          # LIF unit tests
    └── test_snn.py          # SNN integration tests
```

## 9. Key References

### SNN Fundamentals
- [Surrogate Gradient Learning Review](https://arxiv.org/abs/1901.09948)
- [snnTorch Tutorials](https://snntorch.readthedocs.io/en/latest/tutorials/index.html)
- [NMA LIF Tutorial](https://compneuro.neuromatch.io/tutorials/W2D3_BiologicalNeuronModels/student/W2D3_Tutorial1.html)

### Frameworks
- [Spyx GitHub](https://github.com/kmheckel/spyx)
- [Spyx Paper](https://arxiv.org/abs/2402.18994)
- [Framework Benchmarks](https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/)

### Datasets
- [SHD Paper (Heidelberg)](https://arxiv.org/abs/1910.07407)
- [SHD Download](https://zenodo.org/record/4319560)
- [Is N-MNIST a Valid Benchmark?](https://pmc.ncbi.nlm.nih.gov/articles/PMC8027306/)

### SOTA Results
- [STAA-SNN (2025)](https://arxiv.org/abs/2503.02689)
- [SNN Adversarial Robustness](https://www.nature.com/articles/s41467-025-65197-x)

---

*end of specification*
