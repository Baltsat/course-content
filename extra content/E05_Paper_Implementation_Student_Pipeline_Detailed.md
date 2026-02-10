# E05: Paper Implementation – Detailed Student Pipeline

> full specification: student path from start to evaluation
> focus: flash attention 2 (arXiv:2307.08691) + speculative decoding (arXiv:2302.01318)

---

## 1. what student sees at start

### 1.1 task description (interface)

```
TASK: ML Paper Implementation

You will implement algorithms from recent ML papers in JAX.
Each paper comes with:
- Algorithm pseudocode from the paper
- Function signatures to implement
- Test cases for validation

Choose ONE paper to implement:

[A] Flash Attention 2 (Dao 2023)
    - Implement memory-efficient attention with tiling
    - Goal: match naive attention output, achieve speedup
    - Difficulty: HARD

[B] Speculative Decoding (Leviathan et al. 2023)
    - Implement rejection sampling for LLM inference speedup
    - Goal: match target-only sampling distribution, achieve speedup
    - Difficulty: HARD

Requirements:
1. Correctness: output must match reference within tolerance
2. Performance: must demonstrate speedup over baseline
3. Code quality: JAX-idiomatic, no python loops where avoidable

Time limit: 6 hours
Memory limit: 8 GB
Hardware: TPU v5e (2x2) for benchmarking
```

### 1.2 starter code – flash attention

```python
# flash_attention_starter.py
"""
Flash Attention 2 Implementation
================================

Implement memory-efficient attention using tiling (block-wise computation).

Key insight: standard attention materializes O(N²) attention matrix.
Flash attention computes attention in O(N) memory by processing blocks
and maintaining running statistics (online softmax).

Reference: https://arxiv.org/abs/2307.08691
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from functools import partial

# ============== CONSTANTS ==============

DEFAULT_BLOCK_Q = 128
DEFAULT_BLOCK_KV = 64

# ============== HELPER: NAIVE ATTENTION (for reference) ==============

def standard_attention(
    q: jax.Array,  # [batch, seq_q, num_heads, head_dim]
    k: jax.Array,  # [batch, seq_kv, num_heads, head_dim]
    v: jax.Array,  # [batch, seq_kv, num_heads, head_dim]
    mask: Optional[jax.Array] = None,  # [batch, num_heads, seq_q, seq_kv]
) -> jax.Array:
    """
    Standard scaled dot-product attention.
    Memory: O(batch * heads * seq_q * seq_kv)
    """
    scale = q.shape[-1] ** -0.5

    # [batch, num_heads, seq_q, seq_kv]
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale

    if mask is not None:
        scores = jnp.where(mask, scores, -1e9)

    attn_weights = jax.nn.softmax(scores, axis=-1)

    # [batch, seq_q, num_heads, head_dim]
    output = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)
    return output


# ============== YOUR IMPLEMENTATION ==============

def flash_attention_forward(
    q: jax.Array,          # [batch, seq_q, num_heads, head_dim]
    k: jax.Array,          # [batch, seq_kv, num_kv_heads, head_dim]
    v: jax.Array,          # [batch, seq_kv, num_kv_heads, head_dim]
    mask: Optional[jax.Array] = None,
    block_q: int = DEFAULT_BLOCK_Q,
    block_kv: int = DEFAULT_BLOCK_KV,
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
    """
    Memory-efficient attention using tiling.

    Algorithm:
    ---------
    1. Split Q into blocks of size block_q
    2. For each Q block:
       a. Initialize: O = 0, m = -inf, l = 0 (running stats)
       b. For each KV block of size block_kv:
          i.   Compute S = Q_block @ K_block^T / sqrt(d)
          ii.  Compute m_new = max(m, rowmax(S))
          iii. Compute P = exp(S - m_new)
          iv.  Compute l_new = exp(m - m_new) * l + rowsum(P)
          v.   Compute O_new = (l/l_new) * O * exp(m - m_new) + P @ V_block
          vi.  Update m = m_new, l = l_new, O = O_new
       c. Final: O = O / l (normalization)

    Returns:
        output: [batch, seq_q, num_heads, head_dim]
        aux: (logsumexp, max_scores) for backward pass

    Memory: O(batch * heads * block_q * block_kv) instead of O(N²)
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement flash attention forward pass")


def flash_attention_backward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    output: jax.Array,
    logsumexp: jax.Array,
    grad_output: jax.Array,
    block_q: int = DEFAULT_BLOCK_Q,
    block_kv: int = DEFAULT_BLOCK_KV,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Backward pass for flash attention.

    OPTIONAL: Implement for "Excellent" grade.

    Returns:
        grad_q, grad_k, grad_v
    """
    # TODO: Implement this function (optional)
    raise NotImplementedError("Implement flash attention backward pass")


# ============== TEST YOUR SOLUTION ==============

if __name__ == "__main__":
    import time

    key = jax.random.PRNGKey(42)

    # Small test
    batch, seq, heads, dim = 2, 512, 8, 64

    q = jax.random.normal(key, (batch, seq, heads, dim))
    k = jax.random.normal(key, (batch, seq, heads, dim))
    v = jax.random.normal(key, (batch, seq, heads, dim))

    # Reference
    ref_output = standard_attention(q, k, v)

    # Your implementation
    try:
        flash_output, aux = flash_attention_forward(q, k, v)

        # Check correctness
        max_diff = jnp.max(jnp.abs(flash_output - ref_output))
        print(f"Max difference: {max_diff:.2e}")
        print(f"Correct: {max_diff < 1e-4}")

        # Benchmark
        n_warmup, n_runs = 5, 20

        for _ in range(n_warmup):
            _ = standard_attention(q, k, v).block_until_ready()

        start = time.perf_counter()
        for _ in range(n_runs):
            _ = standard_attention(q, k, v).block_until_ready()
        naive_time = (time.perf_counter() - start) / n_runs

        for _ in range(n_warmup):
            _ = flash_attention_forward(q, k, v)[0].block_until_ready()

        start = time.perf_counter()
        for _ in range(n_runs):
            _ = flash_attention_forward(q, k, v)[0].block_until_ready()
        flash_time = (time.perf_counter() - start) / n_runs

        print(f"Naive: {naive_time*1000:.2f}ms, Flash: {flash_time*1000:.2f}ms")
        print(f"Speedup: {naive_time/flash_time:.2f}x")

    except NotImplementedError:
        print("flash_attention_forward not implemented yet")
```

### 1.3 starter code – speculative decoding

```python
# speculative_decode_starter.py
"""
Speculative Decoding Implementation
====================================

Implement speculative decoding for LLM inference speedup.

Key insight: small "draft" model generates γ candidate tokens,
large "target" model verifies ALL in ONE forward pass.
Rejection sampling ensures output distribution = target-only sampling.

Reference: https://arxiv.org/abs/2302.01318
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Dict
from functools import partial

# ============== MODEL INTERFACES ==============

# Draft and target models have this interface:
# model_fn(tokens: jax.Array) -> jax.Array
#   tokens: [seq_len] token IDs
#   returns: [seq_len, vocab_size] logits

# For sampling, we provide:
def sample_token(logits: jax.Array, key: jax.Array) -> jax.Array:
    """Sample single token from logits. Returns token ID."""
    return jax.random.categorical(key, logits)

def get_probs(logits: jax.Array) -> jax.Array:
    """Convert logits to probabilities."""
    return jax.nn.softmax(logits, axis=-1)


# ============== REFERENCE: AUTOREGRESSIVE DECODING ==============

def autoregressive_decode(
    prompt_tokens: jax.Array,
    model_fn: Callable,
    max_new_tokens: int,
    key: jax.Array,
) -> jax.Array:
    """
    Standard autoregressive decoding.

    For each new token:
    1. Forward pass through model
    2. Sample from last position
    3. Append to sequence

    Requires max_new_tokens forward passes.
    """
    tokens = prompt_tokens

    for i in range(max_new_tokens):
        key, subkey = jax.random.split(key)
        logits = model_fn(tokens)
        next_token = sample_token(logits[-1], subkey)
        tokens = jnp.concatenate([tokens, next_token[None]])

    return tokens[len(prompt_tokens):]


# ============== YOUR IMPLEMENTATION ==============

def speculative_decode(
    prompt_tokens: jax.Array,       # [prompt_len]
    draft_model_fn: Callable,       # tokens -> logits
    target_model_fn: Callable,      # tokens -> logits
    gamma: int = 5,                 # number of speculative tokens
    max_new_tokens: int = 100,
    key: jax.Array = None,
) -> Tuple[jax.Array, Dict[str, float]]:
    """
    Speculative decoding with rejection sampling.

    Algorithm (per iteration):
    -------------------------
    1. DRAFT PHASE: Generate γ tokens autoregressively with draft model
       - tokens = [x_1, x_2, ..., x_γ]
       - save draft probs q(x_i | x_{<i}) for each token

    2. VERIFY PHASE: Single forward pass with target model
       - compute target probs p(x_i | x_{<i}) for all γ positions

    3. ACCEPT/REJECT (for i = 1 to γ):
       - r = p(x_i) / q(x_i)
       - if random() < min(1, r): accept x_i, continue
       - else: reject x_i, resample from max(0, p - q), STOP

    4. BONUS TOKEN: if all γ accepted, sample one more from target

    Key property: output distribution EXACTLY matches target-only sampling.

    Args:
        prompt_tokens: Initial prompt token IDs
        draft_model_fn: Small fast model
        target_model_fn: Large accurate model
        gamma: Number of speculative tokens per iteration
        max_new_tokens: Maximum tokens to generate
        key: PRNG key

    Returns:
        generated_tokens: [num_generated] new token IDs
        stats: {
            'acceptance_rate': fraction of draft tokens accepted,
            'tokens_per_target_call': average tokens generated per target forward,
            'num_target_calls': total target forward passes
        }
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement speculative decoding")


def rejection_sample(
    draft_probs: jax.Array,    # [vocab_size] - q(x)
    target_probs: jax.Array,   # [vocab_size] - p(x)
    draft_token: int,          # sampled token from draft
    key: jax.Array,
) -> Tuple[int, bool]:
    """
    Rejection sampling step.

    Given:
    - draft sampled token x ~ q(x)
    - target probs p(x)
    - draft probs q(x)

    Accept x with probability min(1, p(x)/q(x)).
    If rejected, resample from norm(max(0, p - q)).

    Returns:
        token: accepted or resampled token
        accepted: whether draft token was accepted
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement rejection sampling")


# ============== TEST YOUR SOLUTION ==============

if __name__ == "__main__":
    import time

    # Mock models for testing (in real use, these would be actual LLMs)
    vocab_size = 32000

    def make_mock_model(temperature: float, key: jax.Array):
        """Create mock model with fixed distribution."""
        base_logits = jax.random.normal(key, (vocab_size,))
        def model_fn(tokens):
            # Return logits for each position
            return jnp.tile(base_logits / temperature, (len(tokens), 1))
        return model_fn

    key = jax.random.PRNGKey(42)
    key, k1, k2 = jax.random.split(key, 3)

    # Draft: higher temperature (more uniform, less accurate)
    # Target: lower temperature (more peaked, more accurate)
    draft_model = make_mock_model(temperature=1.0, key=k1)
    target_model = make_mock_model(temperature=0.8, key=k2)

    prompt = jnp.array([1, 2, 3])  # mock prompt

    try:
        key, subkey = jax.random.split(key)
        tokens, stats = speculative_decode(
            prompt_tokens=prompt,
            draft_model_fn=draft_model,
            target_model_fn=target_model,
            gamma=5,
            max_new_tokens=50,
            key=subkey,
        )

        print(f"Generated {len(tokens)} tokens")
        print(f"Acceptance rate: {stats['acceptance_rate']:.2%}")
        print(f"Tokens per target call: {stats['tokens_per_target_call']:.2f}")
        print(f"Number of target calls: {stats['num_target_calls']}")

        # Expected for good implementation:
        # - acceptance rate ~60-80% for similar distributions
        # - tokens per target call ~2-5 (higher is better)

    except NotImplementedError:
        print("speculative_decode not implemented yet")
```

### 1.4 paper summaries (provided to student)

#### flash attention 2 – algorithm pseudocode

```
FLASH ATTENTION FORWARD PASS
============================

Input: Q, K, V ∈ R^{N×d}, block sizes B_q, B_kv
Output: O ∈ R^{N×d}

# Split Q into T_q = ceil(N/B_q) blocks
# Split K, V into T_kv = ceil(N/B_kv) blocks

for i = 1 to T_q:
    # Initialize running statistics for block i
    O_i = 0 ∈ R^{B_q × d}      # output accumulator
    m_i = -∞ ∈ R^{B_q}         # running max (for numerical stability)
    l_i = 0 ∈ R^{B_q}          # running sum (denominator)

    Q_i = Q[i*B_q : (i+1)*B_q]  # current Q block

    for j = 1 to T_kv:
        K_j = K[j*B_kv : (j+1)*B_kv]
        V_j = V[j*B_kv : (j+1)*B_kv]

        # Compute attention scores for this block pair
        S_ij = Q_i @ K_j^T / sqrt(d)  # [B_q × B_kv]

        # Online softmax update
        m_ij = rowmax(S_ij)           # [B_q]
        m_new = max(m_i, m_ij)        # new running max

        P_ij = exp(S_ij - m_new[:, None])  # safe softmax numerator
        l_new = exp(m_i - m_new) * l_i + rowsum(P_ij)  # new running sum

        # Update output accumulator with rescaling
        scale = exp(m_i - m_new) * l_i / l_new
        O_i = scale[:, None] * O_i + (P_ij @ V_j) / l_new[:, None]

        # Update running statistics
        m_i = m_new
        l_i = l_new

    # Store result (no final division needed - already normalized)
    O[i*B_q : (i+1)*B_q] = O_i

return O
```

#### speculative decoding – algorithm pseudocode

```
SPECULATIVE DECODING
====================

Input: prompt x_0, draft model M_d, target model M_t, lookahead γ
Output: generated tokens

tokens = x_0
stats = {accepted: 0, total: 0, target_calls: 0}

while len(tokens) - len(x_0) < max_new_tokens:
    # === DRAFT PHASE ===
    # Generate γ tokens autoregressively with draft model
    draft_tokens = []
    draft_probs = []
    current = tokens

    for i in 1 to γ:
        q = softmax(M_d(current)[-1])  # draft distribution at last position
        x_i ~ q                         # sample from draft
        draft_tokens.append(x_i)
        draft_probs.append(q[x_i])
        current = concat(current, x_i)

    # === VERIFY PHASE ===
    # Single forward pass: get target probs for all γ + 1 positions
    verify_input = concat(tokens, draft_tokens)
    target_logits = M_t(verify_input)  # [len(tokens) + γ, vocab]

    # Extract target probs at each draft position
    target_probs = [softmax(target_logits[len(tokens) + i - 1]) for i in 1 to γ]
    stats.target_calls += 1

    # === ACCEPT/REJECT ===
    n_accepted = 0

    for i in 1 to γ:
        p_i = target_probs[i-1][draft_tokens[i-1]]  # p(x_i)
        q_i = draft_probs[i-1]                       # q(x_i)

        r = uniform(0, 1)
        if r < min(1, p_i / q_i):
            # ACCEPT: append token
            tokens = concat(tokens, draft_tokens[i-1])
            n_accepted += 1
            stats.accepted += 1
        else:
            # REJECT: resample from max(0, p - q) and stop
            adjusted = max(0, target_probs[i-1] - draft_probs[i-1] * (p_i/q_i))
            adjusted = adjusted / sum(adjusted)  # normalize
            x_new ~ adjusted
            tokens = concat(tokens, x_new)
            break

        stats.total += 1

    # === BONUS TOKEN ===
    if n_accepted == γ:
        # All accepted! Sample one more from target
        bonus = sample(softmax(target_logits[-1]))
        tokens = concat(tokens, bonus)

return tokens[len(x_0):], stats
```

### 1.5 reference test cases

#### flash attention tests

```python
# test_flash_attention.py

import jax
import jax.numpy as jnp
import pytest

# Import student solution
from flash_attention import flash_attention_forward, standard_attention

class TestFlashAttentionCorrectness:
    """Correctness tests."""

    def test_small_sequence(self):
        """Basic test on small inputs."""
        key = jax.random.PRNGKey(0)
        q = jax.random.normal(key, (1, 64, 4, 32))
        k = jax.random.normal(key, (1, 64, 4, 32))
        v = jax.random.normal(key, (1, 64, 4, 32))

        ref = standard_attention(q, k, v)
        out, _ = flash_attention_forward(q, k, v, block_q=32, block_kv=32)

        assert jnp.allclose(out, ref, atol=1e-4, rtol=1e-4)

    def test_medium_sequence(self):
        """Test on seq=512."""
        key = jax.random.PRNGKey(1)
        q = jax.random.normal(key, (2, 512, 8, 64))
        k = jax.random.normal(key, (2, 512, 8, 64))
        v = jax.random.normal(key, (2, 512, 8, 64))

        ref = standard_attention(q, k, v)
        out, _ = flash_attention_forward(q, k, v)

        assert jnp.allclose(out, ref, atol=1e-5, rtol=1e-5)

    def test_long_sequence(self):
        """Test on seq=2048."""
        key = jax.random.PRNGKey(2)
        q = jax.random.normal(key, (1, 2048, 8, 64))
        k = jax.random.normal(key, (1, 2048, 8, 64))
        v = jax.random.normal(key, (1, 2048, 8, 64))

        ref = standard_attention(q, k, v)
        out, _ = flash_attention_forward(q, k, v)

        assert jnp.allclose(out, ref, atol=1e-5, rtol=1e-5)

    def test_different_block_sizes(self):
        """Test various block size combinations."""
        key = jax.random.PRNGKey(3)
        q = jax.random.normal(key, (1, 256, 4, 64))
        k = jax.random.normal(key, (1, 256, 4, 64))
        v = jax.random.normal(key, (1, 256, 4, 64))

        ref = standard_attention(q, k, v)

        for block_q, block_kv in [(32, 32), (64, 64), (128, 64), (64, 128)]:
            out, _ = flash_attention_forward(q, k, v, block_q=block_q, block_kv=block_kv)
            assert jnp.allclose(out, ref, atol=1e-4)

    def test_causal_mask(self):
        """Test causal attention."""
        key = jax.random.PRNGKey(4)
        seq = 128
        q = jax.random.normal(key, (1, seq, 4, 64))
        k = jax.random.normal(key, (1, seq, 4, 64))
        v = jax.random.normal(key, (1, seq, 4, 64))

        # Causal mask: position i attends only to j <= i
        mask = jnp.tril(jnp.ones((1, 1, seq, seq), dtype=bool))

        ref = standard_attention(q, k, v, mask=mask)
        out, _ = flash_attention_forward(q, k, v, mask=mask)

        assert jnp.allclose(out, ref, atol=1e-4)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    def test_dtypes(self, dtype):
        """Test different dtypes."""
        key = jax.random.PRNGKey(5)
        q = jax.random.normal(key, (1, 128, 4, 64), dtype=dtype)
        k = jax.random.normal(key, (1, 128, 4, 64), dtype=dtype)
        v = jax.random.normal(key, (1, 128, 4, 64), dtype=dtype)

        out, _ = flash_attention_forward(q, k, v)
        assert out.dtype == dtype


class TestFlashAttentionGradients:
    """Gradient tests (for Excellent grade)."""

    def test_gradient_correctness(self):
        """Finite difference vs analytical gradients."""
        key = jax.random.PRNGKey(6)
        q = jax.random.normal(key, (1, 64, 2, 32))
        k = jax.random.normal(key, (1, 64, 2, 32))
        v = jax.random.normal(key, (1, 64, 2, 32))

        def loss_fn(q, k, v):
            out, _ = flash_attention_forward(q, k, v)
            return jnp.sum(out ** 2)

        # Analytical gradient
        grad_q, grad_k, grad_v = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        # Numerical gradient (finite difference)
        eps = 1e-4
        num_grad_q = jnp.zeros_like(q)
        # ... (implementation of finite difference)

        # For this test, we check that gradients are finite and reasonable
        assert jnp.all(jnp.isfinite(grad_q))
        assert jnp.all(jnp.isfinite(grad_k))
        assert jnp.all(jnp.isfinite(grad_v))


class TestFlashAttentionBenchmark:
    """Performance benchmarks."""

    def test_speedup_seq512(self):
        """Minimum speedup on seq=512."""
        import time

        key = jax.random.PRNGKey(7)
        q = jax.random.normal(key, (4, 512, 8, 64))
        k = jax.random.normal(key, (4, 512, 8, 64))
        v = jax.random.normal(key, (4, 512, 8, 64))

        # Warmup
        for _ in range(3):
            _ = standard_attention(q, k, v).block_until_ready()
            _ = flash_attention_forward(q, k, v)[0].block_until_ready()

        # Benchmark
        n_runs = 10

        start = time.perf_counter()
        for _ in range(n_runs):
            _ = standard_attention(q, k, v).block_until_ready()
        naive_time = (time.perf_counter() - start) / n_runs

        start = time.perf_counter()
        for _ in range(n_runs):
            _ = flash_attention_forward(q, k, v)[0].block_until_ready()
        flash_time = (time.perf_counter() - start) / n_runs

        speedup = naive_time / flash_time
        assert speedup > 1.0, f"No speedup: {speedup:.2f}x"

    def test_speedup_seq2048_excellent(self):
        """Excellent grade: >5x speedup on seq=2048."""
        import time

        key = jax.random.PRNGKey(8)
        q = jax.random.normal(key, (2, 2048, 8, 64))
        k = jax.random.normal(key, (2, 2048, 8, 64))
        v = jax.random.normal(key, (2, 2048, 8, 64))

        # ... same benchmark logic ...

        # speedup = naive_time / flash_time
        # assert speedup > 5.0, f"Speedup {speedup:.2f}x < 5x required for excellent"
```

#### speculative decoding tests

```python
# test_speculative_decode.py

import jax
import jax.numpy as jnp
import pytest
from collections import Counter

# Import student solution
from speculative_decode import speculative_decode, autoregressive_decode

class TestSpeculativeDecodeCorrectness:
    """Correctness tests: distribution equivalence."""

    @pytest.fixture
    def models(self):
        """Create test models with known distributions."""
        vocab_size = 100
        key = jax.random.PRNGKey(42)

        # Shared base distribution (so draft and target are similar)
        base = jax.random.normal(key, (vocab_size,))

        def draft_model(tokens):
            # Higher temperature = more uniform
            logits = base / 1.2
            return jnp.tile(logits, (len(tokens), 1))

        def target_model(tokens):
            # Lower temperature = more peaked
            logits = base / 0.8
            return jnp.tile(logits, (len(tokens), 1))

        return draft_model, target_model, vocab_size

    def test_distribution_equivalence_small(self, models):
        """KL divergence between spec and target-only should be small."""
        draft_model, target_model, vocab_size = models
        prompt = jnp.array([0, 1, 2])
        n_samples = 500
        max_tokens = 20

        # Collect speculative samples
        spec_tokens = []
        for i in range(n_samples):
            key = jax.random.PRNGKey(i)
            tokens, _ = speculative_decode(
                prompt, draft_model, target_model,
                gamma=5, max_new_tokens=max_tokens, key=key
            )
            spec_tokens.append(tuple(tokens.tolist()))

        # Collect target-only samples
        target_tokens = []
        for i in range(n_samples):
            key = jax.random.PRNGKey(i + 10000)
            tokens = autoregressive_decode(prompt, target_model, max_tokens, key)
            target_tokens.append(tuple(tokens.tolist()))

        # Compare first-token distributions (simplest check)
        spec_first = Counter([t[0] for t in spec_tokens])
        target_first = Counter([t[0] for t in target_tokens])

        # Chi-squared test or KL divergence
        kl_div = compute_kl_divergence(spec_first, target_first, vocab_size)
        assert kl_div < 0.05, f"KL divergence too high: {kl_div:.4f}"

    def test_distribution_equivalence_strict(self, models):
        """Excellent grade: stricter KL threshold."""
        # ... same as above but with kl_div < 0.005 ...
        pass


class TestSpeculativeDecodeMetrics:
    """Performance metrics tests."""

    def test_acceptance_rate_reasonable(self, models):
        """Acceptance rate should be >50% for similar models."""
        draft_model, target_model, _ = models
        prompt = jnp.array([0, 1, 2])

        key = jax.random.PRNGKey(0)
        _, stats = speculative_decode(
            prompt, draft_model, target_model,
            gamma=5, max_new_tokens=100, key=key
        )

        assert stats['acceptance_rate'] > 0.5

    def test_speedup_measured(self, models):
        """Should achieve speedup over autoregressive."""
        import time

        draft_model, target_model, _ = models
        prompt = jnp.array([0, 1, 2])
        max_tokens = 100

        # Time autoregressive
        key = jax.random.PRNGKey(0)
        start = time.perf_counter()
        for _ in range(10):
            _ = autoregressive_decode(prompt, target_model, max_tokens, key)
        ar_time = (time.perf_counter() - start) / 10

        # Time speculative
        start = time.perf_counter()
        for _ in range(10):
            _ = speculative_decode(
                prompt, draft_model, target_model,
                gamma=5, max_new_tokens=max_tokens, key=key
            )
        spec_time = (time.perf_counter() - start) / 10

        # Note: with mock models, speedup depends on target_calls ratio
        # Real speedup = ar_time / spec_time when using actual LLMs
        tokens_per_call = max_tokens / 10  # average
        expected_speedup = tokens_per_call / 1  # vs 1 token per call

        # Just verify stats are computed correctly
        pass


class TestSpeculativeDecodeEdgeCases:
    """Edge case handling."""

    def test_gamma_one(self, models):
        """gamma=1 should still work."""
        draft_model, target_model, _ = models
        prompt = jnp.array([0])

        key = jax.random.PRNGKey(0)
        tokens, stats = speculative_decode(
            prompt, draft_model, target_model,
            gamma=1, max_new_tokens=10, key=key
        )

        assert len(tokens) == 10

    def test_all_rejected(self):
        """Handle case where draft and target are very different."""
        vocab_size = 100

        # Draft always predicts token 0
        def draft_model(tokens):
            logits = jnp.full((len(tokens), vocab_size), -10.0)
            logits = logits.at[:, 0].set(10.0)
            return logits

        # Target always predicts token 50
        def target_model(tokens):
            logits = jnp.full((len(tokens), vocab_size), -10.0)
            logits = logits.at[:, 50].set(10.0)
            return logits

        prompt = jnp.array([0])
        key = jax.random.PRNGKey(0)

        tokens, stats = speculative_decode(
            prompt, draft_model, target_model,
            gamma=5, max_new_tokens=10, key=key
        )

        # Should still generate tokens (via rejection sampling)
        assert len(tokens) == 10
        # Acceptance rate should be very low
        assert stats['acceptance_rate'] < 0.2


def compute_kl_divergence(counts1, counts2, vocab_size, smoothing=1e-8):
    """Compute KL divergence between two empirical distributions."""
    total1 = sum(counts1.values())
    total2 = sum(counts2.values())

    kl = 0.0
    for token in range(vocab_size):
        p = (counts1.get(token, 0) + smoothing) / (total1 + smoothing * vocab_size)
        q = (counts2.get(token, 0) + smoothing) / (total2 + smoothing * vocab_size)
        if p > 0:
            kl += p * jnp.log(p / q)

    return float(kl)
```

---

## 2. what student does – flash attention

### step 1: understand tiled attention algorithm

student reads paper pseudocode and understands the core insight:

```python
# Standard attention (O(N²) memory):
# S = Q @ K^T           # [N, N] - HUGE matrix
# P = softmax(S)        # [N, N] - still huge
# O = P @ V             # [N, d]

# Flash attention insight:
# Never materialize full S or P matrices!
# Process in blocks, maintain running statistics.

# For softmax: y_i = exp(x_i) / sum(exp(x_j))
# Can compute incrementally using:
#   m = max(x)                    # for numerical stability
#   l = sum(exp(x - m))           # running denominator
#   y = exp(x - m) / l            # final result

# Key: when adding new values, can UPDATE m and l:
#   m_new = max(m_old, m_block)
#   l_new = exp(m_old - m_new) * l_old + sum(exp(x_block - m_new))
```

### step 2: implement online softmax

```python
def online_softmax_update(
    m_prev: jax.Array,      # [B_q] previous max
    l_prev: jax.Array,      # [B_q] previous sum
    o_prev: jax.Array,      # [B_q, d] previous output
    scores: jax.Array,      # [B_q, B_kv] new attention scores
    v_block: jax.Array,     # [B_kv, d] new value block
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Update running softmax statistics with new block.

    This is the CORE of flash attention.
    """
    # New block statistics
    m_block = jnp.max(scores, axis=-1)  # [B_q]

    # New global max
    m_new = jnp.maximum(m_prev, m_block)  # [B_q]

    # Rescale factors
    exp_prev = jnp.exp(m_prev - m_new)[:, None]  # [B_q, 1]
    exp_block = jnp.exp(scores - m_new[:, None])  # [B_q, B_kv]

    # New sum
    l_new = exp_prev.squeeze() * l_prev + jnp.sum(exp_block, axis=-1)  # [B_q]

    # Update output: rescale old + add new contribution
    o_new = (exp_prev * l_prev[:, None] / l_new[:, None]) * o_prev + \
            (exp_block @ v_block) / l_new[:, None]

    return m_new, l_new, o_new
```

### step 3: implement forward pass with block iteration

```python
def flash_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: Optional[jax.Array] = None,
    block_q: int = 128,
    block_kv: int = 64,
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
    """Full flash attention forward pass."""

    batch, seq_q, num_heads, head_dim = q.shape
    _, seq_kv, _, _ = k.shape
    scale = head_dim ** -0.5

    # Number of blocks
    num_blocks_q = (seq_q + block_q - 1) // block_q
    num_blocks_kv = (seq_kv + block_kv - 1) // block_kv

    # Pad sequences to multiple of block size
    pad_q = num_blocks_q * block_q - seq_q
    pad_kv = num_blocks_kv * block_kv - seq_kv

    if pad_q > 0:
        q = jnp.pad(q, ((0, 0), (0, pad_q), (0, 0), (0, 0)))
    if pad_kv > 0:
        k = jnp.pad(k, ((0, 0), (0, pad_kv), (0, 0), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, pad_kv), (0, 0), (0, 0)))

    # Reshape into blocks
    # [batch, num_blocks, block_size, heads, dim]
    q_blocks = q.reshape(batch, num_blocks_q, block_q, num_heads, head_dim)
    k_blocks = k.reshape(batch, num_blocks_kv, block_kv, num_heads, head_dim)
    v_blocks = v.reshape(batch, num_blocks_kv, block_kv, num_heads, head_dim)

    # Initialize outputs
    o = jnp.zeros((batch, num_blocks_q, block_q, num_heads, head_dim))
    m = jnp.full((batch, num_blocks_q, block_q, num_heads), -jnp.inf)
    l = jnp.zeros((batch, num_blocks_q, block_q, num_heads))

    # Process each Q block
    def process_q_block(carry, q_block_idx):
        o, m, l = carry
        q_block = q_blocks[:, q_block_idx]  # [batch, block_q, heads, dim]

        # Initialize for this Q block
        o_i = jnp.zeros((batch, block_q, num_heads, head_dim))
        m_i = jnp.full((batch, block_q, num_heads), -jnp.inf)
        l_i = jnp.zeros((batch, block_q, num_heads))

        # Process each KV block
        def process_kv_block(carry_kv, kv_block_idx):
            o_i, m_i, l_i = carry_kv

            k_block = k_blocks[:, kv_block_idx]  # [batch, block_kv, heads, dim]
            v_block = v_blocks[:, kv_block_idx]

            # Compute attention scores: [batch, heads, block_q, block_kv]
            scores = jnp.einsum("bqhd,bkhd->bhqk", q_block, k_block) * scale

            # Apply mask if provided
            if mask is not None:
                # Extract relevant mask block
                q_start = q_block_idx * block_q
                kv_start = kv_block_idx * block_kv
                mask_block = mask[:, :, q_start:q_start+block_q, kv_start:kv_start+block_kv]
                scores = jnp.where(mask_block, scores, -jnp.inf)

            # Transpose scores for easier processing: [batch, block_q, heads, block_kv]
            scores = scores.transpose(0, 2, 1, 3)

            # Online softmax update
            m_block = jnp.max(scores, axis=-1)  # [batch, block_q, heads]
            m_new = jnp.maximum(m_i, m_block)

            exp_prev = jnp.exp(m_i - m_new)
            exp_scores = jnp.exp(scores - m_new[..., None])

            l_new = exp_prev * l_i + jnp.sum(exp_scores, axis=-1)

            # [batch, block_q, heads, dim]
            o_new = (exp_prev[..., None] * l_i[..., None] * o_i +
                     jnp.einsum("bqhk,bkhd->bqhd", exp_scores, v_block)) / l_new[..., None]

            return (o_new, m_new, l_new), None

        (o_i, m_i, l_i), _ = jax.lax.scan(
            process_kv_block,
            (o_i, m_i, l_i),
            jnp.arange(num_blocks_kv)
        )

        # Store results for this Q block
        o = o.at[:, q_block_idx].set(o_i)
        m = m.at[:, q_block_idx].set(m_i)
        l = l.at[:, q_block_idx].set(l_i)

        return (o, m, l), None

    (o, m, l), _ = jax.lax.scan(
        process_q_block,
        (o, m, l),
        jnp.arange(num_blocks_q)
    )

    # Reshape output back
    o = o.reshape(batch, num_blocks_q * block_q, num_heads, head_dim)

    # Remove padding
    if pad_q > 0:
        o = o[:, :seq_q]

    # Compute logsumexp for backward pass
    logsumexp = m + jnp.log(l)
    logsumexp = logsumexp.reshape(batch, num_blocks_q * block_q, num_heads)
    if pad_q > 0:
        logsumexp = logsumexp[:, :seq_q]

    return o, (logsumexp, m.reshape(batch, -1, num_heads)[:, :seq_q])
```

### step 4: implement backward pass (optional – for excellent grade)

```python
def flash_attention_backward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    output: jax.Array,
    logsumexp: jax.Array,
    grad_output: jax.Array,
    block_q: int = 128,
    block_kv: int = 64,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Backward pass for flash attention.

    Key insight: recompute attention weights during backward
    instead of storing O(N²) matrix.

    See FlashAttention-2 paper Section 3.2 for algorithm.
    """
    batch, seq_q, num_heads, head_dim = q.shape
    _, seq_kv, _, _ = k.shape
    scale = head_dim ** -0.5

    # Initialize gradients
    grad_q = jnp.zeros_like(q)
    grad_k = jnp.zeros_like(k)
    grad_v = jnp.zeros_like(v)

    # Similar block iteration as forward, but computing gradients
    # ... implementation details ...

    return grad_q, grad_k, grad_v
```

### step 5: run correctness tests

```python
if __name__ == "__main__":
    # Run all tests
    import subprocess
    result = subprocess.run(["pytest", "test_flash_attention.py", "-v"])

    # Expected output:
    # test_small_sequence PASSED
    # test_medium_sequence PASSED
    # test_long_sequence PASSED
    # test_different_block_sizes PASSED
    # test_causal_mask PASSED
```

### step 6: benchmark vs naive attention

```python
def benchmark():
    """Compare flash vs naive attention."""
    import time

    results = []

    for seq_len in [512, 1024, 2048, 4096]:
        key = jax.random.PRNGKey(0)
        q = jax.random.normal(key, (2, seq_len, 8, 64))
        k = jax.random.normal(key, (2, seq_len, 8, 64))
        v = jax.random.normal(key, (2, seq_len, 8, 64))

        # Warmup
        for _ in range(3):
            _ = standard_attention(q, k, v).block_until_ready()
            _ = flash_attention_forward(q, k, v)[0].block_until_ready()

        # Benchmark naive
        start = time.perf_counter()
        for _ in range(10):
            _ = standard_attention(q, k, v).block_until_ready()
        naive_ms = (time.perf_counter() - start) / 10 * 1000

        # Benchmark flash
        start = time.perf_counter()
        for _ in range(10):
            _ = flash_attention_forward(q, k, v)[0].block_until_ready()
        flash_ms = (time.perf_counter() - start) / 10 * 1000

        speedup = naive_ms / flash_ms
        results.append({
            'seq_len': seq_len,
            'naive_ms': naive_ms,
            'flash_ms': flash_ms,
            'speedup': speedup
        })

        print(f"Seq {seq_len}: naive={naive_ms:.2f}ms, flash={flash_ms:.2f}ms, speedup={speedup:.2f}x")

    return results

# Expected output:
# Seq 512: naive=1.23ms, flash=0.98ms, speedup=1.26x
# Seq 1024: naive=4.56ms, flash=1.89ms, speedup=2.41x
# Seq 2048: naive=18.2ms, flash=3.21ms, speedup=5.67x
# Seq 4096: naive=72.1ms, flash=6.42ms, speedup=11.2x
```

---

## 3. what student does – speculative decoding

### step 1: understand rejection sampling algorithm

```python
# Key insight: we want samples from target distribution p(x)
# but generating from p is expensive (large model)
#
# Draft model q(x) is cheap but less accurate.
#
# Rejection sampling ensures final distribution = p(x) exactly:
#
# 1. Sample x ~ q(x)
# 2. Accept x with probability min(1, p(x)/q(x))
# 3. If rejected, sample from normalized max(0, p(x) - q(x))
#
# Proof that this works:
# - For tokens where p(x) >= q(x): always accepted, contribution = q(x)
# - For tokens where p(x) < q(x): accepted with prob p(x)/q(x)
#   so contribution = q(x) * p(x)/q(x) = p(x)
# - Rejection resamples from residual, filling in p(x) - accepted
#
# Final distribution = p(x) for all x!
```

### step 2: implement draft model generation (gamma tokens)

```python
def generate_draft_tokens(
    tokens: jax.Array,           # current sequence
    draft_model_fn: Callable,
    gamma: int,
    key: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Generate gamma tokens autoregressively with draft model.

    Returns:
        draft_tokens: [gamma] generated token IDs
        draft_probs: [gamma, vocab_size] probability distributions
    """
    draft_tokens = []
    draft_probs = []
    current = tokens

    for i in range(gamma):
        key, subkey = jax.random.split(key)

        # Get draft logits for last position
        logits = draft_model_fn(current)[-1]  # [vocab_size]
        probs = jax.nn.softmax(logits)

        # Sample token
        token = jax.random.categorical(subkey, logits)

        draft_tokens.append(token)
        draft_probs.append(probs)

        # Append to sequence
        current = jnp.concatenate([current, token[None]])

    return jnp.stack(draft_tokens), jnp.stack(draft_probs)
```

### step 3: implement target model verification (single forward)

```python
def verify_with_target(
    tokens: jax.Array,           # original sequence
    draft_tokens: jax.Array,     # [gamma] draft tokens
    target_model_fn: Callable,
) -> jax.Array:
    """
    Get target probabilities for all positions in ONE forward pass.

    Returns:
        target_probs: [gamma + 1, vocab_size]
        - first gamma: probs for verifying draft tokens
        - last: probs for bonus token if all accepted
    """
    # Concatenate prompt with draft tokens
    full_sequence = jnp.concatenate([tokens, draft_tokens])

    # Single forward pass
    logits = target_model_fn(full_sequence)  # [len(full), vocab_size]

    # Extract relevant positions (last gamma+1)
    # Position i gives p(x_{i+1} | x_{<=i})
    start_idx = len(tokens) - 1  # -1 bc logits[i] predicts position i+1
    relevant_logits = logits[start_idx:start_idx + len(draft_tokens) + 1]

    return jax.nn.softmax(relevant_logits, axis=-1)
```

### step 4: implement rejection/resampling logic

```python
def rejection_sample(
    draft_probs: jax.Array,    # [vocab_size] - q(x)
    target_probs: jax.Array,   # [vocab_size] - p(x)
    draft_token: int,
    key: jax.Array,
) -> Tuple[int, bool]:
    """
    Rejection sampling step.
    """
    key, k1, k2 = jax.random.split(key, 3)

    # Get probabilities for draft token
    q_x = draft_probs[draft_token]
    p_x = target_probs[draft_token]

    # Acceptance probability
    accept_prob = jnp.minimum(1.0, p_x / (q_x + 1e-10))

    # Random threshold
    r = jax.random.uniform(k1)
    accepted = r < accept_prob

    # If rejected, resample from adjusted distribution
    # adjusted = max(0, p - q * accept_prob)
    # But simpler: max(0, p - q) normalized
    adjusted = jnp.maximum(0, target_probs - draft_probs)
    adjusted = adjusted / (jnp.sum(adjusted) + 1e-10)

    # Resample token
    resampled_token = jax.random.categorical(k2, jnp.log(adjusted + 1e-10))

    # Return accepted token or resampled token
    final_token = jnp.where(accepted, draft_token, resampled_token)

    return final_token, accepted


def speculative_decode(
    prompt_tokens: jax.Array,
    draft_model_fn: Callable,
    target_model_fn: Callable,
    gamma: int = 5,
    max_new_tokens: int = 100,
    key: jax.Array = None,
) -> Tuple[jax.Array, Dict[str, float]]:
    """Full speculative decoding implementation."""

    tokens = prompt_tokens
    generated = []

    stats = {
        'accepted': 0,
        'total': 0,
        'target_calls': 0
    }

    while len(generated) < max_new_tokens:
        key, k1, k2 = jax.random.split(key, 3)

        # === DRAFT PHASE ===
        remaining = max_new_tokens - len(generated)
        current_gamma = min(gamma, remaining)

        draft_tokens, draft_probs = generate_draft_tokens(
            tokens, draft_model_fn, current_gamma, k1
        )

        # === VERIFY PHASE ===
        target_probs = verify_with_target(tokens, draft_tokens, target_model_fn)
        stats['target_calls'] += 1

        # === ACCEPT/REJECT ===
        n_accepted = 0
        keys = jax.random.split(k2, current_gamma + 1)

        for i in range(current_gamma):
            token, accepted = rejection_sample(
                draft_probs[i],
                target_probs[i],
                draft_tokens[i],
                keys[i]
            )

            generated.append(int(token))
            tokens = jnp.concatenate([tokens, jnp.array([token])])

            if accepted:
                n_accepted += 1
                stats['accepted'] += 1
            else:
                # Stop at first rejection
                break

            stats['total'] += 1

            if len(generated) >= max_new_tokens:
                break

        # === BONUS TOKEN ===
        if n_accepted == current_gamma and len(generated) < max_new_tokens:
            # All accepted! Sample bonus from target
            bonus_probs = target_probs[-1]
            bonus_token = jax.random.categorical(keys[-1], jnp.log(bonus_probs + 1e-10))
            generated.append(int(bonus_token))
            tokens = jnp.concatenate([tokens, jnp.array([bonus_token])])

    # Compute final stats
    stats['acceptance_rate'] = stats['accepted'] / max(1, stats['total'])
    stats['tokens_per_target_call'] = len(generated) / max(1, stats['target_calls'])
    stats['num_target_calls'] = stats['target_calls']

    return jnp.array(generated[:max_new_tokens]), stats
```

### step 5: verify distribution equivalence

```python
def verify_distribution():
    """
    Statistical test that speculative decoding matches target distribution.
    """
    from collections import Counter
    import numpy as np

    # Setup models
    vocab_size = 100
    key = jax.random.PRNGKey(42)

    base_logits = jax.random.normal(key, (vocab_size,))

    def draft_model(tokens):
        return jnp.tile(base_logits / 1.0, (len(tokens), 1))

    def target_model(tokens):
        return jnp.tile(base_logits / 0.7, (len(tokens), 1))

    prompt = jnp.array([0])
    n_samples = 1000
    max_tokens = 20

    # Collect samples
    spec_first_tokens = []
    target_first_tokens = []

    for i in range(n_samples):
        # Speculative
        key = jax.random.PRNGKey(i)
        tokens, _ = speculative_decode(
            prompt, draft_model, target_model,
            gamma=5, max_new_tokens=max_tokens, key=key
        )
        spec_first_tokens.append(int(tokens[0]))

        # Target-only
        key = jax.random.PRNGKey(i + 100000)
        tokens = autoregressive_decode(prompt, target_model, max_tokens, key)
        target_first_tokens.append(int(tokens[0]))

    # Compare distributions
    spec_counts = Counter(spec_first_tokens)
    target_counts = Counter(target_first_tokens)

    # KL divergence
    kl = 0
    for token in range(vocab_size):
        p = (spec_counts.get(token, 0) + 1) / (n_samples + vocab_size)
        q = (target_counts.get(token, 0) + 1) / (n_samples + vocab_size)
        kl += p * np.log(p / q)

    print(f"KL divergence: {kl:.6f}")
    print(f"Pass threshold (0.05): {'PASS' if kl < 0.05 else 'FAIL'}")
    print(f"Excellent threshold (0.005): {'PASS' if kl < 0.005 else 'FAIL'}")

    return kl

# Expected output:
# KL divergence: 0.002341
# Pass threshold (0.05): PASS
# Excellent threshold (0.005): PASS
```

---

## 4. submission format

### 4.1 flash attention submission

```python
# flash_attention.py
"""
Flash Attention 2 Implementation
Student: [NAME]
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional

def flash_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: Optional[jax.Array] = None,
    block_q: int = 128,
    block_kv: int = 64,
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
    """
    Memory-efficient attention using tiling.
    """
    # ... student implementation ...
    pass


def flash_attention_backward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    output: jax.Array,
    logsumexp: jax.Array,
    grad_output: jax.Array,
    block_q: int = 128,
    block_kv: int = 64,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Backward pass (optional for excellent grade).
    """
    # ... student implementation ...
    pass
```

### 4.2 speculative decoding submission

```python
# speculative_decode.py
"""
Speculative Decoding Implementation
Student: [NAME]
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Dict

def speculative_decode(
    prompt_tokens: jax.Array,
    draft_model_fn: Callable,
    target_model_fn: Callable,
    gamma: int = 5,
    max_new_tokens: int = 100,
    key: jax.Array = None,
) -> Tuple[jax.Array, Dict[str, float]]:
    """
    Speculative decoding with rejection sampling.

    Returns:
        generated_tokens: [num_generated]
        stats: {'acceptance_rate', 'tokens_per_target_call', 'num_target_calls'}
    """
    # ... student implementation ...
    pass


def rejection_sample(
    draft_probs: jax.Array,
    target_probs: jax.Array,
    draft_token: int,
    key: jax.Array,
) -> Tuple[int, bool]:
    """
    Single rejection sampling step.
    """
    # ... student implementation ...
    pass
```

---

## 5. evaluation pipeline

### 5.1 evaluation orchestrator

```python
# evaluator.py

import time
import tracemalloc
from dataclasses import dataclass
from typing import Dict, Any, List
import jax
import jax.numpy as jnp

@dataclass
class EvaluationResult:
    passed: bool
    grade: str  # "Pass", "Good", "Excellent", "Fail"
    score: float
    metrics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]


def evaluate_flash_attention(solution_path: str) -> EvaluationResult:
    """Evaluate flash attention submission."""

    import importlib.util
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    solution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution)

    metrics = {}
    errors = []
    warnings = []

    # === CORRECTNESS TESTS ===

    test_cases = [
        # (batch, seq, heads, dim, block_q, block_kv)
        (1, 64, 4, 32, 32, 32),      # small
        (2, 512, 8, 64, 128, 64),    # medium
        (1, 2048, 8, 64, 128, 64),   # large
        (2, 256, 4, 64, 64, 128),    # different blocks
        # HIDDEN test cases (not shown to student)
        (1, 1024, 16, 64, 128, 64),  # more heads
        (4, 768, 8, 128, 128, 64),   # larger dim
    ]

    correctness_results = []

    for tc in test_cases:
        batch, seq, heads, dim, block_q, block_kv = tc
        key = jax.random.PRNGKey(hash(tc) % 2**31)

        q = jax.random.normal(key, (batch, seq, heads, dim))
        k = jax.random.normal(key, (batch, seq, heads, dim))
        v = jax.random.normal(key, (batch, seq, heads, dim))

        try:
            # Reference
            from flash_attention_starter import standard_attention
            ref = standard_attention(q, k, v)

            # Student solution
            out, _ = solution.flash_attention_forward(q, k, v, block_q=block_q, block_kv=block_kv)

            max_diff = float(jnp.max(jnp.abs(out - ref)))
            correctness_results.append({
                'test': tc,
                'max_diff': max_diff,
                'pass_1e4': max_diff < 1e-4,
                'pass_1e5': max_diff < 1e-5,
            })

        except Exception as e:
            errors.append(f"Test {tc} failed: {str(e)}")
            correctness_results.append({
                'test': tc,
                'max_diff': float('inf'),
                'pass_1e4': False,
                'pass_1e5': False,
            })

    metrics['correctness'] = correctness_results
    metrics['all_pass_1e4'] = all(r['pass_1e4'] for r in correctness_results)
    metrics['all_pass_1e5'] = all(r['pass_1e5'] for r in correctness_results)

    # === BENCHMARK TESTS ===

    benchmark_seqs = [512, 1024, 2048, 4096]
    speedups = []

    for seq_len in benchmark_seqs:
        key = jax.random.PRNGKey(seq_len)
        q = jax.random.normal(key, (2, seq_len, 8, 64))
        k = jax.random.normal(key, (2, seq_len, 8, 64))
        v = jax.random.normal(key, (2, seq_len, 8, 64))

        # Warmup
        for _ in range(3):
            _ = standard_attention(q, k, v).block_until_ready()
            _ = solution.flash_attention_forward(q, k, v)[0].block_until_ready()

        # Benchmark
        n_runs = 10

        start = time.perf_counter()
        for _ in range(n_runs):
            _ = standard_attention(q, k, v).block_until_ready()
        naive_time = (time.perf_counter() - start) / n_runs

        start = time.perf_counter()
        for _ in range(n_runs):
            _ = solution.flash_attention_forward(q, k, v)[0].block_until_ready()
        flash_time = (time.perf_counter() - start) / n_runs

        speedup = naive_time / flash_time
        speedups.append({'seq': seq_len, 'speedup': speedup})

    metrics['benchmarks'] = speedups
    metrics['max_speedup'] = max(s['speedup'] for s in speedups)
    metrics['speedup_4096'] = next((s['speedup'] for s in speedups if s['seq'] == 4096), 0)

    # === GRADIENT TESTS (for Excellent) ===

    has_backward = hasattr(solution, 'flash_attention_backward')
    backward_correct = False

    if has_backward:
        try:
            key = jax.random.PRNGKey(0)
            q = jax.random.normal(key, (1, 64, 2, 32))
            k = jax.random.normal(key, (1, 64, 2, 32))
            v = jax.random.normal(key, (1, 64, 2, 32))

            def loss_fn(q, k, v):
                out, _ = solution.flash_attention_forward(q, k, v)
                return jnp.sum(out ** 2)

            grad_q, grad_k, grad_v = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
            backward_correct = all(jnp.all(jnp.isfinite(g)) for g in [grad_q, grad_k, grad_v])
        except:
            backward_correct = False

    metrics['has_backward'] = has_backward
    metrics['backward_correct'] = backward_correct

    # === DETERMINE GRADE ===

    if not metrics['all_pass_1e4']:
        grade = "Fail"
        passed = False
    elif metrics['all_pass_1e5'] and metrics['speedup_4096'] > 5.0 and backward_correct:
        grade = "Excellent"
        passed = True
    elif metrics['all_pass_1e5'] and metrics['max_speedup'] > 2.0:
        grade = "Good"
        passed = True
    elif metrics['all_pass_1e4'] and metrics['max_speedup'] > 1.0:
        grade = "Pass"
        passed = True
    else:
        grade = "Fail"
        passed = False

    # Compute score
    score = 0
    if metrics['all_pass_1e4']:
        score += 40
    if metrics['all_pass_1e5']:
        score += 20
    score += min(20, metrics['max_speedup'] * 4)
    if backward_correct:
        score += 20

    return EvaluationResult(
        passed=passed,
        grade=grade,
        score=min(100, score),
        metrics=metrics,
        errors=errors,
        warnings=warnings,
    )


def evaluate_speculative_decode(solution_path: str) -> EvaluationResult:
    """Evaluate speculative decoding submission."""

    import importlib.util
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    solution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution)

    metrics = {}
    errors = []
    warnings = []

    # === DISTRIBUTION EQUIVALENCE TESTS ===

    def make_test_models(temp_draft, temp_target, vocab_size=100, seed=42):
        key = jax.random.PRNGKey(seed)
        base = jax.random.normal(key, (vocab_size,))

        def draft(tokens):
            return jnp.tile(base / temp_draft, (len(tokens), 1))

        def target(tokens):
            return jnp.tile(base / temp_target, (len(tokens), 1))

        return draft, target, vocab_size

    test_configs = [
        # (draft_temp, target_temp, expected_acceptance)
        (1.0, 0.8, 0.6),    # similar distributions
        (1.2, 0.7, 0.4),    # more different
        (1.0, 1.0, 0.9),    # nearly identical
    ]

    distribution_results = []

    for draft_temp, target_temp, _ in test_configs:
        draft_model, target_model, vocab_size = make_test_models(draft_temp, target_temp)
        prompt = jnp.array([0])
        n_samples = 500
        max_tokens = 20

        from collections import Counter

        spec_first = Counter()
        target_first = Counter()

        for i in range(n_samples):
            # Speculative
            key = jax.random.PRNGKey(i)
            try:
                tokens, _ = solution.speculative_decode(
                    prompt, draft_model, target_model,
                    gamma=5, max_new_tokens=max_tokens, key=key
                )
                spec_first[int(tokens[0])] += 1
            except Exception as e:
                errors.append(f"Spec decode failed: {e}")
                continue

            # Target-only (reference)
            from speculative_decode_starter import autoregressive_decode
            key = jax.random.PRNGKey(i + 100000)
            tokens = autoregressive_decode(prompt, target_model, max_tokens, key)
            target_first[int(tokens[0])] += 1

        # Compute KL divergence
        kl = 0
        for token in range(vocab_size):
            p = (spec_first.get(token, 0) + 1) / (n_samples + vocab_size)
            q = (target_first.get(token, 0) + 1) / (n_samples + vocab_size)
            kl += p * jnp.log(p / q)

        distribution_results.append({
            'config': (draft_temp, target_temp),
            'kl_div': float(kl),
            'pass_0.05': kl < 0.05,
            'pass_0.005': kl < 0.005,
        })

    metrics['distribution'] = distribution_results
    metrics['all_kl_pass'] = all(r['pass_0.05'] for r in distribution_results)
    metrics['all_kl_excellent'] = all(r['pass_0.005'] for r in distribution_results)

    # === PERFORMANCE METRICS ===

    draft_model, target_model, _ = make_test_models(1.0, 0.8)
    prompt = jnp.array([0])

    acceptance_rates = []
    speedup_proxies = []  # tokens_per_target_call

    for seed in range(10):
        key = jax.random.PRNGKey(seed)
        try:
            _, stats = solution.speculative_decode(
                prompt, draft_model, target_model,
                gamma=5, max_new_tokens=100, key=key
            )
            acceptance_rates.append(stats['acceptance_rate'])
            speedup_proxies.append(stats['tokens_per_target_call'])
        except Exception as e:
            errors.append(f"Stats collection failed: {e}")

    metrics['avg_acceptance'] = sum(acceptance_rates) / len(acceptance_rates) if acceptance_rates else 0
    metrics['avg_tokens_per_call'] = sum(speedup_proxies) / len(speedup_proxies) if speedup_proxies else 0

    # Speedup estimate: tokens_per_call / 1 (vs autoregressive 1 token/call)
    # Real speedup depends on draft/target cost ratio
    metrics['speedup_estimate'] = metrics['avg_tokens_per_call']

    # === DETERMINE GRADE ===

    if not metrics['all_kl_pass']:
        grade = "Fail"
        passed = False
    elif metrics['all_kl_excellent'] and metrics['avg_acceptance'] > 0.7 and metrics['speedup_estimate'] > 2.5:
        grade = "Excellent"
        passed = True
    elif metrics['all_kl_pass'] and metrics['avg_acceptance'] > 0.6 and metrics['speedup_estimate'] > 1.8:
        grade = "Good"
        passed = True
    elif metrics['all_kl_pass'] and metrics['avg_acceptance'] > 0.5 and metrics['speedup_estimate'] > 1.3:
        grade = "Pass"
        passed = True
    else:
        grade = "Fail"
        passed = False

    # Compute score
    score = 0
    if metrics['all_kl_pass']:
        score += 40
    if metrics['all_kl_excellent']:
        score += 20
    score += min(20, metrics['avg_acceptance'] * 25)
    score += min(20, (metrics['speedup_estimate'] - 1) * 10)

    return EvaluationResult(
        passed=passed,
        grade=grade,
        score=min(100, score),
        metrics=metrics,
        errors=errors,
        warnings=warnings,
    )
```

### 5.2 anti-gaming checks

```python
def run_antigaming_checks_flash(solution_module) -> dict:
    """Detect cheating strategies for flash attention."""

    import inspect
    source = inspect.getsource(solution_module.flash_attention_forward)

    results = {
        'trivial_solution': False,
        'calls_standard_attention': False,
        'hardcoded_outputs': False,
        'no_block_iteration': False,
    }

    # Check 1: Calling standard attention internally
    if 'standard_attention' in source or 'jnp.einsum("bqhd,bkhd->bhqk"' in source.replace(' ', ''):
        # einsum pattern from standard attention
        # Need to check if it's applied to full matrices or blocks
        if 'block' not in source.lower():
            results['calls_standard_attention'] = True

    # Check 2: No block iteration (just wrapping naive)
    if 'for' not in source and 'scan' not in source and 'while' not in source:
        results['no_block_iteration'] = True

    # Check 3: Test with unusual input shapes
    key = jax.random.PRNGKey(999)
    q1 = jax.random.normal(key, (1, 100, 4, 64))  # seq=100 not typical
    k1 = jax.random.normal(key, (1, 100, 4, 64))
    v1 = jax.random.normal(key, (1, 100, 4, 64))

    q2 = jax.random.normal(key, (1, 101, 4, 64))  # seq=101 odd number
    k2 = jax.random.normal(key, (1, 101, 4, 64))
    v2 = jax.random.normal(key, (1, 101, 4, 64))

    try:
        out1, _ = solution_module.flash_attention_forward(q1, k1, v1)
        out2, _ = solution_module.flash_attention_forward(q2, k2, v2)

        # If outputs are identical shape regardless of input, suspicious
        if out1.shape == out2.shape:
            results['hardcoded_outputs'] = True
    except:
        pass

    return results


def run_antigaming_checks_speculative(solution_module) -> dict:
    """Detect cheating strategies for speculative decoding."""

    import inspect
    source = inspect.getsource(solution_module.speculative_decode)

    results = {
        'trivial_solution': False,
        'ignores_draft': False,
        'always_accepts': False,
        'always_rejects': False,
    }

    # Check 1: Ignores draft model (just calls target)
    if 'draft_model_fn' not in source or source.count('target_model_fn') > source.count('draft_model_fn') * 3:
        results['ignores_draft'] = True

    # Check 2: Always accepts (no rejection logic)
    if 'reject' not in source.lower() and 'accept' not in source.lower():
        if 'min(1' not in source and 'minimum' not in source:
            results['always_accepts'] = True

    # Check 3: Test with mismatched models
    vocab_size = 100

    # Draft always predicts token 0
    def draft_always_0(tokens):
        logits = jnp.full((len(tokens), vocab_size), -10.0)
        return logits.at[:, 0].set(10.0)

    # Target always predicts token 50
    def target_always_50(tokens):
        logits = jnp.full((len(tokens), vocab_size), -10.0)
        return logits.at[:, 50].set(10.0)

    key = jax.random.PRNGKey(0)
    prompt = jnp.array([0])

    try:
        tokens, stats = solution_module.speculative_decode(
            prompt, draft_always_0, target_always_50,
            gamma=5, max_new_tokens=20, key=key
        )

        # If acceptance rate is very high with mismatched models, suspicious
        if stats['acceptance_rate'] > 0.5:
            results['always_accepts'] = True

        # If all tokens are 0 (draft token), rejection sampling is broken
        if jnp.all(tokens == 0):
            results['trivial_solution'] = True

    except:
        pass

    return results
```

---

## 6. reference solutions

### 6.1 flash attention reference

```python
# reference_flash_attention.py
"""
Reference implementation of Flash Attention 2.
This is the "gold standard" for evaluation.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from functools import partial

@partial(jax.jit, static_argnums=(4, 5))
def flash_attention_forward_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: Optional[jax.Array] = None,
    block_q: int = 128,
    block_kv: int = 64,
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
    """
    Reference flash attention implementation.

    Uses jax.lax.scan for efficient block iteration.
    Handles arbitrary sequence lengths with padding.
    Supports causal masking.
    """
    batch, seq_q, num_heads, head_dim = q.shape
    _, seq_kv, _, _ = k.shape
    scale = head_dim ** -0.5

    # Pad to block multiples
    num_blocks_q = (seq_q + block_q - 1) // block_q
    num_blocks_kv = (seq_kv + block_kv - 1) // block_kv

    pad_q = num_blocks_q * block_q - seq_q
    pad_kv = num_blocks_kv * block_kv - seq_kv

    q_padded = jnp.pad(q, ((0, 0), (0, pad_q), (0, 0), (0, 0)))
    k_padded = jnp.pad(k, ((0, 0), (0, pad_kv), (0, 0), (0, 0)))
    v_padded = jnp.pad(v, ((0, 0), (0, pad_kv), (0, 0), (0, 0)))

    # Reshape to blocks: [batch, num_blocks, block_size, heads, dim]
    q_blocks = q_padded.reshape(batch, num_blocks_q, block_q, num_heads, head_dim)
    k_blocks = k_padded.reshape(batch, num_blocks_kv, block_kv, num_heads, head_dim)
    v_blocks = v_padded.reshape(batch, num_blocks_kv, block_kv, num_heads, head_dim)

    def process_q_block(q_idx):
        """Process single Q block against all KV blocks."""
        q_block = q_blocks[:, q_idx]  # [batch, block_q, heads, dim]

        def inner_loop(carry, kv_idx):
            o_acc, m_acc, l_acc = carry

            k_block = k_blocks[:, kv_idx]
            v_block = v_blocks[:, kv_idx]

            # Compute scores: [batch, block_q, heads, block_kv]
            scores = jnp.einsum("bqhd,bkhd->bqhk", q_block, k_block) * scale

            # Apply mask if provided
            if mask is not None:
                q_start = q_idx * block_q
                kv_start = kv_idx * block_kv
                mask_block = mask[:, :, q_start:q_start+block_q, kv_start:kv_start+block_kv]
                # Transpose mask to [batch, block_q, heads, block_kv]
                mask_block = mask_block.transpose(0, 2, 1, 3)
                scores = jnp.where(mask_block, scores, -1e9)

            # Online softmax
            m_block = jnp.max(scores, axis=-1)  # [batch, block_q, heads]
            m_new = jnp.maximum(m_acc, m_block)

            exp_old = jnp.exp(m_acc - m_new)
            exp_new = jnp.exp(scores - m_new[..., None])

            l_new = exp_old * l_acc + jnp.sum(exp_new, axis=-1)

            # Update output
            o_new = (exp_old[..., None] * l_acc[..., None] * o_acc +
                     jnp.einsum("bqhk,bkhd->bqhd", exp_new, v_block)) / (l_new[..., None] + 1e-10)

            return (o_new, m_new, l_new), None

        # Initialize
        o_init = jnp.zeros((batch, block_q, num_heads, head_dim))
        m_init = jnp.full((batch, block_q, num_heads), -1e9)
        l_init = jnp.zeros((batch, block_q, num_heads))

        (o_final, m_final, l_final), _ = jax.lax.scan(
            inner_loop,
            (o_init, m_init, l_init),
            jnp.arange(num_blocks_kv)
        )

        return o_final, m_final, l_final

    # Process all Q blocks using vmap
    o_all, m_all, l_all = jax.vmap(process_q_block)(jnp.arange(num_blocks_q))

    # Reshape: [num_blocks_q, batch, block_q, heads, dim] -> [batch, seq, heads, dim]
    o_all = o_all.transpose(1, 0, 2, 3, 4)
    o_all = o_all.reshape(batch, num_blocks_q * block_q, num_heads, head_dim)

    # Remove padding
    output = o_all[:, :seq_q]

    # Logsumexp for backward
    m_all = m_all.transpose(1, 0, 2, 3)
    l_all = l_all.transpose(1, 0, 2, 3)
    logsumexp = (m_all + jnp.log(l_all + 1e-10)).reshape(batch, -1, num_heads)[:, :seq_q]

    return output, (logsumexp, m_all.reshape(batch, -1, num_heads)[:, :seq_q])


@partial(jax.jit, static_argnums=(6, 7))
def flash_attention_backward_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    output: jax.Array,
    logsumexp: jax.Array,
    grad_output: jax.Array,
    block_q: int = 128,
    block_kv: int = 64,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Reference backward pass.
    Recomputes attention during backward to save memory.
    """
    # Use JAX autodiff for reference
    def forward_fn(q, k, v):
        out, _ = flash_attention_forward_reference(q, k, v, block_q=block_q, block_kv=block_kv)
        return jnp.sum(out * grad_output)

    grad_q, grad_k, grad_v = jax.grad(forward_fn, argnums=(0, 1, 2))(q, k, v)
    return grad_q, grad_k, grad_v
```

### 6.2 speculative decoding reference

```python
# reference_speculative_decode.py
"""
Reference implementation of Speculative Decoding.
This is the "gold standard" for evaluation.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Dict
from functools import partial


def speculative_decode_reference(
    prompt_tokens: jax.Array,
    draft_model_fn: Callable,
    target_model_fn: Callable,
    gamma: int = 5,
    max_new_tokens: int = 100,
    key: jax.Array = None,
) -> Tuple[jax.Array, Dict[str, float]]:
    """
    Reference speculative decoding implementation.

    Provably maintains target distribution via rejection sampling.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    tokens = prompt_tokens
    generated = []

    accepted_count = 0
    total_draft_tokens = 0
    target_calls = 0

    while len(generated) < max_new_tokens:
        key, k_draft, k_verify = jax.random.split(key, 3)

        remaining = max_new_tokens - len(generated)
        current_gamma = min(gamma, remaining)

        # === DRAFT PHASE ===
        draft_tokens = []
        draft_probs = []
        current_seq = tokens

        draft_keys = jax.random.split(k_draft, current_gamma)

        for i in range(current_gamma):
            logits = draft_model_fn(current_seq)
            last_logits = logits[-1]
            probs = jax.nn.softmax(last_logits)

            token = jax.random.categorical(draft_keys[i], last_logits)

            draft_tokens.append(int(token))
            draft_probs.append(probs)
            current_seq = jnp.concatenate([current_seq, token[None]])

        draft_tokens = jnp.array(draft_tokens)
        draft_probs = jnp.stack(draft_probs)

        # === VERIFY PHASE ===
        verify_seq = jnp.concatenate([tokens, draft_tokens])
        target_logits = target_model_fn(verify_seq)
        target_calls += 1

        # Target probs for positions len(tokens)-1 to len(tokens)+gamma-1
        # (position i predicts token i+1)
        start_pos = len(tokens) - 1
        target_probs = jax.nn.softmax(target_logits[start_pos:start_pos + current_gamma + 1], axis=-1)

        # === REJECTION SAMPLING ===
        verify_keys = jax.random.split(k_verify, current_gamma + 1)
        n_accepted = 0

        for i in range(current_gamma):
            draft_token = draft_tokens[i]
            q_x = draft_probs[i, draft_token]
            p_x = target_probs[i, draft_token]

            # Acceptance probability
            accept_prob = jnp.minimum(1.0, p_x / (q_x + 1e-10))

            r = jax.random.uniform(verify_keys[i])

            if r < accept_prob:
                # Accept
                generated.append(int(draft_token))
                tokens = jnp.concatenate([tokens, draft_token[None]])
                n_accepted += 1
                accepted_count += 1
            else:
                # Reject: sample from adjusted distribution
                adjusted = jnp.maximum(0, target_probs[i] - draft_probs[i])
                adjusted = adjusted / (jnp.sum(adjusted) + 1e-10)

                new_token = jax.random.categorical(verify_keys[i], jnp.log(adjusted + 1e-10))
                generated.append(int(new_token))
                tokens = jnp.concatenate([tokens, new_token[None]])
                break

            total_draft_tokens += 1

            if len(generated) >= max_new_tokens:
                break

        # === BONUS TOKEN ===
        if n_accepted == current_gamma and len(generated) < max_new_tokens:
            bonus_probs = target_probs[-1]
            bonus_token = jax.random.categorical(verify_keys[-1], jnp.log(bonus_probs + 1e-10))
            generated.append(int(bonus_token))
            tokens = jnp.concatenate([tokens, bonus_token[None]])

    # Compute stats
    stats = {
        'acceptance_rate': accepted_count / max(1, total_draft_tokens),
        'tokens_per_target_call': len(generated) / max(1, target_calls),
        'num_target_calls': target_calls,
    }

    return jnp.array(generated[:max_new_tokens]), stats
```

---

## 7. grading summary

### 7.1 flash attention grading

| level | correctness | speedup | backward | score |
|-------|-------------|---------|----------|-------|
| Pass | atol ≤ 1e-4 | >1x anywhere | not required | 60-69 |
| Good | atol ≤ 1e-5 | >2x on seq≥1024 | not required | 70-84 |
| Excellent | atol ≤ 1e-5 | >5x on seq=4096 | correct gradients | 85-100 |
| Fail | atol > 1e-4 OR no speedup | - | - | 0-59 |

### 7.2 speculative decoding grading

| level | KL divergence | acceptance rate | speedup proxy | score |
|-------|---------------|-----------------|---------------|-------|
| Pass | < 0.05 | > 50% | > 1.3x | 60-69 |
| Good | < 0.01 | > 60% | > 1.8x | 70-84 |
| Excellent | < 0.005 | > 70% | > 2.5x | 85-100 |
| Fail | ≥ 0.05 OR acceptance < 50% | - | - | 0-59 |

---

## 8. feedback examples

### 8.1 flash attention – passed

```
═══════════════════════════════════════════════════════════════
                    EVALUATION RESULTS: FLASH ATTENTION
═══════════════════════════════════════════════════════════════

Status: PASSED

Grade: Good
Score: 78/100

Correctness:
  • All tests pass at atol=1e-5
  • Max difference: 3.2e-6 (test: batch=2, seq=512)

Performance:
  • Seq 512:  1.8x speedup
  • Seq 1024: 3.2x speedup
  • Seq 2048: 4.1x speedup
  • Seq 4096: 4.6x speedup (need >5x for Excellent)

Backward Pass:
  • Not implemented (optional for Excellent)

Breakdown:
  ├─ Correctness (1e-4):  40/40 points
  ├─ Correctness (1e-5):  20/20 points
  ├─ Speedup:             18/20 points (4.6x best)
  └─ Backward:            0/20 points

Suggestions for Excellent grade:
  • Optimize block sizes for seq=4096 (try block_kv=128)
  • Implement backward pass for full score
  • Consider using Pallas for TPU-specific optimizations

═══════════════════════════════════════════════════════════════
```

### 8.2 speculative decoding – failed

```
═══════════════════════════════════════════════════════════════
                    EVALUATION RESULTS: SPECULATIVE DECODING
═══════════════════════════════════════════════════════════════

Status: FAILED

Grade: Fail
Score: 42/100

Distribution Equivalence:
  • Config (1.0, 0.8): KL = 0.087 (threshold: 0.05)
  • Config (1.2, 0.7): KL = 0.124 (threshold: 0.05)
  • Config (1.0, 1.0): KL = 0.003 (threshold: 0.05)

Performance:
  • Average acceptance: 68%
  • Tokens per target call: 2.3x

Errors:
  ✗ Distribution mismatch: output does not match target-only sampling
  ✗ KL divergence exceeds threshold in 2/3 test configs

Debug Hints:
  • Check rejection sampling logic: when p(x) < q(x), you must resample
  • Verify adjusted distribution: max(0, p - q) normalized
  • Common bug: using p/q instead of min(1, p/q) for accept probability
  • Common bug: not properly normalizing the adjusted distribution

Anti-Gaming Check:
  • No issues detected

═══════════════════════════════════════════════════════════════
```

---

## 9. full pipeline visualization

```
┌─────────────────────────────────────────────────────────────┐
│                     STUDENT WORKFLOW                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. CHOOSE PAPER                                            │
│     [ ] Flash Attention 2                                   │
│     [ ] Speculative Decoding                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. READ PAPER PSEUDOCODE                                   │
│     - Understand core algorithm                             │
│     - Identify key invariants                               │
│     - Note numerical stability requirements                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. IMPLEMENT STEP BY STEP                                  │
│                                                             │
│  Flash Attention:                                           │
│    3a. Online softmax update function                       │
│    3b. Block iteration over Q                               │
│    3c. Inner loop over KV blocks                            │
│    3d. Handle padding and masking                           │
│    3e. (Optional) Backward pass                             │
│                                                             │
│  Speculative Decoding:                                      │
│    3a. Draft token generation                               │
│    3b. Target verification (single forward)                 │
│    3c. Rejection sampling logic                             │
│    3d. Bonus token handling                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. LOCAL TESTING                                           │
│     - Run provided test cases                               │
│     - Check numerical accuracy                              │
│     - Verify performance metrics                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  5. SUBMIT                                                  │
│     - flash_attention.py OR speculative_decode.py          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  A. CORRECTNESS TESTS                                       │
│     Flash: compare vs standard_attention at atol=1e-4/1e-5 │
│     Spec: KL divergence vs target-only sampling            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  B. PERFORMANCE BENCHMARKS                                  │
│     Flash: speedup at various sequence lengths              │
│     Spec: acceptance rate, tokens per target call          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  C. ANTI-GAMING CHECKS                                      │
│     - Hidden test inputs                                    │
│     - Check for hardcoded solutions                         │
│     - Verify actual algorithm implementation                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  D. GRADE ASSIGNMENT                                        │
│     Fail / Pass / Good / Excellent                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  E. FEEDBACK                                                │
│     - Detailed metrics                                      │
│     - Debug hints for failures                              │
│     - Suggestions for improvement                           │
└─────────────────────────────────────────────────────────────┘
```

---

*end of E05 paper implementation detailed pipeline specification*
