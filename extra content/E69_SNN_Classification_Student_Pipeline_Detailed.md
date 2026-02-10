# E69: spiking neural network classification – detailed student pipeline

> complete specification of student journey from start to evaluation
> tasks: E69-T1 (LIF neuron implementation), E69-T2 (train SNN on SHD)

---

## 1. what student sees at start

### 1.1 task description (in interface)

```
TASK: Spiking Neural Network for Temporal Audio Classification

Spiking Neural Networks (SNNs) are "third generation" neural networks that
communicate via discrete spikes rather than continuous activations. Unlike
traditional ANNs, SNNs process information in the temporal domain – the TIMING
of spikes carries information, not just their rate.

You will work with the Spiking Heidelberg Digits (SHD) dataset:
- 20-class spoken digit classification (0-9 in German + English)
- 700 cochlear channels (spike-encoded audio)
- ~8.3k training samples, ~2k test samples

Your goals:
1. E69-T1: Implement a Leaky Integrate-and-Fire (LIF) neuron from scratch
2. E69-T2: Build and train a 2-layer SNN to classify spoken digits

Why SNNs matter:
- Temporal information is CRUCIAL (shuffling time destroys performance)
- Energy efficient (sparse spike communication)
- Biologically plausible (real neurons spike!)

Time limit: 4 hours
Memory limit: 8 GB
Framework: JAX + Spyx (provided)
```

### 1.2 starter code (what student receives)

```python
# starter_code.py
"""
E69: Spiking Neural Network Classification
==========================================

Part 1 (E69-T1): Implement the LIF neuron dynamics
Part 2 (E69-T2): Train an SNN on the SHD dataset

You have access to:
- JAX for automatic differentiation
- Spyx for SNN utilities (data loading, surrogate gradients)
- optax for optimization
- numpy for array operations

The LIF neuron follows these dynamics:
    τ_m * dV/dt = -(V - V_rest) + R * I

In discrete time (Euler method):
    V[t+1] = V[t] + dt/τ_m * (-(V[t] - V_rest) + R * I[t])

When V >= V_th: emit spike, reset V to V_reset
"""

import jax
import jax.numpy as jnp
from jax import random
import optax
from functools import partial
from typing import Tuple, NamedTuple

# Spyx imports for data loading
import spyx
import spyx.data

# ============== CONSTANTS ==============

# LIF neuron parameters
TAU_M = 20.0      # membrane time constant [ms]
V_REST = 0.0      # resting potential [mV] (normalized)
V_TH = 1.0        # spike threshold [mV] (normalized)
V_RESET = 0.0     # reset potential [mV]
DT = 1.0          # time step [ms]

# Network parameters
N_INPUT = 700     # SHD cochlear channels
N_HIDDEN = 256    # hidden layer neurons
N_OUTPUT = 20     # number of classes

# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
N_EPOCHS = 50


# ============== PART 1: LIF NEURON (E69-T1) ==============

class LIFState(NamedTuple):
    """State of a LIF neuron layer."""
    V: jnp.ndarray      # membrane potential
    spike: jnp.ndarray  # spike output (0 or 1)


def init_lif_state(batch_size: int, n_neurons: int, key: random.PRNGKey) -> LIFState:
    """Initialize LIF neuron state.

    Args:
        batch_size: number of samples in batch
        n_neurons: number of neurons in layer
        key: JAX random key

    Returns:
        Initial LIF state with V at rest, no spikes
    """
    V = jnp.full((batch_size, n_neurons), V_REST)
    spike = jnp.zeros((batch_size, n_neurons))
    return LIFState(V=V, spike=spike)


def lif_step(state: LIFState, I: jnp.ndarray) -> LIFState:
    """
    Single timestep update for LIF neuron.

    Implements the discrete-time LIF dynamics:
    1. Membrane potential update (leaky integration)
    2. Spike generation (threshold crossing)
    3. Reset mechanism

    Args:
        state: current LIF state (V, spike)
        I: input current [batch_size, n_neurons]

    Returns:
        new LIF state after one timestep

    Equations:
        dV/dt = (-(V - V_rest) + I) / τ_m
        V[t+1] = V[t] + dt * dV/dt
        spike = V[t+1] >= V_th
        V[t+1] = V_reset if spike else V[t+1]
    """
    # TODO: Implement this function
    raise NotImplementedError("E69-T1: Implement LIF neuron dynamics")


# ============== SURROGATE GRADIENT ==============

def superspike_surrogate(x: jnp.ndarray, beta: float = 10.0) -> jnp.ndarray:
    """
    SuperSpike surrogate gradient function.

    Forward pass: Heaviside step function (0 if x < 0, 1 if x >= 0)
    Backward pass: smooth approximation for gradient flow

    The surrogate gradient is: 1 / (1 + beta * |x|)^2

    Args:
        x: input (V - V_th)
        beta: sharpness parameter (higher = sharper)

    Returns:
        spike output (0 or 1) with gradient-friendly backward pass
    """
    # Forward: Heaviside
    spike = jnp.where(x >= 0, 1.0, 0.0)
    return spike


@jax.custom_vjp
def spike_fn(x: jnp.ndarray) -> jnp.ndarray:
    """Spike function with custom gradient."""
    return jnp.where(x >= 0, 1.0, 0.0)


def spike_fn_fwd(x):
    return spike_fn(x), x


def spike_fn_bwd(res, g):
    x = res
    # SuperSpike surrogate gradient
    beta = 10.0
    grad = 1.0 / (1.0 + beta * jnp.abs(x)) ** 2
    return (g * grad,)


spike_fn.defvjp(spike_fn_fwd, spike_fn_bwd)


# ============== PART 2: SNN NETWORK (E69-T2) ==============

class SNNParams(NamedTuple):
    """Parameters for 2-layer SNN."""
    W1: jnp.ndarray  # input -> hidden weights
    W2: jnp.ndarray  # hidden -> output weights


def init_snn_params(key: random.PRNGKey) -> SNNParams:
    """Initialize SNN weights with Xavier initialization."""
    key1, key2 = random.split(key)

    # Xavier/Glorot initialization
    W1 = random.normal(key1, (N_INPUT, N_HIDDEN)) * jnp.sqrt(2.0 / (N_INPUT + N_HIDDEN))
    W2 = random.normal(key2, (N_HIDDEN, N_OUTPUT)) * jnp.sqrt(2.0 / (N_HIDDEN + N_OUTPUT))

    return SNNParams(W1=W1, W2=W2)


def snn_forward(
    params: SNNParams,
    spikes_in: jnp.ndarray,  # [T, batch, N_INPUT]
    key: random.PRNGKey
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Forward pass through 2-layer SNN.

    Architecture:
        input spikes (700) -> LIF (256) -> LIF (20) -> output

    Args:
        params: SNN weights
        spikes_in: input spike trains [T, batch_size, N_INPUT]
        key: random key

    Returns:
        output_spikes: [T, batch_size, N_OUTPUT]
        membrane_trace: [T, batch_size, N_OUTPUT] for loss computation
    """
    # TODO: Implement this function
    raise NotImplementedError("E69-T2: Implement SNN forward pass")


# ============== DATA LOADING ==============

def load_shd_data(batch_size: int = BATCH_SIZE):
    """
    Load SHD dataset using Spyx data utilities.

    Returns:
        train_loader: training data iterator
        test_loader: test data iterator

    Data format:
        - x: sparse spike trains [T, batch, 700]
        - y: labels [batch] (0-19)
    """
    # Load SHD dataset
    train_ds, test_ds = spyx.data.shd_loader(
        batch_size=batch_size,
        cache=True
    )
    return train_ds, test_ds


# ============== LOSS FUNCTION ==============

def cross_entropy_loss(
    membrane: jnp.ndarray,  # [T, batch, N_OUTPUT]
    targets: jnp.ndarray    # [batch]
) -> jnp.ndarray:
    """
    Cross-entropy loss on membrane potentials.

    We sum membrane potentials over time and apply softmax,
    which gives a rate-coded readout.

    Args:
        membrane: output membrane potentials [T, batch, N_OUTPUT]
        targets: class labels [batch]

    Returns:
        scalar loss value
    """
    # Sum over time dimension
    logits = jnp.sum(membrane, axis=0)  # [batch, N_OUTPUT]

    # Cross-entropy
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(targets, N_OUTPUT)
    loss = -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))

    return loss


# ============== TRAINING LOOP ==============

def train_step(
    params: SNNParams,
    opt_state,
    optimizer,
    x_batch: jnp.ndarray,
    y_batch: jnp.ndarray,
    key: random.PRNGKey
) -> Tuple[SNNParams, any, float]:
    """
    Single training step.

    Args:
        params: current SNN parameters
        opt_state: optimizer state
        optimizer: optax optimizer
        x_batch: input spikes [T, batch, N_INPUT]
        y_batch: labels [batch]
        key: random key

    Returns:
        new_params, new_opt_state, loss_value
    """
    def loss_fn(params):
        _, membrane = snn_forward(params, x_batch, key)
        return cross_entropy_loss(membrane, y_batch)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss


def evaluate(
    params: SNNParams,
    test_loader,
    key: random.PRNGKey
) -> float:
    """
    Evaluate accuracy on test set.

    Args:
        params: trained SNN parameters
        test_loader: test data iterator
        key: random key

    Returns:
        test accuracy (0-1)
    """
    correct = 0
    total = 0

    for x_batch, y_batch in test_loader:
        _, membrane = snn_forward(params, x_batch, key)
        logits = jnp.sum(membrane, axis=0)
        preds = jnp.argmax(logits, axis=-1)
        correct += jnp.sum(preds == y_batch)
        total += len(y_batch)

    return float(correct / total)


# ============== TEST YOUR SOLUTION ==============

if __name__ == "__main__":
    key = random.PRNGKey(42)

    # Test E69-T1: LIF neuron
    print("Testing LIF neuron...")
    key, subkey = random.split(key)
    state = init_lif_state(batch_size=4, n_neurons=10, key=subkey)
    I = jnp.ones((4, 10)) * 0.15  # constant input current

    # Run for 100 timesteps
    V_trace = []
    spike_trace = []
    for t in range(100):
        state = lif_step(state, I)
        V_trace.append(state.V)
        spike_trace.append(state.spike)

    V_trace = jnp.stack(V_trace)
    spike_trace = jnp.stack(spike_trace)

    print(f"  Total spikes: {spike_trace.sum()}")
    print(f"  Max V reached: {V_trace.max():.3f}")

    # Test E69-T2: SNN forward pass
    print("\nTesting SNN forward pass...")
    key, subkey = random.split(key)
    params = init_snn_params(subkey)

    # Dummy input
    dummy_spikes = random.bernoulli(subkey, 0.1, (100, 4, N_INPUT)).astype(jnp.float32)

    key, subkey = random.split(key)
    out_spikes, membrane = snn_forward(params, dummy_spikes, subkey)

    print(f"  Output shape: {out_spikes.shape}")
    print(f"  Output spike rate: {out_spikes.mean():.4f}")
```

### 1.3 reference documentation (available to student)

```markdown
## LIF Neuron Dynamics

### Continuous-Time Equation

The membrane potential V of a LIF neuron evolves according to:

    τ_m * dV/dt = -(V - V_rest) + R * I(t)

where:
- τ_m: membrane time constant (typically 10-20 ms)
- V_rest: resting potential (often normalized to 0)
- R: membrane resistance (often absorbed into I)
- I(t): input current (synaptic + external)

### Discrete-Time Update (Euler Method)

For simulation with timestep dt:

    V[t+1] = V[t] + (dt / τ_m) * (-(V[t] - V_rest) + I[t])

Or equivalently using decay factor β = exp(-dt / τ_m):

    V[t+1] = β * V[t] + (1 - β) * V_rest + I[t]

### Spike Generation and Reset

When V >= V_th:
1. Emit spike: S[t] = 1
2. Reset: V[t] = V_reset

When V < V_th:
1. No spike: S[t] = 0
2. Continue integration

### Surrogate Gradients

The spike function S = Heaviside(V - V_th) has zero gradient almost everywhere.

Surrogate gradient replaces the backward pass derivative with a smooth function:

**SuperSpike**: ∂S/∂V ≈ 1 / (1 + β|V - V_th|)²

where β controls sharpness (typical: β = 10-25)


## SHD Dataset

### Format
- 20 classes: digits 0-9 in German and English
- 700 input channels (cochlear filterbank)
- Temporal sequences (variable length, ~1 second)
- Sparse spike encoding

### Statistics
- Training: 8,156 samples
- Test: 2,264 samples
- Chance accuracy: 5%
- Good baseline: 70-75%
- State-of-the-art: 85%+

### Loading with Spyx
```python
import spyx.data

train_loader, test_loader = spyx.data.shd_loader(
    batch_size=128,
    cache=True  # cache to disk for faster loading
)

for x, y in train_loader:
    # x: [T, batch, 700] sparse spike trains
    # y: [batch] labels (0-19)
    pass
```

## Network Architecture

Recommended 2-layer SNN:
```
Input (700) → LIF Layer (256) → LIF Layer (20) → Rate Readout
```

Readout: sum membrane potentials over time, apply softmax
- logits = sum(V_output, axis=time)
- probs = softmax(logits)
```

---

## 2. what student does (E69-T1: LIF neuron)

### step 1: understand membrane dynamics

student studies the LIF equation and decay:

```python
# continuous: τ_m * dV/dt = -(V - V_rest) + I
# we want to solve this numerically

# euler method: V[t+1] = V[t] + dt * dV/dt
# substituting: V[t+1] = V[t] + dt/τ_m * (-(V[t] - V_rest) + I[t])

# alternatively, define decay factor:
# β = exp(-dt / τ_m) ≈ 1 - dt/τ_m for small dt

# then: V[t+1] = β * V[t] + (1-β) * V_rest + (1-β) * I[t]
# simplified for V_rest=0: V[t+1] = β * V[t] + (1-β) * I[t]

# let's verify the decay constant
import jax.numpy as jnp

tau_m = 20.0  # ms
dt = 1.0      # ms

# exact decay
beta = jnp.exp(-dt / tau_m)  # ≈ 0.951

# approximate decay (euler)
beta_approx = 1 - dt / tau_m  # = 0.95

print(f"Exact β: {beta:.4f}")
print(f"Approx β: {beta_approx:.4f}")
# both ≈ 0.95, membrane decays 5% per timestep
```

### step 2: implement discrete update

```python
def lif_step(state: LIFState, I: jnp.ndarray) -> LIFState:
    """Single timestep update for LIF neuron."""

    V = state.V

    # decay factor (precomputed for efficiency)
    beta = jnp.exp(-DT / TAU_M)

    # membrane potential update (leaky integration)
    # V[t+1] = β * V[t] + (1-β) * V_rest + I[t]
    # with V_rest = 0: V[t+1] = β * V[t] + I[t]
    V_new = beta * V + (1 - beta) * V_REST + I

    # ... spike generation and reset (step 3-4)
```

### step 3: implement spike generation

```python
    # spike generation: check if V >= threshold
    # use surrogate gradient for backprop
    spike = spike_fn(V_new - V_TH)
```

### step 4: implement reset mechanism

```python
    # reset: if spiked, reset V to V_reset
    # this is a "hard reset" – V goes to V_reset regardless of overshoot
    V_new = jnp.where(spike > 0, V_RESET, V_new)

    return LIFState(V=V_new, spike=spike)
```

### step 5: complete implementation

```python
def lif_step(state: LIFState, I: jnp.ndarray) -> LIFState:
    """
    Single timestep update for LIF neuron.

    Implements:
    1. Leaky integration of membrane potential
    2. Spike generation via threshold crossing
    3. Hard reset after spike
    """
    V = state.V

    # decay factor
    beta = jnp.exp(-DT / TAU_M)

    # leaky integration
    V_new = beta * V + (1 - beta) * V_REST + I

    # spike generation with surrogate gradient
    spike = spike_fn(V_new - V_TH)

    # hard reset
    V_new = jnp.where(spike > 0, V_RESET, V_new)

    return LIFState(V=V_new, spike=spike)
```

### verifying with unit tests

```python
# test 1: decay without input
def test_decay():
    """V should decay toward V_rest when I=0."""
    key = random.PRNGKey(0)
    state = LIFState(
        V=jnp.array([[0.5]]),  # start above rest
        spike=jnp.array([[0.0]])
    )
    I = jnp.array([[0.0]])  # no input

    new_state = lif_step(state, I)

    # V should decay (get closer to 0)
    assert new_state.V[0, 0] < 0.5, "V should decay"
    assert new_state.V[0, 0] > 0.0, "V should not go negative"
    assert new_state.spike[0, 0] == 0, "Should not spike"

    print("PASS: decay test")


# test 2: spike generation
def test_spike():
    """V at threshold should produce spike and reset."""
    state = LIFState(
        V=jnp.array([[0.99]]),  # just below threshold
        spike=jnp.array([[0.0]])
    )
    I = jnp.array([[0.1]])  # enough to push over

    new_state = lif_step(state, I)

    assert new_state.spike[0, 0] == 1.0, "Should spike"
    assert new_state.V[0, 0] == V_RESET, "Should reset"

    print("PASS: spike test")


# test 3: integration
def test_integration():
    """Constant input should accumulate to spike."""
    key = random.PRNGKey(0)
    state = init_lif_state(batch_size=1, n_neurons=1, key=key)
    I = jnp.array([[0.2]])  # moderate input

    spike_times = []
    for t in range(100):
        state = lif_step(state, I)
        if state.spike[0, 0] > 0:
            spike_times.append(t)

    assert len(spike_times) > 0, "Should spike eventually"
    assert len(spike_times) < 50, "Should spike within 50 steps"

    print(f"PASS: integration test (spiked at times {spike_times})")


# run all tests
test_decay()
test_spike()
test_integration()
```

---

## 3. what student does (E69-T2: train on SHD)

### step 1: load SHD data

```python
import spyx.data

# load SHD dataset
train_loader, test_loader = spyx.data.shd_loader(
    batch_size=BATCH_SIZE,
    cache=True
)

# inspect data
for x, y in train_loader:
    print(f"Input shape: {x.shape}")   # [T, batch, 700]
    print(f"Labels shape: {y.shape}")   # [batch]
    print(f"Sequence length: {x.shape[0]}")
    print(f"Spike density: {x.mean():.4f}")
    print(f"Classes: {jnp.unique(y)}")
    break

# Output:
# Input shape: (100, 128, 700) or similar
# Labels shape: (128,)
# Sequence length: 100
# Spike density: 0.02 (sparse!)
# Classes: [0, 1, ..., 19]
```

### step 2: build 2-layer SNN

```python
def snn_forward(
    params: SNNParams,
    spikes_in: jnp.ndarray,
    key: random.PRNGKey
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Forward pass: input -> LIF -> LIF -> output
    """
    T, batch_size, _ = spikes_in.shape

    # initialize layer states
    key1, key2 = random.split(key)
    state1 = init_lif_state(batch_size, N_HIDDEN, key1)
    state2 = init_lif_state(batch_size, N_OUTPUT, key2)

    # collect outputs
    output_spikes = []
    membrane_trace = []

    # unroll through time
    for t in range(T):
        # layer 1: input -> hidden
        I1 = spikes_in[t] @ params.W1  # [batch, N_HIDDEN]
        state1 = lif_step(state1, I1)

        # layer 2: hidden -> output
        I2 = state1.spike @ params.W2  # [batch, N_OUTPUT]
        state2 = lif_step(state2, I2)

        output_spikes.append(state2.spike)
        membrane_trace.append(state2.V)

    return jnp.stack(output_spikes), jnp.stack(membrane_trace)
```

### step 3: implement surrogate gradient (already provided)

student verifies the surrogate gradient works:

```python
# test gradient flow
def test_gradient_flow():
    """Verify gradients flow through spike function."""
    key = random.PRNGKey(42)
    params = init_snn_params(key)

    # dummy batch
    dummy_x = random.bernoulli(key, 0.1, (50, 4, N_INPUT)).astype(jnp.float32)
    dummy_y = random.randint(key, (4,), 0, N_OUTPUT)

    def loss_fn(params):
        _, membrane = snn_forward(params, dummy_x, key)
        return cross_entropy_loss(membrane, dummy_y)

    # compute gradients
    grads = jax.grad(loss_fn)(params)

    # check gradients are not zero
    assert not jnp.allclose(grads.W1, 0), "W1 gradients should be non-zero"
    assert not jnp.allclose(grads.W2, 0), "W2 gradients should be non-zero"

    print(f"W1 grad norm: {jnp.linalg.norm(grads.W1):.4f}")
    print(f"W2 grad norm: {jnp.linalg.norm(grads.W2):.4f}")
    print("PASS: gradients flow correctly")


test_gradient_flow()
```

### step 4: training loop with optax

```python
def train_snn(n_epochs: int = N_EPOCHS):
    """Full training loop."""
    key = random.PRNGKey(42)

    # initialize
    key, init_key = random.split(key)
    params = init_snn_params(init_key)

    # optimizer
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)

    # data
    train_loader, test_loader = load_shd_data()

    # JIT compile training step
    @jax.jit
    def train_step_jit(params, opt_state, x, y, key):
        def loss_fn(params):
            _, membrane = snn_forward(params, x, key)
            return cross_entropy_loss(membrane, y)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # training loop
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in train_loader:
            key, step_key = random.split(key)
            params, opt_state, loss = train_step_jit(
                params, opt_state, x_batch, y_batch, step_key
            )
            epoch_loss += loss
            n_batches += 1

        # evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            key, eval_key = random.split(key)
            test_acc = evaluate(params, test_loader, eval_key)
            print(f"Epoch {epoch+1}: loss={epoch_loss/n_batches:.4f}, test_acc={test_acc:.4f}")

    return params


# train
trained_params = train_snn()
```

### step 5: evaluate on test set

```python
# final evaluation
key = random.PRNGKey(0)
_, test_loader = load_shd_data()
final_acc = evaluate(trained_params, test_loader, key)
print(f"\nFinal test accuracy: {final_acc:.4f}")

# expected output after training:
# Epoch 5: loss=2.1234, test_acc=0.4521
# Epoch 10: loss=1.5678, test_acc=0.5834
# Epoch 15: loss=1.2345, test_acc=0.6523
# Epoch 20: loss=1.0123, test_acc=0.6891
# ...
# Epoch 50: loss=0.6789, test_acc=0.7432
#
# Final test accuracy: 0.7432
```

---

## 4. submission format

### 4.1 E69-T1 submission: `lif.py`

```python
# lif.py
"""E69-T1: LIF Neuron Implementation"""

import jax.numpy as jnp
from typing import NamedTuple

# constants (must match starter code)
TAU_M = 20.0
V_REST = 0.0
V_TH = 1.0
V_RESET = 0.0
DT = 1.0


class LIFState(NamedTuple):
    V: jnp.ndarray
    spike: jnp.ndarray


def lif_step(state: LIFState, I: jnp.ndarray) -> LIFState:
    """Student's LIF implementation."""
    # ... student code ...
    return LIFState(V=V_new, spike=spike)
```

### 4.2 E69-T2 submission: `train_snn.py`

```python
# train_snn.py
"""E69-T2: SNN Training on SHD"""

import jax
import jax.numpy as jnp
from jax import random
import optax
import spyx.data

# student must export:
# - trained_params: SNNParams with W1, W2
# - snn_forward: forward pass function
# - evaluate: evaluation function


def train_and_save():
    """Train SNN and return params + accuracy."""
    # ... student training code ...
    return params, final_accuracy


if __name__ == "__main__":
    params, acc = train_and_save()
    print(f"Test accuracy: {acc:.4f}")

    # save params
    import pickle
    with open("snn_params.pkl", "wb") as f:
        pickle.dump(params, f)
```

---

## 5. evaluation pipeline

### 5.1 E69-T1 evaluation: unit tests

```python
# eval_lif.py
"""Evaluate LIF neuron implementation."""

import jax.numpy as jnp
from jax import random
from solution.lif import lif_step, LIFState

def run_lif_tests() -> dict:
    """Run comprehensive LIF unit tests."""
    results = {
        "decay_test": False,
        "spike_test": False,
        "integration_test": False,
        "numerical_stability": False,
        "vectorization_test": False,
    }

    # Test 1: decay without input
    try:
        state = LIFState(V=jnp.array([[0.8]]), spike=jnp.array([[0.0]]))
        I = jnp.array([[0.0]])
        new_state = lif_step(state, I)

        # should decay toward V_rest (0)
        if new_state.V[0, 0] < 0.8 and new_state.V[0, 0] > 0:
            results["decay_test"] = True
    except Exception as e:
        print(f"decay_test error: {e}")

    # Test 2: spike at threshold
    try:
        state = LIFState(V=jnp.array([[0.95]]), spike=jnp.array([[0.0]]))
        I = jnp.array([[0.2]])  # push over threshold
        new_state = lif_step(state, I)

        if new_state.spike[0, 0] == 1.0 and new_state.V[0, 0] == 0.0:
            results["spike_test"] = True
    except Exception as e:
        print(f"spike_test error: {e}")

    # Test 3: integration to spike
    try:
        state = LIFState(V=jnp.array([[0.0]]), spike=jnp.array([[0.0]]))
        I = jnp.array([[0.15]])

        spike_count = 0
        for _ in range(100):
            state = lif_step(state, I)
            spike_count += int(state.spike[0, 0])

        # should spike multiple times with sustained input
        if spike_count >= 3:
            results["integration_test"] = True
    except Exception as e:
        print(f"integration_test error: {e}")

    # Test 4: numerical stability
    try:
        # test with extreme values
        state = LIFState(V=jnp.array([[100.0]]), spike=jnp.array([[0.0]]))
        I = jnp.array([[1000.0]])
        new_state = lif_step(state, I)

        # should not produce NaN or Inf
        if jnp.isfinite(new_state.V).all() and jnp.isfinite(new_state.spike).all():
            results["numerical_stability"] = True
    except Exception as e:
        print(f"numerical_stability error: {e}")

    # Test 5: vectorization
    try:
        batch_size, n_neurons = 32, 256
        state = LIFState(
            V=jnp.zeros((batch_size, n_neurons)),
            spike=jnp.zeros((batch_size, n_neurons))
        )
        I = random.normal(random.PRNGKey(0), (batch_size, n_neurons)) * 0.2
        new_state = lif_step(state, I)

        if new_state.V.shape == (batch_size, n_neurons):
            results["vectorization_test"] = True
    except Exception as e:
        print(f"vectorization_test error: {e}")

    return results


def evaluate_lif() -> dict:
    """Main evaluation function."""
    results = run_lif_tests()

    passed = sum(results.values())
    total = len(results)

    return {
        "passed": passed == total,
        "score": passed / total * 100,
        "tests": results,
        "grade": "Excellent" if passed == total else ("Pass" if passed >= 3 else "Fail")
    }
```

### 5.2 E69-T2 evaluation: accuracy + anti-gaming

```python
# eval_snn.py
"""Evaluate SNN training on SHD."""

import jax
import jax.numpy as jnp
from jax import random
import spyx.data
import pickle

def evaluate_snn_submission(submission_path: str) -> dict:
    """Full evaluation of SNN submission."""

    # Load student solution
    import importlib.util
    spec = importlib.util.spec_from_file_location("solution", submission_path)
    solution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution)

    # Load trained params
    with open("snn_params.pkl", "rb") as f:
        params = pickle.load(f)

    # Load test data
    _, test_loader = spyx.data.shd_loader(batch_size=128)

    # === Test 1: Accuracy on standard test set ===
    key = random.PRNGKey(42)
    test_acc = solution.evaluate(params, test_loader, key)

    # === Test 2: Accuracy on held-out speakers (anti-gaming) ===
    _, holdout_loader = spyx.data.shd_loader(
        batch_size=128,
        holdout_speakers=True  # different speakers not in train
    )
    holdout_acc = solution.evaluate(params, holdout_loader, key)

    # === Test 3: Temporal shuffle check ===
    # shuffle time dimension - accuracy should DROP significantly
    def shuffle_time(x):
        T = x.shape[0]
        perm = random.permutation(random.PRNGKey(0), T)
        return x[perm]

    shuffled_correct = 0
    shuffled_total = 0
    for x, y in test_loader:
        x_shuffled = shuffle_time(x)
        _, membrane = solution.snn_forward(params, x_shuffled, key)
        logits = jnp.sum(membrane, axis=0)
        preds = jnp.argmax(logits, axis=-1)
        shuffled_correct += jnp.sum(preds == y)
        shuffled_total += len(y)
    shuffled_acc = float(shuffled_correct / shuffled_total)

    # temporal info check: shuffled acc should be much lower
    uses_temporal = (test_acc - shuffled_acc) > 0.1

    # === Test 4: Spike sparsity check ===
    spike_rates = []
    for x, y in test_loader:
        out_spikes, _ = solution.snn_forward(params, x, key)
        spike_rates.append(float(out_spikes.mean()))
    avg_spike_rate = sum(spike_rates) / len(spike_rates)

    # spikes should be sparse (<50% firing rate)
    is_sparse = avg_spike_rate < 0.5

    # === Compute results ===
    results = {
        "test_accuracy": test_acc,
        "holdout_accuracy": holdout_acc,
        "shuffled_accuracy": shuffled_acc,
        "uses_temporal_info": uses_temporal,
        "spike_rate": avg_spike_rate,
        "is_sparse": is_sparse,
    }

    # === Grading ===
    if not uses_temporal:
        grade = "Fail"
        reason = "Does not use temporal information"
    elif not is_sparse:
        grade = "Fail"
        reason = "Spike rate too high (not truly spiking)"
    elif test_acc < 0.50:
        grade = "Fail"
        reason = f"Accuracy {test_acc:.2%} below 50% threshold"
    elif test_acc < 0.70:
        grade = "Pass"
        reason = f"Accuracy {test_acc:.2%} meets baseline"
    elif test_acc < 0.78:
        grade = "Good"
        reason = f"Accuracy {test_acc:.2%} above baseline"
    else:
        grade = "Excellent"
        reason = f"Accuracy {test_acc:.2%} near state-of-art"

    results["grade"] = grade
    results["reason"] = reason
    results["passed"] = grade != "Fail"

    return results
```

### 5.3 anti-gaming checks

```python
def run_antigaming_checks(solution_module, params) -> dict:
    """Detect cheating strategies."""

    import inspect

    results = {
        "hardcoded_outputs": False,
        "trivial_solution": False,
        "no_temporal_processing": False,
        "constant_spikes": False,
    }

    # Check 1: hardcoded class outputs
    # run on two different inputs, should give different outputs
    key = random.PRNGKey(0)
    x1 = random.bernoulli(key, 0.1, (100, 4, 700)).astype(jnp.float32)
    x2 = random.bernoulli(random.PRNGKey(1), 0.1, (100, 4, 700)).astype(jnp.float32)

    _, m1 = solution_module.snn_forward(params, x1, key)
    _, m2 = solution_module.snn_forward(params, x2, key)

    if jnp.allclose(m1, m2, atol=0.01):
        results["hardcoded_outputs"] = True

    # Check 2: trivial solution (always predicts same class)
    logits1 = jnp.sum(m1, axis=0)
    logits2 = jnp.sum(m2, axis=0)
    preds1 = jnp.argmax(logits1, axis=-1)
    preds2 = jnp.argmax(logits2, axis=-1)

    if jnp.all(preds1 == preds1[0]) and jnp.all(preds2 == preds2[0]):
        results["trivial_solution"] = True

    # Check 3: no temporal processing
    # reverse time, should change output
    x_reversed = x1[::-1]
    _, m_reversed = solution_module.snn_forward(params, x_reversed, key)

    if jnp.allclose(m1, m_reversed, atol=0.1):
        results["no_temporal_processing"] = True

    # Check 4: constant spike rate (not truly spiking)
    out_spikes, _ = solution_module.snn_forward(params, x1, key)
    spike_std = jnp.std(out_spikes)

    if spike_std < 0.01:  # almost no variation
        results["constant_spikes"] = True

    return results
```

---

## 6. reference solutions

### 6.1 LIF neuron (E69-T1)

```python
# reference_lif.py
"""Reference LIF neuron implementation."""

import jax
import jax.numpy as jnp
from typing import NamedTuple

TAU_M = 20.0
V_REST = 0.0
V_TH = 1.0
V_RESET = 0.0
DT = 1.0


class LIFState(NamedTuple):
    V: jnp.ndarray
    spike: jnp.ndarray


@jax.custom_vjp
def spike_fn(x: jnp.ndarray) -> jnp.ndarray:
    """Spike function with surrogate gradient."""
    return jnp.where(x >= 0, 1.0, 0.0)


def spike_fn_fwd(x):
    return spike_fn(x), x


def spike_fn_bwd(res, g):
    x = res
    beta = 10.0
    grad = 1.0 / (1.0 + beta * jnp.abs(x)) ** 2
    return (g * grad,)


spike_fn.defvjp(spike_fn_fwd, spike_fn_bwd)


def lif_step(state: LIFState, I: jnp.ndarray) -> LIFState:
    """
    Reference LIF implementation.

    Features:
    - Exponential decay for numerical accuracy
    - SuperSpike surrogate gradient
    - Hard reset mechanism
    """
    V = state.V

    # exponential decay (more accurate than Euler)
    beta = jnp.exp(-DT / TAU_M)

    # leaky integration
    V_new = beta * V + (1 - beta) * V_REST + (1 - beta) * I

    # spike generation with surrogate gradient
    spike = spike_fn(V_new - V_TH)

    # hard reset
    V_new = jnp.where(spike > 0, V_RESET, V_new)

    return LIFState(V=V_new, spike=spike)
```

### 6.2 SNN training (E69-T2)

```python
# reference_snn.py
"""Reference SNN training on SHD."""

import jax
import jax.numpy as jnp
from jax import random
import optax
import spyx.data
from typing import Tuple, NamedTuple
from functools import partial

# import LIF from reference
from reference_lif import LIFState, lif_step, spike_fn

N_INPUT = 700
N_HIDDEN = 256
N_OUTPUT = 20
BATCH_SIZE = 128
LEARNING_RATE = 2e-3
N_EPOCHS = 100


class SNNParams(NamedTuple):
    W1: jnp.ndarray
    W2: jnp.ndarray
    # learnable time constants (optional enhancement)
    tau1: jnp.ndarray
    tau2: jnp.ndarray


def init_snn_params(key: random.PRNGKey) -> SNNParams:
    """Initialize with kaiming initialization."""
    key1, key2 = random.split(key)

    # kaiming init for ReLU-like (surrogate gradient is similar)
    W1 = random.normal(key1, (N_INPUT, N_HIDDEN)) * jnp.sqrt(2.0 / N_INPUT)
    W2 = random.normal(key2, (N_HIDDEN, N_OUTPUT)) * jnp.sqrt(2.0 / N_HIDDEN)

    # learnable time constants
    tau1 = jnp.ones(N_HIDDEN) * 20.0
    tau2 = jnp.ones(N_OUTPUT) * 20.0

    return SNNParams(W1=W1, W2=W2, tau1=tau1, tau2=tau2)


def init_lif_state(batch_size: int, n_neurons: int, key: random.PRNGKey) -> LIFState:
    return LIFState(
        V=jnp.zeros((batch_size, n_neurons)),
        spike=jnp.zeros((batch_size, n_neurons))
    )


def lif_step_with_tau(V: jnp.ndarray, I: jnp.ndarray, tau: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """LIF step with learnable tau."""
    beta = jnp.exp(-1.0 / tau)  # dt=1
    V_new = beta * V + (1 - beta) * I
    spike = spike_fn(V_new - 1.0)
    V_new = jnp.where(spike > 0, 0.0, V_new)
    return V_new, spike


def snn_forward(
    params: SNNParams,
    spikes_in: jnp.ndarray,
    key: random.PRNGKey
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Reference forward pass with learnable dynamics."""
    T, batch_size, _ = spikes_in.shape

    # initialize states
    V1 = jnp.zeros((batch_size, N_HIDDEN))
    V2 = jnp.zeros((batch_size, N_OUTPUT))

    output_spikes = []
    membrane_trace = []

    for t in range(T):
        # layer 1
        I1 = spikes_in[t] @ params.W1
        V1, s1 = lif_step_with_tau(V1, I1, params.tau1)

        # layer 2
        I2 = s1 @ params.W2
        V2, s2 = lif_step_with_tau(V2, I2, params.tau2)

        output_spikes.append(s2)
        membrane_trace.append(V2)

    return jnp.stack(output_spikes), jnp.stack(membrane_trace)


def cross_entropy_loss(membrane: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Cross-entropy with temporal sum readout."""
    logits = jnp.sum(membrane, axis=0)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(targets, N_OUTPUT)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


def evaluate(params: SNNParams, test_loader, key: random.PRNGKey) -> float:
    """Evaluate accuracy."""
    correct = 0
    total = 0

    for x, y in test_loader:
        _, membrane = snn_forward(params, x, key)
        logits = jnp.sum(membrane, axis=0)
        preds = jnp.argmax(logits, axis=-1)
        correct += jnp.sum(preds == y)
        total += len(y)

    return float(correct / total)


def train_reference() -> Tuple[SNNParams, float]:
    """Reference training loop."""
    key = random.PRNGKey(42)

    # init
    key, init_key = random.split(key)
    params = init_snn_params(init_key)

    # optimizer with learning rate schedule
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=100,
        decay_steps=N_EPOCHS * 64,
        end_value=LEARNING_RATE / 10
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(schedule)
    )
    opt_state = optimizer.init(params)

    # data
    train_loader, test_loader = spyx.data.shd_loader(batch_size=BATCH_SIZE)

    @jax.jit
    def train_step(params, opt_state, x, y, key):
        def loss_fn(params):
            _, membrane = snn_forward(params, x, key)
            return cross_entropy_loss(membrane, y)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # training
    best_acc = 0.0
    best_params = params

    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            key, step_key = random.split(key)
            params, opt_state, loss = train_step(params, opt_state, x, y, step_key)
            epoch_loss += loss
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            key, eval_key = random.split(key)
            test_acc = evaluate(params, test_loader, eval_key)
            print(f"Epoch {epoch+1}: loss={epoch_loss/n_batches:.4f}, acc={test_acc:.4f}")

            if test_acc > best_acc:
                best_acc = test_acc
                best_params = params

    return best_params, best_acc


if __name__ == "__main__":
    params, acc = train_reference()
    print(f"\nBest test accuracy: {acc:.4f}")

    # Expected: ~75-80% accuracy
```

---

## 7. grading

### 7.1 E69-T1: LIF neuron

| grade | criteria |
|-------|----------|
| **Fail** | <3 unit tests pass |
| **Pass** | all 3 core tests pass (decay, spike, integration) |
| **Excellent** | all tests pass + numerically stable + efficient vectorization |

### 7.2 E69-T2: SNN training

| grade | criteria |
|-------|----------|
| **Fail** | acc < 50% OR fails anti-gaming checks |
| **Pass** | acc >= 50% AND passes anti-gaming |
| **Good** | acc >= 70% |
| **Excellent** | acc >= 78% |

### 7.3 anti-gaming requirements

all submissions must satisfy:

1. **temporal usage**: shuffling time dimension must reduce accuracy by >10%
2. **spike sparsity**: average spike rate < 50%
3. **generalization**: holdout speaker accuracy within 10% of test accuracy
4. **non-trivial**: different inputs must produce different outputs

### 7.4 score breakdown

```
E69-T1 (30 points):
  - decay test:          6 points
  - spike test:          6 points
  - integration test:    6 points
  - numerical stability: 6 points
  - vectorization:       6 points

E69-T2 (70 points):
  - test accuracy:       40 points (scaled: 50%=0, 80%=40)
  - temporal usage:      10 points (binary)
  - spike sparsity:      10 points (binary)
  - holdout generalization: 10 points (scaled)

Total: 100 points
Pass threshold: 60 points
```

---

## 8. feedback to student

### 8.1 if PASSED (example)

```
===================================================================
                    EVALUATION RESULTS: E69
===================================================================

Status: PASSED

E69-T1 (LIF Neuron): 30/30 points
  - decay_test: PASS
  - spike_test: PASS
  - integration_test: PASS
  - numerical_stability: PASS
  - vectorization_test: PASS
  Grade: Excellent

E69-T2 (SNN Training): 52/70 points
  - Test accuracy: 72.3% (28/40 points)
  - Temporal usage check: PASS (10/10)
  - Spike sparsity: PASS - rate=0.12 (10/10)
  - Holdout generalization: 68.1% (4/10 points)
  Grade: Good

Total Score: 82/100

Suggestions for improvement:
  - Try learnable time constants (tau as parameter)
  - Use learning rate warmup and cosine decay
  - Consider adding dropout or recurrent connections
  - Increase hidden layer size to 512

===================================================================
```

### 8.2 if FAILED (example)

```
===================================================================
                    EVALUATION RESULTS: E69
===================================================================

Status: FAILED

E69-T1 (LIF Neuron): 18/30 points
  - decay_test: PASS
  - spike_test: PASS
  - integration_test: PASS
  - numerical_stability: FAIL - NaN detected
  - vectorization_test: FAIL - shape mismatch
  Grade: Pass (but with issues)

E69-T2 (SNN Training): 0/70 points
  - Test accuracy: 45.2% (below 50% threshold)
  - Temporal usage check: FAIL - shuffled acc only 3% lower
  - Spike sparsity: FAIL - rate=0.78 (too high)

  Anti-gaming failure detected:
    Your model does not properly use temporal information.
    This suggests spike counts are being used rather than
    spike timing. True SNNs should show >10% accuracy drop
    when time is shuffled.

Total Score: 18/100

Debug hints:
  - Check that lif_step uses proper decay (beta = exp(-dt/tau))
  - Ensure surrogate gradient is applied during backward pass
  - Verify weights are being updated (print gradient norms)
  - Your spike rate (78%) is too high - adjust threshold or input scaling

===================================================================
```

---

## 9. summary: full pipeline visualization

```
+-------------------------------------------------------------+
|                     STUDENT WORKFLOW                         |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  1. READ TASK DESCRIPTION                                    |
|     - understand SNNs vs ANNs (temporal vs rate coding)     |
|     - understand LIF neuron dynamics                        |
|     - note SHD dataset format (700 channels, 20 classes)    |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  2. E69-T1: IMPLEMENT LIF NEURON                            |
|     - implement membrane decay                              |
|     - implement spike generation                            |
|     - implement reset mechanism                             |
|     - pass unit tests                                       |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  3. E69-T2: BUILD SNN                                       |
|     - load SHD data with spyx                               |
|     - build 2-layer network (input->LIF->LIF->output)       |
|     - implement forward pass with time unrolling            |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  4. TRAIN SNN                                               |
|     - set up optax optimizer (adam)                         |
|     - implement training loop                               |
|     - use cross-entropy loss on summed membrane             |
|     - train for ~50 epochs                                  |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  5. SUBMIT: lif.py + train_snn.py + snn_params.pkl          |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                    EVALUATION PIPELINE                       |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  A. E69-T1 VALIDATION                                       |
|     - run 5 unit tests on lif_step                          |
|     - check decay, spike, integration, stability, vectors   |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  B. E69-T2 ACCURACY EVALUATION                              |
|     - load trained params                                   |
|     - evaluate on test set                                  |
|     - evaluate on holdout speakers                          |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  C. ANTI-GAMING CHECKS                                      |
|     - temporal shuffle check (acc must drop)                |
|     - spike sparsity check (rate < 50%)                     |
|     - non-trivial output check                              |
|     - holdout generalization check                          |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  D. COMPUTE SCORE AND GRADE                                 |
|     - E69-T1: 30 points from unit tests                     |
|     - E69-T2: 70 points (accuracy + anti-gaming)            |
|     - pass threshold: 60 points                             |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  E. RETURN FEEDBACK                                         |
|     - detailed metrics                                      |
|     - improvement suggestions                               |
|     - debug hints for failures                              |
+-------------------------------------------------------------+
```

---

*end of detailed pipeline specification*
