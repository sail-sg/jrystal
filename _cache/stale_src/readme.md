# Sharded FFTN in JAX using `custom_partitioning`

## Overview

This document explains how we implement a **sharding-compatible `fftn`** in JAX using the modern `jax.experimental.custom_partitioning` API.

The goal is:

* Input tensor shape:

  ```
  [*some_batch_dims, *some_fft_dims]
  ```
* Mesh sharding on the first `some_batch_dims` axes:

  ```python
  PartitionSpec(*some_batch_dims)
  ```
* Perform FFT over the last one or more axes, which can be specified by the argument `axes` in the function `fftn`.
* Ensure FFT axes are not sharded.
* Preserve automatic differentiation
* Avoid hard-coded rank or axis names

This document explains:

* Why native `jnp.fft.fftn` does not support arbitrary sharding
* What `custom_partitioning` does
* How we dynamically control sharding propagation
* How automatic differentiation still works
* How to test and validate correctness

---

# 1. Problem Statement

JAX's built-in `jnp.fft.fftn` does not support arbitrary SPMD sharding.

If FFT axes are sharded, compilation will either:

* Fail
* Insert unwanted communication (e.g., all-gather)
* Produce unsupported layouts

We want to:

✔ Shard only batch-like dimensions.
✔ Keep FFT axes replicated.
✔ Maintain AD support
✔ Work for arbitrary rank and axes

---

# 2. High-Level Strategy

We wrap `jnp.fft.fftn` inside a `custom_partitioning` function.

`custom_partitioning` lets us override:

* How sharding propagates
* What sharding is allowed
* What lowering runs on each device

We do NOT reimplement FFT.

We only change the **sharding semantics**.

---

# 3. Architecture

## 3.1 Layer Structure

We structure the solution into three layers:

### Layer 1 — Semantic Function (Differentiable)

```python
@custom_partitioning(static_argnums=(1, 2, 3))
def fftn(x, s=None, axes=None, norm=None):
    return jnp.fft.fftn(x, s=s, axes=axes, norm=norm)
```

This function:

* Calls standard `jnp.fft.fftn`
* Remains fully differentiable
* Contains no sharding logic

---

### Layer 2 — Sharding Control

We override sharding behavior using:

```python
fftn.def_partition(
    partition=...,
    infer_sharding_from_operands=...,
    sharding_rule=...,
    need_replication_factors=...
)
```

---

# 4. Key Implementation Concepts

## 4.1 Mask FFT Axes to Replicated

We dynamically rewrite the sharding spec:

```python
def mask_fft_axes(sharding, rank, fft_axes):
    spec = list(sharding.spec) + [None] * (rank - len(sharding.spec))
    for ax in fft_axes:
        spec[ax] = None
    return NamedSharding(sharding.mesh, PartitionSpec(*spec))
```

Effect:

| Axis         | Behavior                      |
| ------------ | ----------------------------- |
| Non-FFT axis | Preserved sharding            |
| FFT axis     | Forced to `None` (replicated) |

---

## 4.2 Dynamic Axis Normalization

We support arbitrary axes (including negative indexing):

```python
def normalize_axes(rank, axes):
    return tuple(sorted((a + rank if a < 0 else a) for a in axes))
```

No dimension names are hardcoded.

---

# 5. How Partitioning Works Internally

When JAX compiles with SPMD:

1. It traces the function
2. It infers sharding propagation
3. It calls our `infer_sharding_from_operands`
4. It enforces replication on FFT axes
5. It lowers per-device execution via `partition`

### partition() returns

```python
mesh, lower_fn, out_sharding, in_shardings
```

* `lower_fn` executes locally on each shard
* Sharding is guaranteed valid

---

# 6. Why Automatic Differentiation Still Works

We do NOT define custom JVP or VJP.

Why this works:

* The semantic function still calls `jnp.fft.fftn`
* FFT already has AD rules in JAX
* `custom_partitioning` only changes sharding behavior
* It does not alter primitive semantics

Therefore:

✔ `jax.grad` works
✔ `jax.jvp` works
✔ `jax.vjp` works

If we replaced FFT with a custom primitive, we would need to define VJP.

---

# 7. What This Implementation Guarantees

## 7.1 Guarantees

✔ FFT axes are never sharded
✔ Batch axes preserve input sharding
✔ Works with `jit` and Mesh
✔ Works on fake CPU devices
✔ Fully differentiable
✔ No hardcoded dimension names
✔ Supports arbitrary rank

---

## 7.2 What It Does NOT Do

✘ Does NOT implement distributed FFT (no all-to-all)
✘ Does NOT optimize communication
✘ Does NOT repartition FFT axes across devices

It only ensures safe local FFT execution.

---
