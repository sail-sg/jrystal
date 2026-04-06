# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Density-mixing strategies for SCF convergence.

This module collects all density-mixing tools in one place:

* **DIIS** (direct inversion in the iterative subspace) -- the primary
  accelerator for SCF convergence.
* **Linear mixing** -- simple weighted average of old and new densities.
* **Kerker preconditioner** -- damps long-wavelength charge sloshing.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

# ---------------------------------------------------------------------------
# DIIS  (copied from scf/diis.py to consolidate into calc/)
# ---------------------------------------------------------------------------


def diis_init(max_hist: int, density_shape: tuple, dtype=jnp.float32):
  """Initialise DIIS state.

  Returns a dict with pre-allocated density/error history buffers.
  """
  densities = jnp.zeros((max_hist,) + tuple(density_shape), dtype=dtype)
  errors = jnp.zeros_like(densities)
  size = jnp.array(0, dtype=jnp.int32)
  head = jnp.array(0, dtype=jnp.int32)
  return {
    "densities": densities,
    "errors": errors,
    "size": size,
    "head": head,
  }


def _diis_mix(densities, errors, size, eps=1e-12):
  m = densities.shape[0]
  mask = (jnp.arange(m) < size).astype(errors.dtype)
  mask_mat = mask[:, None] * mask[None, :]

  E = errors.reshape((m, -1))
  B = jnp.matmul(jnp.conj(E), jnp.transpose(E))
  B = B * mask_mat
  B = B + jnp.eye(m, dtype=B.dtype) * (1.0 - mask)
  B = B + jnp.eye(m, dtype=B.dtype) * eps

  a = mask
  A = jnp.zeros((m + 1, m + 1), dtype=B.dtype)
  A = A.at[:m, :m].set(B)
  A = A.at[:m, m].set(a)
  A = A.at[m, :m].set(a)
  rhs = jnp.zeros((m + 1,), dtype=B.dtype)
  rhs = rhs.at[m].set(1.0)
  sol = jnp.linalg.solve(A, rhs)
  coeff = sol[:m]
  return jnp.tensordot(coeff, densities, axes=(0, 0))


@jax.jit
def diis_update(state, density, error, eps=1e-12):
  """Push a new density/error pair and return the DIIS-mixed density.

  Jittable.  Falls back to the raw *density* when fewer than two
  history entries are available.
  """
  densities = state["densities"]
  errors = state["errors"]
  size = state["size"]
  head = state["head"]

  densities = jax.lax.dynamic_update_index_in_dim(
    densities,
    density,
    head,
    axis=0,
  )
  errors = jax.lax.dynamic_update_index_in_dim(
    errors,
    error,
    head,
    axis=0,
  )

  max_hist = densities.shape[0]
  head = (head + 1) % max_hist
  size = jnp.minimum(size + 1, max_hist)

  mixed = jax.lax.cond(
    size < 2,
    lambda _: density,
    lambda _: _diis_mix(densities, errors, size, eps=eps),
    operand=None,
  )
  new_state = {
    "densities": densities,
    "errors": errors,
    "size": size,
    "head": head,
  }
  return new_state, mixed


# ---------------------------------------------------------------------------
# Linear mixing
# ---------------------------------------------------------------------------


def simple_mixing(new_density, old_density, beta: float = 0.7):
  """Linear mixing: ``beta * new + (1 - beta) * old``."""
  return new_density * beta + old_density * (1.0 - beta)


# ---------------------------------------------------------------------------
# Kerker preconditioner
# ---------------------------------------------------------------------------


def kerker_preconditioner(
  g_vec: Float[Array, "x y z 3"],
  freq_mask: Bool[Array, "x y z"],
):
  """Build a Kerker preconditioner vector in masked G-space.

  The Kerker filter ``|G|^2 / (1 + |G|^2)`` suppresses the long-wavelength
  (small |G|) components of the density residual, preventing charge sloshing
  in metallic systems.

  Returns a 1-D array of length ``sum(freq_mask)`` suitable for use as a
  LOBPCG elementwise preconditioner.
  """
  eff_g = g_vec[freq_mask]
  g2 = jnp.sum(eff_g**2, axis=-1)
  return g2 / (1.0 + g2)


__all__ = [
  "diis_init",
  "diis_update",
  "kerker_preconditioner",
  "simple_mixing",
]
