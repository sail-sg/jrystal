"""Unit test for packing the atomic density matrix."""
from __future__ import annotations

import jax.numpy as jnp

from jrystal.pseudopotential.utils import pack


def test_pack_density_matrix_values() -> None:
  n_kpts = 2
  n_spins = 1
  n_proj = 3

  base = jnp.array(
    [
      [1.0, 0.2, -0.3],
      [0.2, 2.0, 0.4],
      [-0.3, 0.4, 3.0],
    ]
  )
  alt = jnp.array(
    [
      [4.0, -0.1, 0.0],
      [-0.1, 5.0, 0.6],
      [0.0, 0.6, 6.0],
    ]
  )

  D = jnp.zeros((n_kpts, n_spins, n_proj, n_proj))
  D = D.at[0, 0].set(base)
  D = D.at[1, 0].set(alt)

  packed = pack(D)
  expected = jnp.array([1.0, 0.4, -0.6, 2.0, 0.8, 3.0])

  assert packed.shape == (n_proj * (n_proj + 1) // 2,)
  assert jnp.allclose(packed, expected, rtol=0, atol=1e-12)
