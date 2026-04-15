from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from einops import einsum

from jrystal.pseudopotential.augmentation import (
  channel_pair_multipoles,
  effective_channel_matrix,
)


def _legacy_channel_pair_multipoles(
  pair_density,
  channel_coupling,
  channel_beta,
  max_beta,
):
  spin = pair_density.shape[0]
  atom = pair_density.shape[1]
  l_dim = channel_coupling.shape[3]
  m_dim = channel_coupling.shape[4]
  output = jnp.zeros(
    (spin, atom, max_beta, max_beta, l_dim, m_dim),
    dtype=pair_density.dtype,
  )
  channel_beta_np = np.asarray(channel_beta, dtype=np.int32)
  num_channel = pair_density.shape[2]
  for atom_idx in range(atom):
    for channel_i in range(num_channel):
      beta_i = int(channel_beta_np[atom_idx, channel_i])
      for channel_j in range(num_channel):
        beta_j = int(channel_beta_np[atom_idx, channel_j])
        output = output.at[:, atom_idx, beta_i, beta_j].add(
          pair_density[:, atom_idx, channel_i, channel_j][..., None, None] *
          channel_coupling[atom_idx, channel_i, channel_j],
        )
  return output


def _legacy_effective_channel_matrix(
  local_potential_r,
  channel_dii,
  channel_coupling,
  channel_beta,
  radial_fields,
  harmonics,
  vol,
  channel_mask=None,
):

  def _single(local_potential_spin):
    num_grids = np.prod(local_potential_spin.shape)
    basis_integrals = einsum(
      radial_fields,
      harmonics,
      local_potential_spin,
      "a i j l x y z, a l m x y z, x y z -> a i j l m",
    ) * (vol / num_grids)

    atom_matrices = []
    channel_beta_np = np.asarray(channel_beta, dtype=np.int32)
    mask = jnp.asarray(channel_mask) if channel_mask is not None else None

    for atom_idx in range(channel_dii.shape[0]):
      beta_indices = channel_beta_np[atom_idx]
      gathered = basis_integrals[atom_idx][
        beta_indices[:, None],
        beta_indices[None, :],
      ]
      atom_matrix = channel_dii[atom_idx] + jnp.sum(
        channel_coupling[atom_idx] * gathered,
        axis=(-1, -2),
      )
      if mask is not None:
        atom_mask = mask[atom_idx]
        atom_matrix = atom_matrix * (atom_mask[:, None] * atom_mask[None, :])
      atom_matrices.append(atom_matrix)

    return jnp.stack(atom_matrices, axis=0)

  local_potential_r = jnp.asarray(local_potential_r)
  if local_potential_r.ndim == 3:
    return _single(local_potential_r)
  return jax.vmap(_single)(local_potential_r)


@pytest.mark.parametrize(
  "atom,num_channel,max_beta,l_dim,m_dim",
  [(2, 4, 3, 2, 3), (3, 8, 4, 3, 5)],
)
def test_channel_pair_multipoles_matches_legacy(
  atom,
  num_channel,
  max_beta,
  l_dim,
  m_dim,
):
  rng = np.random.default_rng(0)
  pair_density = jnp.asarray(
    rng.normal(size=(2, atom, num_channel, num_channel)) +
    1j * rng.normal(size=(2, atom, num_channel, num_channel)),
    dtype=jnp.complex64,
  )
  channel_coupling = jnp.asarray(
    rng.normal(size=(atom, num_channel, num_channel, l_dim, m_dim)),
    dtype=jnp.float32,
  )
  channel_beta = jnp.asarray(
    rng.integers(0, max_beta, size=(atom, num_channel)),
    dtype=jnp.int32,
  )

  expected = _legacy_channel_pair_multipoles(
    pair_density,
    channel_coupling,
    channel_beta,
    max_beta,
  )
  actual = channel_pair_multipoles(
    pair_density,
    channel_coupling,
    channel_beta,
    max_beta,
  )

  np.testing.assert_allclose(
    np.asarray(actual),
    np.asarray(expected),
    rtol=1e-5,
    atol=1e-5,
  )


@pytest.mark.parametrize("with_spin_axis", [False, True])
def test_effective_channel_matrix_matches_legacy(with_spin_axis):
  rng = np.random.default_rng(1)
  atom = 2
  num_channel = 4
  max_beta = 3
  l_dim = 2
  m_dim = 3
  grid_shape = (2, 3, 2)
  vol = 12.0

  radial_fields = jnp.asarray(
    rng.normal(size=(atom, max_beta, max_beta, l_dim, *grid_shape)),
    dtype=jnp.float32,
  )
  harmonics = jnp.asarray(
    rng.normal(size=(atom, l_dim, m_dim, *grid_shape)),
    dtype=jnp.float32,
  )
  channel_coupling = jnp.asarray(
    rng.normal(size=(atom, num_channel, num_channel, l_dim, m_dim)),
    dtype=jnp.float32,
  )
  channel_dii = jnp.asarray(
    rng.normal(size=(atom, num_channel, num_channel)),
    dtype=jnp.float32,
  )
  channel_beta = jnp.asarray(
    rng.integers(0, max_beta, size=(atom, num_channel)),
    dtype=jnp.int32,
  )
  channel_mask = jnp.asarray(
    [[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0]],
    dtype=jnp.float32,
  )
  if with_spin_axis:
    local_potential_r = jnp.asarray(
      rng.normal(size=(2, *grid_shape)),
      dtype=jnp.float32,
    )
  else:
    local_potential_r = jnp.asarray(
      rng.normal(size=grid_shape),
      dtype=jnp.float32,
    )

  expected = _legacy_effective_channel_matrix(
    local_potential_r,
    channel_dii,
    channel_coupling,
    channel_beta,
    radial_fields,
    harmonics,
    vol,
    channel_mask=channel_mask,
  )
  actual = effective_channel_matrix(
    local_potential_r,
    channel_dii,
    channel_coupling,
    channel_beta,
    radial_fields,
    harmonics,
    vol,
    channel_mask=channel_mask,
  )

  np.testing.assert_allclose(
    np.asarray(actual),
    np.asarray(expected),
    rtol=1e-5,
    atol=1e-5,
  )
