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
"""Ultrasoft one-center augmentation helpers."""

from __future__ import annotations

import numpy as np

import jax.numpy as jnp
from einops import einsum
from jaxtyping import Array, Complex, Float, Int

from .._src import kinetic as _kinetic
from .._src import pw as _pw


def local_potential_apply(
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  local_potential_r: Float[Array, "x y z"],
  vol: float,
) -> Complex[Array, "spin kpt band x y z"]:
  """Apply a real-space local potential to plane-wave coefficients."""
  wave_grid = _pw.wave_grid(pw_coefficients, vol)
  local_wave = wave_grid * local_potential_r[None, None, None, ...]
  coeff = jnp.fft.fftn(local_wave, axes=range(-3, 0))
  return coeff * (jnp.sqrt(jnp.asarray(vol, dtype=coeff.real.dtype)) / np.prod(pw_coefficients.shape[-3:]))


def kinetic_apply(
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  g_vec: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
) -> Complex[Array, "spin kpt band x y z"]:
  """Apply the diagonal kinetic operator in reciprocal space."""
  kinetic = _kinetic.kinetic_operator(g_vec, kpts)
  return pw_coefficients * kinetic[None, :, None, ...]


def channel_nonlocal_apply(
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  projector_channels: Complex[Array, "atom kpt channel x y z"],
  channel_dii: Float[Array, "atom channel channel"],
  vol: float,
  channel_mask: Float[Array, "atom channel"] | None = None,
) -> Complex[Array, "spin kpt band x y z"]:
  """Apply a channel-space separable nonlocal operator."""
  from .ultrasoft import projector_channel_overlap

  f_matrix = projector_channel_overlap(
    pw_coefficients,
    projector_channels,
    channel_mask=channel_mask,
  )
  df_matrix = einsum(
    channel_dii,
    f_matrix,
    "a i j, s a k band j -> s a k band i",
  )
  correction = einsum(
    jnp.conj(projector_channels),
    df_matrix,
    "a k i x y z, s a k band i -> s k band x y z",
  ) / vol
  return correction


def channel_nonlocal_apply_compact(
  pw_coefficients: Complex[Array, "spin kpt gpt band"],
  projector_channels: Complex[Array, "atom kpt channel gpt"],
  channel_dii: Float[Array, "atom channel channel"],
  vol: float,
  channel_mask: Float[Array, "atom channel"] | None = None,
) -> Complex[Array, "spin kpt gpt band"]:
  """Apply a channel-space separable nonlocal operator in compact G-space."""
  from .ultrasoft import projector_channel_overlap_compact

  f_matrix = projector_channel_overlap_compact(
    pw_coefficients,
    projector_channels,
    channel_mask=channel_mask,
  )
  df_matrix = einsum(
    channel_dii,
    f_matrix,
    "a i j, s a k band j -> s a k band i",
  )
  correction = einsum(
    jnp.conj(projector_channels),
    df_matrix,
    "a k i g, s a k band i -> s k band g",
  ) / vol
  return jnp.swapaxes(correction, -1, -2)


def channel_nonlocal_energy_compact(
  pw_coefficients: Complex[Array, "spin kpt gpt band"],
  projector_channels: Complex[Array, "atom kpt channel gpt"],
  channel_dii: Float[Array, "atom channel channel"],
  vol: float,
  occupation: Float[Array, "spin kpt band"] | None = None,
  kpts_weights: Float[Array, "kpt"] | None = None,
  channel_mask: Float[Array, "atom channel"] | None = None,
) -> Float:
  """Return the compact-G ultrasoft nonlocal energy expectation value."""
  from .ultrasoft import projector_channel_overlap_compact

  f_matrix = projector_channel_overlap_compact(
    pw_coefficients,
    projector_channels,
    channel_mask=channel_mask,
  )
  df_matrix = einsum(
    channel_dii,
    f_matrix,
    "a i j, s a k band j -> s a k band i",
  )
  diag_nl = einsum(
    jnp.conj(f_matrix),
    df_matrix,
    "s a k band i, s a k band i -> s k band",
  ).real / vol
  if occupation is None:
    occupation = jnp.ones(diag_nl.shape, dtype=diag_nl.dtype)
  if kpts_weights is not None:
    occupation = occupation * kpts_weights[None, :, None]
  return jnp.sum(diag_nl * occupation).real


def channel_pair_density(
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  occupation: Float[Array, "spin kpt band"],
  projector_channels: Complex[Array, "atom kpt channel x y z"],
  channel_mask: Float[Array, "atom channel"] | None = None,
  kpts_weights: Float[Array, "kpt"] | None = None,
) -> Complex[Array, "spin atom channel channel"]:
  """Return the occupied projector-space density matrix in channel basis."""
  from .ultrasoft import projector_channel_overlap

  f_matrix = projector_channel_overlap(
    pw_coefficients,
    projector_channels,
    channel_mask=channel_mask,
  )
  if kpts_weights is not None:
    occupation = occupation * kpts_weights[None, :, None]
  return einsum(
    jnp.conj(f_matrix),
    occupation,
    f_matrix,
    "s a k band i, s k band, s a k band j -> s a i j",
  )


def channel_pair_density_compact(
  pw_coefficients: Complex[Array, "spin kpt gpt band"],
  occupation: Float[Array, "spin kpt band"],
  projector_channels: Complex[Array, "atom kpt channel gpt"],
  channel_mask: Float[Array, "atom channel"] | None = None,
  kpts_weights: Float[Array, "kpt"] | None = None,
) -> Complex[Array, "spin atom channel channel"]:
  """Return occupied projector-space density in channel basis from compact G."""
  from .ultrasoft import projector_channel_overlap_compact

  f_matrix = projector_channel_overlap_compact(
    pw_coefficients,
    projector_channels,
    channel_mask=channel_mask,
  )
  if kpts_weights is not None:
    occupation = occupation * kpts_weights[None, :, None]
  return einsum(
    jnp.conj(f_matrix),
    occupation,
    f_matrix,
    "s a k band i, s k band, s a k band j -> s a i j",
  )


def channel_pair_multipoles(
  pair_density: Complex[Array, "spin atom channel channel"],
  channel_coupling: Float[Array, "atom channel channel l m"],
  channel_beta: Int[Array, "atom channel"],
  max_beta: int,
) -> Complex[Array, "spin atom beta beta l m"]:
  """Accumulate channel-pair density into one-center beta-pair multipoles."""
  spin = pair_density.shape[0]
  atom = pair_density.shape[1]
  l_dim = channel_coupling.shape[3]
  m_dim = channel_coupling.shape[4]
  output = jnp.zeros((spin, atom, max_beta, max_beta, l_dim, m_dim), dtype=pair_density.dtype)

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


def augmentation_density_from_multipoles(
  multipoles: Complex[Array, "spin atom beta beta l m"],
  radial_fields: Float[Array, "atom beta beta l x y z"],
  harmonics: Float[Array, "atom l m x y z"],
  vol: float,
) -> Float[Array, "spin x y z"]:
  """Reconstruct augmentation density from one-center multipoles."""
  weighted_radial = einsum(
    multipoles.real,
    radial_fields,
    "s a i j l m, a i j l x y z -> s a l m x y z",
  )
  return einsum(
    weighted_radial,
    harmonics,
    "s a l m x y z, a l m x y z -> s x y z",
  ) / vol


def augmentation_density(
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  occupation: Float[Array, "spin kpt band"],
  projector_channels: Complex[Array, "atom kpt channel x y z"],
  radial_fields: Float[Array, "atom beta beta l x y z"] | None,
  harmonics: Float[Array, "atom l m x y z"] | None,
  channel_coupling: Float[Array, "atom channel channel l m"] | None,
  channel_beta: Int[Array, "atom channel"] | None,
  vol: float,
  kpts_weights: Float[Array, "kpt"] | None = None,
  channel_mask: Float[Array, "atom channel"] | None = None,
) -> Float[Array, "spin x y z"]:
  """Build ultrasoft augmentation density using one-center basis caches."""
  if (
    radial_fields is None or harmonics is None or channel_coupling is None or
    channel_beta is None
  ):
    return jnp.zeros(
      (pw_coefficients.shape[0], *pw_coefficients.shape[-3:]),
      dtype=pw_coefficients.real.dtype,
    )

  pair_density = channel_pair_density(
    pw_coefficients,
    occupation,
    projector_channels,
    channel_mask=channel_mask,
    kpts_weights=kpts_weights,
  )
  multipoles = channel_pair_multipoles(
    pair_density,
    channel_coupling,
    channel_beta,
    max_beta=radial_fields.shape[1],
  )
  return augmentation_density_from_multipoles(
    multipoles,
    radial_fields,
    harmonics,
    vol,
  )


def augmentation_density_compact(
  pw_coefficients: Complex[Array, "spin kpt gpt band"],
  occupation: Float[Array, "spin kpt band"],
  projector_channels: Complex[Array, "atom kpt channel gpt"],
  radial_fields: Float[Array, "atom beta beta l x y z"] | None,
  harmonics: Float[Array, "atom l m x y z"] | None,
  channel_coupling: Float[Array, "atom channel channel l m"] | None,
  channel_beta: Int[Array, "atom channel"] | None,
  vol: float,
  kpts_weights: Float[Array, "kpt"] | None = None,
  channel_mask: Float[Array, "atom channel"] | None = None,
) -> Float[Array, "spin x y z"]:
  """Build ultrasoft augmentation density using compact G-space projectors."""
  if (
    radial_fields is None or harmonics is None or channel_coupling is None or
    channel_beta is None
  ):
    if radial_fields is not None:
      grid_shape = radial_fields.shape[-3:]
    elif harmonics is not None:
      grid_shape = harmonics.shape[-3:]
    else:
      grid_shape = (1, 1, 1)
    return jnp.zeros(
      (pw_coefficients.shape[0], *grid_shape),
      dtype=pw_coefficients.real.dtype,
    )

  pair_density = channel_pair_density_compact(
    pw_coefficients,
    occupation,
    projector_channels,
    channel_mask=channel_mask,
    kpts_weights=kpts_weights,
  )
  multipoles = channel_pair_multipoles(
    pair_density,
    channel_coupling,
    channel_beta,
    max_beta=radial_fields.shape[1],
  )
  return augmentation_density_from_multipoles(
    multipoles,
    radial_fields,
    harmonics,
    vol,
  )


def effective_channel_matrix(
  local_potential_r: Float[Array, "x y z"],
  channel_dii: Float[Array, "atom channel channel"],
  channel_coupling: Float[Array, "atom channel channel l m"],
  channel_beta: Int[Array, "atom channel"],
  radial_fields: Float[Array, "atom beta beta l x y z"],
  harmonics: Float[Array, "atom l m x y z"],
  vol: float,
  channel_mask: Float[Array, "atom channel"] | None = None,
) -> Float[Array, "atom channel channel"]:
  """Build the fixed-density ultrasoft effective channel matrix."""
  num_grids = np.prod(local_potential_r.shape)
  basis_integrals = einsum(
    radial_fields,
    harmonics,
    local_potential_r,
    "a i j l x y z, a l m x y z, x y z -> a i j l m",
  ) * (vol / num_grids)

  atom_matrices = []
  channel_beta_np = np.asarray(channel_beta, dtype=np.int32)
  num_channel = channel_dii.shape[1]
  if channel_mask is not None:
    channel_mask = jnp.asarray(channel_mask)

  for atom_idx in range(channel_dii.shape[0]):
    beta_indices = channel_beta_np[atom_idx]
    gathered = basis_integrals[atom_idx][beta_indices[:, None], beta_indices[None, :]]
    atom_matrix = channel_dii[atom_idx] + jnp.sum(
      channel_coupling[atom_idx] * gathered,
      axis=(-1, -2),
    )
    if channel_mask is not None:
      atom_mask = channel_mask[atom_idx]
      atom_matrix = atom_matrix * (atom_mask[:, None] * atom_mask[None, :])
    atom_matrices.append(atom_matrix)

  return jnp.stack(atom_matrices, axis=0)


__all__ = [
  "augmentation_density",
  "augmentation_density_compact",
  "augmentation_density_from_multipoles",
  "channel_nonlocal_apply",
  "channel_nonlocal_apply_compact",
  "channel_nonlocal_energy_compact",
  "channel_pair_density",
  "channel_pair_density_compact",
  "channel_pair_multipoles",
  "effective_channel_matrix",
  "kinetic_apply",
  "local_potential_apply",
]
