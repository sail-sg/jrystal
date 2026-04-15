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

import jax
import jax.numpy as jnp
import numpy as np
from einops import einsum
from jaxtyping import Array, Complex, Float, Int

from .._src import kinetic as _kinetic
from .._src import pw as _pw


def local_potential_apply(
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  local_potential_r: Float[Array, "... x y z"],
  vol: float,
) -> Complex[Array, "spin kpt band x y z"]:
  """Apply a real-space local potential to plane-wave coefficients."""
  wave_grid = _pw.wave_grid(pw_coefficients, vol)
  local_potential_r = jnp.asarray(local_potential_r)
  if local_potential_r.ndim == 3:
    local_wave = wave_grid * local_potential_r[None, None, None, ...]
  elif local_potential_r.ndim == 4:
    local_wave = wave_grid * local_potential_r[:, None, None, ...]
  else:
    raise ValueError(
      "local_potential_r must have shape [x, y, z] or [spin, x, y, z]. "
      f"Got {local_potential_r.shape}."
    )
  coeff = jnp.fft.fftn(local_wave, axes=range(-3, 0))
  return coeff * (
    jnp.sqrt(jnp.asarray(vol, dtype=coeff.real.dtype)) /
    np.prod(pw_coefficients.shape[-3:])
  )


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
  channel_dii: Float[Array, "... atom channel channel"],
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
  channel_dii = jnp.asarray(channel_dii)
  if channel_dii.ndim == 3:
    df_matrix = einsum(
      channel_dii,
      f_matrix,
      "a i j, s a k band j -> s a k band i",
    )
  elif channel_dii.ndim == 4:
    df_matrix = einsum(
      channel_dii,
      f_matrix,
      "s a i j, s a k band j -> s a k band i",
    )
  else:
    raise ValueError(
      "channel_dii must have shape [atom, ch, ch] or [spin, atom, ch, ch]. "
      f"Got {channel_dii.shape}."
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
  channel_dii: Float[Array, "... atom channel channel"],
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
  channel_dii = jnp.asarray(channel_dii)
  if channel_dii.ndim == 3:
    df_matrix = einsum(
      channel_dii,
      f_matrix,
      "a i j, s a k band j -> s a k band i",
    )
  elif channel_dii.ndim == 4:
    df_matrix = einsum(
      channel_dii,
      f_matrix,
      "s a i j, s a k band j -> s a k band i",
    )
  else:
    raise ValueError(
      "channel_dii must have shape [atom, ch, ch] or [spin, atom, ch, ch]. "
      f"Got {channel_dii.shape}."
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
  channel_dii: Float[Array, "... atom channel channel"],
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
  channel_dii = jnp.asarray(channel_dii)
  if channel_dii.ndim == 3:
    df_matrix = einsum(
      channel_dii,
      f_matrix,
      "a i j, s a k band j -> s a k band i",
    )
  elif channel_dii.ndim == 4:
    df_matrix = einsum(
      channel_dii,
      f_matrix,
      "s a i j, s a k band j -> s a k band i",
    )
  else:
    raise ValueError(
      "channel_dii must have shape [atom, ch, ch] or [spin, atom, ch, ch]. "
      f"Got {channel_dii.shape}."
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


def _channel_to_beta_one_hot(
  channel_beta: Int[Array, "atom channel"],
  num_beta: int,
  dtype,
) -> Float[Array, "atom channel beta"]:
  """Return a one-hot channel->beta map for vectorized contractions."""
  return jax.nn.one_hot(
    jnp.asarray(channel_beta, dtype=jnp.int32),
    num_classes=num_beta,
    dtype=dtype,
  )


def channel_pair_multipoles(
  pair_density: Complex[Array, "spin atom channel channel"],
  channel_coupling: Float[Array, "atom channel channel l m"],
  channel_beta: Int[Array, "atom channel"],
  max_beta: int,
) -> Complex[Array, "spin atom beta beta l m"]:
  """Accumulate channel-pair density into one-center beta-pair multipoles."""
  one_hot = _channel_to_beta_one_hot(
    channel_beta,
    max_beta,
    pair_density.real.dtype,
  )
  return einsum(
    pair_density,
    one_hot,
    one_hot,
    channel_coupling,
    "s a i j, a i p, a j q, a i j l m -> s a p q l m",
  )


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
  local_potential_r: Float[Array, "... x y z"],
  channel_dii: Float[Array, "atom channel channel"],
  channel_coupling: Float[Array, "atom channel channel l m"],
  channel_beta: Int[Array, "atom channel"],
  radial_fields: Float[Array, "atom beta beta l x y z"],
  harmonics: Float[Array, "atom l m x y z"],
  vol: float,
  channel_mask: Float[Array, "atom channel"] | None = None,
) -> Float[Array, "atom channel channel"]:
  """Build the fixed-density ultrasoft effective channel matrix."""

  def _single(local_potential_spin):
    num_grids = np.prod(local_potential_spin.shape)
    basis_integrals = einsum(
      radial_fields,
      harmonics,
      local_potential_spin,
      "a i j l x y z, a l m x y z, x y z -> a i j l m",
    ) * (vol / num_grids)
    mask = jnp.asarray(channel_mask) if channel_mask is not None else None
    one_hot = _channel_to_beta_one_hot(
      channel_beta,
      basis_integrals.shape[1],
      basis_integrals.dtype,
    )
    gathered = einsum(
      basis_integrals,
      one_hot,
      one_hot,
      "a p q l m, a i p, a j q -> a i j l m",
    )
    atom_matrix = channel_dii + jnp.sum(
      channel_coupling * gathered,
      axis=(-1, -2),
    )
    if mask is not None:
      atom_matrix = atom_matrix * (mask[..., None] * mask[:, None, :])
    return atom_matrix

  local_potential_r = jnp.asarray(local_potential_r)
  if local_potential_r.ndim == 3:
    return _single(local_potential_r)
  if local_potential_r.ndim == 4:
    return jax.vmap(_single)(local_potential_r)
  raise ValueError(
    "local_potential_r must have shape [x, y, z] or [spin, x, y, z]. "
    f"Got {local_potential_r.shape}."
  )


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
