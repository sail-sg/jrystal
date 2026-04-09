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
"""Ultrasoft overlap helpers and compatibility wrappers."""

from __future__ import annotations

import jax.numpy as jnp
from einops import einsum
from jaxtyping import Array, Complex, Float, Int

from .augmentation import (
  augmentation_density as _augmentation_density_low_rank,
  augmentation_density_compact as _augmentation_density_compact,
  channel_nonlocal_apply,
  channel_nonlocal_apply_compact,
  channel_nonlocal_energy_compact,
  effective_channel_matrix,
  kinetic_apply,
  local_potential_apply,
)


def projector_channel_overlap(
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  projector_channels: Complex[Array, "atom kpt channel x y z"],
  channel_mask: Float[Array, "atom channel"] | None = None,
) -> Complex[Array, "spin atom kpt band channel"]:
  """Project wavefunctions onto full ultrasoft channel projectors."""
  f_matrix = einsum(
    pw_coefficients,
    projector_channels,
    "s k band x y z, a k channel x y z -> s a k band channel",
  )
  if channel_mask is not None:
    f_matrix = f_matrix * channel_mask[None, :, None, None, :]
  return f_matrix


def projector_channel_overlap_compact(
  pw_coefficients: Complex[Array, "spin kpt gpt band"],
  projector_channels: Complex[Array, "atom kpt channel gpt"],
  channel_mask: Float[Array, "atom channel"] | None = None,
) -> Complex[Array, "spin atom kpt band channel"]:
  """Project compact-G wavefunctions onto compact ultrasoft channel projectors."""
  coeff_band_g = jnp.swapaxes(pw_coefficients, -1, -2)
  f_matrix = einsum(
    coeff_band_g,
    projector_channels,
    "s k band g, a k channel g -> s a k band channel",
  )
  if channel_mask is not None:
    f_matrix = f_matrix * channel_mask[None, :, None, None, :]
  return f_matrix


def overlap_apply(
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  projector_channels: Complex[Array, "atom kpt channel x y z"],
  channel_qii: Float[Array, "atom channel channel"],
  vol: float,
  channel_mask: Float[Array, "atom channel"] | None = None,
) -> Complex[Array, "spin kpt band x y z"]:
  """Apply the ultrasoft overlap operator ``S`` in full channel basis."""
  if channel_qii is None or projector_channels is None:
    return pw_coefficients

  f_matrix = projector_channel_overlap(
    pw_coefficients,
    projector_channels,
    channel_mask=channel_mask,
  )

  qf_matrix = einsum(
    channel_qii,
    f_matrix,
    "a i j, s a k band j -> s a k band i",
  )
  correction = einsum(
    jnp.conj(projector_channels),
    qf_matrix,
    "a k i x y z, s a k band i -> s k band x y z",
  ) / vol
  return pw_coefficients + correction


def overlap_apply_compact(
  pw_coefficients: Complex[Array, "spin kpt gpt band"],
  projector_channels: Complex[Array, "atom kpt channel gpt"],
  channel_qii: Float[Array, "atom channel channel"],
  vol: float,
  channel_mask: Float[Array, "atom channel"] | None = None,
) -> Complex[Array, "spin kpt gpt band"]:
  """Apply the ultrasoft overlap operator in compact G-space."""
  if channel_qii is None or projector_channels is None:
    return pw_coefficients

  f_matrix = projector_channel_overlap_compact(
    pw_coefficients,
    projector_channels,
    channel_mask=channel_mask,
  )
  qf_matrix = einsum(
    channel_qii,
    f_matrix,
    "a i j, s a k band j -> s a k band i",
  )
  correction = einsum(
    jnp.conj(projector_channels),
    qf_matrix,
    "a k i g, s a k band i -> s k band g",
  ) / vol
  return pw_coefficients + jnp.swapaxes(correction, -1, -2)


def subspace_overlap_matrix(
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  projector_channels: Complex[Array, "atom kpt channel x y z"],
  channel_qii: Float[Array, "atom channel channel"],
  vol: float,
  channel_mask: Float[Array, "atom channel"] | None = None,
) -> Complex[Array, "spin kpt band band"]:
  """Return the projected overlap matrix ``C^H S C``."""
  s_coefficients = overlap_apply(
    pw_coefficients,
    projector_channels,
    channel_qii,
    vol,
    channel_mask=channel_mask,
  )
  overlap_matrix = einsum(
    jnp.conj(pw_coefficients),
    s_coefficients,
    "s k b1 x y z, s k b2 x y z -> s k b1 b2",
  )
  return 0.5 * (
    overlap_matrix + jnp.swapaxes(jnp.conj(overlap_matrix), -1, -2)
  )


def subspace_overlap_matrix_compact(
  pw_coefficients: Complex[Array, "spin kpt gpt band"],
  projector_channels: Complex[Array, "atom kpt channel gpt"],
  channel_qii: Float[Array, "atom channel channel"],
  vol: float,
  channel_mask: Float[Array, "atom channel"] | None = None,
) -> Complex[Array, "spin kpt band band"]:
  """Return the projected overlap matrix ``C^H S C`` in compact G-space."""
  s_coefficients = overlap_apply_compact(
    pw_coefficients,
    projector_channels,
    channel_qii,
    vol,
    channel_mask=channel_mask,
  )
  coeff_band_g = jnp.swapaxes(pw_coefficients, -1, -2)
  s_band_g = jnp.swapaxes(s_coefficients, -1, -2)
  overlap_matrix = einsum(
    jnp.conj(coeff_band_g),
    s_band_g,
    "s k b1 g, s k b2 g -> s k b1 b2",
  )
  return 0.5 * (
    overlap_matrix + jnp.swapaxes(jnp.conj(overlap_matrix), -1, -2)
  )


def subspace_overlap_eigenvalues(
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  projector_channels: Complex[Array, "atom kpt channel x y z"],
  channel_qii: Float[Array, "atom channel channel"],
  vol: float,
  channel_mask: Float[Array, "atom channel"] | None = None,
) -> Float[Array, "spin kpt band"]:
  """Return eigenvalues of ``C^H S C`` for overlap diagnostics."""
  overlap_matrix = subspace_overlap_matrix(
    pw_coefficients,
    projector_channels,
    channel_qii,
    vol,
    channel_mask=channel_mask,
  )
  return jnp.linalg.eigvalsh(overlap_matrix).real


def subspace_overlap_eigenvalues_compact(
  pw_coefficients: Complex[Array, "spin kpt gpt band"],
  projector_channels: Complex[Array, "atom kpt channel gpt"],
  channel_qii: Float[Array, "atom channel channel"],
  vol: float,
  channel_mask: Float[Array, "atom channel"] | None = None,
) -> Float[Array, "spin kpt band"]:
  """Return eigenvalues of ``C^H S C`` for compact-G overlap diagnostics."""
  overlap_matrix = subspace_overlap_matrix_compact(
    pw_coefficients,
    projector_channels,
    channel_qii,
    vol,
    channel_mask=channel_mask,
  )
  return jnp.linalg.eigvalsh(overlap_matrix).real


def overlap_inv_sqrt_apply(
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  projector_channels: Complex[Array, "atom kpt channel x y z"],
  channel_qii: Float[Array, "atom channel channel"],
  vol: float,
  channel_mask: Float[Array, "atom channel"] | None = None,
  eps: float = 1e-10,
) -> Complex[Array, "spin kpt band x y z"]:
  """Canonicalize a coefficient batch so that ``C^H S C = I``."""
  overlap_matrix = subspace_overlap_matrix(
    pw_coefficients,
    projector_channels,
    channel_qii,
    vol,
    channel_mask=channel_mask,
  )

  eigvals, eigvecs = jnp.linalg.eigh(overlap_matrix)
  inv_sqrt = jnp.reciprocal(jnp.sqrt(jnp.clip(eigvals.real, eps, None)))
  overlap_inv_sqrt = jnp.einsum(
    "skbi,ski,skci->skbc",
    eigvecs,
    inv_sqrt,
    jnp.conj(eigvecs),
  )
  return jnp.einsum(
    "skbxyz,skbc->skcxyz",
    pw_coefficients,
    overlap_inv_sqrt,
  )


def overlap_inv_sqrt_apply_compact(
  pw_coefficients: Complex[Array, "spin kpt gpt band"],
  projector_channels: Complex[Array, "atom kpt channel gpt"],
  channel_qii: Float[Array, "atom channel channel"],
  vol: float,
  channel_mask: Float[Array, "atom channel"] | None = None,
  eps: float = 1e-10,
) -> Complex[Array, "spin kpt gpt band"]:
  """Canonicalize compact coefficients so that ``C^H S C = I``."""
  overlap_matrix = subspace_overlap_matrix_compact(
    pw_coefficients,
    projector_channels,
    channel_qii,
    vol,
    channel_mask=channel_mask,
  )
  eigvals, eigvecs = jnp.linalg.eigh(overlap_matrix)
  inv_sqrt = jnp.reciprocal(jnp.sqrt(jnp.clip(eigvals.real, eps, None)))
  overlap_inv_sqrt = jnp.einsum(
    "skbi,ski,skci->skbc",
    eigvecs,
    inv_sqrt,
    jnp.conj(eigvecs),
  )
  return jnp.einsum(
    "skgb,skbc->skgc",
    pw_coefficients,
    overlap_inv_sqrt,
  )


def augmentation_density(
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  occupation: Float[Array, "spin kpt band"],
  projector_channels: Complex[Array, "atom kpt channel x y z"],
  augmentation_radial_fields_g: Float[Array, "atom beta beta l x y z"] | None,
  augmentation_harmonics_g: Float[Array, "atom l m x y z"] | None,
  channel_coupling: Float[Array, "atom channel channel l m"] | None,
  channel_beta: Int[Array, "atom channel"] | None,
  vol: float,
  kpts_weights: Float[Array, "kpt"] | None = None,
  channel_mask: Float[Array, "atom channel"] | None = None,
) -> Float[Array, "spin x y z"]:
  """Build ultrasoft augmentation density from the one-center cache."""
  return _augmentation_density_low_rank(
    pw_coefficients,
    occupation,
    projector_channels,
    augmentation_radial_fields_g,
    augmentation_harmonics_g,
    channel_coupling,
    channel_beta,
    vol,
    kpts_weights=kpts_weights,
    channel_mask=channel_mask,
  )


def augmentation_density_compact(
  pw_coefficients: Complex[Array, "spin kpt gpt band"],
  occupation: Float[Array, "spin kpt band"],
  projector_channels: Complex[Array, "atom kpt channel gpt"],
  augmentation_radial_fields_g: Float[Array, "atom beta beta l x y z"] | None,
  augmentation_harmonics_g: Float[Array, "atom l m x y z"] | None,
  channel_coupling: Float[Array, "atom channel channel l m"] | None,
  channel_beta: Int[Array, "atom channel"] | None,
  vol: float,
  kpts_weights: Float[Array, "kpt"] | None = None,
  channel_mask: Float[Array, "atom channel"] | None = None,
) -> Float[Array, "spin x y z"]:
  """Build ultrasoft augmentation density from compact-G projectors."""
  return _augmentation_density_compact(
    pw_coefficients,
    occupation,
    projector_channels,
    augmentation_radial_fields_g,
    augmentation_harmonics_g,
    channel_coupling,
    channel_beta,
    vol,
    kpts_weights=kpts_weights,
    channel_mask=channel_mask,
  )


__all__ = [
  "augmentation_density",
  "augmentation_density_compact",
  "channel_nonlocal_apply",
  "channel_nonlocal_apply_compact",
  "channel_nonlocal_energy_compact",
  "effective_channel_matrix",
  "kinetic_apply",
  "local_potential_apply",
  "overlap_apply",
  "overlap_apply_compact",
  "overlap_inv_sqrt_apply",
  "overlap_inv_sqrt_apply_compact",
  "projector_channel_overlap",
  "projector_channel_overlap_compact",
  "subspace_overlap_eigenvalues_compact",
  "subspace_overlap_eigenvalues",
  "subspace_overlap_matrix_compact",
  "subspace_overlap_matrix",
]
