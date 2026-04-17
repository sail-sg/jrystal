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
"""Normalized pseudopotential setup objects shared across families."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Optional

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float, Int

from .._src.crystal import Crystal
from .load import find_upf, parse_upf
from .spherical import batch_sph_harm_real, cartesian_to_spherical

PseudoFamily = Literal["nc", "us", "paw"]


def _normalize_family(family: str) -> PseudoFamily:
  family_key = family.lower()
  aliases = {
    "nc": "nc",
    "normcons": "nc",
    "normconserving": "nc",
    "us": "us",
    "ultrasoft": "us",
    "paw": "paw",
  }
  try:
    return aliases[family_key]
  except KeyError as exc:
    raise ValueError(f"Unsupported pseudopotential family: {family}") from exc


def divide_on_positive_grid(
  values: Array,
  grid: Array,
  *,
  power: int = 1,
  dtype=np.float64,
) -> np.ndarray:
  """Safely divide by ``grid**power`` while keeping masked entries zero."""
  numerator = np.asarray(values, dtype=dtype)
  denominator = np.asarray(grid, dtype=dtype)**power
  output = np.zeros_like(numerator, dtype=dtype)
  np.divide(numerator, denominator, where=(denominator > 0), out=output)
  return output


@dataclass(frozen=True)
class RadialMesh:
  r_g: Float[Array, "num_r"]
  dr_g: Float[Array, "num_r"]


@dataclass(frozen=True)
class LocalChannel:
  vloc_r: Float[Array, "num_r"]
  z_valence: int


@dataclass(frozen=True)
class ProjectorChannel:
  beta_jr: Float[Array, "num_beta num_r"]
  l_j: Int[Array, "num_beta"]
  d_jj: Float[Array, "num_beta num_beta"]
  cutoff_radii: tuple[float, ...]
  channel_map: "ProjectorChannelMap"


@dataclass(frozen=True)
class ProjectorChannelMap:
  channel_beta: Int[Array, "num_channel"]
  channel_l: Int[Array, "num_channel"]
  channel_m: Int[Array, "num_channel"]


@dataclass(frozen=True)
class AugmentationChannel:
  q_jj: Float[Array, "num_beta num_beta"]
  q_jjlr: Float[Array, "num_beta num_beta num_l num_r"]
  q_with_l: bool


@dataclass(frozen=True)
class PseudoSpeciesSetup:
  family: PseudoFamily
  symbol: str
  valence_charge: int
  radial: RadialMesh
  local: LocalChannel
  projectors: ProjectorChannel
  l_max: int
  l_max_rho: int | None
  valence_configuration: tuple[dict, ...]
  source_path: str
  num_pseudo_waves: int | None = None
  augmentation: Optional[AugmentationChannel] = None
  nlcc_r: Optional[Float[Array, "num_r"]] = None


@dataclass(frozen=True)
class AtomSpeciesMap:
  positions: Float[Array, "atom 3"]
  species_index: Int[Array, "atom"]
  species_symbols: tuple[str, ...]


@dataclass(frozen=True)
class BasePseudoCache:
  family: PseudoFamily
  species_setups: tuple[PseudoSpeciesSetup, ...]
  atom_species_map: AtomSpeciesMap
  vloc_g: Float[Array, "x y z"]
  beta_radial_gk: tuple[Float[Array, "kpt beta x y z"], ...]
  projector_gk: Optional[object] = None
  projector_mask: Optional[Float[Array, "atom beta"]] = None


@dataclass(frozen=True)
class UltrasoftBaseCache(BasePseudoCache):
  q_matrices: Float[Array, "atom beta beta"] | None = None
  channel_qii: Float[Array, "atom channel channel"] | None = None
  channel_mask: Float[Array, "atom channel"] | None = None
  channel_beta: Int[Array, "atom channel"] | None = None
  channel_dii: Float[Array, "atom channel channel"] | None = None
  channel_coupling: Float[Array, "atom channel channel l m"] | None = None
  augmentation_radial_fields_g: Float[Array, "atom beta beta l x y z"] | None = None
  augmentation_harmonics_g: Float[Array, "atom l m x y z"] | None = None
  nlcc_g: Optional[Float[Array, "x y z"]] = None
  augmentation_l_max: int = 0
  has_angular_augmentation: bool = False


@dataclass(frozen=True)
class UltrasoftMeshCache(UltrasoftBaseCache):
  channel_projectors_gk: Complex[Array, "atom kpt channel x y z"] | None = None
  channel_projectors_compact_gk: Complex[Array, "atom kpt channel gpt"] | None = None


@dataclass(frozen=True)
class UltrasoftPathCache(UltrasoftBaseCache):
  pass


@dataclass(frozen=True)
class PAWCache(BasePseudoCache):
  ghat_g: Optional[tuple[Array, ...]] = None


def build_atom_species_map(crystal: Crystal) -> AtomSpeciesMap:
  """Map each atom in the crystal to a deduplicated species index."""
  species_symbols: list[str] = []
  species_lookup: dict[str, int] = {}
  species_index: list[int] = []

  for symbol in crystal.symbols:
    idx = species_lookup.get(symbol)
    if idx is None:
      idx = len(species_symbols)
      species_lookup[symbol] = idx
      species_symbols.append(symbol)
    species_index.append(idx)

  return AtomSpeciesMap(
    positions=np.asarray(crystal.positions, dtype=np.float64),
    species_index=np.asarray(species_index, dtype=np.int32),
    species_symbols=tuple(species_symbols),
  )


def _build_projector_channel(pp_dict: dict) -> ProjectorChannel:
  mesh = np.asarray(pp_dict["PP_MESH"]["PP_R"], dtype=np.float64)
  radial_mask = mesh > 0

  beta_entries = pp_dict["PP_NONLOCAL"].get("PP_BETA", [])
  if beta_entries:
    beta_values = np.stack(
      [divide_on_positive_grid(beta["values"], mesh) for beta in beta_entries],
      axis=0,
    )[:, radial_mask]
    l_j = np.asarray(
      [int(beta["angular_momentum"]) for beta in beta_entries],
      dtype=np.int32,
    )
    cutoff_radii = tuple(float(beta["cutoff_radius"]) for beta in beta_entries)
    num_beta = len(beta_entries)
    d_jj = np.asarray(
      pp_dict["PP_NONLOCAL"]["PP_DIJ"],
      dtype=np.float64,
    ).reshape(num_beta, num_beta) / 2.0
  else:
    beta_values = np.zeros((1, int(np.sum(radial_mask))), dtype=np.float64)
    l_j = np.zeros((1,), dtype=np.int32)
    cutoff_radii = (0.0,)
    d_jj = np.zeros((1, 1), dtype=np.float64)

  return ProjectorChannel(
    beta_jr=beta_values,
    l_j=l_j,
    d_jj=d_jj,
    cutoff_radii=cutoff_radii,
    channel_map=_build_projector_channel_map(l_j),
  )


def _build_projector_channel_map(
  l_j: Int[Array, "num_beta"],
) -> ProjectorChannelMap:
  channel_beta: list[int] = []
  channel_l: list[int] = []
  channel_m: list[int] = []

  for beta_idx, angular_momentum in enumerate(np.asarray(l_j, dtype=np.int32)):
    l_val = int(angular_momentum)
    for m_val in range(-l_val, l_val + 1):
      channel_beta.append(beta_idx)
      channel_l.append(l_val)
      channel_m.append(m_val)

  return ProjectorChannelMap(
    channel_beta=np.asarray(channel_beta, dtype=np.int32),
    channel_l=np.asarray(channel_l, dtype=np.int32),
    channel_m=np.asarray(channel_m, dtype=np.int32),
  )


def _build_channel_overlap_matrix(
  q_jj: np.ndarray,
  channel_map: ProjectorChannelMap,
) -> np.ndarray:
  num_channel = len(channel_map.channel_beta)
  q_ii = np.zeros((num_channel, num_channel), dtype=np.float64)

  for i in range(num_channel):
    beta_i = int(channel_map.channel_beta[i])
    l_i = int(channel_map.channel_l[i])
    m_i = int(channel_map.channel_m[i])
    for j in range(num_channel):
      beta_j = int(channel_map.channel_beta[j])
      l_j = int(channel_map.channel_l[j])
      m_j = int(channel_map.channel_m[j])
      if l_i == l_j and m_i == m_j:
        q_ii[i, j] = q_jj[beta_i, beta_j]

  return q_ii


def _pad_channel_tensor(
  matrix: np.ndarray,
  max_channel: int,
) -> np.ndarray:
  return np.pad(
    matrix,
    (
      (0, max_channel - matrix.shape[0]),
      (0, max_channel - matrix.shape[1]),
    ),
  )


def _pad_channel_coupling(
  coupling: np.ndarray,
  *,
  max_channel: int,
  l_max_aug: int,
) -> np.ndarray:
  output = np.zeros(
    (max_channel, max_channel, l_max_aug + 1, 2 * l_max_aug + 1),
    dtype=np.float64,
  )
  local_l_max = coupling.shape[2] - 1
  local_m = coupling.shape[3]
  m_offset = l_max_aug - local_l_max
  output[
    :coupling.shape[0],
    :coupling.shape[1],
    :local_l_max + 1,
    m_offset:m_offset + local_m,
  ] = coupling
  return output


def _pad_harmonics_grid(
  harmonics: list[np.ndarray],
  l_max: int,
) -> np.ndarray:
  sample_shape = harmonics[0].shape[:-1]
  output = np.zeros((l_max + 1, 2 * l_max + 1, *sample_shape), dtype=np.float64)
  for l_val, values in enumerate(harmonics):
    m_offset = l_max - l_val
    output[l_val, m_offset:m_offset + 2 * l_val + 1] = np.moveaxis(
      values,
      -1,
      0,
    )
  return output


@lru_cache(maxsize=None)
def _real_gaunt_lookup(
  l_max_projector: int,
  l_max_aug: int,
  num_theta: int = 32,
  num_phi: int = 64,
) -> dict[tuple[int, int, int, int, int, int], float]:
  cos_theta, theta_weights = np.polynomial.legendre.leggauss(num_theta)
  theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
  phi = np.linspace(0.0, 2.0 * np.pi, num_phi, endpoint=False)
  theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
  quad_weight = theta_weights[:, None] * (2.0 * np.pi / num_phi)

  flat_theta = jnp.asarray(theta_grid.reshape(-1))
  flat_phi = jnp.asarray(phi_grid.reshape(-1))
  max_l = max(l_max_projector, l_max_aug)
  harmonics = {
    # ``batch_sph_harm_real()`` uses the local convention
    # (theta=azimuth, phi=polar).  The quadrature grid above is generated in
    # the SciPy/JAX convention (theta=polar, phi=azimuth), so swap them here.
    l_val: np.asarray(batch_sph_harm_real(l_val, flat_phi, flat_theta)).reshape(
      num_theta,
      num_phi,
      2 * l_val + 1,
    )
    for l_val in range(max_l + 1)
  }
  gram_inverse = {
    l_val: np.linalg.inv(
      np.einsum(
        "tpm,tpn,tp->mn",
        harmonics[l_val],
        harmonics[l_val],
        quad_weight,
      ),
    )
    for l_val in range(l_max_aug + 1)
  }

  coeffs: dict[tuple[int, int, int, int, int, int], float] = {}
  for l1 in range(l_max_projector + 1):
    for m1 in range(-l1, l1 + 1):
      y1 = harmonics[l1][..., m1 + l1]
      for l2 in range(l_max_projector + 1):
        for m2 in range(-l2, l2 + 1):
          y2 = harmonics[l2][..., m2 + l2]
          product = y1 * y2
          l_min = abs(l1 - l2)
          l_hi = min(l1 + l2, l_max_aug)
          for l3 in range(l_min, l_hi + 1):
            rhs = np.einsum(
              "tp,tpm,tp->m",
              product,
              harmonics[l3],
              quad_weight,
            )
            expansion = gram_inverse[l3] @ rhs
            for m3 in range(-l3, l3 + 1):
              coeffs[(l1, m1, l2, m2, l3, m3)] = float(expansion[m3 + l3])

  return coeffs


def _build_channel_augmentation_coupling(
  channel_map: ProjectorChannelMap,
  l_max_aug: int,
) -> np.ndarray:
  num_channel = len(channel_map.channel_beta)
  coupling = np.zeros(
    (num_channel, num_channel, l_max_aug + 1, 2 * l_max_aug + 1),
    dtype=np.float64,
  )
  l_values = np.asarray(channel_map.channel_l, dtype=np.int32)
  l_max_projector = int(np.max(l_values)) if l_values.size else 0
  gaunt = _real_gaunt_lookup(l_max_projector, l_max_aug)

  for i in range(num_channel):
    l_i = int(channel_map.channel_l[i])
    m_i = int(channel_map.channel_m[i])
    for j in range(num_channel):
      l_j = int(channel_map.channel_l[j])
      m_j = int(channel_map.channel_m[j])
      l_min = abs(l_i - l_j)
      l_hi = min(l_i + l_j, l_max_aug)
      for l_val in range(l_min, l_hi + 1):
        for m_val in range(-l_val, l_val + 1):
          coupling[i, j, l_val, m_val + l_max_aug] = gaunt[
            (l_i, m_i, l_j, m_j, l_val, m_val)
          ]

  return coupling


def _real_spherical_harmonics_grid(
  relative_vectors: np.ndarray,
  l_max: int,
) -> list[np.ndarray]:
  spherical = cartesian_to_spherical(jnp.asarray(relative_vectors))
  theta = spherical[..., 1]
  phi = spherical[..., 2]
  return [
    np.asarray(batch_sph_harm_real(l_val, theta, phi), dtype=np.float64)
    for l_val in range(l_max + 1)
  ]


def _expand_projector_channels(
  projector_gk,
  atom_setups: tuple[PseudoSpeciesSetup, ...],
) -> tuple[jnp.ndarray, jnp.ndarray]:
  max_channel = max(
    len(setup.projectors.channel_map.channel_beta)
    for setup in atom_setups
  )
  phi_center = projector_gk.projectors.shape[3] // 2

  expanded = []
  channel_mask = []

  for atom_idx, setup in enumerate(atom_setups):
    channel_map = setup.projectors.channel_map
    num_channel = len(channel_map.channel_beta)
    atom_projectors = np.asarray(projector_gk.projectors[atom_idx])
    atom_expanded = np.zeros(
      (atom_projectors.shape[0], max_channel, *atom_projectors.shape[-3:]),
      dtype=atom_projectors.dtype,
    )

    for channel_idx, (beta_idx, m_val) in enumerate(
      zip(channel_map.channel_beta, channel_map.channel_m)
    ):
      phi_idx = int(phi_center + m_val)
      atom_expanded[:, channel_idx] = atom_projectors[:, int(beta_idx), phi_idx]

    expanded.append(atom_expanded)
    channel_mask.append(
      np.pad(
        np.ones(num_channel, dtype=np.float64),
        ((0, max_channel - num_channel),),
      )
    )

  return jnp.asarray(expanded), jnp.asarray(channel_mask)


def _build_compact_projector_channels(
  beta_radial_gk: tuple[jnp.ndarray, ...],
  atom_setups: tuple[PseudoSpeciesSetup, ...],
  atom_species_map: AtomSpeciesMap,
  g_vec: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  freq_mask: Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Build compact ultrasoft channel projectors directly on the active G mask."""
  max_channel = max(
    len(setup.projectors.channel_map.channel_beta)
    for setup in atom_setups
  )
  g_indices = np.flatnonzero(np.asarray(freq_mask).reshape(-1))
  active_g = np.asarray(g_vec).reshape(-1, 3)[g_indices]
  kpts_np = np.asarray(kpts)
  gk_vectors = kpts_np[:, None, :] + active_g[None, :, :]

  l_max = max(
    int(np.max(np.asarray(setup.projectors.l_j)))
    if np.asarray(setup.projectors.l_j).size else 0
    for setup in atom_setups
  )
  spherical = cartesian_to_spherical(jnp.asarray(gk_vectors))
  theta = spherical[..., 1]
  phi = spherical[..., 2]
  harmonics_by_l = {
    l_val: np.asarray(batch_sph_harm_real(l_val, theta, phi))
    for l_val in range(l_max + 1)
  }

  species_beta_compact = []
  for beta in beta_radial_gk:
    beta_flat = np.asarray(beta).reshape(beta.shape[:2] + (-1,))
    species_beta_compact.append(beta_flat[..., g_indices])

  atom_positions = np.asarray(atom_species_map.positions, dtype=np.float64)
  projector_channels = []
  channel_mask = []
  for atom_idx, setup in enumerate(atom_setups):
    channel_map = setup.projectors.channel_map
    num_channel = len(channel_map.channel_beta)
    species_idx = int(atom_species_map.species_index[atom_idx])
    beta_compact = species_beta_compact[species_idx]
    structure_factor = np.exp(
      -1.0j * np.einsum("kgd,d->kg", gk_vectors, atom_positions[atom_idx])
    )
    atom_projectors = np.zeros(
      (kpts_np.shape[0], max_channel, active_g.shape[0]),
      dtype=np.result_type(beta_compact.dtype, np.complex64),
    )

    for channel_idx, (beta_idx, l_val, m_val) in enumerate(
      zip(
        np.asarray(channel_map.channel_beta, dtype=np.int32),
        np.asarray(channel_map.channel_l, dtype=np.int32),
        np.asarray(channel_map.channel_m, dtype=np.int32),
        strict=True,
      )
    ):
      y_lm = harmonics_by_l[int(l_val)][..., int(m_val) + int(l_val)]
      atom_projectors[:, channel_idx] = (
        4.0 * np.pi *
        (1.0j ** int(l_val)) *
        y_lm *
        beta_compact[:, int(beta_idx)] *
        structure_factor
      )

    projector_channels.append(atom_projectors)
    channel_mask.append(
      np.pad(
        np.ones(num_channel, dtype=np.float64),
        ((0, max_channel - num_channel),),
      )
    )

  return jnp.asarray(projector_channels), jnp.asarray(channel_mask)


def squeeze_projector_channels_to_mask(
  projector_channels_gk: Complex[Array, "atom kpt channel x y z"],
  freq_mask: Array,
) -> jnp.ndarray:
  """Project full reciprocal-grid channel projectors onto the active G basis."""
  flat = np.asarray(projector_channels_gk).reshape(
    projector_channels_gk.shape[:3] + (-1,),
  )
  g_indices = np.flatnonzero(np.asarray(freq_mask).reshape(-1))
  return jnp.asarray(flat[..., g_indices])


def _build_channel_mask(
  atom_setups: tuple[PseudoSpeciesSetup, ...],
  max_channel: int,
) -> jnp.ndarray:
  masks = []
  for setup in atom_setups:
    num_channel = len(setup.projectors.channel_map.channel_beta)
    masks.append(
      np.pad(
        np.ones(num_channel, dtype=np.float64),
        ((0, max_channel - num_channel),),
      )
    )
  return jnp.asarray(masks)


def build_uspp_projector_channels_for_k(
  cache: UltrasoftBaseCache,
  g_vec: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  kpt_index: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Build full ultrasoft channel projectors lazily for a single k-point."""
  from .nloc import potential_nonlocal_psi_reciprocal

  atom_setups = expand_species_setups(cache.species_setups, cache.atom_species_map)
  atom_positions = jnp.asarray(cache.atom_species_map.positions)
  atom_r_grid = [setup.radial.r_g for setup in atom_setups]
  atom_beta = [setup.projectors.beta_jr for setup in atom_setups]
  atom_l = [setup.projectors.l_j for setup in atom_setups]
  atom_d = [setup.projectors.d_jj for setup in atom_setups]
  species_beta_at_k = tuple(
    beta.at[kpt_index:kpt_index + 1].get() for beta in cache.beta_radial_gk
  )
  atom_beta_gk = [
    species_beta_at_k[int(idx)] for idx in cache.atom_species_map.species_index
  ]

  projector_gk = potential_nonlocal_psi_reciprocal(
    atom_positions,
    g_vec,
    kpts[kpt_index:kpt_index + 1],
    atom_r_grid,
    atom_beta,
    atom_l,
    atom_d,
    atom_beta_gk,
  )
  return _expand_projector_channels(projector_gk, atom_setups)


def _build_augmentation_channel(
  pp_dict: dict,
  radial_mask: np.ndarray,
) -> Optional[AugmentationChannel]:
  augmentation = pp_dict["PP_NONLOCAL"].get("PP_AUGMENTATION")
  if augmentation is None:
    return None

  num_beta = len(pp_dict["PP_NONLOCAL"].get("PP_BETA", []))
  if num_beta == 0:
    return None

  q_values = augmentation.get("PP_Q")
  if q_values is None:
    q_jj = np.zeros((num_beta, num_beta), dtype=np.float64)
  else:
    q_jj = np.asarray(q_values, dtype=np.float64).reshape(num_beta, num_beta)

  q_with_l = bool(augmentation["q_with_l"])
  num_l = (
    int(pp_dict["PP_HEADER"].get("l_max_rho") or 0) + 1
    if q_with_l else 1
  )
  q_jjlr = np.zeros(
    (num_beta, num_beta, num_l, int(np.sum(radial_mask))),
    dtype=np.float64,
  )
  mesh = np.asarray(pp_dict["PP_MESH"]["PP_R"], dtype=np.float64)

  for qij in augmentation.get("PP_QIJ", []):
    i = int(qij["first_index"]) - 1
    j = int(qij["second_index"]) - 1
    angular_momentum = 0 if not q_with_l else int(qij["angular_momentum"])
    values = divide_on_positive_grid(qij["values"], mesh, power=2)[radial_mask]
    q_jjlr[i, j, angular_momentum] = values
    if i != j:
      q_jjlr[j, i, angular_momentum] = values

  return AugmentationChannel(
    q_jj=q_jj,
    q_jjlr=q_jjlr,
    q_with_l=q_with_l,
  )


def _build_nlcc_channel(
  pp_dict: dict,
  radial_mask: np.ndarray,
) -> Optional[np.ndarray]:
  nlcc = pp_dict.get("PP_NLCC")
  if nlcc is None:
    return None
  # QE USPP/PAW files store NLCC on the radial mesh directly as the
  # pseudized core charge density used for XC, unlike PP_RHOATOM which is
  # multiplied by 4*pi*r^2.
  return np.asarray(nlcc, dtype=np.float64)[radial_mask]


def _build_species_setup(
  symbol: str,
  pp_dict: dict,
  family: PseudoFamily,
  source_path: str,
) -> PseudoSpeciesSetup:
  mesh = np.asarray(pp_dict["PP_MESH"]["PP_R"], dtype=np.float64)
  radial_mask = mesh > 0
  valence_configuration = tuple(pp_dict["PP_INFO"]["Valence configuration"])
  number_of_wfc = pp_dict["PP_HEADER"].get("number_of_wfc")
  num_pseudo_waves = (
    int(number_of_wfc)
    if number_of_wfc is not None else
    (len(valence_configuration) if valence_configuration else None)
  )
  radial = RadialMesh(
    r_g=mesh[radial_mask],
    dr_g=np.asarray(pp_dict["PP_MESH"]["PP_RAB"], dtype=np.float64)[radial_mask],
  )

  valence_charge = int(float(pp_dict["PP_HEADER"]["z_valence"]))
  local = LocalChannel(
    vloc_r=np.asarray(pp_dict["PP_LOCAL"], dtype=np.float64)[radial_mask] / 2.0,
    z_valence=valence_charge,
  )

  l_max_rho = pp_dict["PP_HEADER"].get("l_max_rho")
  return PseudoSpeciesSetup(
    family=family,
    symbol=symbol,
    valence_charge=valence_charge,
    radial=radial,
    local=local,
    projectors=_build_projector_channel(pp_dict),
    l_max=int(pp_dict["PP_HEADER"]["l_max"]),
    l_max_rho=None if l_max_rho is None else int(l_max_rho),
    valence_configuration=valence_configuration,
    source_path=source_path,
    num_pseudo_waves=num_pseudo_waves,
    augmentation=_build_augmentation_channel(pp_dict, radial_mask)
    if family == "us" else None,
    nlcc_r=_build_nlcc_channel(pp_dict, radial_mask),
  )


def load_species_setups(
  crystal: Crystal,
  pseudo_dir: str,
  family: str,
) -> tuple[tuple[PseudoSpeciesSetup, ...], AtomSpeciesMap]:
  """Load one normalized setup per unique species in the crystal."""
  normalized_family = _normalize_family(family)
  atom_species_map = build_atom_species_map(crystal)

  setups = []
  for symbol in atom_species_map.species_symbols:
    pp_path = find_upf(pseudo_dir, symbol)
    setups.append(
      _build_species_setup(
        symbol,
        parse_upf(pp_path),
        normalized_family,
        pp_path,
      )
    )

  return tuple(setups), atom_species_map


def expand_species_setups(
  setups: tuple[PseudoSpeciesSetup, ...],
  atom_species_map: AtomSpeciesMap,
) -> tuple[PseudoSpeciesSetup, ...]:
  """Expand per-species setup data to atom order."""
  return tuple(setups[int(idx)] for idx in atom_species_map.species_index)


def build_pseudo_cache(
  setups: tuple[PseudoSpeciesSetup, ...],
  atom_species_map: AtomSpeciesMap,
  g_vec: Float[Array, "x y z 3"],
  r_vec: Float[Array, "x y z 3"],
  ksampling,
  vol: float,
  freq_mask: Array | None = None,
) -> BasePseudoCache | UltrasoftBaseCache:
  """Build reusable pseudo caches from normalized species setups."""
  from .beta import beta_sbt_grid
  from .local import potential_local_reciprocal
  from .nloc import potential_nonlocal_psi_reciprocal

  family = setups[0].family if setups else "nc"
  atom_setups = expand_species_setups(setups, atom_species_map)
  atom_positions = jnp.asarray(atom_species_map.positions)

  atom_r_grid = [setup.radial.r_g for setup in atom_setups]
  atom_vloc = [setup.local.vloc_r for setup in atom_setups]
  atom_z = [setup.local.z_valence for setup in atom_setups]
  atom_beta = [setup.projectors.beta_jr for setup in atom_setups]
  atom_l = [setup.projectors.l_j for setup in atom_setups]
  atom_d = [setup.projectors.d_jj for setup in atom_setups]

  vloc_g = potential_local_reciprocal(
    atom_positions,
    g_vec,
    atom_r_grid,
    atom_vloc,
    atom_z,
    vol,
  )

  beta_radial_gk = tuple(
    jnp.asarray(arr)
    for arr in beta_sbt_grid(
      [setup.radial.r_g for setup in setups],
      [setup.projectors.beta_jr for setup in setups],
      [setup.projectors.l_j for setup in setups],
      np.asarray(g_vec),
      np.asarray(ksampling.kpts),
    )
  )
  atom_beta_gk = [beta_radial_gk[int(idx)] for idx in atom_species_map.species_index]

  projector_gk = None
  projector_mask = None
  if ksampling.mode == "mesh":
    if not (family == "us" and freq_mask is not None):
      projector_gk = potential_nonlocal_psi_reciprocal(
        atom_positions,
        g_vec,
        ksampling.kpts,
        atom_r_grid,
        atom_beta,
        atom_l,
        atom_d,
        atom_beta_gk,
      )
      projector_mask = projector_gk.projector_mask

  if family != "us":
    return BasePseudoCache(
      family=family,
      species_setups=setups,
      atom_species_map=atom_species_map,
      vloc_g=vloc_g,
      beta_radial_gk=beta_radial_gk,
      projector_gk=projector_gk,
      projector_mask=projector_mask,
    )

  max_beta = max(setup.projectors.beta_jr.shape[0] for setup in atom_setups)
  max_channel = max(
    len(setup.projectors.channel_map.channel_beta)
    for setup in atom_setups
  )
  l_max_aug = max(
    (
      setup.augmentation.q_jjlr.shape[2] - 1
      for setup in atom_setups
      if setup.augmentation is not None
    ),
    default=0,
  )
  atom_radius_grid = np.linalg.norm(
    np.asarray(r_vec)[None, ...] - np.asarray(atom_positions)[:, None, None, None, :],
    axis=-1,
  )

  def _interpolate_radial_field(
    radius_grid: np.ndarray,
    radial_grid: np.ndarray,
    values: np.ndarray,
  ) -> np.ndarray:
    return np.interp(
      radius_grid.reshape(-1),
      radial_grid,
      values,
      left=float(values[0]),
      right=0.0,
    ).reshape(radius_grid.shape)

  q_matrices = []
  channel_qii = []
  channel_beta = []
  channel_dii = []
  channel_coupling = []
  augmentation_radial_fields = []
  augmentation_harmonics = []
  nlcc_grid = np.zeros(r_vec.shape[:3], dtype=np.float64)
  has_nlcc = False
  has_angular_augmentation = False

  atom_relative_vectors = (
    np.asarray(r_vec)[None, ...] - np.asarray(atom_positions)[:, None, None, None, :]
  )

  for setup, radius_grid, relative_vectors in zip(
    atom_setups,
    atom_radius_grid,
    atom_relative_vectors,
  ):
    num_beta = setup.projectors.beta_jr.shape[0]
    num_channel = len(setup.projectors.channel_map.channel_beta)
    q_atom = np.zeros((max_beta, max_beta), dtype=np.float64)
    d_channel_atom = _pad_channel_tensor(
      _build_channel_overlap_matrix(
        np.asarray(setup.projectors.d_jj),
        setup.projectors.channel_map,
      ),
      max_channel,
    )
    channel_beta_atom = np.pad(
      np.asarray(setup.projectors.channel_map.channel_beta, dtype=np.int32),
      ((0, max_channel - num_channel),),
    )
    radial_fields_atom = np.zeros(
      (max_beta, max_beta, l_max_aug + 1, *r_vec.shape[:3]),
      dtype=np.float64,
    )
    harmonics_atom = np.zeros(
      (l_max_aug + 1, 2 * l_max_aug + 1, *r_vec.shape[:3]),
      dtype=np.float64,
    )
    coupling_atom = np.zeros(
      (max_channel, max_channel, l_max_aug + 1, 2 * l_max_aug + 1),
      dtype=np.float64,
    )

    if setup.augmentation is not None:
      q_atom[:num_beta, :num_beta] = np.asarray(setup.augmentation.q_jj)
      q_channel = _build_channel_overlap_matrix(
        np.asarray(setup.augmentation.q_jj),
        setup.projectors.channel_map,
      )
      local_l_max = setup.augmentation.q_jjlr.shape[2] - 1
      coupling_atom = _pad_channel_coupling(
        _build_channel_augmentation_coupling(
          setup.projectors.channel_map,
          local_l_max,
        ),
        max_channel=max_channel,
        l_max_aug=l_max_aug,
      )
      harmonics_atom = _pad_harmonics_grid(
        _real_spherical_harmonics_grid(relative_vectors, l_max_aug),
        l_max_aug,
      )

      for i in range(num_beta):
        for j in range(num_beta):
          for l_val in range(local_l_max + 1):
            radial_values = np.asarray(setup.augmentation.q_jjlr[i, j, l_val])
            if not np.any(np.abs(radial_values) > 0):
              continue
            radial_fields_atom[i, j, l_val] = _interpolate_radial_field(
              radius_grid,
              np.asarray(setup.radial.r_g),
              radial_values,
            )

      if setup.augmentation.q_with_l and local_l_max > 0:
        has_angular_augmentation = has_angular_augmentation or bool(
          np.any(np.abs(np.asarray(setup.augmentation.q_jjlr[:, :, 1:])) > 0)
        )
    else:
      q_channel = np.zeros((num_channel, num_channel), dtype=np.float64)

    if setup.nlcc_r is not None:
      nlcc_grid += _interpolate_radial_field(
        radius_grid,
        np.asarray(setup.radial.r_g),
        np.asarray(setup.nlcc_r),
      )
      has_nlcc = True

    q_matrices.append(q_atom)
    channel_qii.append(_pad_channel_tensor(q_channel, max_channel))
    channel_beta.append(channel_beta_atom)
    channel_dii.append(d_channel_atom)
    channel_coupling.append(coupling_atom)
    augmentation_radial_fields.append(radial_fields_atom)
    augmentation_harmonics.append(harmonics_atom)

  common_kwargs = dict(
    family=family,
    species_setups=setups,
    atom_species_map=atom_species_map,
    vloc_g=vloc_g,
    beta_radial_gk=beta_radial_gk,
    projector_gk=projector_gk,
    projector_mask=projector_mask,
    q_matrices=jnp.asarray(q_matrices),
    channel_qii=jnp.asarray(channel_qii),
    channel_mask=_build_channel_mask(atom_setups, max_channel),
    channel_beta=jnp.asarray(channel_beta),
    channel_dii=jnp.asarray(channel_dii),
    channel_coupling=jnp.asarray(channel_coupling),
    augmentation_radial_fields_g=jnp.asarray(augmentation_radial_fields),
    augmentation_harmonics_g=jnp.asarray(augmentation_harmonics),
    nlcc_g=jnp.asarray(nlcc_grid) if has_nlcc else None,
    augmentation_l_max=l_max_aug,
    has_angular_augmentation=has_angular_augmentation,
  )

  if projector_gk is None:
    if ksampling.mode == "mesh" and freq_mask is not None:
      channel_projectors_compact_gk, channel_mask = _build_compact_projector_channels(
        beta_radial_gk,
        atom_setups,
        atom_species_map,
        g_vec,
        ksampling.kpts,
        freq_mask,
      )
      return UltrasoftMeshCache(
        **{
          **common_kwargs,
          "channel_mask": channel_mask,
        },
        channel_projectors_gk=None,
        channel_projectors_compact_gk=channel_projectors_compact_gk,
      )
    return UltrasoftPathCache(**common_kwargs)

  channel_projectors_gk, _ = _expand_projector_channels(
    projector_gk,
    atom_setups,
  )
  channel_projectors_compact_gk = None
  if freq_mask is not None:
    channel_projectors_compact_gk = squeeze_projector_channels_to_mask(
      channel_projectors_gk,
      freq_mask,
    )

  return UltrasoftMeshCache(
    **common_kwargs,
    channel_projectors_gk=channel_projectors_gk,
    channel_projectors_compact_gk=channel_projectors_compact_gk,
  )


__all__ = [
  "AtomSpeciesMap",
  "AugmentationChannel",
  "BasePseudoCache",
  "LocalChannel",
  "PAWCache",
  "ProjectorChannel",
  "ProjectorChannelMap",
  "PseudoSpeciesSetup",
  "RadialMesh",
  "UltrasoftBaseCache",
  "UltrasoftMeshCache",
  "UltrasoftPathCache",
  "build_uspp_projector_channels_for_k",
  "build_atom_species_map",
  "build_pseudo_cache",
  "expand_species_setups",
  "divide_on_positive_grid",
  "load_species_setups",
  "squeeze_projector_channels_to_mask",
]
