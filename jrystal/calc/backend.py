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
"""Electronic-structure backends (AE, NC, US).

Each backend encapsulates the physics that differs between all-electron,
norm-conserving pseudopotential, and ultrasoft calculations.
Workflow solvers only call the backend protocol methods, so they are
agnostic to which backend is active.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from chex import dataclass as chex_dataclass

from .._src import energy as _energy
from .._src import hamiltonian as _hamiltonian
from .._src import potential as _potential
from .._src import pw as _pw
from .._src.utils import expand_coefficient, squeeze_coefficient
from ..pseudopotential import normcons as _normcons
from ..pseudopotential import ultrasoft as _ultrasoft
from ..pseudopotential.kernel import (
  UltrasoftBaseCache,
  UltrasoftMeshCache,
  UltrasoftPathCache,
  build_pseudo_cache,
  build_uspp_projector_channels_for_k,
)
from ..terminal_ui import stage_line
from .types import KPointOperatorBundle

if TYPE_CHECKING:
  from ..config import JrystalConfigDict
  from .runtime import RuntimeContext


def _ion_charge_dtype():
  return jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32


# ---------------------------------------------------------------------------
# All-electron backend
# ---------------------------------------------------------------------------


@chex_dataclass
class UltrasoftIterationState:
  """Fixed-density operator state reused across one SCF diagonalisation."""

  local_potential_r: object
  channel_dii: object


class AllElectronBackend:
  """All-electron (no pseudopotential) backend."""

  def __init__(self, config: JrystalConfigDict):
    self.xc = config.method.xc

  def build_potentials(self, ctx: RuntimeContext) -> RuntimeContext:
    # All-electron has no extra potentials to precompute.
    return ctx

  def ion_charges(self, ctx: RuntimeContext):
    """Bare nuclear charges for all-electron Ewald."""
    return jnp.asarray(ctx.crystal.charges, dtype=_ion_charge_dtype())

  def total_energy(self, coeff, occ, ctx: RuntimeContext):
    """Electronic total energy (excluding Ewald)."""
    crystal = ctx.crystal
    g_vec = ctx.g_vec
    k_vec = ctx.ksampling.kpts
    k_weights = ctx.ksampling.weights
    vol = crystal.vol

    density = _pw.density_grid(coeff, vol, occ, k_weights=k_weights)
    density_reciprocal = _pw.density_grid_reciprocal(
      coeff,
      vol,
      occ,
      k_weights=k_weights,
    )
    kin = _energy.kinetic(
      coeff,
      g_vec,
      k_vec,
      kpts_weights=k_weights,
      occupation=occ,
    )
    hart = _energy.hartree(density_reciprocal, g_vec, vol)
    ext = _energy.external(
      density_reciprocal,
      crystal.positions,
      crystal.charges,
      g_vec,
      vol,
    )
    xc = _energy.xc_energy(density, g_vec, vol, self.xc, kohn_sham=False)
    return kin + hart + ext + xc

  def hamiltonian_apply(self, coeff, density, ctx: RuntimeContext):
    """H|psi> via grad of hamiltonian trace (used by SCF eigensolver)."""
    crystal = ctx.crystal
    g_vec = ctx.g_vec
    k_vec = ctx.ksampling.kpts
    k_weights = ctx.ksampling.weights

    def _trace(c):
      return _hamiltonian.hamiltonian_matrix_trace(
        c,
        crystal.positions,
        crystal.charges,
        density,
        crystal.vol,
        g_vec,
        k_vec,
        kpts_weights=k_weights,
        xc=self.xc,
        kohn_sham=True,
        keep_spin_axis=False,
      )

    return jax.grad(_trace)(coeff) / 2.0

  def prepare_iteration(self, density, ctx: RuntimeContext):
    del ctx
    return density

  def prepare_nscf(self, density, ctx: RuntimeContext):
    """Reuse the SCF fixed-density preparation for band workflows."""
    return self.prepare_iteration(density, ctx)

  def build_kpoint_operator(
    self, kpt_index: int, nscf_state, ctx: RuntimeContext
  ):
    del kpt_index, nscf_state, ctx
    raise NotImplementedError(
      "Band operator bundles are not implemented for the all-electron backend."
    )

  def overlap_apply(self, coeff, ctx: RuntimeContext):
    """All-electron calculations use the identity overlap."""
    del ctx
    return coeff

  def overlap_inv_sqrt_apply(self, coeff, ctx: RuntimeContext):
    """Canonical transform is a no-op for identity overlap."""
    del ctx
    return coeff

  def energy_decomposition(self, coeff, occ, ctx: RuntimeContext) -> dict:
    """Return individual energy terms for logging."""
    crystal = ctx.crystal
    g_vec = ctx.g_vec
    k_vec = ctx.ksampling.kpts
    k_weights = ctx.ksampling.weights
    vol = crystal.vol

    density = _pw.density_grid(coeff, vol, occ, k_weights=k_weights)
    density_reciprocal = _pw.density_grid_reciprocal(
      coeff,
      vol,
      occ,
      k_weights=k_weights,
    )
    return {
      "kinetic":
        _energy.kinetic(
          coeff,
          g_vec,
          k_vec,
          kpts_weights=k_weights,
          occupation=occ,
        ),
      "hartree":
        _energy.hartree(density_reciprocal, g_vec, vol),
      "external":
        _energy.external(
          density_reciprocal,
          crystal.positions,
          crystal.charges,
          g_vec,
          vol,
        ),
      "xc":
        _energy.xc_energy(density, g_vec, vol, self.xc, kohn_sham=False),
    }

  def num_electrons(self, ctx: RuntimeContext) -> int:
    """Total number of electrons for occupation calculation."""
    return int(ctx.crystal.num_electron)


# ---------------------------------------------------------------------------
# Norm-conserving pseudopotential backend
# ---------------------------------------------------------------------------


class NormConservingBackend:
  """Norm-conserving pseudopotential backend."""

  def __init__(self, config: JrystalConfigDict):
    self.xc = config.method.xc
    self._config = config

  def build_potentials(self, ctx: RuntimeContext) -> RuntimeContext:
    """Precompute local + nonlocal NC potentials and attach to ctx."""
    from .opt_utils import create_pseudopotential
    crystal = ctx.crystal
    pseudopot = create_pseudopotential(self._config, crystal=crystal)

    stage_line(
      "Init",
      "Initializing pseudopotential cache...",
      level="verbose",
    )
    pseudo_cache = build_pseudo_cache(
      pseudopot.species_setups,
      pseudopot.atom_species_map,
      ctx.g_vec,
      ctx.r_vec,
      ctx.ksampling,
      crystal.vol,
      freq_mask=ctx.basis.freq_mask,
      cell_vectors=crystal.cell_vectors,
    )

    potential_loc = pseudo_cache.vloc_g
    if ctx.ksampling.mode == "mesh":
      potential_nl = pseudo_cache.projector_gk
    else:
      potential_nl = pseudo_cache.beta_radial_gk

    return ctx.replace(
      pseudopotential=pseudopot,
      pseudo_cache=pseudo_cache,
      potential_local=potential_loc,
      potential_nonlocal=potential_nl,
    )

  def ion_charges(self, ctx: RuntimeContext):
    """Valence ionic charges for pseudopotential Ewald."""
    return jnp.asarray(
      ctx.pseudopotential.valence_charges,
      dtype=_ion_charge_dtype(),
    )

  def total_energy(self, coeff, occ, ctx: RuntimeContext):
    """Electronic total energy (excluding Ewald)."""
    crystal = ctx.crystal
    g_vec = ctx.g_vec
    k_vec = ctx.ksampling.kpts
    k_weights = ctx.ksampling.weights
    vol = crystal.vol

    density = _pw.density_grid(coeff, vol, occ, k_weights=k_weights)
    density_reciprocal = _pw.density_grid_reciprocal(
      coeff,
      vol,
      occ,
      k_weights=k_weights,
    )
    kin = _energy.kinetic(
      coeff,
      g_vec,
      k_vec,
      kpts_weights=k_weights,
      occupation=occ,
    )
    hart = _energy.hartree(density_reciprocal, g_vec, vol)
    ext_loc = _normcons.energy_local(
      density_reciprocal,
      ctx.potential_local,
      vol=vol,
    )
    projector_channels_compact = getattr(
      ctx.pseudo_cache,
      "channel_projectors_compact_gk",
      None,
    )
    if projector_channels_compact is not None:
      ext_nloc = _ultrasoft.channel_nonlocal_energy_compact(
        squeeze_coefficient(coeff, ctx.basis.freq_mask),
        projector_channels_compact,
        ctx.pseudo_cache.channel_dii,
        vol=vol,
        occupation=occ,
        kpts_weights=k_weights,
        channel_mask=ctx.pseudo_cache.channel_mask,
      )
    else:
      ext_nloc = _normcons.energy_nonlocal(
        coeff,
        ctx.potential_nonlocal,
        vol=vol,
        occupation=occ,
        kpts_weights=k_weights,
      )
    xc = _energy.xc_energy(density, g_vec, vol, self.xc, kohn_sham=False)
    return kin + hart + ext_loc + ext_nloc + xc

  def hamiltonian_apply(self, coeff, density, ctx: RuntimeContext):
    """H|psi> for SCF eigensolver."""
    crystal = ctx.crystal
    g_vec = ctx.g_vec
    k_vec = ctx.ksampling.kpts
    k_weights = ctx.ksampling.weights

    def _trace(c):
      return _normcons.hamiltonian_trace(
        c,
        density,
        ctx.potential_local,
        ctx.potential_nonlocal,
        g_vec,
        k_vec,
        crystal.vol,
        kpts_weights=k_weights,
        xc=self.xc,
        kohn_sham=True,
      )

    return jax.grad(_trace)(coeff) / 2.0

  def prepare_iteration(self, density, ctx: RuntimeContext):
    del ctx
    return density

  def prepare_nscf(self, density, ctx: RuntimeContext):
    """Reuse the SCF fixed-density preparation for band workflows."""
    return self.prepare_iteration(density, ctx)

  def build_kpoint_operator(
    self, kpt_index: int, nscf_state, ctx: RuntimeContext
  ):
    del kpt_index, nscf_state, ctx
    raise NotImplementedError(
      "Band operator bundles are not implemented for the norm-conserving backend."
    )

  def overlap_apply(self, coeff, ctx: RuntimeContext):
    """Norm-conserving calculations use the identity overlap."""
    del ctx
    return coeff

  def overlap_inv_sqrt_apply(self, coeff, ctx: RuntimeContext):
    """Canonical transform is a no-op for identity overlap."""
    del ctx
    return coeff

  def energy_decomposition(self, coeff, occ, ctx: RuntimeContext) -> dict:
    """Return individual energy terms for logging."""
    crystal = ctx.crystal
    g_vec = ctx.g_vec
    k_vec = ctx.ksampling.kpts
    k_weights = ctx.ksampling.weights
    vol = crystal.vol

    density = _pw.density_grid(coeff, vol, occ, k_weights=k_weights)
    density_reciprocal = _pw.density_grid_reciprocal(
      coeff,
      vol,
      occ,
      k_weights=k_weights,
    )
    return {
      "kinetic":
        _energy.kinetic(
          coeff,
          g_vec,
          k_vec,
          kpts_weights=k_weights,
          occupation=occ,
        ),
      "hartree":
        _energy.hartree(density_reciprocal, g_vec, vol),
      "external_local":
        _normcons.energy_local(
          density_reciprocal,
          ctx.potential_local,
          vol=vol,
        ),
      "external_nonlocal":
        (
          _ultrasoft.channel_nonlocal_energy_compact(
            squeeze_coefficient(coeff, ctx.basis.freq_mask),
            ctx.pseudo_cache.channel_projectors_compact_gk,
            ctx.pseudo_cache.channel_dii,
            vol=vol,
            occupation=occ,
            kpts_weights=k_weights,
            channel_mask=ctx.pseudo_cache.channel_mask,
          ) if getattr(ctx.pseudo_cache, "channel_projectors_compact_gk", None)
          is not None else _normcons.energy_nonlocal(
            coeff,
            ctx.potential_nonlocal,
            vol=vol,
            occupation=occ,
            kpts_weights=k_weights,
          )
        ),
      "xc":
        _energy.xc_energy(density, g_vec, vol, self.xc, kohn_sham=False),
    }

  def num_electrons(self, ctx: RuntimeContext) -> int:
    """Valence electron count from pseudopotential."""
    return int(np.rint(np.sum(ctx.pseudopotential.valence_charges)))


# ---------------------------------------------------------------------------
# Ultrasoft pseudopotential backend
# ---------------------------------------------------------------------------


class UltrasoftBackend:
  """Ultrasoft pseudopotential backend.

  Ground-state workflows support mesh mode through the compact-projector path.
  Path-mode runtime/cache construction is also supported for band workflows.
  """

  requires_canonical_transform = True

  def __init__(self, config: JrystalConfigDict):
    self.xc = config.method.xc
    self._config = config

  def build_potentials(self, ctx: RuntimeContext) -> RuntimeContext:
    """Precompute ultrasoft local/nonlocal caches and augmentation grids."""
    from .opt_utils import create_pseudopotential

    if ctx.ksampling.mode not in ("mesh", "path"):
      raise NotImplementedError(
        f"Unsupported ultrasoft sampling mode: {ctx.ksampling.mode}"
      )

    crystal = ctx.crystal
    pseudopot = create_pseudopotential(self._config, crystal=crystal)

    stage_line("Init", "Initializing ultrasoft cache...", level="verbose")
    pseudo_cache = build_pseudo_cache(
      pseudopot.species_setups,
      pseudopot.atom_species_map,
      ctx.g_vec,
      ctx.r_vec,
      ctx.ksampling,
      crystal.vol,
      freq_mask=ctx.basis.freq_mask,
      cell_vectors=crystal.cell_vectors,
    )
    if not isinstance(pseudo_cache, UltrasoftBaseCache):
      raise TypeError("Ultrasoft cache construction returned the wrong type.")
    if pseudo_cache.nlcc_g is not None:
      stage_line(
        "Init",
        "Ultrasoft NLCC will be included in XC density.",
        level="verbose",
      )
    if ctx.ksampling.mode == "mesh":
      if not isinstance(pseudo_cache, UltrasoftMeshCache):
        raise TypeError("USPP mesh mode requires an UltrasoftMeshCache.")
      if pseudo_cache.channel_projectors_compact_gk is None:
        raise ValueError(
          "USPP mesh mode requires compact projector channels for ground-state "
          "workflows."
        )
      num_kpts = int(ctx.ksampling.kpts.shape[0])
      if pseudo_cache.channel_projectors_compact_gk.shape[1] != num_kpts:
        raise ValueError(
          "USPP compact projector k-point dimension does not match runtime "
          "k-sampling."
        )
      if pseudo_cache.channel_qii is None or pseudo_cache.channel_qii.ndim != 3:
        raise ValueError(
          "USPP channel_qii must be a k-independent [atom, ch, ch] tensor."
        )
      if pseudo_cache.channel_dii is None or pseudo_cache.channel_dii.ndim != 3:
        raise ValueError(
          "USPP channel_dii must be a k-independent [atom, ch, ch] tensor."
        )
      if pseudo_cache.channel_mask is None or pseudo_cache.channel_mask.ndim != 2:
        raise ValueError(
          "USPP channel_mask must be a k-independent [atom, ch] tensor."
        )
      stage_line(
        "Init",
        (
          "USPP mesh cache: "
          f"ik={num_kpts} sym={self._config.ksampling.symmetry_reduction} "
          "compact_projectors=on"
        ),
        level="verbose",
      )

    potential_nl = (
      pseudo_cache.projector_gk if isinstance(pseudo_cache, UltrasoftMeshCache)
      else pseudo_cache.beta_radial_gk
    )

    return ctx.replace(
      pseudopotential=pseudopot,
      pseudo_cache=pseudo_cache,
      potential_local=pseudo_cache.vloc_g,
      potential_nonlocal=potential_nl,
    )

  def ion_charges(self, ctx: RuntimeContext):
    """Valence ionic charges for pseudopotential Ewald."""
    return jnp.asarray(
      ctx.pseudopotential.valence_charges,
      dtype=_ion_charge_dtype(),
    )

  def _xc_density(self, total_density, ctx: RuntimeContext):
    xc_density = total_density
    if ctx.pseudo_cache.nlcc_g is not None:
      num_spin = total_density.shape[0] if total_density.ndim == 4 else 1
      xc_density = total_density + (
        ctx.pseudo_cache.nlcc_g[None, ...] / float(num_spin)
      )
    return xc_density

  def _effective_local_potential(
    self,
    total_density,
    ctx: RuntimeContext,
    *,
    kohn_sham: bool,
  ):
    density_reciprocal = jnp.fft.fftn(total_density, axes=range(-3, 0))
    v_hartree = jnp.fft.ifftn(
      _potential.hartree_reciprocal(
        density_reciprocal,
        ctx.g_vec,
        kohn_sham=kohn_sham,
      ),
      axes=range(-3, 0),
    ).real
    v_local = jnp.fft.ifftn(ctx.potential_local, axes=range(-3, 0)).real
    v_xc = _potential.xc_density(
      self._xc_density(total_density, ctx),
      ctx.g_vec,
      xc_type=self.xc,
      kohn_sham=kohn_sham,
    )
    return v_xc + v_local[None, ...] + v_hartree[None, ...]

  def _total_density(self, coeff, occ, ctx: RuntimeContext):
    """Build smooth + augmentation valence density."""
    smooth_density = _pw.density_grid(
      coeff,
      ctx.crystal.vol,
      occ,
      k_weights=ctx.ksampling.weights,
    )
    projector_channels_compact = getattr(
      ctx.pseudo_cache,
      "channel_projectors_compact_gk",
      None,
    )
    if projector_channels_compact is not None:
      augmentation = _ultrasoft.augmentation_density_compact(
        squeeze_coefficient(coeff, ctx.basis.freq_mask),
        occ,
        projector_channels_compact,
        ctx.pseudo_cache.augmentation_radial_fields_g,
        ctx.pseudo_cache.augmentation_harmonics_g,
        ctx.pseudo_cache.channel_coupling,
        ctx.pseudo_cache.channel_beta,
        ctx.crystal.vol,
        kpts_weights=ctx.ksampling.weights,
        channel_mask=ctx.pseudo_cache.channel_mask,
      )
    else:
      augmentation = _ultrasoft.augmentation_density(
        coeff,
        occ,
        ctx.pseudo_cache.channel_projectors_gk,
        ctx.pseudo_cache.augmentation_radial_fields_g,
        ctx.pseudo_cache.augmentation_harmonics_g,
        ctx.pseudo_cache.channel_coupling,
        ctx.pseudo_cache.channel_beta,
        ctx.crystal.vol,
        kpts_weights=ctx.ksampling.weights,
        channel_mask=ctx.pseudo_cache.channel_mask,
      )
    return smooth_density + augmentation

  def prepare_iteration(self, density, ctx: RuntimeContext):
    """Prepare fixed-density USPP operators for one SCF eigensolve."""
    v_eff_local = self._effective_local_potential(
      density,
      ctx,
      kohn_sham=True,
    )
    d_eff_channel = _ultrasoft.effective_channel_matrix(
      v_eff_local,
      ctx.pseudo_cache.channel_dii,
      ctx.pseudo_cache.channel_coupling,
      ctx.pseudo_cache.channel_beta,
      ctx.pseudo_cache.augmentation_radial_fields_g,
      ctx.pseudo_cache.augmentation_harmonics_g,
      ctx.crystal.vol,
      channel_mask=ctx.pseudo_cache.channel_mask,
    )
    return UltrasoftIterationState(
      local_potential_r=v_eff_local,
      channel_dii=d_eff_channel,
    )

  def prepare_nscf(self, density, ctx: RuntimeContext):
    """Reuse the SCF fixed-density USPP preparation for band workflows."""
    return self.prepare_iteration(density, ctx)

  def build_kpoint_operator(
    self,
    kpt_index: int,
    nscf_state: UltrasoftIterationState,
    ctx: RuntimeContext,
  ) -> KPointOperatorBundle:
    """Build a data-only ultrasoft operator bundle for one path k-point."""
    if not isinstance(ctx.pseudo_cache, UltrasoftPathCache):
      raise TypeError(
        "USPP path operator construction requires an UltrasoftPathCache."
      )

    projector_channels_g, channel_mask = build_uspp_projector_channels_for_k(
      ctx.pseudo_cache,
      ctx.g_vec,
      ctx.ksampling.kpts,
      kpt_index,
    )
    return KPointOperatorBundle(
      kpt=ctx.ksampling.kpts[kpt_index:kpt_index + 1],
      local_potential_r=nscf_state.local_potential_r,
      projector_channels_g=projector_channels_g,
      channel_qii=ctx.pseudo_cache.channel_qii,
      channel_dii_eff=nscf_state.channel_dii,
      channel_mask=channel_mask,
      solver_kind="generalized",
    )

  def total_energy(self, coeff, occ, ctx: RuntimeContext):
    """Electronic total energy (excluding Ewald) for ultrasoft PP."""
    crystal = ctx.crystal
    g_vec = ctx.g_vec
    k_vec = ctx.ksampling.kpts
    k_weights = ctx.ksampling.weights
    vol = crystal.vol

    density = self._total_density(coeff, occ, ctx)
    density_reciprocal = jnp.fft.fftn(density, axes=range(-3, 0))

    smooth_density = _pw.density_grid(
      coeff,
      vol,
      occ,
      k_weights=k_weights,
    )
    smooth_density_reciprocal = jnp.fft.fftn(
      smooth_density,
      axes=range(-3, 0),
    )

    kin = _energy.kinetic(
      coeff,
      g_vec,
      k_vec,
      kpts_weights=k_weights,
      occupation=occ,
    )
    hart = _energy.hartree(density_reciprocal, g_vec, vol)
    ext_loc = _normcons.energy_local(
      smooth_density_reciprocal,
      ctx.potential_local,
      vol=vol,
    )
    projector_channels_compact = getattr(
      ctx.pseudo_cache,
      "channel_projectors_compact_gk",
      None,
    )
    if projector_channels_compact is not None:
      ext_nloc = _ultrasoft.channel_nonlocal_energy_compact(
        squeeze_coefficient(coeff, ctx.basis.freq_mask),
        projector_channels_compact,
        ctx.pseudo_cache.channel_dii,
        vol=vol,
        occupation=occ,
        kpts_weights=k_weights,
        channel_mask=ctx.pseudo_cache.channel_mask,
      )
    else:
      ext_nloc = _normcons.energy_nonlocal(
        coeff,
        ctx.potential_nonlocal,
        vol=vol,
        occupation=occ,
        kpts_weights=k_weights,
      )
    xc = _energy.xc_energy(
      self._xc_density(density, ctx),
      g_vec,
      vol,
      self.xc,
      kohn_sham=False,
    )
    return kin + hart + ext_loc + ext_nloc + xc

  def hamiltonian_apply(self, coeff, iteration_state, ctx: RuntimeContext):
    """Apply the fixed-density ultrasoft Hamiltonian explicitly."""
    kinetic = _ultrasoft.kinetic_apply(
      coeff,
      ctx.g_vec,
      ctx.ksampling.kpts,
    )
    local = _ultrasoft.local_potential_apply(
      coeff,
      iteration_state.local_potential_r,
      ctx.crystal.vol,
    )
    projector_channels_compact = getattr(
      ctx.pseudo_cache,
      "channel_projectors_compact_gk",
      None,
    )
    if projector_channels_compact is not None:
      nonlocal_term = expand_coefficient(
        _ultrasoft.channel_nonlocal_apply_compact(
          squeeze_coefficient(coeff, ctx.basis.freq_mask),
          projector_channels_compact,
          iteration_state.channel_dii,
          ctx.crystal.vol,
          channel_mask=ctx.pseudo_cache.channel_mask,
        ),
        ctx.basis.freq_mask,
      )
    else:
      nonlocal_term = _ultrasoft.channel_nonlocal_apply(
        coeff,
        ctx.pseudo_cache.channel_projectors_gk,
        iteration_state.channel_dii,
        ctx.crystal.vol,
        channel_mask=ctx.pseudo_cache.channel_mask,
      )
    return jnp.conj(kinetic + local + nonlocal_term)

  def overlap_apply(self, coeff, ctx: RuntimeContext):
    """Apply the ultrasoft overlap operator."""
    projector_channels_compact = getattr(
      ctx.pseudo_cache,
      "channel_projectors_compact_gk",
      None,
    )
    if projector_channels_compact is not None:
      return expand_coefficient(
        _ultrasoft.overlap_apply_compact(
          squeeze_coefficient(coeff, ctx.basis.freq_mask),
          projector_channels_compact,
          ctx.pseudo_cache.channel_qii,
          ctx.crystal.vol,
          channel_mask=ctx.pseudo_cache.channel_mask,
        ),
        ctx.basis.freq_mask,
      )
    return _ultrasoft.overlap_apply(
      coeff,
      ctx.pseudo_cache.channel_projectors_gk,
      ctx.pseudo_cache.channel_qii,
      ctx.crystal.vol,
      channel_mask=ctx.pseudo_cache.channel_mask,
    )

  def overlap_inv_sqrt_apply(self, coeff, ctx: RuntimeContext):
    """Canonicalize a coefficient batch in the ultrasoft overlap metric."""
    projector_channels_compact = getattr(
      ctx.pseudo_cache,
      "channel_projectors_compact_gk",
      None,
    )
    if projector_channels_compact is not None:
      return expand_coefficient(
        _ultrasoft.overlap_inv_sqrt_apply_compact(
          squeeze_coefficient(coeff, ctx.basis.freq_mask),
          projector_channels_compact,
          ctx.pseudo_cache.channel_qii,
          ctx.crystal.vol,
          channel_mask=ctx.pseudo_cache.channel_mask,
        ),
        ctx.basis.freq_mask,
      )
    return _ultrasoft.overlap_inv_sqrt_apply(
      coeff,
      ctx.pseudo_cache.channel_projectors_gk,
      ctx.pseudo_cache.channel_qii,
      ctx.crystal.vol,
      channel_mask=ctx.pseudo_cache.channel_mask,
    )

  def energy_decomposition(self, coeff, occ, ctx: RuntimeContext) -> dict:
    """Return individual energy terms for logging."""
    crystal = ctx.crystal
    g_vec = ctx.g_vec
    k_vec = ctx.ksampling.kpts
    k_weights = ctx.ksampling.weights
    vol = crystal.vol

    density = self._total_density(coeff, occ, ctx)
    density_reciprocal = jnp.fft.fftn(density, axes=range(-3, 0))

    smooth_density = _pw.density_grid(
      coeff,
      vol,
      occ,
      k_weights=k_weights,
    )
    smooth_density_reciprocal = jnp.fft.fftn(
      smooth_density,
      axes=range(-3, 0),
    )

    return {
      "kinetic":
        _energy.kinetic(
          coeff,
          g_vec,
          k_vec,
          kpts_weights=k_weights,
          occupation=occ,
        ),
      "hartree":
        _energy.hartree(density_reciprocal, g_vec, vol),
      "external_local":
        _normcons.energy_local(
          smooth_density_reciprocal,
          ctx.potential_local,
          vol=vol,
        ),
      "external_nonlocal":
        (
          _ultrasoft.channel_nonlocal_energy_compact(
            squeeze_coefficient(coeff, ctx.basis.freq_mask),
            ctx.pseudo_cache.channel_projectors_compact_gk,
            ctx.pseudo_cache.channel_dii,
            vol=vol,
            occupation=occ,
            kpts_weights=k_weights,
            channel_mask=ctx.pseudo_cache.channel_mask,
          ) if getattr(ctx.pseudo_cache, "channel_projectors_compact_gk", None)
          is not None else _normcons.energy_nonlocal(
            coeff,
            ctx.potential_nonlocal,
            vol=vol,
            occupation=occ,
            kpts_weights=k_weights,
          )
        ),
      "xc":
        _energy.xc_energy(
          self._xc_density(density, ctx),
          g_vec,
          vol,
          self.xc,
          kohn_sham=False,
        ),
    }

  def num_electrons(self, ctx: RuntimeContext) -> int:
    """Valence electron count from pseudopotential."""
    return int(np.rint(np.sum(ctx.pseudopotential.valence_charges)))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_backend(
  config: JrystalConfigDict
) -> AllElectronBackend | NormConservingBackend | UltrasoftBackend:
  """Select the appropriate backend from config."""
  family = str(config.method.family).lower()
  if family == "ae":
    return AllElectronBackend(config)
  if family == "nc":
    return NormConservingBackend(config)
  if family == "us":
    return UltrasoftBackend(config)
  raise NotImplementedError(f"Method family '{family}' is not yet supported.")


__all__ = [
  "AllElectronBackend",
  "NormConservingBackend",
  "get_backend",
]
