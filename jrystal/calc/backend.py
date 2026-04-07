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
"""Electronic-structure backends (AE, NC).

Each backend encapsulates the physics that differs between all-electron,
norm-conserving pseudopotential, and (future) ultrasoft calculations.
Workflow solvers only call the backend protocol methods, so they are
agnostic to which backend is active.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np

from .._src import energy as _energy
from .._src import hamiltonian as _hamiltonian
from .._src import pw as _pw
from ..pseudopotential import normcons as _normcons
from ..terminal_ui import stage_line
from .pre_calc import pre_calc_beta_sbt

if TYPE_CHECKING:
  from ..config import JrystalConfigDict
  from .runtime import RuntimeContext

# ---------------------------------------------------------------------------
# All-electron backend
# ---------------------------------------------------------------------------


class AllElectronBackend:
  """All-electron (no pseudopotential) backend."""

  def __init__(self, config: JrystalConfigDict):
    self.xc = config.method.xc

  def build_potentials(self, ctx: RuntimeContext) -> RuntimeContext:
    # All-electron has no extra potentials to precompute.
    return ctx

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
    g_vec = ctx.g_vec
    ksampling = ctx.ksampling

    pseudopot = create_pseudopotential(self._config, crystal=crystal)

    stage_line("Init", "Initializing pseudopotential (local)...")
    potential_loc = _normcons.potential_local_reciprocal(
      crystal.positions,
      g_vec,
      pseudopot.r_grid,
      pseudopot.local_potential_grid,
      pseudopot.local_potential_charge,
      crystal.vol,
    )

    stage_line(
      "Init", "Initializing pseudopotential (Spherical Bessel Transform)..."
    )
    beta_gk = pre_calc_beta_sbt(
      pseudopot,
      np.array(g_vec),
      np.array(ksampling.kpts),
    )

    if ksampling.mode == "mesh":
      stage_line("Init", "Initializing pseudopotential (nonlocal)...")
      potential_nl = _normcons.potential_nonlocal_psi_reciprocal(
        crystal.positions,
        g_vec,
        ksampling.kpts,
        pseudopot.r_grid,
        pseudopot.nonlocal_beta_grid,
        pseudopot.nonlocal_angular_momentum,
        pseudopot.nonlocal_d_matrix,
        beta_gk,
      )
    else:
      # For band-path mode, keep SBT cache; nonlocal is built per k-point
      # in the band solver.
      potential_nl = beta_gk

    return ctx.replace(
      pseudopotential=pseudopot,
      potential_local=potential_loc,
      potential_nonlocal=potential_nl,
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

    def _trace(c):
      return _normcons.hamiltonian_trace(
        c,
        density,
        ctx.potential_local,
        ctx.potential_nonlocal,
        g_vec,
        k_vec,
        crystal.vol,
        xc=self.xc,
        kohn_sham=True,
      )

    return jax.grad(_trace)(coeff) / 2.0

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
        _normcons.energy_nonlocal(
          coeff,
          ctx.potential_nonlocal,
          vol=vol,
          occupation=occ,
          kpts_weights=k_weights,
        ),
      "xc":
        _energy.xc_energy(density, g_vec, vol, self.xc, kohn_sham=False),
    }

  def num_electrons(self, ctx: RuntimeContext) -> int:
    """Valence electron count from pseudopotential."""
    return int(np.sum(ctx.pseudopotential.valence_charges))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_backend(
  config: JrystalConfigDict
) -> AllElectronBackend | NormConservingBackend:
  """Select the appropriate backend from config."""
  if config.method.use_pseudopotential:
    pp_type = config.method.pseudopotential_type
    if pp_type in ("nc", "normcons", "normconserving"):
      return NormConservingBackend(config)
    raise NotImplementedError(
      f"Pseudopotential type '{pp_type}' is not yet supported. "
      "Only norm-conserving ('nc') is available."
    )
  return AllElectronBackend(config)


__all__ = [
  "AllElectronBackend",
  "NormConservingBackend",
  "get_backend",
]
