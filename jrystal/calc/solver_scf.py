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
"""Self-consistent field (SCF) ground-state solver.

Iterates:  diagonalise -> update occupations -> mix density
until self-consistency in the density is reached.

The Hamiltonian application (H|psi>) is delegated to the
:class:`~jrystal.calc.backend.ElectronicBackend`, making this
solver agnostic to AE vs NC vs USPP physics.
"""
from __future__ import annotations

import time
from math import ceil
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging

from .._src import pw as _pw
from .._src.linalg import batched_lobpcg
from .._src.utils import expand_coefficient
from ..smearing import fermi_dirac, find_chemical_potential
from .types import EnergyDecomposition, GroundStateResult

if TYPE_CHECKING:
  from ..config import JrystalConfigDict
  from .backend import AllElectronBackend, NormConservingBackend
  from .runtime import RuntimeContext

from .density_mixing import (
  diis_init,
  diis_update,
  kerker_preconditioner,
  simple_mixing,
)


def _fixed_occupation(evals, num_electrons, occ_max=2.0):
  """Assign identical occupied-band count at every k-point (insulator)."""
  num_occ = int(round(num_electrons / occ_max))
  order = jnp.argsort(evals, axis=-1)
  rank = jnp.argsort(order, axis=-1)
  return jnp.where(rank < num_occ, occ_max, 0.0).astype(evals.dtype)


def _compute_occupation(evals, num_electrons, k_weights, smearing):
  """Compute Fermi-Dirac or fixed occupation."""
  if smearing > 0:
    mu = find_chemical_potential(
      evals, num_electrons, smearing=smearing, k_weights=k_weights,
    )
    return fermi_dirac(evals, mu, smearing=smearing)
  return _fixed_occupation(evals, num_electrons)


# ---------------------------------------------------------------------------
# SCF solver
# ---------------------------------------------------------------------------

def run_scf(
  config: JrystalConfigDict,
  ctx: RuntimeContext,
  backend: AllElectronBackend | NormConservingBackend,
) -> GroundStateResult:
  """Run a ground-state calculation via the SCF loop.

  The loop:
    1. Diagonalise H using LOBPCG (matrix-free).
    2. Update occupation numbers.
    3. Compute new density.
    4. Check convergence.
    5. Mix density (DIIS + linear mixing).

  Args:
    config: Jrystal configuration.
    ctx: Fully initialised runtime context.
    backend: Electronic-structure backend.

  Returns:
    Ground-state result with converged energy, density, eigenvalues.
  """
  key = jax.random.PRNGKey(config.execution.seed)
  crystal = ctx.crystal

  if crystal.spin != 0 or not config.system.spin_restricted:
    raise NotImplementedError(
      "SCF currently supports only spin-restricted calculations "
      "with system.spin == 0."
    )

  g_vec = ctx.g_vec
  freq_mask = ctx.basis.freq_mask
  ew = ctx.ewald_energy
  k_weights = ctx.ksampling.weights

  num_electrons = backend.num_electrons(ctx)
  num_kpts = ctx.ksampling.kpts.shape[0]
  num_bands = ceil(num_electrons / 2) + config.occupation.empty_bands
  smearing = config.occupation.smearing

  scf_max_iter = config.solver.get("scf_max_iter", config.solver.epoch)
  lobpcg_max_iter = config.solver.get("lobpcg_max_iter", 6)
  mixing_beta = config.solver.get("mixing_beta", 0.8)
  diis_max_hist = config.solver.get("diis_max_hist", 8)
  convergence_tol = config.solver.convergence_condition

  occ_max = 2.0  # spin-restricted
  logging.info(f"Crystal: {crystal.symbols}")
  logging.info(f"SCF: max_iter={scf_max_iter}, lobpcg_iter={lobpcg_max_iter}")
  logging.info(f"num_bands: {num_bands}, smearing: {smearing}")

  # --- Init wavefunctions (compact, in masked G-space) ---
  pw_params = _pw.param_init(
    key, num_bands, num_kpts, freq_mask,
    spin_restricted=config.system.spin_restricted,
  )
  coeff_compact = pw_params["w_re"] + 1.0j * pw_params["w_im"]
  coeff_compact = jnp.linalg.qr(coeff_compact)[0]  # [s, k, g, band]

  # --- Init eigenvalues, occupation, density ---
  evals = jax.random.normal(key, [1, num_kpts, num_bands])
  evals = jnp.sort(evals, axis=-1)
  occ = _fixed_occupation(evals, num_electrons, occ_max)

  def _density_from_compact(c, occ):
    coeff_full = expand_coefficient(c, freq_mask)
    return _pw.density_grid(
      coeff_full, crystal.vol, occ, k_weights=k_weights,
    )

  density = _density_from_compact(coeff_compact, occ)

  # --- Preconditioner ---
  precond = kerker_preconditioner(g_vec, freq_mask)

  # --- DIIS state ---
  diis_state = diis_init(
    max_hist=diis_max_hist, density_shape=density.shape, dtype=density.dtype,
  )

  # --- Hvp via backend ---
  def _hvp(coeff_compact_conj, dens):
    coeff_full = expand_coefficient(coeff_compact_conj.conj(), freq_mask)
    return backend.hamiltonian_apply(coeff_full, dens, ctx)

  def _lobpcg_matmul(c, dens):
    """Wraps Hvp for batched LOBPCG input shape (s*k, g, band)."""
    return _hvp(jnp.expand_dims(c, axis=0), dens)

  @jax.jit
  def _diagonalise(coeff_compact, dens):
    s, k, g, b = coeff_compact.shape
    eigval, evec = batched_lobpcg(
      matmul=lambda c: _lobpcg_matmul(c, dens).reshape(s * k, g, -1),
      k=b,
      v0=coeff_compact.reshape(s * k, g, b),
      which="smallest",
      preconditioner=precond,
      maxit=lobpcg_max_iter,
      tol=1e-8,
    )
    return evec.reshape(s, k, g, b), eigval.reshape(s, k, b)

  # --- SCF loop ---
  converged = False
  total_energy_history = []
  band_energy = jnp.sum(evals * occ * k_weights[None, :, None])

  for i in range(scf_max_iter):
    t0 = time.time()

    # 1. Diagonalise
    coeff_new, evals_new = _diagonalise(coeff_compact, density)
    coeff_new = coeff_new.conj()

    # 2. Update occupation
    occ = _compute_occupation(
      evals_new, num_electrons, k_weights, smearing,
    )

    # 3. New density
    density_new = _density_from_compact(coeff_new, occ)

    # 4. Check convergence
    band_energy_new = jnp.sum(evals_new * occ * k_weights[None, :, None])
    d_density = float(jnp.mean(jnp.abs(density_new - density)))
    d_energy = float(jnp.abs(band_energy_new - band_energy))
    dt = time.time() - t0

    logging.info(
      f"SCF iter {i + 1}: d_density={d_density:.2e} "
      f"d_energy={d_energy:.2e} ({dt:.2f}s)"
    )

    if d_density < 1e-3 and d_energy < convergence_tol:
      converged = True
      logging.info(f"SCF converged in {i + 1} iterations.")
      coeff_compact = coeff_new
      density = density_new
      band_energy = band_energy_new
      break

    # 5. Density mixing (DIIS + linear)
    dens_error = density_new - density
    diis_state, density_mixed = diis_update(
      diis_state, density_new, dens_error,
    )
    density = simple_mixing(density_mixed, density, beta=mixing_beta)
    band_energy = band_energy_new
    coeff_compact = coeff_new

    total_energy_history.append(float(band_energy))

  if not converged:
    logging.warning("SCF did not converge.")

  # --- Final energy decomposition via backend ---
  coeff_full = expand_coefficient(coeff_compact, freq_mask)
  decomp = backend.energy_decomposition(coeff_full, occ, ctx)
  total_e = float(sum(decomp.values()) + ew)
  density = _density_from_compact(coeff_compact, occ)

  for name, val in decomp.items():
    logging.info(f"{name}: {float(val):.4f} Ha")
  logging.info(f"Ewald: {float(ew):.4f} Ha")
  logging.info(f"Total Energy: {total_e:.4f} Ha")

  return GroundStateResult(
    config=config,
    crystal=crystal,
    params_pw={"w_re": coeff_compact.real, "w_im": coeff_compact.imag},
    params_occ={},
    total_energy=total_e,
    energy_terms=EnergyDecomposition(
      ewald=float(ew),
      **{k: float(v) for k, v in decomp.items()},
    ),
    converged=converged,
    density=density,
    eigenvalues=evals_new,
    total_energy_history=total_energy_history,
  )
