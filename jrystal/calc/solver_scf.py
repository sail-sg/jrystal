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
from absl import logging
from tqdm import tqdm

from .._src import pw as _pw
from .._src.linalg import batched_lobpcg
from .._src.utils import expand_coefficient, squeeze_coefficient
from ..smearing import fermi_dirac, find_chemical_potential
from .types import EnergyDecomposition, GroundStateResult
from .workflow_logging import (
  format_ground_state_iteration,
  log_energy_breakdown,
  log_ground_state_finish,
  log_ground_state_start,
)

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
  overall_start = time.time()
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

  scf_config = config.solver.scf
  if scf_config.eigensolver.method != "lobpcg":
    raise NotImplementedError(
      "SCF currently supports only solver.scf.eigensolver.method='lobpcg'."
    )
  if scf_config.mixing.method != "diis":
    raise NotImplementedError(
      "SCF currently supports only solver.scf.mixing.method='diis'."
    )

  scf_max_iter = scf_config.max_iter
  lobpcg_max_iter = scf_config.eigensolver.max_iter
  mixing_beta = scf_config.mixing.beta
  diis_max_hist = scf_config.mixing.history_size
  density_tol = scf_config.convergence.density_tol
  energy_tol = scf_config.convergence.energy_tol

  occ_max = 2.0  # spin-restricted
  logging.info(f"Crystal: {crystal.symbols}")
  log_ground_state_start(
    "SCF",
    max_steps=scf_max_iter,
    num_bands=num_bands,
    smearing=smearing,
    xc=config.method.xc,
    controls=(
      f"eigensolver=lobpcg(max_iter={lobpcg_max_iter}) | "
      f"mixing=diis(beta={mixing_beta:.3f}, hist={diis_max_hist}) | "
      f"density_tol={density_tol:.2e} | energy_tol={energy_tol:.2e}"
    ),
  )

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
  evals_new = evals
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
    hpsi_full = backend.hamiltonian_apply(coeff_full, dens, ctx)
    return squeeze_coefficient(hpsi_full, freq_mask)

  @jax.jit
  def _diagonalise(coeff_compact, dens):
    s, k, g, b = coeff_compact.shape

    def _lobpcg_matmul(c):
      coeff_batch = c.reshape(s, k, g, -1)
      return _hvp(coeff_batch, dens).reshape(s * k, g, -1)

    eigval, evec = batched_lobpcg(
      matmul=_lobpcg_matmul,
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

  iters = tqdm(
    range(scf_max_iter),
    disable=not config.execution.verbose,
  )
  for i in iters:
    t0 = time.time()

    # 1. Diagonalise
    coeff_new, evals_new = _diagonalise(coeff_compact, density)
    coeff_new = coeff_new.conj()

    # 2. Update occupation
    occ = _compute_occupation(
      evals_new, num_electrons, k_weights, smearing,
    )

    # 3. New density
    coeff_full_new = expand_coefficient(coeff_new, freq_mask)
    density_new = _pw.density_grid(
      coeff_full_new, crystal.vol, occ, k_weights=k_weights,
    )

    # 4. Check convergence
    band_energy_new = jnp.sum(evals_new * occ * k_weights[None, :, None])
    total_energy_new = float(backend.total_energy(coeff_full_new, occ, ctx) + ew)
    delta_total_energy = None
    if total_energy_history:
      delta_total_energy = abs(total_energy_new - total_energy_history[-1])
    total_energy_history.append(total_energy_new)
    d_density = float(jnp.mean(jnp.abs(density_new - density)))
    d_energy = float(jnp.abs(band_energy_new - band_energy))
    dt = time.time() - t0

    iters.set_description(
      format_ground_state_iteration(
        "SCF",
        step=i + 1,
        max_steps=scf_max_iter,
        total_energy=total_energy_new,
        delta_energy=delta_total_energy,
        step_time=dt,
        density_delta=d_density,
      ),
      refresh=False,
    )

    if d_density < density_tol and d_energy < energy_tol:
      converged = True
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

  # --- Final energy decomposition via backend ---
  coeff_full = expand_coefficient(coeff_compact, freq_mask)
  decomp = backend.energy_decomposition(coeff_full, occ, ctx)
  total_e = float(sum(decomp.values()) + ew)
  density = _density_from_compact(coeff_compact, occ)
  wall_time = time.time() - overall_start

  log_ground_state_finish(
    "SCF",
    converged=converged,
    steps_completed=len(total_energy_history),
    max_steps=scf_max_iter,
    total_energy=total_e,
    wall_time=wall_time,
  )
  log_energy_breakdown(
    "SCF",
    decomp,
    ewald=ew,
    total_energy=total_e,
  )

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
