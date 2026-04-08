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

import signal
import time
from functools import partial
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .._src import pw as _pw
from .._src.linalg import batched_lobpcg
from .._src.utils import expand_coefficient, squeeze_coefficient
from ..io import make_checkpoint_manager, save_checkpoint
from ..pseudopotential.kernel import UltrasoftMeshCache
from ..pseudopotential import ultrasoft as _ultrasoft
from ..smearing import fermi_dirac, find_chemical_potential
from ..terminal_ui import Spinner, stage_line, stage_warning
from .types import EnergyDecomposition, GroundStateResult
from .workflow_logging import (
  format_ground_state_iteration,
  log_energy_breakdown,
  log_ground_state_finish,
  log_ground_state_start,
)

if TYPE_CHECKING:
  from ..config import JrystalConfigDict
  from .runtime import RuntimeContext
  from .types import ElectronicBackend

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
      evals,
      num_electrons,
      smearing=smearing,
      k_weights=k_weights,
    )
    return fermi_dirac(evals, mu, smearing=smearing)
  return _fixed_occupation(evals, num_electrons)


def _occupation_max(spin_restricted: bool) -> float:
  """Maximum occupation per state for the requested spin treatment."""
  return 2.0 if spin_restricted else 1.0


def _compact_g_indices(freq_mask) -> jax.Array:
  """Flattened reciprocal-grid indices for the active plane-wave mask."""
  return jnp.asarray(
    np.flatnonzero(np.asarray(freq_mask).reshape(-1)),
    dtype=jnp.int32,
  )


def _expand_compact_with_indices(
  coeff_compact,
  grid_shape: tuple[int, int, int],
  g_indices,
):
  """JIT-safe version of expand_coefficient using integer gather/scatter."""
  coeff_compact = jnp.swapaxes(coeff_compact, -1, -2)
  flat_size = grid_shape[0] * grid_shape[1] * grid_shape[2]
  flat = jnp.zeros(
    coeff_compact.shape[:-1] + (flat_size,),
    dtype=coeff_compact.dtype,
  )
  flat = flat.at[..., g_indices].set(coeff_compact)
  return flat.reshape(coeff_compact.shape[:-1] + grid_shape)


def _squeeze_full_with_indices(coeff_full, g_indices):
  """JIT-safe version of squeeze_coefficient using integer indexing."""
  flat = coeff_full.reshape(coeff_full.shape[:-3] + (-1,))
  coeff_compact = jnp.take(flat, g_indices, axis=-1)
  return jnp.swapaxes(coeff_compact, -1, -2)


@partial(jax.jit, inline=False, static_argnums=(2,))
def _uspp_hvp_compact(
  coeff_compact_conj,
  iteration_state,
  grid_shape,
  g_indices,
  g_vec,
  kpts,
  projector_channels,
  channel_mask,
  vol,
):
  """USPP-specific compact-space H|psi> using only explicit operator arrays."""
  coeff_full = _expand_compact_with_indices(
    coeff_compact_conj.conj(),
    grid_shape,
    g_indices,
  )
  kinetic = _ultrasoft.kinetic_apply(coeff_full, g_vec, kpts)
  local = _ultrasoft.local_potential_apply(
    coeff_full,
    iteration_state.local_potential_r,
    vol,
  )
  nonlocal_term = _ultrasoft.channel_nonlocal_apply(
    coeff_full,
    projector_channels,
    iteration_state.channel_dii,
    vol,
    channel_mask=channel_mask,
  )
  return _squeeze_full_with_indices(
    jnp.conj(kinetic + local + nonlocal_term),
    g_indices,
  )


@partial(jax.jit, inline=False, static_argnums=(1,))
def _uspp_svp_compact(
  coeff_compact_conj,
  grid_shape,
  g_indices,
  projector_channels,
  channel_qii,
  channel_mask,
  vol,
):
  """USPP-specific compact-space S|psi> using only explicit operator arrays."""
  coeff_full = _expand_compact_with_indices(
    coeff_compact_conj.conj(),
    grid_shape,
    g_indices,
  )
  spsi_full = _ultrasoft.overlap_apply(
    coeff_full,
    projector_channels,
    channel_qii,
    vol,
    channel_mask=channel_mask,
  )
  return _squeeze_full_with_indices(spsi_full.conj(), g_indices)


# ---------------------------------------------------------------------------
# SCF solver
# ---------------------------------------------------------------------------


def run_scf(
  config: JrystalConfigDict,
  ctx: RuntimeContext,
  backend: ElectronicBackend,
  *,
  requested_solver_mode: Optional[str] = None,
  restart_state: Optional[dict] = None,
  output_dir: Optional[Path] = None,
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
  occ_max = _occupation_max(config.system.spin_restricted)
  num_kpts = ctx.ksampling.kpts.shape[0]
  num_bands = ceil(num_electrons / occ_max) + config.occupation.empty_bands
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

  stage_line("Init", f"Crystal: {crystal.symbols}")
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

  checkpoint_manager = None
  if output_dir is not None and config.io.save_checkpoint:
    checkpoint_manager = make_checkpoint_manager(output_dir)

  # --- Init wavefunctions (compact, in masked G-space) ---
  start_step = 0
  previous_total_energy = None
  if restart_state is None:
    pw_params = _pw.param_init(
      key,
      num_bands,
      num_kpts,
      freq_mask,
      spin_restricted=config.system.spin_restricted,
    )
    coeff_compact = pw_params["w_re"] + 1.0j * pw_params["w_im"]
    coeff_compact = jnp.linalg.qr(coeff_compact)[0]  # [s, k, g, band]

    # --- Init eigenvalues, occupation, density ---
    evals = jax.random.normal(key, [1, num_kpts, num_bands])
    evals = jnp.sort(evals, axis=-1)
    evals_new = evals
    occ = _fixed_occupation(evals, num_electrons, occ_max)
  else:
    coeff_compact = (
      jnp.asarray(restart_state["coefficients"]["w_re"]) +
      1.0j * jnp.asarray(restart_state["coefficients"]["w_im"])
    )
    occ = jnp.asarray(restart_state["occupations"])
    if bool(restart_state["has_eigenvalues"]):
      evals_new = jnp.asarray(restart_state["eigenvalues"])
    else:
      evals_new = jnp.sort(
        jax.random.normal(key, [1, num_kpts, num_bands]), axis=-1
      )
    start_step = int(restart_state["step"]) + 1
    previous_total_energy = float(restart_state["total_energy"])
    stage_warning(
      "SCF",
      f"Restarting from step {start_step} using checkpoint state.",
    )

  def _density_from_compact(c, occ):
    coeff_full = expand_coefficient(c, freq_mask)
    total_density_fn = getattr(backend, "_total_density", None)
    if total_density_fn is not None:
      return total_density_fn(coeff_full, occ, ctx)
    return _pw.density_grid(
      coeff_full,
      crystal.vol,
      occ,
      k_weights=k_weights,
    )

  density = (
    _density_from_compact(coeff_compact, occ)
    if restart_state is None else jnp.asarray(restart_state["density"])
  )

  # --- Preconditioner ---
  precond = kerker_preconditioner(g_vec, freq_mask)
  uspp_cache = (
    ctx.pseudo_cache
    if isinstance(ctx.pseudo_cache, UltrasoftMeshCache)
    else None
  )
  compact_g_indices = _compact_g_indices(freq_mask)
  reciprocal_grid_shape = tuple(int(x) for x in freq_mask.shape)

  # --- DIIS state ---
  diis_state = diis_init(
    max_hist=diis_max_hist,
    density_shape=density.shape,
    dtype=density.dtype,
  )

  # --- Hvp via backend ---
  def _generic_hvp(coeff_compact_conj, iteration_state):
    coeff_full = expand_coefficient(coeff_compact_conj.conj(), freq_mask)
    hpsi_full = backend.hamiltonian_apply(coeff_full, iteration_state, ctx)
    return squeeze_coefficient(hpsi_full, freq_mask)

  def _generic_svp(coeff_compact_conj):
    coeff_full = expand_coefficient(coeff_compact_conj.conj(), freq_mask)
    spsi_full = backend.overlap_apply(coeff_full, ctx)
    return squeeze_coefficient(spsi_full.conj(), freq_mask)

  @jax.jit
  def _diagonalise_generic(coeff_compact, iteration_state, precond):
    s, k, g, b = coeff_compact.shape

    def _lobpcg_matmul(c):
      coeff_batch = c.reshape(s, k, g, -1)
      return _generic_hvp(coeff_batch, iteration_state).reshape(s * k, g, -1)

    def _lobpcg_bmatmul(c):
      coeff_batch = c.reshape(s, k, g, -1)
      return _generic_svp(coeff_batch).reshape(s * k, g, -1)

    eigval, evec = batched_lobpcg(
      matmul=_lobpcg_matmul,
      b_matmul=_lobpcg_bmatmul,
      k=b,
      v0=coeff_compact.reshape(s * k, g, b),
      which="smallest",
      preconditioner=precond,
      maxit=lobpcg_max_iter,
      tol=1e-8,
    )
    return evec.reshape(s, k, g, b), eigval.reshape(s, k, b)

  @jax.jit
  def _diagonalise_uspp(
    coeff_compact,
    iteration_state,
    precond,
    g_vec,
    kpts,
    projector_channels,
    channel_qii,
    channel_mask,
    vol,
  ):
    s, k, g, b = coeff_compact.shape

    def _lobpcg_matmul(c):
      coeff_batch = c.reshape(s, k, g, -1)
      return _uspp_hvp_compact(
        coeff_batch,
        iteration_state,
        reciprocal_grid_shape,
        compact_g_indices,
        g_vec,
        kpts,
        projector_channels,
        channel_mask,
        vol,
      ).reshape(s * k, g, -1)

    def _lobpcg_bmatmul(c):
      coeff_batch = c.reshape(s, k, g, -1)
      return _uspp_svp_compact(
        coeff_batch,
        reciprocal_grid_shape,
        compact_g_indices,
        projector_channels,
        channel_qii,
        channel_mask,
        vol,
      ).reshape(s * k, g, -1)

    eigval, evec = batched_lobpcg(
      matmul=_lobpcg_matmul,
      b_matmul=_lobpcg_bmatmul,
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
  convergence_history = []
  last_completed_step = start_step - 1
  last_checkpointed_step = None
  spinner = Spinner("SCF")
  interrupted = False

  def _handle_sigint(sig, frame):
    del sig, frame
    nonlocal interrupted
    interrupted = True

  old_handler = signal.signal(signal.SIGINT, _handle_sigint)
  try:
    if config.execution.verbose:
      spinner.start("initialising SCF state...")

    for step in range(start_step, scf_max_iter):
      t0 = time.time()

      # 1. Diagonalise
      iteration_state = backend.prepare_iteration(density, ctx)
      if uspp_cache is not None:
        coeff_new, evals_new = _diagonalise_uspp(
          coeff_compact,
          iteration_state,
          precond,
          g_vec,
          ctx.ksampling.kpts,
          uspp_cache.channel_projectors_gk,
          uspp_cache.channel_qii,
          uspp_cache.channel_mask,
          crystal.vol,
        )
      else:
        coeff_new, evals_new = _diagonalise_generic(
          coeff_compact,
          iteration_state,
          precond,
        )
      coeff_new = coeff_new.conj()

      # 2. Update occupation
      occ = _compute_occupation(
        evals_new,
        num_electrons,
        k_weights,
        smearing,
      )

      # 3. New density
      coeff_full_new = expand_coefficient(coeff_new, freq_mask)
      density_new = _density_from_compact(coeff_new, occ)

      # 4. Check convergence
      total_energy_new = float(
        backend.total_energy(coeff_full_new, occ, ctx) + ew
      )
      delta_total_energy = None
      if total_energy_history:
        delta_total_energy = abs(total_energy_new - total_energy_history[-1])
      elif previous_total_energy is not None:
        delta_total_energy = abs(total_energy_new - previous_total_energy)
      total_energy_history.append(total_energy_new)
      d_density = float(jnp.mean(jnp.abs(density_new - density)))
      dt = time.time() - t0
      display_step = step + 1
      last_completed_step = step
      convergence_history.append(
        {
          "step": display_step,
          "total_energy": total_energy_new,
          "delta_energy": delta_total_energy,
          "delta_density": d_density,
          "wall_time": dt,
        }
      )

      if config.execution.verbose:
        spinner.update(
          format_ground_state_iteration(
            "SCF",
            step=display_step,
            max_steps=scf_max_iter,
            total_energy=total_energy_new,
            delta_energy=delta_total_energy,
            step_time=dt,
            density_delta=d_density,
          )
        )

      if (
        delta_total_energy is not None and d_density < density_tol and
        delta_total_energy < energy_tol
      ):
        converged = True
        coeff_compact = coeff_new
        density = density_new
        break

      # 5. Density mixing (DIIS + linear)
      dens_error = density_new - density
      diis_state, density_mixed = diis_update(
        diis_state, density_new, dens_error,
      )
      density = simple_mixing(density_mixed, density, beta=mixing_beta)
      coeff_compact = coeff_new
      previous_total_energy = total_energy_new

      if checkpoint_manager is not None and (
        display_step % config.io.checkpoint_interval == 0 or interrupted
      ):
        save_checkpoint(
          checkpoint_manager,
          {
            "density": density,
            "coefficients":
              {
                "w_re": coeff_compact.real,
                "w_im": coeff_compact.imag,
              },
            "occupations": occ,
            "eigenvalues": evals_new,
            "has_eigenvalues": True,
            "step": step,
            "total_energy": total_energy_new,
          },
          step,
        )
        last_checkpointed_step = step

      if interrupted:
        stage_warning("SCF", "Interrupted — checkpoint saved")
        break

    if checkpoint_manager is not None and (
      last_completed_step >= 0 and last_checkpointed_step != last_completed_step
    ):
      save_checkpoint(
        checkpoint_manager,
        {
          "density": density,
          "coefficients":
            {
              "w_re": coeff_compact.real,
              "w_im": coeff_compact.imag,
            },
          "occupations": occ,
          "eigenvalues": evals_new,
          "has_eigenvalues": True,
          "step": last_completed_step,
          "total_energy": total_energy_history[-1],
        },
        last_completed_step,
      )
  finally:
    signal.signal(signal.SIGINT, old_handler)
    if checkpoint_manager is not None:
      checkpoint_manager.wait_until_finished()
    spinner.stop()

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
    params_pw={
      "w_re": coeff_compact.real, "w_im": coeff_compact.imag
    },
    params_occ={},
    total_energy=total_e,
    energy_terms=EnergyDecomposition(
      ewald=float(ew),
      **{
        k: float(v) for k, v in decomp.items()
      },
    ),
    converged=converged,
    density=density,
    coefficients={
      "w_re": coeff_compact.real,
      "w_im": coeff_compact.imag,
    },
    eigenvalues=evals_new,
    occupations=occ,
    actual_solver="scf",
    requested_solver_mode=requested_solver_mode or config.solver.mode,
    num_iterations=max(last_completed_step + 1, 0),
    wall_time=wall_time,
    convergence_history=convergence_history,
    total_energy_history=total_energy_history,
  )
