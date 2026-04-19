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
from ..pseudopotential import ultrasoft as _ultrasoft
from ..pseudopotential.kernel import UltrasoftMeshCache
from ..smearing import fermi_dirac, find_chemical_potential
from ..terminal_ui import Spinner, stage_warning
from .timer import PhaseTimer
from .types import EnergyDecomposition, GroundStateResult
from .workflow_logging import (
  format_ground_state_iteration,
  log_energy_breakdown,
  log_ground_state_finish,
  log_ground_state_start,
  log_timing_breakdown,
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


def _spin_channel_electron_counts(
  num_electrons: int,
  spin: int,
  spin_restricted: bool,
) -> tuple[int, ...]:
  """Return the per-spin electron counts for the requested spin treatment."""
  if spin_restricted:
    return (num_electrons,)
  return ((num_electrons + spin) // 2, (num_electrons - spin) // 2)


def _fixed_occupation(
  evals,
  num_electrons,
  *,
  spin: int,
  spin_restricted: bool,
):
  """Assign identical occupied-band count at every k-point (insulator)."""
  evals = jnp.asarray(evals)
  occ_max = _occupation_max(spin_restricted)
  electron_counts = _spin_channel_electron_counts(
    num_electrons,
    spin,
    spin_restricted,
  )
  if evals.shape[0] != len(electron_counts):
    raise ValueError(
      "Eigenvalue spin axis does not match the requested occupation mode. "
      f"Got evals.shape[0]={evals.shape[0]} and "
      f"{len(electron_counts)} spin channel(s)."
    )

  occupations = []
  for spin_index, electron_count in enumerate(electron_counts):
    num_occ = int(round(electron_count / occ_max))
    order = jnp.argsort(evals[spin_index], axis=-1)
    rank = jnp.argsort(order, axis=-1)
    occupations.append(
      jnp.where(rank < num_occ, occ_max, 0.0).astype(evals.dtype)
    )
  return jnp.stack(occupations, axis=0)


def _compute_occupation(
  evals,
  num_electrons,
  k_weights,
  smearing,
  *,
  spin: int,
  spin_restricted: bool,
):
  """Compute Fermi-Dirac or fixed occupation."""
  if smearing > 0:
    if spin_restricted:
      mu = find_chemical_potential(
        evals,
        num_electrons,
        smearing=smearing,
        k_weights=k_weights,
      )
      return fermi_dirac(evals, mu, smearing=smearing), mu

    mus = []
    occupations = []
    for spin_index, electron_count in enumerate(
      _spin_channel_electron_counts(num_electrons, spin, spin_restricted)
    ):
      spin_evals = evals[spin_index:spin_index + 1]
      mu = find_chemical_potential(
        spin_evals,
        electron_count,
        smearing=smearing,
        k_weights=k_weights,
        spin_polorized=True,
      )
      occ = fermi_dirac(
        spin_evals,
        mu,
        smearing=smearing,
        spin_polorized=True,
      )
      mus.append(mu)
      occupations.append(occ[0])
    return jnp.stack(occupations, axis=0), jnp.stack(mus, axis=0)
  return _fixed_occupation(
    evals,
    num_electrons,
    spin=spin,
    spin_restricted=spin_restricted,
  ), None


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


def _density_dtype_from_context(density, ctx):
  """Choose a stable real dtype for SCF density storage and DIIS buffers."""
  density_dtype = jnp.result_type(
    jnp.real(jnp.asarray(density)),
    jnp.real(jnp.asarray(ctx.g_vec)),
    jnp.asarray(ctx.crystal.vol),
  )
  if ctx.potential_local is not None:
    density_dtype = jnp.result_type(
      density_dtype,
      jnp.real(jnp.asarray(ctx.potential_local)),
    )
  nlcc = getattr(getattr(ctx, "pseudo_cache", None), "nlcc_g", None)
  if nlcc is not None:
    density_dtype = jnp.result_type(density_dtype, jnp.asarray(nlcc))
  return density_dtype


def _kinetic_preconditioner_batch(
  g_vec,
  freq_mask,
  kpts,
  *,
  num_spin: int,
  shift: float = 1.0,
):
  """Return a batched masked-G kinetic preconditioner for flattened (spin,k)."""
  eff_g = jnp.asarray(g_vec)[freq_mask]
  kpts = jnp.asarray(kpts, dtype=eff_g.dtype)
  kinetic = 0.5 * jnp.sum((eff_g[None, :, :] + kpts[:, None, :])**2, axis=-1)
  shift = jnp.asarray(shift, dtype=kinetic.dtype)
  floor = jnp.maximum(shift, jnp.finfo(kinetic.dtype).eps)
  precond = jnp.reciprocal(jnp.maximum(kinetic + shift, floor))
  precond = jnp.broadcast_to(
    precond[None, :, :, None],
    (num_spin, precond.shape[0], precond.shape[1], 1),
  )
  return precond.reshape(num_spin * kpts.shape[0], eff_g.shape[0], 1)


def _density_integral(density, vol: float) -> float:
  """Integrate a real-space density over the unit cell."""
  density = jnp.asarray(density)
  total_grid = int(np.prod(density.shape[-3:]))
  return float(jnp.sum(density).real) * float(vol) / float(total_grid)


def _uspp_charge_diagnostics(
  coeff_full,
  total_density,
  occ,
  ctx,
  *,
  target_charge: float,
):
  """Return smooth/augmentation/total charge integrals for USPP SCF debug."""
  smooth_density = _pw.density_grid(
    coeff_full,
    ctx.crystal.vol,
    occ,
    k_weights=ctx.ksampling.weights,
  )
  augmentation_density = jnp.asarray(total_density
                                    ) - jnp.asarray(smooth_density)
  total_charge = _density_integral(total_density, ctx.crystal.vol)
  smooth_charge = _density_integral(smooth_density, ctx.crystal.vol)
  augmentation_charge = _density_integral(
    augmentation_density,
    ctx.crystal.vol,
  )
  return {
    "total_charge_e": total_charge,
    "smooth_charge_e": smooth_charge,
    "augmentation_charge_e": augmentation_charge,
    "charge_delta_e": total_charge - target_charge,
  }


def _uspp_overlap_diagnostics(coeff_compact, ctx, occupied_bands: int):
  """Return min/max eigenvalues of C^H S C on the occupied subspace."""
  if occupied_bands <= 0:
    return {}
  overlap_eigs = _ultrasoft.subspace_overlap_eigenvalues_compact(
    coeff_compact[..., :occupied_bands],
    ctx.pseudo_cache.channel_projectors_compact_gk,
    ctx.pseudo_cache.channel_qii,
    ctx.crystal.vol,
    channel_mask=ctx.pseudo_cache.channel_mask,
  )
  return {
    "overlap_eig_min": float(jnp.min(overlap_eigs).real),
    "overlap_eig_max": float(jnp.max(overlap_eigs).real),
  }


def _uspp_projector_charge_diagnostics(
  coeff_compact,
  occ,
  ctx,
  *,
  smooth_charge: float,
  augmentation_charge: float,
  target_charge: float,
):
  """Compare real-space augmentation charge to projector-space Q expectation."""
  f_matrix = _ultrasoft.projector_channel_overlap_compact(
    coeff_compact,
    ctx.pseudo_cache.channel_projectors_compact_gk,
    channel_mask=ctx.pseudo_cache.channel_mask,
  )
  qf_matrix = jnp.einsum(
    "aij,sakbj->sakbi",
    ctx.pseudo_cache.channel_qii,
    f_matrix,
  )
  q_diag = jnp.einsum(
    "sakbi,sakbi->skb",
    jnp.conj(f_matrix),
    qf_matrix,
  ).real / ctx.crystal.vol
  weighted_occ = occ * ctx.ksampling.weights[None, :, None]
  augmentation_charge_q_expected = float(jnp.sum(q_diag * weighted_occ).real)
  return {
    "augmentation_charge_q_expected_e":
      augmentation_charge_q_expected,
    "augmentation_charge_residual_e":
      (float(augmentation_charge) - augmentation_charge_q_expected),
    "charge_closure_error_e":
      (
        float(smooth_charge) + augmentation_charge_q_expected -
        float(target_charge)
      ),
  }


def _require_finite(name: str, value) -> None:
  """Raise a clear error when a solver intermediate becomes non-finite."""
  arr = np.asarray(value)
  if np.all(np.isfinite(arr)):
    return
  raise FloatingPointError(
    f"Encountered non-finite values in {name}; "
    "this usually indicates the eigensolver or density reconstruction "
    "has become unstable."
  )


@partial(jax.jit, inline=False, static_argnums=(2,))
def _uspp_hvp_compact(
  coeff_compact_conj,
  iteration_state,
  grid_shape,
  g_indices,
  g_vec,
  kpts,
  projector_channels_compact,
  channel_mask,
  vol,
):
  """USPP-specific compact-space H|psi> using only explicit operator arrays."""
  coeff_compact = coeff_compact_conj.conj()
  coeff_full = _expand_compact_with_indices(
    coeff_compact,
    grid_shape,
    g_indices,
  )
  kinetic = _ultrasoft.kinetic_apply(coeff_full, g_vec, kpts)
  local = _ultrasoft.local_potential_apply(
    coeff_full,
    iteration_state.local_potential_r,
    vol,
  )
  nonlocal_compact = _ultrasoft.channel_nonlocal_apply_compact(
    coeff_compact,
    projector_channels_compact,
    iteration_state.channel_dii,
    vol,
    channel_mask=channel_mask,
  )
  smooth_compact = _squeeze_full_with_indices(
    jnp.conj(kinetic + local), g_indices
  )
  return smooth_compact + jnp.conj(nonlocal_compact)


@partial(jax.jit, inline=False)
def _uspp_svp_compact(
  coeff_compact_conj,
  projector_channels_compact,
  channel_qii,
  channel_mask,
  vol,
):
  """USPP-specific compact-space S|psi> using only explicit operator arrays."""
  spsi_compact = _ultrasoft.overlap_apply_compact(
    coeff_compact_conj.conj(),
    projector_channels_compact,
    channel_qii,
    vol,
    channel_mask=channel_mask,
  )
  return spsi_compact.conj()


@partial(jax.jit, inline=False, static_argnums=(3, 11))
def _diagonalise_uspp_explicit(
  coeff_compact,
  iteration_state,
  precond,
  grid_shape,
  g_indices,
  g_vec,
  kpts,
  projector_channels_compact,
  channel_qii,
  channel_mask,
  vol,
  lobpcg_max_iter,
):
  """USPP LOBPCG solve driven only by explicit operator arrays.

  Keeping this kernel at module scope avoids rebuilding a large jitted
  function object for every SCF run, and avoids closing over the full
  runtime/backend object graph.
  """
  s, k, g, b = coeff_compact.shape

  def _lobpcg_matmul(c):
    coeff_batch = c.reshape(s, k, g, -1)
    return _uspp_hvp_compact(
      coeff_batch,
      iteration_state,
      grid_shape,
      g_indices,
      g_vec,
      kpts,
      projector_channels_compact,
      channel_mask,
      vol,
    ).reshape(s * k, g, -1)

  def _lobpcg_bmatmul(c):
    coeff_batch = c.reshape(s, k, g, -1)
    return _uspp_svp_compact(
      coeff_batch,
      projector_channels_compact,
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

  g_vec = ctx.g_vec
  freq_mask = ctx.basis.freq_mask
  ew = ctx.ewald_energy
  k_weights = ctx.ksampling.weights

  num_electrons = backend.num_electrons(ctx)
  num_spin = 1 if config.system.spin_restricted else 2
  occ_max = _occupation_max(config.system.spin_restricted)
  num_kpts = ctx.ksampling.kpts.shape[0]
  num_bands = ceil(num_electrons / occ_max) + config.occupation.empty_bands
  occupied_bands = max(
    ceil(count / occ_max) for count in _spin_channel_electron_counts(
      num_electrons,
      crystal.spin,
      config.system.spin_restricted,)
  )
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
  show_progress = config.io.log_level != "quiet"
  phase_timer = PhaseTimer(enabled=config.execution.profile)

  log_ground_state_start(
    "SCF",
    max_steps=scf_max_iter,
    num_bands=num_bands,
    smearing=smearing,
    xc=config.method.xc,
    controls=(
      f"lobpcg={lobpcg_max_iter} mix=diis "
      f"beta={mixing_beta:.2f} hist={diis_max_hist} "
      f"tol=({density_tol:.1e},{energy_tol:.1e})"
    ),
  )

  checkpoint_manager = None
  if output_dir is not None and config.io.save_checkpoint:
    checkpoint_manager = make_checkpoint_manager(output_dir)

  uspp_cache = (
    ctx.pseudo_cache
    if isinstance(ctx.pseudo_cache, UltrasoftMeshCache) else None
  )

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
    if uspp_cache is not None:
      coeff_compact = _ultrasoft.overlap_inv_sqrt_apply_compact(
        coeff_compact,
        uspp_cache.channel_projectors_compact_gk,
        uspp_cache.channel_qii,
        crystal.vol,
        channel_mask=uspp_cache.channel_mask,
      )

    # --- Init eigenvalues, occupation, density ---
    evals = jax.random.normal(key, [num_spin, num_kpts, num_bands])
    evals = jnp.sort(evals, axis=-1)
    evals_new = evals
    occ = _fixed_occupation(
      evals,
      num_electrons,
      spin=crystal.spin,
      spin_restricted=config.system.spin_restricted,
    )
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
        jax.random.normal(key, [num_spin, num_kpts, num_bands]), axis=-1
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
  density = jnp.asarray(
    density,
    dtype=_density_dtype_from_context(density, ctx),
  )

  # --- Preconditioner ---
  if uspp_cache is not None:
    precond = _kinetic_preconditioner_batch(
      g_vec,
      freq_mask,
      ctx.ksampling.kpts,
      num_spin=num_spin,
    )
  else:
    precond = kerker_preconditioner(g_vec, freq_mask)
  if uspp_cache is not None and uspp_cache.channel_projectors_compact_gk is None:
    raise ValueError(
      "USPP SCF requires compact active-G channel projectors in the mesh cache."
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

  # --- SCF loop ---
  converged = False
  total_energy_history = []
  convergence_history = []
  last_completed_step = start_step - 1
  last_checkpointed_step = None
  spinner = Spinner("SCF")
  interrupted = False
  last_chemical_potential = None

  def _handle_sigint(sig, frame):
    del sig, frame
    nonlocal interrupted
    interrupted = True

  old_handler = signal.signal(signal.SIGINT, _handle_sigint)
  try:
    if show_progress:
      spinner.start("initialising SCF state...")

    for step in range(start_step, scf_max_iter):
      t0 = time.time()

      # 1. Diagonalise
      with phase_timer.phase("potential"):
        iteration_state = backend.prepare_iteration(density, ctx)
      diagonalise_phase = (
        "first_call_overhead" if
        (config.execution.profile and step == start_step) else "diagonalize"
      )
      with phase_timer.phase(diagonalise_phase):
        if uspp_cache is not None:
          coeff_new, evals_new = _diagonalise_uspp_explicit(
            coeff_compact,
            iteration_state,
            precond,
            reciprocal_grid_shape,
            compact_g_indices,
            g_vec,
            ctx.ksampling.kpts,
            uspp_cache.channel_projectors_compact_gk,
            uspp_cache.channel_qii,
            uspp_cache.channel_mask,
            crystal.vol,
            lobpcg_max_iter,
          )
        else:
          coeff_new, evals_new = _diagonalise_generic(
            coeff_compact,
            iteration_state,
            precond,
          )
        if config.execution.profile:
          coeff_new, evals_new = jax.block_until_ready((coeff_new, evals_new))
        if uspp_cache is not None:
          _require_finite(f"USPP SCF eigensolve (step {step + 1})", coeff_new)
          _require_finite(
            f"USPP SCF eigenvalues (step {step + 1})",
            evals_new,
          )
      coeff_new = coeff_new.conj()

      # 2. Update occupation
      with phase_timer.phase("occupation"):
        occ, chemical_potential = _compute_occupation(
          evals_new,
          num_electrons,
          k_weights,
          smearing,
          spin=crystal.spin,
          spin_restricted=config.system.spin_restricted,
        )
        if config.execution.profile:
          occ = jax.block_until_ready(occ)
      scalar_chemical_potential = None
      if chemical_potential is not None:
        chemical_potential_arr = np.asarray(chemical_potential)
        if chemical_potential_arr.ndim == 0:
          scalar_chemical_potential = float(chemical_potential_arr)
          last_chemical_potential = scalar_chemical_potential

      # 3. New density
      with phase_timer.phase("density"):
        coeff_full_new = expand_coefficient(coeff_new, freq_mask)
        density_new = jnp.asarray(
          _density_from_compact(coeff_new, occ),
          dtype=density.dtype,
        )
        if config.execution.profile:
          density_new = jax.block_until_ready(density_new)
        if uspp_cache is not None:
          _require_finite(
            f"USPP SCF density update (step {step + 1})",
            density_new,
          )

      # 4. Check convergence
      with phase_timer.phase("energy"):
        total_energy_new = float(
          backend.total_energy(coeff_full_new, occ, ctx) + ew
        )
      if uspp_cache is not None:
        _require_finite(
          f"USPP SCF total energy (step {step + 1})",
          total_energy_new,
        )
      delta_total_energy = None
      if total_energy_history:
        delta_total_energy = abs(total_energy_new - total_energy_history[-1])
      elif previous_total_energy is not None:
        delta_total_energy = abs(total_energy_new - previous_total_energy)
      total_energy_history.append(total_energy_new)
      d_density = float(jnp.mean(jnp.abs(density_new - density)))
      dt = time.time() - t0
      cumulative_time = time.time() - overall_start
      display_step = step + 1
      last_completed_step = step
      record = {
        "step": display_step,
        "total_energy": total_energy_new,
        "delta_energy": delta_total_energy,
        "delta_density": d_density,
        "wall_time": dt,
        "cumulative_time_s": cumulative_time,
      }
      charge_delta = None
      overlap_eig_min = None
      overlap_eig_max = None
      if uspp_cache is not None:
        charge_info = _uspp_charge_diagnostics(
          coeff_full_new,
          density_new,
          occ,
          ctx,
          target_charge=float(num_electrons),
        )
        projector_charge_info = _uspp_projector_charge_diagnostics(
          coeff_new,
          occ,
          ctx,
          smooth_charge=charge_info["smooth_charge_e"],
          augmentation_charge=charge_info["augmentation_charge_e"],
          target_charge=float(num_electrons),
        )
        overlap_info = _uspp_overlap_diagnostics(
          coeff_new,
          ctx,
          occupied_bands=occupied_bands,
        )
        record.update(charge_info)
        record.update(projector_charge_info)
        record.update(overlap_info)
        charge_delta = charge_info["charge_delta_e"]
        overlap_eig_min = overlap_info.get("overlap_eig_min")
        overlap_eig_max = overlap_info.get("overlap_eig_max")
      if chemical_potential is not None:
        chemical_potential_arr = np.asarray(chemical_potential)
        if chemical_potential_arr.ndim == 0:
          record["chemical_potential_ha"] = float(chemical_potential_arr)
        else:
          record["chemical_potential_ha"] = [
            float(x) for x in chemical_potential_arr.reshape(-1)
          ]
      convergence_history.append(record)

      if show_progress:
        spinner.update(
          format_ground_state_iteration(
            "SCF",
            step=display_step,
            max_steps=scf_max_iter,
            total_energy=total_energy_new,
            delta_energy=delta_total_energy,
            step_time=dt,
            cumulative_time=cumulative_time,
            density_delta=d_density,
            charge_delta=charge_delta,
            overlap_eig_min=overlap_eig_min,
            overlap_eig_max=overlap_eig_max,
            chemical_potential=scalar_chemical_potential,
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
      with phase_timer.phase("mixing"):
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
        with phase_timer.phase("checkpoint"):
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
  density = jnp.asarray(
    _density_from_compact(coeff_compact, occ),
    dtype=_density_dtype_from_context(density, ctx),
  )
  wall_time = time.time() - overall_start
  profiling = {}

  log_ground_state_finish(
    "SCF",
    converged=converged,
    steps_completed=len(total_energy_history),
    max_steps=scf_max_iter,
    total_energy=total_e,
    wall_time=wall_time,
  )
  if config.execution.profile:
    profiling = {
      "kind": "ground_state",
      "enabled": True,
      "wall_time_sec": wall_time,
      "phases": phase_timer.summary(total_wall_time=wall_time),
    }
    log_timing_breakdown(
      "SCF",
      profiling["phases"],
      total_wall_time=wall_time,
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
    fermi_energy=last_chemical_potential,
    convergence_history=convergence_history,
    total_energy_history=total_energy_history,
    profiling=profiling,
  )
