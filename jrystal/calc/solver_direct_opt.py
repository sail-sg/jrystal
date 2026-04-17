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
"""Direct-optimisation ground-state solver.

Minimises the total (free) energy directly with respect to
plane-wave coefficients and occupation parameters using a
first-order ``optax`` optimiser.  The physics (which energy
terms to compute) is delegated entirely to an
:class:`~jrystal.calc.backend.ElectronicBackend`.
"""
from __future__ import annotations

import signal
import time
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .._src import entropy as _entropy
from .._src import occupation as _occupation
from .._src import pw as _pw
from .._src.utils import squeeze_coefficient
from ..io import make_checkpoint_manager, save_checkpoint
from ..terminal_ui import Spinner, stage_line, stage_warning
from .convergence import create_convergence_checker
from .opt_utils import (
  create_direct_opt_optimizer,
  has_occupation_scheduler,
  reset_occupation_plateau_state,
)
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


@dataclass(frozen=True)
class _OccupationSetup:
  fn: Callable[[dict], jax.Array]
  params: dict
  trainable: bool


def _occupation_max(spin_restricted: bool) -> float:
  """Maximum occupation per state for the requested spin treatment."""
  return 2.0 if spin_restricted else 1.0


def _build_occupation_setup(
  *,
  num_electrons: int,
  spin: int,
  spin_restricted: bool,
  num_bands: int,
  num_kpts: int,
  smearing: float,
) -> _OccupationSetup:
  """Create either trainable or fixed occupations for direct optimisation."""
  if float(smearing) == 0.0 and num_kpts == 1:
    fixed_occ = _occupation._get_fixed_occupation(
      num_k=num_kpts,
      num_electrons=num_electrons,
      spin=spin,
      num_bands=num_bands,
      spin_restricted=spin_restricted,
    )

    def occ_fn(_params: dict) -> jax.Array:
      return fixed_occ

    return _OccupationSetup(fn=occ_fn, params={}, trainable=False)

  occ_fn = _occupation.get_occupation_fn(
    num_electrons,
    spin=spin,
    spin_restricted=spin_restricted,
  )
  params_occ = _occupation.params_init(num_bands, num_kpts)
  return _OccupationSetup(fn=occ_fn, params=params_occ, trainable=True)


def _free_energy_from_total_energy(
  total_energy: jax.Array,
  occupation: jax.Array,
  smearing: float,
) -> tuple[jax.Array, jax.Array]:
  """Return free energy and entropy for the current occupations."""
  total_energy = jnp.asarray(total_energy)
  if float(smearing) == 0.0:
    return total_energy, jnp.zeros((), dtype=total_energy.dtype)
  entropy = _entropy.von_neumann(occupation)
  free_energy = total_energy - jnp.asarray(
    smearing, dtype=total_energy.dtype
  ) * entropy
  return free_energy, entropy


def _total_energy_metric(
  total_energy: jax.Array,
  ewald_energy: float | jax.Array,
) -> jax.Array:
  """Return the total-energy metric used for convergence and LR scheduling."""
  total_energy = jnp.asarray(total_energy)
  return total_energy + jnp.asarray(ewald_energy, dtype=total_energy.dtype)


def _freeze_occupation_gradient(grad: dict) -> dict:
  """Zero occupation gradients during the direct-opt warmup phase."""
  return {
    **grad,
    "occ": jax.tree_util.tree_map(jnp.zeros_like, grad["occ"]),
  }


def _occupation_params_from_tensor(
  occupation: jax.Array,
  *,
  spin_restricted: bool,
) -> dict:
  """Construct simplex-projector parameters from a saved occupation tensor."""
  eps = 1e-6
  occupation = jnp.asarray(occupation)
  if spin_restricted:
    occ_up = jnp.clip(occupation[0] / 2.0, eps, 1.0 - eps)
    occ_down = occ_up
  else:
    occ_up = jnp.clip(occupation[0], eps, 1.0 - eps)
    occ_down = jnp.clip(occupation[1], eps, 1.0 - eps)
  return {
    "param_up": jnp.log(occ_up / (1.0 - occ_up)),
    "param_down": jnp.log(occ_down / (1.0 - occ_down)),
  }


def run_direct_opt(
  config: JrystalConfigDict,
  ctx: RuntimeContext,
  backend: ElectronicBackend,
  *,
  requested_solver_mode: str | None = None,
  restart_state: dict | None = None,
  output_dir: Path | None = None,
) -> GroundStateResult:
  """Run a ground-state calculation via direct energy minimisation.

  This function is backend-agnostic: the same optimisation loop works
  for all-electron and norm-conserving pseudopotential calculations
  because ``backend.total_energy`` encapsulates the physics.

  Args:
    config: Jrystal configuration.
    ctx: Fully initialised runtime context.
    backend: Electronic-structure backend.

  Returns:
    Ground-state result with converged energy, density, and parameters.
  """
  overall_start = time.time()
  key = jax.random.PRNGKey(config.execution.seed)

  crystal = ctx.crystal
  freq_mask = ctx.basis.freq_mask
  ew = ctx.ewald_energy
  k_vec = ctx.ksampling.kpts
  k_weights = ctx.ksampling.weights

  num_electrons = backend.num_electrons(ctx)
  num_kpts = k_vec.shape[0]
  num_spin = 1 if config.system.spin_restricted else 2
  occ_max = _occupation_max(config.system.spin_restricted)
  num_bands = ceil(num_electrons / occ_max) + config.occupation.empty_bands
  smearing = config.occupation.smearing
  occupation_warmup_steps = (
    config.solver.direct_opt.occupation_optimizer.warmup_steps
  )
  show_progress = config.io.log_level != "quiet"
  phase_timer = PhaseTimer(enabled=config.execution.profile)
  canonical_transform = bool(
    config.solver.direct_opt.canonical_transform or
    getattr(backend, "requires_canonical_transform", False)
  )
  if (
    getattr(backend, "requires_canonical_transform", False) and
    not config.solver.direct_opt.canonical_transform
  ):
    stage_warning(
      "DirectOpt",
      "Enabling canonical overlap transform required by the backend.",
    )

  log_ground_state_start(
    "DirectOpt",
    max_steps=config.solver.direct_opt.max_steps,
    num_bands=num_bands,
    smearing=config.occupation.smearing,
    xc=config.method.xc,
    controls=(
      f"opt={config.solver.direct_opt.optimizer.name} "
      f"lr={config.solver.direct_opt.optimizer.learning_rate:g} "
      f"occ_lr={config.solver.direct_opt.occupation_optimizer.learning_rate:g} "
      f"warmup={occupation_warmup_steps} "
      f"win={config.solver.direct_opt.convergence.window_size} "
      f"tol={config.solver.direct_opt.convergence.energy_std_tol:.1e}"
    ),
  )

  # --- Sharding / device setup ---
  num_devices = ctx.execution.num_devices
  util_devices = num_devices if ctx.execution.parallel_over_k else 1
  stage_line(
    "Init",
    f"Parallel over k: {ctx.execution.parallel_over_k}. "
    f"Devices: {num_devices} (used {util_devices}).",
    level="verbose",
  )

  mesh = Mesh(
    np.array(jax.devices()[:util_devices]).reshape([1, -1]),
    ("s", "k"),
  )
  sharding = NamedSharding(mesh, P("s", "k"))

  k_vec = jax.device_put(k_vec, NamedSharding(mesh, P("k")))
  k_weights = jax.device_put(k_weights, NamedSharding(mesh, P("k")))

  def _to_physical_coeff(coeff_full):
    if canonical_transform:
      return backend.overlap_inv_sqrt_apply(coeff_full, ctx)
    return coeff_full

  def _density_from_coeff(coeff_full, occ):
    total_density_fn = getattr(backend, "_total_density", None)
    if total_density_fn is not None:
      return total_density_fn(coeff_full, occ, ctx)
    return _pw.density_grid(
      coeff_full,
      crystal.vol,
      occ,
      k_weights=ctx.ksampling.weights,
    )

  # For NC, deploy nonlocal potential to devices.
  if ctx.potential_nonlocal is not None:
    potential_nl = ctx.potential_nonlocal
    if hasattr(potential_nl,
               "projectors") and hasattr(potential_nl, "d_matrices"):
      from ..pseudopotential.nloc import NonlocalProjectorGrid

      potential_nl = NonlocalProjectorGrid(
        projectors=jax.device_put(
          potential_nl.projectors,
          NamedSharding(mesh, P(None, "k")),
        ),
        d_matrices=jax.device_put(
          potential_nl.d_matrices,
          NamedSharding(mesh, P()),
        ),
        projector_mask=jax.device_put(
          potential_nl.projector_mask,
          NamedSharding(mesh, P()),
        ),
      )
    else:
      potential_nl = jax.device_put(
        potential_nl,
        NamedSharding(mesh, P("k")),
      )
    ctx = ctx.replace(potential_nonlocal=potential_nl)

  # Update ksampling on ctx with sharded arrays
  ctx = ctx.replace(
    ksampling=ctx.ksampling.replace(kpts=k_vec, weights=k_weights),
  )

  # --- Occupation function ---
  occupation_setup = _build_occupation_setup(
    num_electrons=num_electrons,
    spin=crystal.spin,
    spin_restricted=config.system.spin_restricted,
    num_bands=num_bands,
    num_kpts=num_kpts,
    smearing=smearing,
  )
  occ_fn = occupation_setup.fn

  # --- Energy function (delegates to backend) ---
  def free_energy(params_pw, params_occ):
    coeff = _pw.coeff(params_pw, freq_mask, sharding=sharding)
    coeff = _to_physical_coeff(coeff)
    occ = occ_fn(params_occ)
    total = backend.total_energy(coeff, occ, ctx)
    free, _ = _free_energy_from_total_energy(total, occ, smearing)
    return free, (total, free)

  # --- Init params + optimiser ---
  optimizer = create_direct_opt_optimizer(config)
  start_step = 0
  previous_total_energy = None
  if restart_state is None:
    params_pw = _pw.param_init(
      key,
      num_bands,
      num_kpts,
      freq_mask,
      spin_restricted=config.system.spin_restricted,
      sharding=sharding,
    )
    params_occ = occupation_setup.params
  else:
    params_pw = {
      "w_re": restart_state["coefficients"]["w_re"],
      "w_im": restart_state["coefficients"]["w_im"],
    }
    if occupation_setup.trainable:
      params_occ = _occupation_params_from_tensor(
        restart_state["occupations"],
        spin_restricted=config.system.spin_restricted,
      )
    else:
      params_occ = occupation_setup.params
    start_step = int(restart_state["step"]) + 1
    previous_total_energy = float(restart_state["total_energy"])
    stage_warning(
      "DirectOpt",
      f"Restarting from step {start_step} using checkpoint state.",
    )
  params = {"pw": params_pw, "occ": params_occ}
  opt_state = optimizer.init(params)
  checkpoint_manager = None
  if output_dir is not None and config.io.save_checkpoint:
    checkpoint_manager = make_checkpoint_manager(output_dir)

  # --- Training loop ---
  convergence_checker = create_convergence_checker(config)
  converged = False
  total_energy_history = []
  convergence_history = []
  last_completed_step = start_step - 1
  last_checkpointed_step = None
  spinner = Spinner("DirectOpt")
  interrupted = False
  nonfinite_abort = False
  last_finite_params = params
  last_finite_opt_state = opt_state

  def _handle_sigint(sig, frame):
    del sig, frame
    nonlocal interrupted
    interrupted = True

  with mesh:
    freeze_allowed = bool(
      occupation_setup.trainable and occupation_warmup_steps > 0
    )

    @jax.jit
    def update(params, opt_state, freeze_occupation):

      def loss_fn(x):
        return free_energy(
          x["pw"],
          x["occ"],
        )

      (_loss_val, aux), grad = jax.value_and_grad(
        loss_fn, has_aux=True,
      )(params)
      grad = jax.lax.cond(
        freeze_occupation,
        _freeze_occupation_gradient,
        lambda g: g,
        grad,
      )
      total_energy, free_energy_value = aux
      total_energy_metric = _total_energy_metric(total_energy, ew)
      updates, new_opt_state = optimizer.update(
        grad,
        opt_state,
        params,
        value=total_energy_metric,
      )
      new_params = optax.apply_updates(params, updates)
      return (
        new_params,
        new_opt_state,
        total_energy,
        free_energy_value,
      )

    old_handler = signal.signal(signal.SIGINT, _handle_sigint)

    def _physical_state_from_params(params, step, total_energy):
      coeff_full_param = _pw.coeff(params["pw"], freq_mask, sharding=sharding)
      coeff_full = _to_physical_coeff(coeff_full_param)
      coeff_compact = squeeze_coefficient(coeff_full_param, freq_mask)
      current_occ = occ_fn(params["occ"])
      density = _density_from_coeff(coeff_full, current_occ)
      return {
        "density":
          density,
        "coefficients":
          {
            "w_re": coeff_compact.real,
            "w_im": coeff_compact.imag,
          },
        "occupations":
          current_occ,
        "eigenvalues":
          jnp.zeros(
            (num_spin, num_kpts, num_bands),
            dtype=coeff_compact.real.dtype,
          ),
        "has_eigenvalues":
          False,
        "step":
          step,
        "total_energy":
          total_energy,
      }

    try:
      if show_progress:
        spinner.start("initialising optimiser state...")

      for step in range(start_step, config.solver.direct_opt.max_steps):
        start = time.time()
        if (
          occupation_setup.trainable and has_occupation_scheduler(config) and
          step == occupation_warmup_steps
        ):
          opt_state = reset_occupation_plateau_state(
            optimizer, opt_state, params
          )
        freeze_now = jnp.asarray(
          freeze_allowed and step < occupation_warmup_steps,
          dtype=jnp.bool_,
        )
        update_phase = (
          "first_call_overhead" if
          (config.execution.profile and step == start_step) else "update_step"
        )
        with phase_timer.phase(update_phase):
          params, opt_state, total_val, free_val = update(
            params,
            opt_state,
            freeze_now,
          )
          total_val, free_val = jax.block_until_ready((total_val, free_val))
        total_energy = float(_total_energy_metric(total_val, ew))
        free_energy_display = float(free_val + ew)
        if not np.isfinite(total_energy) or not np.isfinite(free_energy_display):
          nonfinite_abort = True
          params = last_finite_params
          opt_state = last_finite_opt_state
          stage_warning(
            "DirectOpt",
            (
              f"Non-finite energy at step {step + 1}; "
              "restoring previous finite state and stopping. "
              "Reduce `solver.direct_opt.optimizer.learning_rate`."
            ),
          )
          break
        delta_energy = None
        if total_energy_history:
          delta_energy = abs(total_energy - total_energy_history[-1])
        elif previous_total_energy is not None:
          delta_energy = abs(total_energy - previous_total_energy)
        total_energy_history.append(total_energy)
        converged = convergence_checker.check(total_energy)
        energy_std = convergence_checker.current_std()
        dt = time.time() - start
        cumulative_time = time.time() - overall_start
        display_step = step + 1
        last_completed_step = step
        convergence_history.append(
          {
            "step": display_step,
            "total_energy": total_energy,
            "free_energy": free_energy_display,
            "delta_energy": delta_energy,
            "energy_std": energy_std,
            "wall_time": dt,
            "cumulative_time_s": cumulative_time,
          }
        )
        last_finite_params = params
        last_finite_opt_state = opt_state

        if show_progress:
          spinner.update(
            format_ground_state_iteration(
              "DirectOpt",
              step=display_step,
              max_steps=config.solver.direct_opt.max_steps,
              total_energy=total_energy,
              delta_energy=delta_energy,
              step_time=dt,
              cumulative_time=cumulative_time,
              energy_std=energy_std,
            )
          )

        if checkpoint_manager is not None and (
          display_step % config.io.checkpoint_interval == 0 or interrupted
        ):
          with phase_timer.phase("checkpoint"):
            save_checkpoint(
              checkpoint_manager,
              _physical_state_from_params(params, step, total_energy),
              step,
            )
          last_checkpointed_step = step

        if interrupted:
          stage_warning("DirectOpt", "Interrupted — checkpoint saved")
          break

        if nonfinite_abort:
          break

        if converged:
          break

      if checkpoint_manager is not None and (
        last_completed_step >= 0 and
        last_checkpointed_step != last_completed_step
      ):
        save_checkpoint(
          checkpoint_manager,
          _physical_state_from_params(
            params,
            last_completed_step,
            total_energy_history[-1],
          ),
          last_completed_step,
        )
    finally:
      signal.signal(signal.SIGINT, old_handler)
      if checkpoint_manager is not None:
        checkpoint_manager.wait_until_finished()
      spinner.stop()

  # --- Final energy decomposition ---
  coeff_param = _pw.coeff(params["pw"], freq_mask)
  coeff = _to_physical_coeff(coeff_param)
  occ = occ_fn(params["occ"])
  coeff_compact = squeeze_coefficient(coeff_param, freq_mask)
  density = _density_from_coeff(coeff, occ)
  decomp = backend.energy_decomposition(coeff, occ, ctx)
  total_e = float(sum(decomp.values()) + ew)
  wall_time = time.time() - overall_start
  profiling = {}

  log_ground_state_finish(
    "DirectOpt",
    converged=converged,
    steps_completed=len(total_energy_history),
    max_steps=config.solver.direct_opt.max_steps,
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
      "DirectOpt",
      profiling["phases"],
      total_wall_time=wall_time,
    )
  log_energy_breakdown(
    "DirectOpt",
    decomp,
    ewald=ew,
    total_energy=total_e,
  )

  return GroundStateResult(
    config=config,
    crystal=crystal,
    params_pw=params["pw"],
    params_occ=params["occ"],
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
    occupations=occ,
    actual_solver="direct_opt",
    requested_solver_mode=requested_solver_mode or config.solver.mode,
    num_iterations=max(last_completed_step + 1, 0),
    wall_time=wall_time,
    convergence_history=convergence_history,
    total_energy_history=total_energy_history,
    profiling=profiling,
  )
