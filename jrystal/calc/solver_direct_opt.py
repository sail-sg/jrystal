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

import time
from math import ceil
from typing import TYPE_CHECKING

import jax
import numpy as np
import optax
from absl import logging
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

from .._src import occupation as _occupation
from .._src import pw as _pw
from .convergence import create_convergence_checker
from .opt_utils import create_optimizer
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


def run_direct_opt(
  config: JrystalConfigDict,
  ctx: RuntimeContext,
  backend: AllElectronBackend | NormConservingBackend,
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
  g_vec = ctx.g_vec
  freq_mask = ctx.basis.freq_mask
  ew = ctx.ewald_energy
  k_vec = ctx.ksampling.kpts
  k_weights = ctx.ksampling.weights

  num_electrons = backend.num_electrons(ctx)
  num_kpts = k_vec.shape[0]
  num_bands = ceil(num_electrons / 2) + config.occupation.empty_bands

  logging.info(f"Crystal: {crystal.symbols}")
  log_ground_state_start(
    "DirectOpt",
    max_steps=config.solver.direct_opt.max_steps,
    num_bands=num_bands,
    smearing=config.occupation.smearing,
    xc=config.method.xc,
    controls=(
      f"optimizer={config.solver.direct_opt.optimizer.name} "
      f"(lr={config.solver.direct_opt.optimizer.learning_rate:g}) | "
      f"window={config.solver.direct_opt.convergence.window_size} | "
      f"energy_std_tol="
      f"{config.solver.direct_opt.convergence.energy_std_tol:.2e}"
    ),
  )

  # --- Sharding / device setup ---
  num_devices = ctx.execution.num_devices
  util_devices = num_devices if ctx.execution.parallel_over_k else 1
  logging.info(
    f"Parallel over k: {ctx.execution.parallel_over_k}. "
    f"Devices: {num_devices} (used {util_devices})."
  )

  mesh = Mesh(
    np.array(jax.devices()[:util_devices]).reshape([1, -1]), ("s", "k"),
  )
  sharding = NamedSharding(mesh, P("s", "k"))

  k_vec = jax.device_put(k_vec, NamedSharding(mesh, P("k")))
  k_weights = jax.device_put(k_weights, NamedSharding(mesh, P("k")))

  # For NC, deploy nonlocal potential to devices
  if ctx.potential_nonlocal is not None:
    potential_nl = jax.device_put(
      ctx.potential_nonlocal, NamedSharding(mesh, P("k")),
    )
    ctx = ctx.replace(potential_nonlocal=potential_nl)

  # Update ksampling on ctx with sharded arrays
  ctx = ctx.replace(
    ksampling=ctx.ksampling.replace(kpts=k_vec, weights=k_weights),
  )

  # --- Occupation function ---
  occ_fn = _occupation.get_occupation_fn(
    num_electrons,
    spin=crystal.spin,
    spin_restricted=config.system.spin_restricted,
  )

  # --- Energy function (delegates to backend) ---
  def free_energy(params_pw, params_occ):
    coeff = _pw.coeff(params_pw, freq_mask, sharding=sharding)
    occ = occ_fn(params_occ)
    etot = backend.total_energy(coeff, occ, ctx)
    return etot, etot

  # --- Init params + optimiser ---
  optimizer = create_optimizer(config)
  params_pw = _pw.param_init(
    key, num_bands, num_kpts, freq_mask,
    spin_restricted=config.system.spin_restricted,
    sharding=sharding,
  )
  params_occ = _occupation.params_init(num_bands, num_kpts)
  params = {"pw": params_pw, "occ": params_occ}
  opt_state = optimizer.init(params)

  # --- Training loop ---
  convergence_checker = create_convergence_checker(config)
  converged = False
  total_energy_history = []

  with mesh:

    @jax.jit
    def update(params, opt_state):
      loss_fn = lambda x: free_energy(x["pw"], x["occ"])
      (loss_val, etot), grad = jax.value_and_grad(
        loss_fn, has_aux=True,
      )(params)
      updates, new_opt_state = optimizer.update(grad, opt_state)
      new_params = optax.apply_updates(params, updates)
      return new_params, new_opt_state, loss_val, etot

    iters = tqdm(
      range(config.solver.direct_opt.max_steps),
      disable=not config.execution.verbose,
    )
    for i in iters:
      start = time.time()
      params, opt_state, loss_val, etot = update(params, opt_state)
      etot = jax.block_until_ready(etot)
      total_energy = float(etot + ew)
      delta_energy = None
      if total_energy_history:
        delta_energy = abs(total_energy - total_energy_history[-1])
      total_energy_history.append(total_energy)
      converged = convergence_checker.check(float(etot))
      energy_std = convergence_checker.current_std()
      dt = time.time() - start
      iters.set_description(
        format_ground_state_iteration(
          "DirectOpt",
          step=i + 1,
          max_steps=config.solver.direct_opt.max_steps,
          total_energy=total_energy,
          delta_energy=delta_energy,
          step_time=dt,
          energy_std=energy_std,
        ),
        refresh=False,
      )
      if converged:
        break

  # --- Final energy decomposition ---
  coeff = _pw.coeff(params["pw"], freq_mask)
  occ = occ_fn(params["occ"])
  density = _pw.density_grid(
    coeff, crystal.vol, occ, k_weights=ctx.ksampling.weights,
  )
  decomp = backend.energy_decomposition(coeff, occ, ctx)
  total_e = float(sum(decomp.values()) + ew)
  wall_time = time.time() - overall_start

  log_ground_state_finish(
    "DirectOpt",
    converged=converged,
    steps_completed=len(total_energy_history),
    max_steps=config.solver.direct_opt.max_steps,
    total_energy=total_e,
    wall_time=wall_time,
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
      **{k: float(v) for k, v in decomp.items()},
    ),
    converged=converged,
    density=density,
    total_energy_history=total_energy_history,
  )
