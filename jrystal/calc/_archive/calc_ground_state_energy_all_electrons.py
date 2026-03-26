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

import time
from math import ceil

import jax
import numpy as np
import optax
from absl import logging
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

from .._src import energy, occupation, pw
from .._src.grid import proper_grid_size
from ..config import JrystalConfigDict
from .convergence import create_convergence_checker
from .opt_utils import (
  create_optimizer,
  set_env_params,
)
from .runtime import build_runtime_context
from .types import EnergyDecomposition, GroundStateResult


def calc(config: JrystalConfigDict) -> GroundStateResult:
  """Calculate the ground state energy of a crystal with norm-conserving pseudopotential.

  Args:
      config (JrystalConfigDict): The configuration for the calculation.

  Returns:
      GroundStateResult: The ground state energy output of the crystal.
  """
  # Initialize and Prepare variables.
  set_env_params(config)
  key = jax.random.PRNGKey(config.execution.seed)
  temp = config.occupation.smearing

  ctx = build_runtime_context(config)
  crystal = ctx.crystal
  g_vec = ctx.g_vec
  freq_mask = ctx.basis.freq_mask
  ew = ctx.ewald_energy
  k_vec = ctx.ksampling.kpts
  k_weights = ctx.ksampling.weights
  num_electrons = crystal.num_electron
  logging.info(f"Crystal: {crystal.symbols}")

  # Initialize the mesh and sharding for the parallelization.
  num_devices = len(jax.devices())
  util_devices = num_devices if config.execution.parallel_over_k_mesh else 1
  logging.info(
    f"Parallel over k-mesh: {config.execution.parallel_over_k_mesh}."
  )
  logging.info(f"Number of devices (used): {num_devices} ({util_devices}).")

  mesh = Mesh(
    np.array(jax.devices()[:util_devices]).reshape([1, -1]), ('s', 'k')
  )
  sharding = NamedSharding(mesh, P('s', 'k'))  # shard by the kpt dimension.

  num_kpts = k_vec.shape[0]
  logging.info(
    f"Number of G-vectors: {proper_grid_size(config.basis.grid_sizes)}"
  )
  logging.info(
    f"Number of k-vectors: {proper_grid_size(config.ksampling.k_grid_sizes)}"
  )
  num_bands = ceil(num_electrons / 2) + config.occupation.empty_bands
  logging.info(f"num_bands: {num_bands}")
  logging.info(f"XC functional: {config.method.xc}")
  logging.info(f"Occupation method: {config.occupation.method}")
  convergence_checker = create_convergence_checker(config)
  converged = False
  total_energy_history = []
  k_vec = jax.device_put(k_vec, NamedSharding(mesh, P('k')))
  k_weights = jax.device_put(k_weights, NamedSharding(mesh, P('k')))

  # Define functions for energy calculation.
  occ_fn = occupation.get_occupation_fn(
    int(num_electrons), spin=crystal.spin,
    spin_restricted=config.system.spin_restricted,
  )

  def total_energy(params_pw, params_occ, g_vec):
    coeff = pw.coeff(params_pw, freq_mask, sharding=sharding)
    occ = occ_fn(params_occ)
    density = pw.density_grid(coeff, crystal.vol, occ, k_weights=k_weights)
    density_reciprocal = pw.density_grid_reciprocal(
      coeff, crystal.vol, occ, k_weights=k_weights
    )
    kinetic = energy.kinetic(
      coeff, g_vec, k_vec, kpts_weights=k_weights, occupation=occ
    )
    hartree = energy.hartree(density_reciprocal, g_vec, crystal.vol)
    external = energy.external(
      density_reciprocal,
      crystal.positions,
      crystal.charges,
      g_vec,
      crystal.vol
    )

    xc = energy.xc_energy(
      density, g_vec, crystal.vol, config.method.xc, kohn_sham=False
    )
    return kinetic + hartree + external + xc

  def free_energy(params_pw, params_occ, g_vec):
    total = total_energy(params_pw, params_occ, g_vec)
    return total, (total, 0.0)

  # Initialize parameters and optimizer.
  optimizer = create_optimizer(config)
  params_pw = pw.param_init(
    key,
    num_bands,
    num_kpts,
    freq_mask,
    spin_restricted=config.system.spin_restricted,
    sharding=sharding
  )
  params_occ = occupation.params_init(num_bands, num_kpts)
  params = {"pw": params_pw, "occ": params_occ}
  opt_state = optimizer.init(params)

  # Define update function.
  with mesh:

    @jax.jit
    def update(params, opt_state, g_vec):
      loss = lambda x: free_energy(x["pw"], x["occ"], g_vec)
      (loss_val, es), grad = jax.value_and_grad(loss, has_aux=True)(params)
      updates, opt_state = optimizer.update(grad, opt_state)
      params = optax.apply_updates(params, updates)
      return params, opt_state, loss_val, es

    # The main loop for optimization.
    if config.execution.verbose:
      iters = tqdm(range(config.solver.epoch))
    else:
      iters = tqdm(range(config.solver.epoch), disable=True)

    train_time = 0
    for i in iters:
      start = time.time()
      params, opt_state, loss_val, es = update(params, opt_state, g_vec)
      etot, entro = es
      etot = jax.block_until_ready(etot)
      total_energy_history.append(float(etot + ew))
      train_time += time.time() - start
      converged = convergence_checker.check(etot)
      if converged:
        logging.info("Converged.")
        break

      iters.set_description(
        f"Loss: {loss_val:.4f}|Energy: {etot+ew:.4f}|"
        f"Entropy: {entro:.4f}|T: {temp:.2E}"
      )

  if not converged:
    logging.warning("Did not converge.")

  #####################################
  #        END OF OPTIMIZATION        #
  #####################################
  coeff = pw.coeff(params["pw"], freq_mask)
  occ = occ_fn(params["occ"])
  density = pw.density_grid(coeff, crystal.vol, occ, k_weights=k_weights)
  density_reciprocal = pw.density_grid_reciprocal(
    coeff, crystal.vol, occ, k_weights=k_weights
  )
  kinetic = energy.kinetic(
    coeff, g_vec, k_vec, kpts_weights=k_weights, occupation=occ
  )
  hartree = energy.hartree(density_reciprocal, g_vec, crystal.vol)
  external = energy.external(
    density_reciprocal, crystal.positions, crystal.charges, g_vec, crystal.vol
  )

  xc = energy.xc_energy(
    density, g_vec, crystal.vol, config.method.xc, kohn_sham=False
  )
  total_energy = float(kinetic + hartree + external + xc + ew)

  logging.info(f"Hartree Energy: {hartree:.4f} Ha")
  logging.info(f"External Energy: {external:.4f} Ha")
  logging.info(f"XC Energy: {xc:.4f} Ha")
  logging.info(f"Kinetic Energy: {kinetic:.4f} Ha")
  logging.info(f"Nuclear repulsion Energy: {ew:.4f} Ha")
  logging.info(f"Total Energy: {total_energy:.4f} Ha")

  return GroundStateResult(
    config=config,
    crystal=crystal,
    params_pw=params["pw"],
    params_occ=params["occ"],
    total_energy=total_energy,
    energy_terms=EnergyDecomposition(
      kinetic=float(kinetic),
      hartree=float(hartree),
      xc=float(xc),
      external=float(external),
      ewald=float(ew),
    ),
    converged=converged,
    density=density,
    total_energy_history=total_energy_history,
  )
