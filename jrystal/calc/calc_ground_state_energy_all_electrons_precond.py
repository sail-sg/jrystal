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
from dataclasses import dataclass
from math import ceil
from typing import List, Union, Callable, Any

import numpy as np
import jax
import jax.numpy as jnp

import optax
from absl import logging
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from einops import einsum
from tqdm import tqdm

from jrystal._src import energy, entropy, occupation, pw
from jrystal._src.crystal import Crystal
from jrystal._src.grid import proper_grid_size
# from jrystal._src.hessian import hessian_diag_pytree
from jrystal._src.preconditioner import preconditioner_neumann
from jrystal.config import JrystalConfigDict
from jrystal.calc.convergence import create_convergence_checker
from jrystal.calc.opt_utils import (
  create_crystal,
  create_freq_mask,
  create_grids,
  # create_optimizer,
  set_env_params,
  get_ewald_coulomb_repulsion
)


@dataclass
class GroundStateEnergyOutput:
  """Output of the ground state energy calculation.

  Args:
    config (JrystalConfigDict): The configuration for the calculation.
    crystal (Crystal): The crystal object.
    params_pw (dict): Parameters for the plane wave basis.
    params_occ (dict): Parameters for the occupation.
    total_energy (Union[float, jax.Array]): The total energy of the crystal.
    total_energy_history (List[float]): The optimization history of the total
    energy.
  """
  config: JrystalConfigDict
  crystal: Crystal
  params_pw: dict
  params_occ: dict
  total_energy: Union[float, jax.Array]
  total_energy_history: List[float]


def calc(config: JrystalConfigDict) -> GroundStateEnergyOutput:
  """Calculate the ground state energy of a crystal with norm-conserving
  pseudopotential.

  Args:
    config (JrystalConfigDict): The configuration for the calculation.

  Returns:
    GroundStateEnergyOutput: The ground state energy output of the crystal.
  """
  # Initialize and Prepare variables.
  set_env_params(config)
  key = jax.random.PRNGKey(config.seed)
  temp = config.smearing

  crystal = create_crystal(config)
  num_electrons = crystal.num_electron
  logging.info(f"Crystal: {crystal.symbols}")
  EPS = config.eps

  # Initialize the mesh and sharding for the parallelization.
  num_devices = len(jax.devices())
  util_devices = num_devices if config.parallel_over_k_mesh else 1
  logging.info(f"Parallel over k-mesh: {config.parallel_over_k_mesh}.")
  logging.info(f"Number of devices (used): {num_devices} ({util_devices}).")

  mesh = Mesh(
    np.array(jax.devices()[:util_devices]).reshape([1, -1]), ('s', 'k')
  )
  sharding = NamedSharding(mesh, P('s', 'k'))  # shard by the kpt dimension.

  g_vec, r_vec, k_vec = create_grids(config)
  num_kpts = k_vec.shape[0]
  logging.info(f"Number of G-vectors: {proper_grid_size(config.grid_sizes)}")
  logging.info(f"Number of k-vectors: {proper_grid_size(config.k_grid_sizes)}")
  num_bands = ceil(num_electrons / 2) + config.empty_bands
  logging.info(f"num_bands: {num_bands}")
  logging.info(f"XC functional: {config.xc}")
  logging.info(f"Occupation method: {config.occupation}")
  freq_mask = create_freq_mask(config)
  ew = get_ewald_coulomb_repulsion(config)

  convergence_checker = create_convergence_checker(config)
  converged = False
  k_vec = jax.device_put(k_vec, NamedSharding(mesh, P('k')))

  # Define functions for energy calculation.
  def get_occupation(params):
    return occupation.occupation(
      params,
      num_kpts,
      num_electrons,
      spin=crystal.spin,
      method=config.occupation,
      spin_restricted=config.spin_restricted
    )

  def total_energy(params_pw, params_occ, g_vec):
    coeff = pw.coeff(params_pw, freq_mask, sharding=sharding)
    occ = get_occupation(params_occ)
    density = pw.density_grid(coeff, crystal.vol, occ)
    density_reciprocal = pw.density_grid_reciprocal(coeff, crystal.vol, occ)
    kinetic = energy.kinetic(g_vec, k_vec, coeff, occ)
    hartree = energy.hartree(density_reciprocal, g_vec, crystal.vol)
    external = energy.external(
      density_reciprocal,
      crystal.positions,
      crystal.charges,
      g_vec,
      crystal.vol
    )

    xc = energy.xc_energy(
      density, g_vec, crystal.vol, config.xc, kohn_sham=False
    )
    return kinetic + hartree + external + xc

  def get_entropy(params_occ):
    occ = get_occupation(params_occ)
    return entropy.fermi_dirac(occ, eps=EPS)

  def free_energy(params_pw, params_occ, temp, g_vec):
    total = total_energy(params_pw, params_occ, g_vec)
    etro = get_entropy(params_occ)
    free = total - temp * etro
    return free, (total, etro)

  params_pw = pw.param_init(
    key,
    num_bands,
    num_kpts,
    freq_mask,
    spin_restricted=config.spin_restricted,
    sharding=sharding
  )
  params_occ = occupation.param_init(
    key,
    num_bands,
    num_electrons,
    num_kpts,
    crystal.spin,
    config.occupation,
    spin_restricted=config.spin_restricted
  )
  # params_occ = jax.device_put(params_occ, sharding)
  params = {"pw": params_pw, "occ": params_occ}
  logging.info("Computing preconditioner...")
  start = time.time()

  occ_init = get_occupation(params_occ)

  def total_energy_shifted(params_pw, g_vec):
    coeff = pw.coeff(params_pw, freq_mask, sharding=sharding)
    shift = einsum(
      coeff, coeff.conj(), "s k b x y z, s k b x y z -> "
    ).real * 200
    # occ = get_occupation(params_occ)
    density = pw.density_grid(coeff, crystal.vol, occ_init)
    density_reciprocal = pw.density_grid_reciprocal(
      coeff, crystal.vol, occ_init
    )
    kinetic = energy.kinetic(g_vec, k_vec, coeff, occ_init)
    hartree = energy.hartree(density_reciprocal, g_vec, crystal.vol)
    external = energy.external(
      density_reciprocal,
      crystal.positions,
      crystal.charges,
      g_vec,
      crystal.vol
    )

    xc = energy.xc_energy(
      density, g_vec, crystal.vol, config.xc, kohn_sham=False
    )
    return kinetic + hartree + external + xc + shift

  def get_precond_fun(
    params_pw,
    max_iter: int = 500,
    max_eigval: float = 300
  ) -> Callable:
    return preconditioner_neumann(
      lambda x: total_energy_shifted(x, g_vec),
      params_pw,
      max_iter=max_iter,
      max_eigval=max_eigval,
    )

  # grad_init = jax.grad(total_energy_shifted)(params_pw, g_vec)
  # preconditioner = get_precond_fun(params_pw)
  # precond = preconditioner(grad_init)
  # precond = jax.tree.map(lambda x: jnp.clip(x, 1e-6, 1e6), precond)

  def create_preconditioned_sgd(
    learning_rate: float,
    precond: Any = None,
  ):
    def scale_by_preconditioner(eps: float = 1e-12):

      def init_fn(params):
        if precond is None:
          pass
        else:
          assert jax.tree.structure(
            precond
          ) == jax.tree.structure(params['pw'])
        return ()

      def update_fn(updates, state, params):
        if precond is None:
          preconditioner = get_precond_fun(params['pw'], max_iter=20)
          precond_grad_pw = preconditioner(updates['pw'])
          precond_grad_pw = jax.tree.map(
            lambda x: jnp.clip(x, -1e6, 1e6), precond_grad_pw
          )
        else:
          precond_grad_pw = jax.tree.map(
            lambda x, y: x * y, precond, updates['pw']
          )
        return {'pw': precond_grad_pw, 'occ': updates['occ']}, state

      return optax.GradientTransformation(init_fn, update_fn)

    return optax.chain(
      scale_by_preconditioner(),
      optax.adam(learning_rate),
    )

  logging.info(
    f"Computing preconditioner time: {time.time() - start: .3f} seconds"
  )

  optimizer = create_preconditioned_sgd(1e2)
  # optimizer = optax.sgd(1e-2)
  opt_state = optimizer.init(params)

  # Define update function.
  with mesh:

    @jax.jit
    def update(params, opt_state, temp, g_vec):
      loss = lambda x: free_energy(x["pw"], x["occ"], temp, g_vec)
      (loss_val, es), grad = jax.value_and_grad(loss, has_aux=True)(params)
      updates, opt_state = optimizer.update(grad, opt_state, params)
      params = optax.apply_updates(params, updates)
      return params, opt_state, loss_val, es

    # Define scheduler for temperature annealing.
    if config.smearing > 0.:
      temperature_scheduler = optax.exponential_decay(
        init_value=100.,
        transition_steps=config.epoch // 2,
        decay_rate=config.smearing / 100,
        end_value=config.smearing
      )
    else:

      def temperature_scheduler(i):
        return 0.

    logging.info(f"smearing: {config.smearing}")

    # The main loop for optimization.
    if config.verbose:
      iters = tqdm(range(config.epoch))
    else:
      iters = tqdm(range(config.epoch), disable=True)

    train_time = 0
    for i in iters:
      temp = temperature_scheduler(i)
      start = time.time()
      params, opt_state, loss_val, es = update(params, opt_state, temp, g_vec)
      etot, entro = es
      etot = jax.block_until_ready(etot)
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
  occ = get_occupation(params["occ"])
  density = pw.density_grid(coeff, crystal.vol, occ)
  density_reciprocal = pw.density_grid_reciprocal(coeff, crystal.vol, occ)
  kinetic = energy.kinetic(g_vec, k_vec, coeff, occ)
  hartree = energy.hartree(density_reciprocal, g_vec, crystal.vol)
  external = energy.external(
    density_reciprocal, crystal.positions, crystal.charges, g_vec, crystal.vol
  )

  xc = energy.xc_energy(density, g_vec, crystal.vol, config.xc, kohn_sham=False)

  logging.info(f"Hartree Energy: {hartree:.4f} Ha")
  logging.info(f"External Energy: {external:.4f} Ha")
  logging.info(f"XC Energy: {xc:.4f} Ha")
  logging.info(f"Kinetic Energy: {kinetic:.4f} Ha")
  logging.info(f"Nuclear repulsion Energy: {ew:.4f} Ha")
  logging.info(f"Total Energy: {etot+ew:.4f} Ha")

  return density


if __name__ == "__main__":
  import jax
  jax.config.update("jax_debug_nans", True)
  import jrystal
  path = jrystal.get_pkg_path()
  config = jrystal.config.get_config(path + "/config_precond.yaml")
  calc(config)
