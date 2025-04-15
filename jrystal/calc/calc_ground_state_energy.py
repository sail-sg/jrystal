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
from typing import List, Union

import cloudpickle as pickle
import jax
import optax
from absl import logging
from tqdm import tqdm

from .._src import energy, entropy, occupation, pw
from .._src.crystal import Crystal
from .._src.grid import proper_grid_size
from ..config import JrystalConfigDict
from .opt_utils import (
  create_crystal,
  create_freq_mask,
  create_grids,
  create_optimizer,
  get_ewald_coulomb_repulsion,
  set_env_params
)


@dataclass
class GroundStateEnergyOutput:
  """Output of the ground state energy calculation.

  Args:
    config (JrystalConfigDict): Configuration for the calculation.
    crystal (Crystal): The crystal object.
    params_pw (dict): Parameters for the plane wave basis.
    params_occ (dict): Parameters for the occupation.
    total_energy (Union[float, jax.Array]): Total energy of the crystal.
    total_energy_history (List[float]): The optimization history of the total energy.
  """
  config: JrystalConfigDict
  crystal: Crystal
  params_pw: dict
  params_occ: dict
  total_energy: Union[float, jax.Array]
  total_energy_history: List[float]


def calc(config: JrystalConfigDict) -> GroundStateEnergyOutput:
  """Calculate the ground state energy of a crystal.

  Args:
    config (JrystalConfigDict): Configuration for the calculation.

  Returns:
    GroundStateEnergyOutput: The ground state energy of the crystal.
  """

  # check if using correct calculator.
  if config.use_pseudopotential:
    raise RuntimeError(
      "This calculator does not support pseudopotential. It only supports all electron calculations. For norm-conserving pseudopotential calculations, please use the `calc_ground_state_energy_normcons.calc` function."
    )

  # Initialize and Prepare variables.
  set_env_params(config)
  key = jax.random.PRNGKey(config.seed)
  temp = config.smearing

  crystal = create_crystal(config)
  logging.info(f"Crystal: {crystal.symbol}")
  EPS = config.eps

  g_vec, r_vec, k_vec = create_grids(config)
  num_kpts = k_vec.shape[0]
  logging.info(f"Number of G-vectors: {proper_grid_size(config.grid_sizes)}")
  logging.info(f"Number of k-vectors: {proper_grid_size(config.k_grid_sizes)}")
  num_bands = ceil(crystal.num_electron / 2) + config.empty_bands
  freq_mask = create_freq_mask(config)
  ew = get_ewald_coulomb_repulsion(config)

  # Define functions for energy calculation.
  # assume fermi-dirac occupation.
  logging.info(f"Potentially-occupied bands: {num_bands}")

  def get_occupation(params):
    return occupation.idempotent(
      params,
      crystal.num_electron,
      num_kpts,
      crystal.spin,
      config.spin_restricted,
    )

  def total_energy(params_pw, params_occ, g_vec=g_vec):
    coeff = pw.coeff(params_pw, freq_mask)
    occ = get_occupation(params_occ)
    return energy.total_energy(
      coeff,
      crystal.positions,
      crystal.charges,
      g_vec,
      k_vec,
      crystal.vol,
      occ,
      kohn_sham=False,
      xc=config.xc,
      spin_restricted=config.spin_restricted,
    )

  def get_entropy(params_occ):
    occ = get_occupation(params_occ)
    return entropy.fermi_dirac(occ, eps=EPS)

  def free_energy(params_pw, params_occ, temp, g_vec=g_vec):
    total = total_energy(params_pw, params_occ, g_vec)
    etro = get_entropy(params_occ)
    free = total - temp * etro
    return free, (total, etro)

  # Initialize parameters and optimizer.
  optimizer = create_optimizer(config)
  params_pw = pw.param_init(
    key, num_bands, num_kpts, freq_mask, config.spin_restricted
  )
  params_occ = occupation.idempotent_param_init(key, num_bands, num_kpts)
  params = {"pw": params_pw, "occ": params_occ}
  opt_state = optimizer.init(params)

  # Define update function.
  @jax.jit
  def update(params, opt_state, temp, g_vec):
    loss = lambda x: free_energy(x["pw"], x["occ"], temp, g_vec)
    (loss_val, es), grad = jax.value_and_grad(loss, has_aux=True)(params)
    updates, opt_state = optimizer.update(grad, opt_state)
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

    iters.set_description(
      f"Loss: {loss_val:.4f}|Energy: {etot+ew:.4f}|"
      f"Entropy: {entro:.4f}|T: {temp:.2E}"
    )

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
  exc = energy.exc_functional(density, g_vec, crystal.vol, config.xc)

  logging.info(f"Hartree Energy: {hartree:.4f} Ha")
  logging.info(f"External Energy: {external:.4f} Ha")
  logging.info(f"XC Energy: {exc:.4f} Ha")
  logging.info(f"Kinetic Energy: {kinetic:.4f} Ha")
  logging.info(f"Nuclear repulsion Energy: {ew:.4f} Ha")
  logging.info(f"Total Energy: {etot+ew:.4f} Ha")

  output = GroundStateEnergyOutput(
    config, crystal, params["pw"], params["occ"], etot + ew, []
  )

  save_file = ''.join(crystal.symbol) + "_ground_state.pkl"
  with open(save_file, "wb") as f:
    pickle.dump(output, f)

  return output
