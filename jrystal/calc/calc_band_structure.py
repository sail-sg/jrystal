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
"""Band Structure Calculator. """

import time
from dataclasses import dataclass
from math import ceil
from typing import Optional

import jax
import jax.numpy as jnp
import optax
from absl import logging
from tqdm import tqdm

from .._src import hamiltonian, occupation, pw
from .._src.band import get_k_path
from .._src.crystal import Crystal
from .._src.utils import wave_to_density
from ..config import JrystalConfigDict
from .calc_ground_state_energy import GroundStateEnergyOutput
from .calc_ground_state_energy import calc as energy_calc
from .opt_utils import (
  create_crystal,
  create_freq_mask,
  create_grids,
  create_optimizer,
  set_env_params
)


@dataclass
class BandStructureOutput:
  """Output of the band structure calculation.

  Args:
    config (JrystalConfigDict): Configuration for the calculation.
    crystal (Crystal): The crystal object.
    params_pw (dict): Parameters for the plane wave basis.
    ground_state_energy_output (GroundStateEnergyOutput): The output of the ground state energy calculation.
    k_path (jax.Array): The K-path.
    band_structure (jax.Array): The band structure.
  """
  config: JrystalConfigDict
  crystal: Crystal
  params_pw: dict
  ground_state_energy_output: GroundStateEnergyOutput
  k_path: jax.Array
  band_structure: jax.Array


def calc(
  config: JrystalConfigDict,
  ground_state_energy_output: Optional[GroundStateEnergyOutput] = None
) -> BandStructureOutput:
  """Calculate the band structure of a crystal.

  Args:
      config (JrystalConfigDict): The configuration for the calculation.
      ground_state_energy_output (Optional[GroundStateEnergyOutput], optional): The output of the ground state energy calculation. Defaults to None.

  Returns:
      BandStructureOutput: The band structure output of the crystal.
  """
  if config.use_pseudopotential:
    raise RuntimeError(
      "This calculator does not support pseudopotential. It only supports all electron calculations. For norm-conserving pseudopotential calculations, please use the `calc_band_structure_normcons.calc` function."
    )

  set_env_params(config)
  key = jax.random.PRNGKey(config.seed)
  crystal = create_crystal(config)
  g_vec, _, k_vec = create_grids(config)
  freq_mask = create_freq_mask(config)
  xc = config.xc

  # generate K-path.
  logging.info("===> Generating K-path...")
  k_path = get_k_path(
    crystal.cell_vectors,
    path=config.k_path_special_points,
    num=config.num_kpoints,
    fractional=False
  )
  logging.info(
    f"{k_path.shape[0]} k-points generated. ({config.k_path_special_points})"
  )

  # optimitimize ground state energy if not provided.
  if ground_state_energy_output is None:
    logging.info("===> Starting total energy minimization...")
    start = time.time()
    ground_state_energy_output = energy_calc(config)
    logging.info(
      f" Total energy minimization done. "
      f"Time: {time.time()-start:.3f} "
    )

  params_pw_ground_state = ground_state_energy_output.params_pw
  params_occ_ground_state = ground_state_energy_output.params_occ

  # Calculate ground state density at grid points.
  def get_occupation(params):
    return occupation.idempotent(
      params,
      crystal.num_electron,
      k_vec.shape[0],
      crystal.spin,
      config.spin_restricted
    )

  coeff_ground_state = pw.coeff(params_pw_ground_state, freq_mask)
  occ_ground_state = get_occupation(params_occ_ground_state)
  ground_state_density_grid = pw.density_grid(
    coeff_ground_state, crystal.vol, occ_ground_state
  )
  if not config.spin_restricted:
    wave_grid_arr = pw.wave_grid(coeff_ground_state, crystal.vol)
    o_alpha, o_beta = occ_ground_state
    n_alpha_grid = wave_to_density(wave_grid_arr[0], o_alpha)
    n_beta_grid = wave_to_density(wave_grid_arr[1], o_beta)
    ground_state_density_grid = jnp.vstack([n_alpha_grid, n_beta_grid])

  # Define the objective function for band structure calculation.
  def hamiltonian_trace(params_pw_band, kpts, g_vec=g_vec):
    coeff_band = pw.coeff(params_pw_band, freq_mask)
    energy = hamiltonian.hamiltonian_matrix_trace(
      coeff_band,
      crystal.positions,
      crystal.charges,
      ground_state_density_grid,
      g_vec,
      kpts,
      crystal.vol,
      xc,
      config.spin_restricted,
    )
    return jnp.sum(energy).real

  # Initialize parameters and optimizer.
  optimizer = create_optimizer(config)
  num_bands = ceil(crystal.num_electron / 2) + config.band_structure_empty_bands
  params_pw_band = pw.param_init(
    key, num_bands, 1, freq_mask, config.spin_restricted
  )
  opt_state = optimizer.init(params_pw_band)

  # define update function
  @jax.jit
  def update(params, opt_state, kpts, g_vec):
    hamil_trace, grad = jax.value_and_grad(hamiltonian_trace)(
      params, kpts, g_vec
    )

    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, hamil_trace

  # the main loop for band structure calculation.
  logging.info("===> Starting band structure calculation...")
  logging.info(f"Number of bands: {num_bands}")
  logging.info("Optimizing the first K point...")

  params_kpoint_list = []
  eigen_values = []

  if config.verbose:
    iters = tqdm(range(config.band_structure_epoch))
  else:
    iters = tqdm(range(config.band_structure_epoch), disable=True)

  start = time.time()
  for i in iters:
    params_pw_band, opt_state, hamil_trace = update(
      params_pw_band, opt_state, k_path[0:1], g_vec
    )
    iters.set_description(f"Hamiltonian trace: {hamil_trace:.4E}")

  # TODO: introduce convergence tracker.
  converged = True

  params_kpoint_list.append(params_pw_band)
  logging.info(f" Converged: {converged}.")
  logging.info(f" Total epochs run: {i+1}.")
  logging.info(f" Training Time: {(time.time() - start):.3f}s.")
  logging.info("===> Starting fine-tuning the rest k points...")
  logging.info(f" fine tuning steps: {config.k_path_fine_tuning_epoch}")

  if config.verbose:
    iters = tqdm(range(1, k_path.shape[0]))
  else:
    iters = tqdm(range(1, k_path.shape[0]), disable=True)

  for i in iters:
    for _ in range(config.k_path_fine_tuning_epoch):
      params_pw_band, opt_state, hamil_trace = update(
        params_pw_band, opt_state, k_path[i:(i+1)], g_vec
      )
    iters.set_description(f" Loss(the {i+1}th k point): {hamil_trace:.4E}")
    params_kpoint_list.append(params_pw_band)

  logging.info("===> Band structure calculation done.")

  ########################################################
  # One-time eigen decomposition
  logging.info("===> Diagonalizing the Hamiltonian matrix...")

  @jax.jit
  def eig_fn(
    coeff_k,
    k,
    ground_state_density_grid,
    g_vec,
  ):
    hamil_matrix = hamiltonian.hamiltonian_matrix(
      coeff_k,
      crystal.positions,
      crystal.charges,
      ground_state_density_grid,
      g_vec,
      k,
      crystal.vol,
      xc,
      config.spin_restricted,
      kohn_sham=True,
    )

    eigen_values = jnp.linalg.eigvalsh(
      hamil_matrix.reshape(-1, num_bands, num_bands)
    )

    return eigen_values

  iters = tqdm(range(len(params_kpoint_list)))
  for i in iters:
    prm = params_kpoint_list[i]
    k = k_path[i:(i + 1), :]
    iters.set_description(f"Diagonolizing the {i+1}th k points")
    coeff_k = pw.coeff(prm, freq_mask)
    eig = eig_fn(coeff_k, k, ground_state_density_grid, g_vec)
    eigen_values.append(eig)

  # eigen_values = jnp.vstack(eigen_values)
  logging.info("===> Eigen decomposition done.")
  save_file = ''.join(crystal.symbol) + "_band_structure.npy"
  logging.info(f"Results saved in {save_file}")
  jnp.save(save_file, eigen_values)

  return BandStructureOutput(
    config=config,
    crystal=crystal,
    params_pw=params_pw_band,
    ground_state_energy_output=ground_state_energy_output,
    k_path=k_path,
    band_structure=eigen_values,
  )
