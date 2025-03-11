"""Band Structure Calculator. """
import jax
import jax.numpy as jnp
import optax
from math import ceil

import time
from typing import Optional
from dataclasses import dataclass
from absl import logging
from tqdm import tqdm

from .calc_ground_state_energy import calc as energy_calc
from .calc_ground_state_energy import GroundStateEnergyOutput
from .opt_utils import set_env_params, create_crystal, create_freq_mask
from .opt_utils import create_grids, create_optimizer, create_pseudopotential

from ..config import JrystalConfigDict
from .._src import pw, hamiltonian, occupation
from .._src.band import get_k_path
from ..pseudopotential import local, normcons


@dataclass
class BandStructureOutput:
  """Output of the band structure calculation. 
  
  Args:
    config (JrystalConfigDict): The configuration for the calculation.
    params_pw (dict): Parameters for the plane wave basis.
    ground_state_energy_output (GroundStateEnergyOutput): The output of the ground state energy calculation.
    k_path (jax.Array): The K-path.
    band_structure (jax.Array): The band structure.
  """
  config: JrystalConfigDict
  params_pw: dict
  ground_state_energy_output: GroundStateEnergyOutput
  k_path: jax.Array
  band_structure: jax.Array


def calc(
  config: JrystalConfigDict,
  ground_state_energy_output: Optional[GroundStateEnergyOutput] = None
) -> BandStructureOutput:
  """Calculate the band structure of a crystal with norm-conserving pseudopotential.

  Args:
      config (JrystalConfigDict): The configuration for the calculation.
      ground_state_energy_output (Optional[GroundStateEnergyOutput], optional): The output of the ground state energy calculation. Defaults to None.

  Returns:
      BandStructureOutput: The band structure output of the crystal.
  """
  set_env_params(config)
  key = jax.random.PRNGKey(config.seed)
  crystal = create_crystal(config)
  g_vec, r_vec, k_vec = create_grids(config)
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
  logging.info(f"{k_path.shape[0]} k-points generated.")

  # initialize pseudopotential
  pseudopot = create_pseudopotential(config)
  potential_loc = local._potential_local_reciprocal(
    crystal.positions,
    r_vec,
    g_vec,
    pseudopot.r_grid,
    pseudopot.local_potential_grid,
    pseudopot.local_potential_charge,
    crystal.vol
  )
  potential_nl = normcons._potential_nonlocal_square_root(
    crystal.positions,
    g_vec,
    k_vec,
    pseudopot.r_grid,
    pseudopot.nonlocal_beta_grid,
    pseudopot.nonlocal_angular_momentum,
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
      params, crystal.num_electron, k_vec.shape[0], crystal.spin
    )

  coeff_ground_state = pw.coeff(params_pw_ground_state, freq_mask)
  occ_ground_state = get_occupation(params_occ_ground_state)
  ground_state_density_grid = pw.density_grid(
    coeff_ground_state, crystal.vol, occ_ground_state
  )

  # Define the objective function for band structure calculation.
  def hamiltonian_trace(params_pw_band, kpts):
    coeff_band = pw.coeff(params_pw_band, freq_mask)

    return normcons._hamiltonian_trace(
      coeff_band,
      ground_state_density_grid,
      potential_loc,
      potential_nl,
      g_vec,
      kpts,
      pseudopot.nonlocal_d_matrix,
      crystal.vol
    )

  # Initialize parameters and optimizer.
  optimizer = create_optimizer(config)
  num_bands = ceil(crystal.num_electron / 2) + config.band_structure_empty_bands
  params_pw_band = pw.param_init(key, num_bands, k_path.shape[0], freq_mask)
  opt_state = optimizer.init(params_pw_band)

  # define update function
  @jax.jit
  def update(params, opt_state, kpts):
    hamil_trace, grad = jax.value_and_grad(hamiltonian_trace)(params, kpts)

    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, hamil_trace

  # the main loop for band structure calculation.
  logging.info("===> Starting band structure calculation...")
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
      params_pw_band, opt_state, k_path[0:1]
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
        params_pw_band, opt_state, k_path[i:(i+1)]
      )
    iters.set_description(f" Loss(the {i+1}th k point): {hamil_trace:.4E}")
    params_kpoint_list.append(params_pw_band)

  logging.info("===> Band structure calculation done.")
  ########################################################
  # One-time eigen decomposition
  logging.info("===> Diagonalizing the Hamiltonian matrix...")

  @jax.jit
  def eig_fn(param, kpts):
    coeff_i = pw.coeff(param, freq_mask)
    hamil_matrix = normcons._hamiltonian_matrix(
      coeff_i,
      ground_state_density_grid,
      potential_loc,
      potential_nl,
      g_vec,
      kpts,
      pseudopot.nonlocal_d_matrix,
      crystal.vol
    )

    hamil_matrix = hamiltonian.hamiltonian_matrix(
      coeff_i,
      crystal.positions,
      crystal.charges,
      ground_state_density_grid,
      g_vec,
      kpts,
      crystal.vol,
      xc
    )
    return jnp.linalg.eigvalsh(hamil_matrix[0])

  iters = tqdm(range(len(params_kpoint_list)))
  for i in iters:
    prm = params_kpoint_list[i]
    k = k_path[i:(i + 1), :]
    iters.set_description(f"Diagonolizing the {i+1}th k points")
    eig = eig_fn(prm, k)
    eigen_values.append(eig)

  eigen_values = jnp.vstack(eigen_values)
  logging.info("===> Eigen decomposition done.")
  save_file = ''.join(crystal.symbol) + "_band_structure.npy"
  logging.info(f"results is saved in {save_file}")
  jnp.save(save_file, eigen_values)

  return BandStructureOutput(
    config=config,
    params_pw=params_pw_band,
    ground_state_energy_output=ground_state_energy_output,
    k_path=k_path,
    band_structure=eigen_values,
  )
