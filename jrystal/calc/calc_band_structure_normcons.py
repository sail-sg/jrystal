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
from functools import partial
from math import ceil
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging

from .._src import pw
from .._src.band import get_k_path
from .._src.crystal import Crystal
from ..config import JrystalConfigDict
from ..pseudopotential import normcons
from .calc_ground_state_energy_normcons import GroundStateEnergyOutput
from .calc_ground_state_energy_normcons import calc as energy_calc
from .opt_utils import (
    create_crystal,
    create_freq_mask,
    create_grids,
    create_optimizer,
    create_pseudopotential,
    set_env_params,
)
from .pre_calc import pre_calc_beta_sbt


@dataclass
class BandStructureOutput:
  """Output of the band structure calculation.

  Args:
    config (JrystalConfigDict): The configuration for the calculation.
    crystal (Crystal): The crystal object.
    params_pw (dict): Parameters for the plane wave basis.
    ground_state_energy_output (GroundStateEnergyOutput): The output of the
    ground state energy calculation.
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
  """Calculate the band structure of a crystal with norm-conserving
  pseudopotential.

  Args:
      config (JrystalConfigDict): The configuration for the calculation.
      ground_state_energy_output (Optional[GroundStateEnergyOutput], optional):
      The output of the ground state energy calculation. Defaults to None.

  Returns:
      BandStructureOutput: The band structure output of the crystal.
  """
  set_env_params(config)
  key = jax.random.PRNGKey(config.seed)
  crystal = create_crystal(config)
  pseudopot = create_pseudopotential(config)
  valence_charges = np.sum(pseudopot.valence_charges)
  g_vec, r_vec, k_vec = create_grids(config)
  freq_mask = create_freq_mask(config)
  xc = config.xc
  logging.info(f"XC functional: {config.xc}")

  # generate K-path.
  logging.info("===> Generating K-path...")
  k_path = get_k_path(
    crystal.cell_vectors,
    path=config.k_path_special_points,
    num=config.num_kpoints,
    fractional=False
  )
  logging.info(f"{k_path.shape[0]} k-points generated.")

  # Initialize the mesh and sharding for the parallelization.
  num_devices = len(jax.devices())
  util_devices = num_devices if config.parallel_over_k_path else 1
  logging.info(f"Parallel over k-path: {config.parallel_over_k_path}.")
  logging.info(f"Number of GPU devices {num_devices}, used: {util_devices}.")
  logging.info("Initializing pseudopotential (local)...")
  potential_loc = normcons.potential_local_reciprocal(
    crystal.positions,
    g_vec,
    pseudopot.r_grid,
    pseudopot.local_potential_grid,
    pseudopot.local_potential_charge,
    crystal.vol
  )
  logging.info("Initializing pseudopotential (Spherical Bessel Transform)...")
  start = time.time()
  beta_gk = pre_calc_beta_sbt(
    pseudopot,
    np.array(g_vec),
    np.array(k_path),
  )  # shape: [kpt beta x y z]
  end = time.time()
  logging.info(
    f"Spherical Bessel Transform done. Times: {end - start:.2f} seconds"
  )

  # optimitimize ground state energy if not provided.
  if ground_state_energy_output is None:
    logging.info("===> Starting total energy minimization...")
    start = time.time()
    ground_state_density_grid = energy_calc(config)
  jax.clear_caches()

  def select_beta_gk(beta_gk, k_idx):
    return jax.tree.map(lambda x: x.at[k_idx:(k_idx+1)].get(), beta_gk)

  def get_potential_nl(kpt, beta_gk):
    return normcons.potential_nonlocal_psi_reciprocal(
      crystal.positions,
      g_vec,
      kpt,
      pseudopot.r_grid,
      pseudopot.nonlocal_beta_grid,
      pseudopot.nonlocal_angular_momentum,
      pseudopot.nonlocal_d_matrix,
      beta_gk
    )

  def hamiltonian_trace(params_pw_band, kpts, g_vector_grid, potential_nl):
    coeff_band = pw.coeff(params_pw_band, freq_mask)
    return normcons.hamiltonian_trace(
      coeff_band,
      ground_state_density_grid,
      potential_loc,
      potential_nl,
      g_vector_grid,
      kpts,
      crystal.vol,
      xc=xc,
      kohn_sham=True
    )

  # Initialize parameters and optimizer.
  optimizer = create_optimizer(config)
  num_bands = ceil(valence_charges / 2) + config.band_structure_empty_bands
  params_pw_band = pw.param_init(key, num_bands, 1, freq_mask)
  opt_state = optimizer.init(params_pw_band)

  # define update function
  @jax.jit
  def update(params, opt_state, kpts, g_vector_grid, potential_nl):
    hamil_trace, grad = jax.value_and_grad(hamiltonian_trace)(
      params, kpts, g_vector_grid, potential_nl
    )

    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, hamil_trace

  # the main loop for band structure calculation.
  logging.info("===> Starting band structure calculation...")
  logging.info(f"Number of bands: {num_bands}")
  logging.info("Optimizing the first K point...")

  # the main function for band structure calculation.
  @partial(jax.pmap, in_axes=(0, 0, 0, 0), devices=jax.devices()[:util_devices])
  def optimize_eigenvalues(kpts, beta_gk, params_pw_band, opt_state):

    logging.info("===> Optimizing the first K point...")
    eigen_values = []

    def update_scan(carry, xs):
      params_pw_band, opt_state, potential_nl_k, kpts = carry
      params_pw_band, opt_state, hamil_trace = update(
        params_pw_band, opt_state, kpts, g_vec, potential_nl_k
      )
      return (params_pw_band, opt_state, potential_nl_k, kpts), None

    # optimize the first k point.
    potential_nl_k = get_potential_nl(kpts[0:1], select_beta_gk(beta_gk, 0))
    carry, _ = jax.lax.scan(
      update_scan, (params_pw_band, opt_state, potential_nl_k, kpts[0:1]),
      length=config.band_structure_epoch, unroll=1
    )
    params_first_kpt, opt_state, _, _ = carry

    logging.info("===> Fine-tuning the rest K points...")

    def finetuning(carry, x):
      kpts, beta_gk = x
      kpts = jnp.expand_dims(kpts, axis=0)
      beta_gk = [jnp.expand_dims(_b, axis=0) for _b in beta_gk]
      params_pw_band, opt_state = carry
      potential_nl_k = get_potential_nl(kpts, beta_gk)

      carry, _ = jax.lax.scan(
        update_scan, (params_pw_band, opt_state, potential_nl_k, kpts),
        length=config.k_path_fine_tuning_epoch, unroll=1
      )
      params_pw_band, opt_state, _, _ = carry

      return (params_pw_band, opt_state), params_pw_band

    carry, params_fine_tuning = jax.lax.scan(
      finetuning, (params_first_kpt, opt_state),
      xs=(kpts[1:], [b[1:] for b in beta_gk])
    )

    logging.info("===> Eigen decomposition...")

    def eig_fn(param, kpts, potential_nl, g_vector_grid):
      coeff_i = pw.coeff(param, freq_mask)
      hamil_matrix = normcons.hamiltonian_matrix(
        coeff_i,
        ground_state_density_grid,
        potential_loc,
        potential_nl,
        g_vector_grid,
        kpts,
        crystal.vol,
        xc,
        kohn_sham=True
      )
      return jnp.linalg.eigvalsh(hamil_matrix[0])

    eigen_values = [
      eig_fn(
        params_first_kpt,
        kpts[0:1],
        get_potential_nl(kpts[0:1], select_beta_gk(beta_gk, 0)),
        g_vec
      )
    ]

    def eig_scan(carry, x):
      k, b, prm = x
      k = jnp.expand_dims(k, axis=0)
      b = [jnp.expand_dims(_b, axis=0) for _b in b]
      potential_nl_k = get_potential_nl(k, b)
      eig = eig_fn(prm, k, potential_nl_k, g_vec)
      return None, eig

    carry, ys = jax.lax.scan(
      eig_scan, None,
      xs=(kpts[1:], [b[1:] for b in beta_gk], params_fine_tuning)
    )
    eigen_values += list(ys)

    return eigen_values

  # reshape the k-path, beta_gk, and params_pw_band for parallelization.
  k_path = jnp.reshape(k_path, (util_devices, -1, 3))
  beta_gk = [jnp.reshape(b, (util_devices, -1, *b.shape[1:])) for b in beta_gk]
  params_pw_band = jax.tree.map(
    lambda x: jnp.stack([x] * util_devices, axis=0), params_pw_band
  )
  opt_state = jax.tree.map(
    lambda x: jnp.stack([x] * util_devices, axis=0), opt_state
  )
  time_start = time.time()

  # perform the band structure calculation. it is parallelized over k-path.
  eigen_values = optimize_eigenvalues(
    k_path, beta_gk, params_pw_band, opt_state
  )
  time_end = time.time()
  logging.info(
    f"Band structure calculation: {time_end - time_start:.2f} seconds"
  )
  eigen_values = jnp.stack(eigen_values)
  eigen_values = jnp.reshape(
    eigen_values, (-1, eigen_values.shape[-1]), order="F"
  )

  logging.info("===> Eigen decomposition done.")
  save_file = ''.join(crystal.symbols) + "_band_structure.npy"
  logging.info(f"Results saved in {save_file}")
  jnp.save(save_file, eigen_values)
