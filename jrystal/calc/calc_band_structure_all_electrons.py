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
from functools import partial
from math import ceil

import jax
import jax.numpy as jnp
import optax
from absl import logging

from .._src import pw, hamiltonian
from .._src.band import get_k_path
from ..config import JrystalConfigDict
from .calc_ground_state_energy_all_electrons import calc as energy_calc
from .opt_utils import (
  create_crystal,
  create_freq_mask,
  create_grids,
  create_optimizer,
  set_env_params,
)


def calc(config: JrystalConfigDict):
  """Calculate the band structure of a crystal with norm-conserving pseudopotential.

  Args:
      config (JrystalConfigDict): The configuration for the calculation.

  Returns:
      BandStructureOutput: The band structure output of the crystal.
  """
  set_env_params(config)
  key = jax.random.PRNGKey(config.seed)
  crystal = create_crystal(config)
  num_electrons = crystal.num_electron
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

  # Initialize the mesh and sharding for the parallelization.
  num_devices = len(jax.devices())
  util_devices = num_devices if config.parallel_over_k_path else 1
  logging.info(f"Parallel over k-path: {config.parallel_over_k_path}.")
  logging.info(f"Number of devices (used): {num_devices} ({util_devices}).")

  # optimitimize ground state energy if not provided.
  logging.info("===> Starting total energy minimization...")
  ground_state_density_grid = energy_calc(config)
  jax.clear_caches()

  def hamiltonian_trace(params_pw_band, kpts, g_vector_grid):
    coeff_band = pw.coeff(params_pw_band, freq_mask)
    output = hamiltonian.hamiltonian_matrix_trace(
      coeff_band,
      crystal.positions,
      crystal.charges,
      ground_state_density_grid,
      g_vector_grid,
      kpts,
      crystal.vol,
      xc,
      kohn_sham=True
    )
    return jnp.sum(output)

  # Initialize parameters and optimizer.
  optimizer = create_optimizer(config)
  num_bands = ceil(num_electrons / 2) + config.band_structure_empty_bands
  params_pw_band = pw.param_init(
    key, num_bands, 1, freq_mask, spin_restricted=config.spin_restricted
  )
  opt_state = optimizer.init(params_pw_band)

  # define update function
  @jax.jit
  def update(params, opt_state, kpts, g_vector_grid):
    hamil_trace, grad = jax.value_and_grad(hamiltonian_trace)(
      params, kpts, g_vector_grid
    )

    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, hamil_trace

  # the main loop for band structure calculation.
  logging.info("===> Starting band structure calculation...")
  logging.info(f"Number of bands: {num_bands}")
  logging.info("Optimizing the first K point...")

  @partial(jax.pmap, in_axes=(0, 0, 0), devices=jax.devices()[:util_devices])
  def optimize_eigenvalues(kpts, params_pw_band, opt_state):
    logging.info("===> Optimizing the first K point...")
    eigen_values = []

    def update_scan(carry, xs):
      params_pw_band, opt_state, kpts = carry
      params_pw_band, opt_state, hamil_trace = update(
        params_pw_band, opt_state, kpts, g_vec
      )
      return (params_pw_band, opt_state, kpts), None

    carry, _ = jax.lax.scan(
      update_scan, (params_pw_band, opt_state, kpts[0:1]),
      length=config.band_structure_epoch, unroll = 1
    )
    params_first_kpt, opt_state, _ = carry

    logging.info("===> Fine-tuning the rest K points...")

    def finetuning(carry, x):
      kpts = x
      kpts = jnp.expand_dims(kpts, axis=0)

      params_pw_band, opt_state = carry

      carry, _ = jax.lax.scan(
        update_scan, (params_pw_band, opt_state, kpts),
        length=config.k_path_fine_tuning_epoch, unroll = 1
      )
      params_pw_band, opt_state, _ = carry

      return (params_pw_band, opt_state), params_pw_band

    carry, params_fine_tuning = jax.lax.scan(
      finetuning, (params_first_kpt, opt_state),
      xs = (kpts[1:])
    )

    logging.info("===> Eigen decomposition...")

    def eig_fn(param, kpts, g_vector_grid):
      coeff_i = pw.coeff(param, freq_mask)
      hamil_matrix = hamiltonian.hamiltonian_matrix(
        coeff_i,
        crystal.positions,
        crystal.charges,
        ground_state_density_grid,
        g_vector_grid,
        kpts,
        crystal.vol,
        xc,
        kohn_sham=True
      )
      return jax.vmap(jnp.linalg.eigvalsh)(hamil_matrix)

    eigen_values = [eig_fn(params_first_kpt, kpts[0:1], g_vec)]

    def eig_scan(carry, x):
      k, prm = x
      k = jnp.expand_dims(k, axis=0)
      eig = eig_fn(prm, k, g_vec)
      return None, eig

    carry, ys = jax.lax.scan(eig_scan, None, xs=(kpts[1:], params_fine_tuning))
    eigen_values += list(ys)

    return eigen_values

  num_devices = jax.device_count()
  k_path = jnp.reshape(k_path, (num_devices, -1, 3))
  params_pw_band = jax.tree.map(
    lambda x: jnp.stack([x] * num_devices, axis=0), params_pw_band
  )
  opt_state = jax.tree.map(
    lambda x: jnp.stack([x] * num_devices, axis=0), opt_state
  )
  time_start = time.time()
  eigen_values = optimize_eigenvalues(k_path, params_pw_band, opt_state)
  time_end = time.time()
  logging.info(
    f"Band structure calculation: {time_end - time_start:.2f} seconds"
  )

  eigen_values = jnp.stack(eigen_values)
  num_spin = eigen_values.shape[2]
  eigen_values = jnp.reshape(
    eigen_values, (config.num_kpoints, num_spin, num_bands), order="F"
  )
  eigen_values = jnp.transpose(eigen_values, (1, 0, 2))

  logging.info("===> Eigen decomposition done.")
  save_file = ''.join(crystal.symbols) + "_band_structure.npy"
  logging.info(f"Results saved in {save_file}")
  jnp.save(save_file, eigen_values)
