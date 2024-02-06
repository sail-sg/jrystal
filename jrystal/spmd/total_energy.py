"""Optimization with SPMD parallelism.

  This module is for optimizing the parameters for total eneryg minimization.
"""
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from math import ceil
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from tqdm import tqdm
from absl import logging
from ml_collections import ConfigDict
import optax

from jrystal.config import get_config
from jrystal.training_utils import create_crystal, create_optimizer
from jrystal.spmd.wave import PlaneWave
from jrystal._src import energy
from jrystal._src.grid import (
  k_vectors,
  g_vectors,
  get_grid_sizes,
  half_frequency_shape,
  translation_vectors
)
from jrystal._src.occupation import occupation_gamma


def parse_args(config: ConfigDict) -> ConfigDict:
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser(
    description='Jrystal total energy minimization (SPMD).'
  )
  for key, value in config.items():
    parser.add_argument(f"--{key}", type=type(value), default=value)
  args = parser.parse_args()

  for key, value in vars(args).items():
    config[key] = value

  return config


def set_env_params(config):
  if config.verbose:
    logging.set_verbosity(logging.INFO)
  jax.config.update("jax_enable_x64", config.jax_enable_x64)


config = parse_args(get_config())


def initialize_mesh_and_sharding():
  """Initialize mesh and sharding for parallel computations."""
  mesh = Mesh(np.array(jax.devices()).reshape(1, 1, -1), ('s', 'k', 'i'))
  shd = NamedSharding(mesh, P('s', 'k', 'i'))
  return mesh, shd


def create_module(config):
  num_gpus = jax.device_count()
  grid_sizes = get_grid_sizes(config.grid_sizes)
  k_grid_sizes = get_grid_sizes(config.k_grid_sizes)
  half_shape = np.array(half_frequency_shape(grid_sizes), dtype=int)
  logging.info(f"grid_sizes: {grid_sizes}")
  logging.info(f"half_shape: {half_shape}")

  crystal = create_crystal(config)
  empty_bands = config.empty_bands

  kpts = k_vectors(crystal.cell_vectors, get_grid_sizes(config.k_grid_sizes))
  kpts = jnp.array(kpts)

  num_bands = ceil(
    crystal.num_electrons * (1. + empty_bands) // 2 // num_gpus
  ) * num_gpus

  logging.info(f"num_gpts: {np.prod(np.array(half_shape), dtype=int).item()}")
  logging.info(f"num_bands: {num_bands}")
  logging.info(f"Polarize: {config.polarize}")

  return PlaneWave(
    num_bands, grid_sizes, k_grid_sizes, polarize=config.polarize
  )


def get_occupation(config):
  num_gpus = jax.device_count()
  k_grid_sizes = get_grid_sizes(config.k_grid_sizes)
  num_k = np.prod(k_grid_sizes).item()
  crystal = create_crystal(config)
  num_bands = ceil(
    crystal.num_electrons * (1. + config.empty_bands) // 2 // num_gpus
  ) * num_gpus
  return occupation_gamma(
    num_k, crystal.num_electrons, crystal.spin, num_bands, config.polarize
  )


def total_energy(config: ConfigDict):
  mesh, shd = initialize_mesh_and_sharding()
  crystal = create_crystal(config)
  # crystal = extend_carbon_crystal([4, 4, 4])
  grid_sizes = get_grid_sizes(config.grid_sizes)
  g_vector_grid = g_vectors(crystal.cell_vectors, grid_sizes)

  pw = create_module(config)
  key = jax.random.PRNGKey(123)
  params = pw.init(key)
  optimizer = create_optimizer(config)
  opt_state = optimizer.init(params)
  occupation = get_occupation(config)

  # calculate ewald
  ewald_grid = translation_vectors(crystal.cell_vectors, cutoff=2e4)
  ew = energy.ewald_coulomb_repulsion(
    crystal.positions,
    crystal.charges,
    g_vector_grid,
    crystal.vol,
    ewald_eta=0.1,
    ewald_grid=ewald_grid
  )

  @jax.jit
  def update(params, opt_state, g_vector_grid):

    def total_energy(params):
      density2 = pw.apply(
        params, crystal.cell_vectors, occupation, shd, method='density'
      )
      rec_density = jnp.fft.fftn(density2, axes=range(-3, 0))
      eh = energy.hartree(rec_density, g_vector_grid, crystal.vol)
      ee = energy.external(
        rec_density,
        crystal.positions,
        crystal.charges,
        g_vector_grid,
        crystal.vol
      )
      ex = energy.xc_lda(density2, crystal.vol)
      ek = pw.apply(
        params, crystal.cell_vectors, occupation, shd, method='kinetic'
      )
      return eh + ee + ex + ek, (eh, ee, ex, ek)

    (e_tot, es), grads = jax.value_and_grad(total_energy, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, e_tot, es

  iters = tqdm(range(config.epoch))

  with mesh:
    for i in iters:
      params, opt_state, e_tot, es = update(params, opt_state, g_vector_grid)
      iters.set_description(f"Total energy: {e_tot+ew:.3f}")

  eh, ee, ex, ek = es
  logging.info(f" - Ground State: {e_tot+ew}")
  logging.info(f" - Kinetic: {ek}")
  logging.info(f" - External: {ee}")
  logging.info(f" - Exchange-Correlation: {ex}")
  logging.info(f" - Hartree: {eh}")
  logging.info(f" - Nucleus Repulsion: {ew}")

  return params


if __name__ == "__main__":
  total_energy(config)
