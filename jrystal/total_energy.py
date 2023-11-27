"""Optimization 
  
  This module is for optimizing the parameters.
"""
import os
import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import ml_collections

from jrystal.config import get_config
from jrystal.wave import PlaneWaveDensity, PlaneWaveFermiDirac
from jrystal._src.grid import get_grid_sizes
from jrystal._src.grid import k_vectors
from jrystal._src.functional import get_mask_radial
from jrystal.training_utils import create_crystal, get_ewald_coulomb_repulsion
from jrystal.training_utils import create_optimizer

import time
from tqdm import tqdm
from absl import logging
from ml_collections import ConfigDict

logging.set_verbosity(logging.INFO)


def create_module(config: ConfigDict):
  crystal = create_crystal(config)
  cutoff_energy = config.cutoff_energy
  g_grid_sizes = get_grid_sizes(config.grid_sizes)
  xc_functional = config.xc
  ni = crystal.num_electrons
  k_grid_sizes = get_grid_sizes(config.k_grid_sizes)
  spin = int(ni % 2)
  mask = get_mask_radial(crystal.cell_vectors, g_grid_sizes, cutoff_energy)
  k_vector_grid = k_vectors(crystal.cell_vectors, k_grid_sizes)

  if config.occupation in ["fermi dirac", "fermi_dirac", "fermi"]:
    density_module = PlaneWaveFermiDirac(
      crystal.num_electrons,
      mask,
      crystal.cell_vectors,
      k_vector_grid,
      spin,
      xc_functional=xc_functional
    )
  else:
    density_module = PlaneWaveDensity(
      crystal.num_electrons,
      mask,
      crystal.cell_vectors,
      k_vector_grid,
      spin,
      occupation_method=config.occupation,
      xc_functional=xc_functional
    )

  return density_module


def create_train_state(rng, config):
  """Creates initial `TrainState`."""
  density = create_module(config)
  # _, r_vector_grid, _ = create_grids(config)
  crystal = create_crystal(config)
  optimizer = create_optimizer(config)
  grid_point_num = jnp.prod(jnp.array(density.mask.shape))
  mask_ratio = jnp.sum(density.mask) / grid_point_num

  logging.info(f'{jnp.sum(density.mask)} G points selected.')
  logging.info(f'{mask_ratio*100:.2f}% frequency masked.')
  logging.info(f'the mesh of G grid: {density.mask.shape}')

  variables, params = flax.core.pop(density.init(rng, crystal), 'params')
  state = train_state.TrainState.create(
    apply_fn=density.apply, params=params, tx=optimizer
  )
  return state, variables


def train(config: ml_collections.ConfigDict, return_fn=None):

  jax.config.update("jax_enable_x64", config.jax_enable_x64)
  if config.xla_preallocate is False:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

  rng = jax.random.key(config.seed)
  rng, init_rng = jax.random.split(rng)
  state, variables = create_train_state(init_rng, config)
  crystal = create_crystal(config)
  start_time = time.time()
  iters = tqdm(range(config.epoch))
  ew = get_ewald_coulomb_repulsion(config)

  @jax.jit
  def update(
    state: train_state.TrainState, variables
  ) -> train_state.TrainState:

    def loss(params):
      # e_har = state.apply_fn({'params': params}, method='hartree')
      # e_ext = state.apply_fn({'params': params}, crystal, method='external')
      # e_kin = state.apply_fn({'params': params}, method='kinetic')
      # e_xc = state.apply_fn({'params': params}, method='xc')
      energy, new_variables = state.apply_fn(
        {'params': params, **variables}, crystal,
        method='total_energy', mutable=list(variables.keys())
      )
      total_energy, energies = energy
      # e_tot = e_kin + e_har + e_ext + e_xc + ew
      return total_energy, (energies, new_variables)

    (e_total, energies_and_variables), grads = jax.value_and_grad(
      loss, has_aux=True)(state.params)
    energies, variables = energies_and_variables
    return state.apply_gradients(grads=grads), (e_total, energies, variables)

  for i in iters:
    state, es = update(state, variables)
    # e_tot, e_kin, e_har, e_ext, e_xc, e_ew = es
    e_tot, energies, variables = es
    e_ew = ew
    e_tot += ew
    iters.set_description(f"Total energy: {e_tot:.3f}")

  e_kin = energies["kinetic"]
  e_ext = energies["external"]
  e_xc = energies["xc"]
  e_har = energies["hartree"]
  converged = True
  # TODO(tianbo): include a convergence check module.

  logging.info(
    f"Converged: {converged}. \n"
    f"Total epochs run: {i+1}. \n"
    f"Training Time: {(time.time() - start_time):.3f}s. \n"
  )
  logging.info("Energy:")
  logging.info(f" Ground State: {e_tot}")
  logging.info(f" Kinetic: {e_kin}")
  logging.info(f" External: {e_ext}")
  logging.info(f" Exchange-Correlation: {e_xc}")
  logging.info(f" Hartree: {e_har}")
  logging.info(f" Nucleus Repulsion: {e_ew}")

  if return_fn:
    def density(r):
      return state.apply_fn(
          {'params': state.params, **variables}, r, method=return_fn
      )

    return density


if __name__ == '__main__':
  config = get_config()
  train(config)
