"""Optimization 
  
  This module is for optimizing the parameters.
"""
import os
import jax
import jax.numpy as jnp
from optax._src import alias
from flax.training import train_state
import ml_collections
from ase.dft.kpoints import monkhorst_pack

import jrystal
from jrystal.config import get_config
from jrystal.wave import PlaneWaveDensity
from jrystal._src.energy import ewald_coulomb_repulsion
from jrystal._src.grid import translation_vectors, get_grid_sizes
from jrystal._src.grid import g_vectors, r_vectors
from jrystal._src.functional import get_mask_radius
from jrystal.crystal import Crystal
import time
from tqdm import tqdm
from absl import logging

logging.set_verbosity(logging.INFO)


def create_crystal(config):
  _pkg_path = jrystal.get_pkg_path()
  path = _pkg_path + '/geometries/' + config.crystal + '.xyz'
  crystal = Crystal(xyz_file=path)
  return crystal


def create_density_module(config):
  crystal = create_crystal(config)
  cutoff_energy = config.cutoff_energy
  g_grid_sizes = get_grid_sizes(config.grid_sizes)
  occupation_method = config.occupation
  xc_method = config.xc
  ni = crystal.num_electrons
  k_grid_sizes = get_grid_sizes(config.k_grid_sizes)
  spin = int(ni % 2)
  mask = get_mask_radius(crystal.A, g_grid_sizes, cutoff_energy)
  k_vector_grid = monkhorst_pack(k_grid_sizes)

  density_module = PlaneWaveDensity(
    crystal.num_electrons,
    mask,
    crystal.cell_vectors,
    k_vector_grid,
    spin,
    crystal.vol,
    occupation_method,
    xc_method=xc_method
  )

  return density_module


def create_grids(config):
  crystal = create_crystal(config)
  grid_sizes = get_grid_sizes(config.grid_sizes)
  k_grid_sizes = get_grid_sizes(config.k_grid_sizes)
  g_vector_grid = g_vectors(crystal.cell_vectors, grid_sizes)
  r_vector_grid = r_vectors(crystal.cell_vectors, grid_sizes)
  k_vector_grid = monkhorst_pack(k_grid_sizes)
  return g_vector_grid, r_vector_grid, k_vector_grid


def create_train_state(rng, config):
  """Creates initial `TrainState`."""
  density = create_density_module(config)
  _, r_vector_grid, _ = create_grids(config)

  opt = getattr(alias, config.optimizer, None)
  if opt:
    optimizer = opt(**config.optimizer_args)
  else:
    raise NotImplementedError(f'"{config.optimizer}" is not included in optax')

  grid_point_num = jnp.prod(jnp.array(density.mask.shape))
  mask_ratio = jnp.sum(density.mask) / grid_point_num

  logging.info(f'{jnp.sum(density.mask)} G points selected.')
  logging.info(f'{mask_ratio*100:.2f}% frequency masked.')
  logging.info(f'the mesh of G grid: {density.mask.shape}')

  params = density.init(rng, r_vector_grid)['params']
  state = train_state.TrainState.create(
    apply_fn=density.apply, params=params, tx=optimizer
  )
  return state


def get_ewald_coulomb_repulsion(config):
  crystal = create_crystal(config)

  ewald_grid = translation_vectors(
    crystal.cell_vectors, config.ewald_args['ewald_cut']
  )

  g_vector_grid, _, _ = create_grids(config)

  ew = ewald_coulomb_repulsion(
    crystal.positions,
    crystal.charges,
    g_vector_grid,
    crystal.vol,
    ewald_eta=config.ewald_args['ewald_eta'],
    ewald_grid=ewald_grid
  )

  return ew


def train(config: ml_collections.ConfigDict):

  jax.config.update("jax_enable_x64", config.jax_enable_x64)
  if config.xla_preallocate is False:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

  rng = jax.random.key(config.seed)
  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng, config)
  crystal = create_crystal(config)
  start_time = time.time()
  iters = tqdm(range(config.epoch))
  ew = get_ewald_coulomb_repulsion(config)

  @jax.jit
  def update(state: train_state.TrainState) -> train_state.TrainState:

    def loss(params):
      e_har = state.apply_fn({'params': params}, method='hartree')
      e_ext = state.apply_fn({'params': params}, crystal, method='external')
      e_kin = state.apply_fn({'params': params}, method='kinetic')
      e_xc = state.apply_fn({'params': params}, method='xc')
      e_tot = e_kin + e_har + e_ext + e_xc + ew
      return e_tot, (e_kin, e_har, e_ext, e_xc, ew)

    (e_total, e_splits), grads = jax.value_and_grad(loss, has_aux=True)(
      state.params
    )
    return state.apply_gradients(grads=grads), (e_total, *e_splits)

  for i in iters:
    state, es = update(state)
    e_tot, e_kin, e_har, e_ext, e_xc, e_ew = es
    iters.set_description(f"Total energy: {e_tot:.3f}")

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


if __name__ == '__main__':
  config = get_config()
  train(config)
