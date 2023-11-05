"""Optimization 
  
  This module is for optimizing the parameters.
"""
import jax
import jax.numpy as jnp
import numpy as np

import optax
from optax._src import alias
import flax
import ml_collections
from flax.training import train_state

import jrystal
from jrystal.config import get_config
from jrystal.wave import PlaneWaveDensity
from jrystal._src.energies import energy_total
from jrystal._src.pw import get_cg
from jrystal._src.paramdict import PWDArgs, EwaldArgs


def create_crystal(config):
  _pkg_path = jrystal.get_pkg_path()
  path = _pkg_path + '/geometries/' + config.crystal + '.xyz'
  crystal = jrystal.Crystal(xyz_file=path)
  return crystal


def create_train_state(rng, config):
  """Creates initial `TrainState`."""
  crystal = create_crystal(config)
  pwd_args, grids = PWDArgs.get_PWD_args(
    crystal, config.ecut, config.grid_size, config.k_grid_size
  )
  r_grid, g_grid = grids

  opt = getattr(alias, config.optimizer, None)
  if opt:
    optimizer = opt(**config.optimizer_args)
  else:
    raise NotImplementedError(f'"{config.optimizer}" is not included in optax')

  ew_args = EwaldArgs(**config.ewald_args)

  density = PlaneWaveDensity(**pwd_args)
  params = density.init(rng, r_grid)['params']
  state = train_state.TrainState.create(
    apply_fn=density.apply,
    params=params,
    tx=optimizer,
    grids=grids,
    ew_args=ew_args
  )
  return state, density


def update(
  state: train_state.TrainState,
  density: PlaneWaveDensity,
):
  # ew_args

  def e_tot(params):
    nr, ng = state.apply_fn({'params': params}, r_grid)
    cg = density.get_cg(params)
    occ = density.get_occ(params)
    g_grids = density.shape

    return energy_total(
      nr,
      ng,
      cg,
      occ,
      crystal,
      g_grids,
      density.k_grid,
      ew_args,
    )

  grads = jax.grad(e_tot)
  return state.apply_gradients(grads=grads)


def train(config: ml_collections.ConfigDict) -> train_state.TrainState:
  """_summary_

  Args:
      config (ml_collections.ConfigDict): Hyperparameter configuration for 
      training and evaluation.

  Returns:
      train_state.TrainState: the train state
  """
  pass


if __name__ == '__main__':
  config = get_config()

  # Define the planewavefunction.
  crystal = create_crystal(config)
  pwd_args, grids = PWDArgs.get_PWD_args(
    crystal, config.ecut, config.grid_size, config.k_grid_size
  )
  r_grid, g_grid = grids
  ewald_args = EwaldArgs(**config.ewald_args)
