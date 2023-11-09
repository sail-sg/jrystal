"""Optimization 
  
  This module is for optimizing the parameters.
"""
import jax
import jax.numpy as jnp
from optax._src import alias
from flax.training import train_state
import ml_collections

import jrystal
from jrystal.config import get_config
from jrystal.wave import PlaneWaveDensity
from jrystal._src.energy import total
from jrystal._src.paramdict import PWDArgs, EwaldArgs
from jrystal._src.crystal import Crystal
import time
from tqdm import tqdm
from absl import logging

logging.set_verbosity(logging.INFO)
jax.config.update("jax_enable_x64", False)


def create_crystal(config):
  _pkg_path = jrystal.get_pkg_path()
  path = _pkg_path + '/geometries/' + config.crystal + '.xyz'
  crystal = Crystal(xyz_file=path)
  return crystal


def create_train_state(rng, config):
  """Creates initial `TrainState`."""
  crystal = create_crystal(config)
  pwd_args, grids = PWDArgs.get_PWD_args(
    crystal, config.ecut, config.grid_size, config.k_grid_size
  )
  r_grid, _ = grids

  opt = getattr(alias, config.optimizer, None)
  if opt:
    optimizer = opt(**config.optimizer_args)
  else:
    raise NotImplementedError(f'"{config.optimizer}" is not included in optax')

  grid_point_num = jnp.prod(jnp.array(pwd_args.mask.shape))
  mask_ratio = jnp.sum(pwd_args.mask) / grid_point_num

  logging.info(f'{jnp.sum(pwd_args.mask)} G points selected.')
  logging.info(f'{mask_ratio*100:.2f}% frequency masked.')
  logging.info(f'the mesh of G grid: {pwd_args.mask.shape}')

  density = PlaneWaveDensity(**pwd_args)
  params = density.init(rng, r_grid)['params']
  state = train_state.TrainState.create(
    apply_fn=density.apply, params=params, tx=optimizer
  )

  ew_args = EwaldArgs(**config.ewald_args)
  variable_dict = {'grids': grids, 'ew_args': ew_args, 'crystal': crystal}
  return state, density, variable_dict


def train(config: ml_collections.ConfigDict):
  """_summary_

  Args:
      config (ml_collections.ConfigDict): Hyperparameter configuration for 
      training and evaluation.

  Returns:
      train_state.TrainState: the train state
  """
  rng = jax.random.key(config.seed)
  rng, init_rng = jax.random.split(rng)
  state, density, variable_dict = create_train_state(init_rng, config)
  ew_args = variable_dict['ew_args']
  grids = variable_dict['grids']
  crystal = variable_dict['crystal']

  start_time = time.time()
  iters = tqdm(range(config.epoch))

  @jax.jit
  def update(state: train_state.TrainState, grids) -> train_state.TrainState:
    r_grid, g_grid = grids

    def loss(params):
      nr, ng = state.apply_fn({'params': params}, r_grid)
      cg = density.get_cg(params)
      occ = density.get_occ(params)
      e_tot = total(
        nr,
        ng,
        cg,
        occ,
        crystal,
        g_grid,
        density.k_grid,
        ew_args,
      )
      return e_tot

    (e_total, e_splits), grads = jax.value_and_grad(loss, has_aux=True)(
      state.params
    )
    return state.apply_gradients(grads=grads), (e_total, *e_splits)

  for i in iters:
    state, es = update(state, grids)
    e_tol, e_kin, e_har, e_ext, e_xc, e_ew = es
    iters.set_description(f"Total energy: {e_tol:.3f}")

  converged = True
  # TODO(tianbo): include a convergence check module.

  logging.info(
    f"Converged: {converged}. \n"
    f"Total epochs run: {i+1}. \n"
    f"Training Time: {(time.time() - start_time):.3f}s. \n"
  )
  logging.info("Energy:")
  logging.info(f" Ground State: {e_tol}")
  logging.info(f" Kinetic: {e_kin}")
  logging.info(f" External: {e_ext}")
  logging.info(f" Exchange-Correlation: {e_xc}")
  logging.info(f" Hartree: {e_har}")
  logging.info(f" Nucleus Repulsion: {e_ew}")


if __name__ == '__main__':
  config = get_config()

  train(config)
