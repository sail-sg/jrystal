"""Module for band structure calculation. """
import os
import jax
import flax
from flax.training import train_state
import jrystal

from jrystal.wave import PlaneWaveBandStructure
from jrystal.training_utils import create_crystal, get_ewald_coulomb_repulsion
from jrystal.training_utils import create_optimizer
from jrystal._src.band_structure import get_k_path
from jrystal._src.grid import get_grid_sizes
import time
from tqdm import tqdm
from absl import logging
from ml_collections import ConfigDict

logging.set_verbosity(logging.INFO)


def create_module(config: ConfigDict, density_fn: callable):
  crystal = create_crystal(config)
  g_grid_sizes = get_grid_sizes(config.grid_sizes)
  xc_functional = config.xc

  k_vectors = get_k_path(
    crystal.cell_vectors, path=config.k_path,
    num=config.num_kpoints, fractional=False
  )

  band_structure_module = PlaneWaveBandStructure(
    density_fn,
    config.num_unoccupied_bands + crystal.num_electrons//2,
    crystal.A,
    g_grid_sizes,
    k_vectors,
    xc_functional=xc_functional
  )

  return band_structure_module


def create_train_state(
  rng, config: ConfigDict, density_fn: callable
):
  """Creates initial `TrainState`."""
  band_structure_module = create_module(config, density_fn)
  logging.info(f"{band_structure_module.k_vectors.shape[0]} k points sampled.")
  logging.info(f"{band_structure_module.num_bands} bands will be computed.")

  crystal = create_crystal(config)
  optimizer = create_optimizer(config)
  variables, params = flax.core.pop(
    band_structure_module.init(rng, crystal), 'params'
  )
  state = train_state.TrainState.create(
    apply_fn=band_structure_module.apply, params=params, tx=optimizer
  )
  return state, variables


def train(config: ConfigDict):
  jax.config.update("jax_enable_x64", config.jax_enable_x64)
  if config.xla_preallocate is False:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

  logging.info("Starting total energy minimization...")
  density_fn = jrystal.total_energy.train(config, return_fn="density")
  logging.info("Done total energy minimization. "
               "Now optimize the sum of band energies...")

  rng = jax.random.key(config.seed)
  rng, init_rng = jax.random.split(rng)
  state, variables = create_train_state(init_rng, config, density_fn)
  crystal = create_crystal(config)
  start_time = time.time()
  iters = tqdm(range(config.band_structure_epoch))
  ew = get_ewald_coulomb_repulsion(config)

  @jax.jit
  def update(
    state: train_state.TrainState, variables
  ) -> train_state.TrainState:

    def loss(params):
      energy, new_variables = state.apply_fn(
        {'params': params, **variables}, crystal,
        method='energy_trace', mutable=list(variables.keys())
      )
      total_energy, energies = energy
      # e_tot = e_kin + e_har + e_ext + e_xc + ew
      return total_energy, (energies, new_variables)

    (e_total, energies), grads = jax.value_and_grad(loss, has_aux=True)(
      state.params
    )
    energies, variables = energies
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


if __name__ == '__main__':
  config = jrystal.config.get_config()
  train(config)
