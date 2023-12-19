"""Module for band structure calculation. """
import os
import jax
import jax.numpy as jnp
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


def create_module(config: ConfigDict, density_fn: callable):
  crystal = create_crystal(config)
  g_grid_sizes = get_grid_sizes(config.grid_sizes)
  xc_functional = config.xc

  k_vectors = get_k_path(
    crystal.cell_vectors,
    path=config.k_path,
    num=config.num_kpoints,
    fractional=False
  )

  if config.k_path_fine_tuning:
    band_structure_module = PlaneWaveBandStructure(
      density_fn,
      config.num_unoccupied_bands + crystal.num_electrons // 2,
      crystal.A,
      g_grid_sizes,
      k_vectors[:1],
      xc_functional=xc_functional
    )
  else:
    band_structure_module = PlaneWaveBandStructure(
      density_fn,
      config.num_unoccupied_bands + crystal.num_electrons // 2,
      crystal.A,
      g_grid_sizes,
      k_vectors,
      xc_functional=xc_functional
    )

  return band_structure_module


def create_train_state(rng, config: ConfigDict, density_fn: callable):
  """Creates initial `TrainState`."""
  if config.verbose == "true":
    logging.set_verbosity(logging.INFO)

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
  if config.verbose:
    logging.set_verbosity(logging.INFO)
  else:
    logging.set_verbosity(logging.WARNING)

  jax.config.update("jax_enable_x64", config.jax_enable_x64)
  if config.xla_preallocate is False:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

  logging.info("===> Starting total energy minimization...")
  density_fn = jrystal.total_energy.train(config, return_fn="density")
  logging.info(" Done total energy minimization. ")
  logging.info("===> Optimize the sum of band energies...")

  rng = jax.random.key(config.seed)
  rng, init_rng = jax.random.split(rng)
  state, variables = create_train_state(init_rng, config, density_fn)
  crystal = create_crystal(config)

  e_ew = get_ewald_coulomb_repulsion(config)

  k_vectors = get_k_path(
    crystal.cell_vectors,
    path=config.k_path,
    num=config.num_kpoints,
    fractional=False
  )

  @jax.jit
  def update(
    state: train_state.TrainState, variables, k_vector
  ) -> train_state.TrainState:

    def loss(params):
      energy, new_variables = state.apply_fn(
        {'params': params, **variables}, crystal, k_vectors=k_vector,
        method='energy_trace', mutable=list(variables.keys())
      )
      total_energy, energies = energy
      return total_energy, (energies, new_variables)

    (e_total, energies), grads = jax.value_and_grad(loss, has_aux=True)(
      state.params
    )
    energies, variables = energies
    return state.apply_gradients(grads=grads), (e_total, energies, variables)

  if config.k_path_fine_tuning:
    params_kpoint_list = []
    logging.info("===> Optimizing the first k point...")
    if config.verbose:
      iters = tqdm(range(config.band_structure_epoch))
    else:
      iters = tqdm(range(config.band_structure_epoch), disable=True)

    start_time = time.time()
    for i in iters:
      state, es = update(state, variables, k_vectors[:1])
      e_tot, energies, variables = es
      e_tot += e_ew
      iters.set_description(f"Total energy: {e_tot:.3f}")

    e_kin = energies["kinetic"]
    e_ext = energies["external"]
    e_xc = energies["xc"]
    e_har = energies["hartree"]
    converged = True

    logging.info(f" Converged: {converged}.")
    logging.info(f" Total epochs run: {i+1}.")
    logging.info(f" Training Time: {(time.time() - start_time):.3f}s.")
    logging.info(" Energy:")
    logging.info(f" - Ground State: {e_tot}")
    logging.info(f" - Kinetic: {e_kin}")
    logging.info(f" - External: {e_ext}")
    logging.info(f" - Exchange-Correlation: {e_xc}")
    logging.info(f" - Hartree: {e_har}")
    logging.info("The first k point converged.")
    logging.info("===> Starting fine tuning the others...")

    params_kpoint_list.append(state.params)
    num_k = k_vectors.shape[0]

    if config.verbose:
      iters = tqdm(range(1, num_k))
    else:
      iters = tqdm(range(1, num_k), disable=True)

    for i in iters:
      for j in range(config.k_path_fine_tuning_epoch):
        state, es = update(state, variables, k_vectors[i:(i + 1)])
        e_tot, energies, variables = es
        e_tot += e_ew
      iters.set_description(f"Total Energy(the {i+1}th k point): {e_tot:.3f}")
      params_kpoint_list.append(state.params)

  else:
    if config.verbose:
      iters = tqdm(range(config.band_structure_epoch))
    else:
      iters = tqdm(range(config.band_structure_epoch), disable=True)

    start_time = time.time()
    for i in iters:
      state, es = update(state, variables)
      # e_tot, e_kin, e_har, e_ext, e_xc, e_ew = es
      e_tot, energies, variables = es
      e_tot += e_ew
      iters.set_description(f"Total energy: {e_tot:.3f}")

    e_kin = energies["kinetic"]
    e_ext = energies["external"]
    e_xc = energies["xc"]
    e_har = energies["hartree"]

    converged = True
    # TODO(tianbo): include a convergence check module.

    logging.info(f" Converged: {converged}.")
    logging.info(f" Total epochs run: {i+1}.")
    logging.info(f" Training Time: {(time.time() - start_time):.3f}s.")
    logging.info(" Energy:")
    logging.info(f" - Ground State: {e_tot}")
    logging.info(f" - Kinetic: {e_kin}")
    logging.info(f" - External: {e_ext}")
    logging.info(f" - Exchange-Correlation: {e_xc}")
    logging.info(f" - Hartree: {e_har}")

  if config.verbose:
    iters = tqdm(range(len(params_kpoint_list)))
  else:
    iters = tqdm(range(len(params_kpoint_list)), disable=True)

  eigen_values = []

  @jax.jit
  def eig_fn(params, k_vector):
    fork_k, _ = state.apply_fn(
        {'params': params, **variables}, crystal, k_vectors=k_vector,
        method='fork_matrix', mutable=list(variables.keys())
      )
    return jnp.linalg.eigvalsh(fork_k[0])

  for i in iters:
    prm = params_kpoint_list[i]
    k = k_vectors[i:(i + 1), :]
    iters.set_description(f"Diagonolizing the {i+1}th k points")
    eig = eig_fn(prm, k)
    eigen_values.append(eig)

  return jnp.vstack(eigen_values)


if __name__ == '__main__':
  config = jrystal.config.get_config()
  train(config)
