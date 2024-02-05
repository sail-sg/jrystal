"""Optimization with SPMD parallelism.

  This module is for optimizing the parameters for total eneryg minimization.
"""
# import os
import jax
import jax.numpy as jnp
import numpy as np
from math import ceil
# import flax
# from flax.training import train_state
# import ml_collections

from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh
from jrystal.config import get_config
from jrystal._src.spmd.wave import PlaneWave
from jrystal._src.occupation import occupation_gamma
from jrystal.training_utils import create_crystal
# from jrystal.training_utils import create_optimizer
from jrystal._src.grid import k_vectors, r_vectors, g_vectors
from jrystal._src.grid import half_frequency_shape
from jrystal._src import energy
from jrystal._src.grid import translation_vectors
# from jrystal.utils import view_hlo

# import time
from tqdm import tqdm
from absl import logging
# from ml_collections import ConfigDict
import argparse

import optax

logging.set_verbosity(logging.INFO)
jax.config.update("jax_enable_x64", False)

ns = 2
nk = 1
num_gpus = jax.device_count()
mesh = Mesh(np.array(jax.devices()).reshape(1, 1, -1), ('s', 'k', 'i'))
shd = NamedSharding(mesh, P('s', 'k', 'i'))

parser = argparse.ArgumentParser(prog='Jrystal_test')
parser.add_argument("--empty_bands", "-e", type=float, default=0.1)
parser.add_argument("--grid", "-g", type=int, default=96)
parser.add_argument("--epoch", type=int, default=10000)
args = parser.parse_args()


def extend_carbon_crystal(shape):
  config = get_config()
  config.crystal = "diamond"
  crystal = create_crystal(config)
  crystal.symbols = "C" + str(np.prod(shape).item())

  shift = np.array(np.meshgrid(*(range(i) for i in shape))).T.reshape(-1, 3)
  crystal.scaled_positions = np.vstack(
    [i + shift for i in crystal.scaled_positions]
  )
  crystal.positions = np.matmul(crystal.scaled_positions, crystal.A)
  crystal.charges = np.ones(crystal.positions.shape[0]) * 6

  crystal.num_atoms = crystal.positions.shape[0]
  crystal.num_electrons = np.sum(crystal.charges, dtype=int).item()

  crystal.cell_vectors = np.matmul(crystal.cell_vectors, np.diag(shape))
  crystal.A = crystal.cell_vectors
  crystal.reciprocal_vectors = np.linalg.inv(crystal.A).T * 2 * np.pi
  crystal.B = crystal.reciprocal_vectors
  crystal.vol = np.linalg.det(crystal.A)

  return crystal


grid_sizes = np.ones(3, dtype=int) * args.grid
half_shape = np.array(half_frequency_shape(grid_sizes), dtype=int)
print("grid_sizes: ", grid_sizes)
print("half_shape: ", half_shape)

crystal = extend_carbon_crystal([4, 4, 4])
empty_bands = args.empty_bands
kpts = k_vectors(crystal.cell_vectors, [nk, 1, 1])
kpts = jnp.array(kpts)
num_bands = ceil(
  crystal.num_electrons * (1. + empty_bands) // 2 // num_gpus
) * num_gpus

print("num_gpts: ", np.prod(np.array(half_shape), dtype=int).item())
print("num_bands: ", num_bands)

r_vector_grid = r_vectors(crystal.cell_vectors, grid_sizes)

occupation = occupation_gamma(nk, num_bands, ns)

pw = PlaneWave(num_bands, grid_sizes, [1, 1, 1])
key = jax.random.PRNGKey(123)
params = pw.init(key, crystal.cell_vectors)
g_vector_grid = g_vectors(crystal.cell_vectors, grid_sizes)

optimizer = optax.yogi(1e-2)
opt_state = optimizer.init(params)

ewald_grid = translation_vectors(crystal.cell_vectors, cutoff=2e4)
# ew = energy.ewald_coulomb_repulsion(
#   crystal.positions,
#   crystal.charges,
#   g_vector_grid,
#   crystal.vol,
#   ewald_eta=0.1,
#   ewald_grid=ewald_grid
# )
ew = 0

jnp.linalg.qr


@jax.jit
def update(params, opt_state, g_vector_grid):

  def total_energy(params):
    density2 = pw.apply(params, crystal.cell_vectors, shd, method='density')
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
    ek = pw.apply(params, crystal.cell_vectors, kpts, shd, method='kinetic')
    return eh + ee + ex + ek, (eh, ee, ex, ek)

  (e_tot, es), grads = jax.value_and_grad(total_energy, has_aux=True)(params)
  updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)

  return params, opt_state, e_tot, es


iters = tqdm(range(args.epoch))

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
