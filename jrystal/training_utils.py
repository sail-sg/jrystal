"Utility functions for training."

import jrystal
from jrystal._src.energy import ewald_coulomb_repulsion
from jrystal._src.grid import get_grid_sizes, translation_vectors
from jrystal._src.grid import g_vectors, r_vectors, k_vectors
from jrystal.crystal import Crystal
from ml_collections import ConfigDict

from optax._src import alias


def create_crystal(config: ConfigDict):
  _pkg_path = jrystal.get_pkg_path()
  path = _pkg_path + '/geometries/' + config.crystal + '.xyz'
  crystal = Crystal(xyz_file=path)
  return crystal


def create_grids(config: ConfigDict):
  crystal = create_crystal(config)
  grid_sizes = get_grid_sizes(config.grid_sizes)
  k_grid_sizes = get_grid_sizes(config.k_grid_sizes)
  g_vector_grid = g_vectors(crystal.cell_vectors, grid_sizes)
  r_vector_grid = r_vectors(crystal.cell_vectors, grid_sizes)
  k_vector_grid = k_vectors(crystal.cell_vectors, k_grid_sizes)
  return g_vector_grid, r_vector_grid, k_vector_grid


def create_optimizer(config: ConfigDict):
  opt = getattr(alias, config.optimizer, None)
  if opt:
    optimizer = opt(**config.optimizer_args)
  else:
    raise NotImplementedError(f'"{config.optimizer}" is not included in optax.')

  return optimizer


def get_ewald_coulomb_repulsion(config: ConfigDict):
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
