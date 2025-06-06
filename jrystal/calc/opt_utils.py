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
"""Utility functions for optimization. """
import argparse
from typing import Callable

import jax
import jrystal as jr
import numpy as np
import optax
from absl import logging
from optax._src import alias

from .._src.crystal import Crystal
from .._src.ewald import ewald_coulomb_repulsion
from .._src.grid import (
  cubic_mask,
  estimate_max_cutoff_energy,
  g_vectors,
  k_vectors,
  proper_grid_size,
  r_vectors,
  spherical_mask,
  translation_vectors
)
from .._src.utils import check_spin_number
from ..config import JrystalConfigDict


def set_env_params(config: JrystalConfigDict):
  if config.debug:
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_disable_jit", True)

  if config.verbose:
    logging.set_verbosity(logging.INFO)
    logging.info('Versbose mode is on.')
    if config.jax_enable_x64:
      logging.info("Precision: Double (64 bit).")
  else:
    logging.set_verbosity(logging.WARNING)
    logging.warning('Versbose mode is off.')

  jax.config.update("jax_enable_x64", config.jax_enable_x64)


def parse_args(config: JrystalConfigDict) -> JrystalConfigDict:
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser(description='Jrystal energy optimization.')
  for key, value in config.items():
    parser.add_argument(f"--{key}", type=type(value), default=value)
  args = parser.parse_args()

  for key, value in vars(args).items():
    config[key] = value
  return config


def create_freq_mask(config: JrystalConfigDict):
  crystal = create_crystal(config)
  grid_sizes = proper_grid_size(config.grid_sizes)
  logging.info(f"freq_mask_method: {config.freq_mask_method}")

  if config.freq_mask_method == "cubic":
    mask = np.array(cubic_mask(grid_sizes))
    max_cutoff = estimate_max_cutoff_energy(crystal.cell_vectors, mask)
    logging.info(f"maxmum cutoff: {max_cutoff:.2f} Ha")
    logging.info(f"number of g points: {np.sum(mask)}")

  elif config.freq_mask_method == "spherical":
    mask = spherical_mask(
      crystal.cell_vectors, grid_sizes, config.cutoff_energy
    )
    logging.info(f"mask percentage: {np.mean(mask)*100:.2f}%")
    logging.info(f"maxmum cutoff: {config.cutoff_energy}")
    logging.info(f"number of g points: {np.sum(mask)}")

  else:
    raise ValueError("freq_mask_method must be either cubic or spherical.")

  return mask


def create_crystal(config: JrystalConfigDict) -> Crystal:
  _pkg_path = jr.get_pkg_path()
  path = _pkg_path + '/geometry/' + config.crystal + '.xyz'
  crystal = Crystal.create_from_file(file_path=path)
  check_spin_number(crystal.num_electron, crystal.spin)
  return crystal


def create_pseudopotential(config: JrystalConfigDict):
  assert config.use_pseudopotential
  crystal = create_crystal(config)
  _pkg_path = jr.get_pkg_path()
  path = _pkg_path + '/pseudopotential/normconserving/'
  logging.info(f"Pseudopotential path: {path}")
  pp = jr.pseudopotential.NormConservingPseudopotential.create(crystal, path)
  return pp


def create_grids(config: JrystalConfigDict):
  crystal = create_crystal(config)
  grid_sizes = proper_grid_size(config.grid_sizes)
  k_grid_sizes = proper_grid_size(config.k_grid_sizes)
  g_vector_grid = g_vectors(crystal.cell_vectors, grid_sizes)
  r_vector_grid = r_vectors(crystal.cell_vectors, grid_sizes)
  kpts = k_vectors(crystal.cell_vectors, k_grid_sizes)
  return g_vector_grid, r_vector_grid, kpts


def create_optimizer(config: JrystalConfigDict) -> optax.GradientTransformation:
  logging.info(f"optimization method: {config.optimizer}")
  config_dict = dict(config.optimizer_args)
  opt = getattr(alias, config.optimizer, None)
  lr = config_dict.pop("learning_rate")
  if config.scheduler:
    raise NotImplementedError("Scheduler is not implemented yet.")

  # TODO: Add scheduler

  if opt:
    optimizer = opt(learning_rate=lr, **config_dict)
  else:
    raise NotImplementedError(f'"{config.optimizer}" is not found in optax.')
  return optimizer


def create_occupation(config: JrystalConfigDict) -> Callable:
  occupation_method = config.occupation
  if occupation_method == "fermi-dirac":
    return jr.occupation.idempotent
  elif occupation_method == "uniform":
    return jr.occupation.uniform
  elif occupation_method == "gamma":
    return jr.occupation.gamma
  else:
    raise NotImplementedError(
      f"Occupation method {occupation_method} is not implemented."
    )


def get_ewald_coulomb_repulsion(config: JrystalConfigDict):
  crystal = create_crystal(config)
  ewald_grid = translation_vectors(
    crystal.cell_vectors, config.ewald_args['ewald_cutoff']
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
