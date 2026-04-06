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
import os
from typing import Optional

import jax
import numpy as np
import optax
from optax._src import alias

import jrystal as jr

from ..__init__ import get_pkg_path
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
  translation_vectors,
)
from .._src.utils import check_spin_number
from ..config import JrystalConfigDict
from ..terminal_ui import stage_line, stage_warning
from .types import KSampling


def set_env_params(config: JrystalConfigDict):
  os.environ["OPENBLAS_NUM_THREADS"] = "4"
  os.environ["MKL_NUM_THREADS"] = "4"
  os.environ["OMP_NUM_THREADS"] = "4"
  os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(
    config.execution.xla_preallocate
  ).lower()
  jax.config.update("jax_debug_nans", config.execution.jax_debug_nans)

  if config.execution.verbose:
    stage_line("Init", "Verbose mode is on.")
    if config.execution.jax_enable_x64:
      stage_line("Init", "Precision: Double (64 bit).")
    else:
      stage_line("Init", "Precision: Single (32 bit).")
  else:
    stage_warning("Init", "Verbose mode is off.")

  jax.config.update("jax_enable_x64", config.execution.jax_enable_x64)


def create_freq_mask(
  config: JrystalConfigDict,
  crystal: Optional[Crystal] = None,
):
  crystal = create_crystal(config) if crystal is None else crystal
  grid_sizes = proper_grid_size(config.basis.grid_sizes)
  stage_line("Init", f"freq_mask_method: {config.basis.freq_mask_method}")

  if config.basis.freq_mask_method == "cubic":
    mask = np.array(cubic_mask(grid_sizes))
    max_cutoff = estimate_max_cutoff_energy(crystal.cell_vectors, mask)
    stage_line(
      "Init",
      f"Maximum cutoff: {max_cutoff:.0f} Ha ({max_cutoff*27.2114:.0f} eV)",
    )
    stage_line("Init", f"Number of g points: {np.sum(mask)}")

  elif config.basis.freq_mask_method == "spherical":
    mask = spherical_mask(
      crystal.cell_vectors, grid_sizes, config.basis.cutoff_energy
    )
    stage_line("Init", f"Mask percentage: {np.mean(mask)*100:.2f}%")
    stage_line(
      "Init",
      f"Maximum cutoff: {config.basis.cutoff_energy:.0f} Ha "
      f"({config.basis.cutoff_energy*27.2114:.0f} eV)",
    )
    stage_line("Init", f"Number of g points: {np.sum(mask)}")

  else:
    raise ValueError("freq_mask_method must be either cubic or spherical.")

  return mask


def create_crystal(config: JrystalConfigDict) -> Crystal:
  _pkg_path = jr.get_pkg_path()
  if config.system.crystal is not None:
    path = _pkg_path + '/geometry/' + config.system.crystal + '.xyz'
  else:
    path = config.system.crystal_file_path
  crystal = Crystal.create_from_file(file_path=path, spin=config.system.spin)
  check_spin_number(crystal.num_electron, crystal.spin)
  return crystal


def create_pseudopotential(
  config: JrystalConfigDict,
  crystal: Optional[Crystal] = None,
):
  assert config.method.use_pseudopotential
  crystal = create_crystal(config) if crystal is None else crystal
  _pkg_path = jr.get_pkg_path()
  if config.method.pseudopotential_type in ["normcons", "normconserving", "nc"]:
    if config.method.pseudopotential_file_dir is None:
      path = _pkg_path + '/pseudopotential/normconserving/'
    else:
      path = config.method.pseudopotential_file_dir
    pp = jr.pseudopotential.NormConservingPseudopotential.create(crystal, path)
  elif config.method.pseudopotential_type in ["ultrasoft", "us"]:
    if config.method.pseudopotential_file_dir is None:
      path = _pkg_path + '/pseudopotential/ultrasoft/'
    else:
      path = config.method.pseudopotential_file_dir
    pp = jr.pseudopotential.UltrasoftPseudopotential.create(crystal, path)
  else:
    raise ValueError(
      f"Pseudopotential type {config.method.pseudopotential_type} is not "
      f"supported."
    )

  stage_line("Init", f"Pseudopotential path: {path}")

  return pp


def create_grids(
  config: JrystalConfigDict,
  crystal: Optional[Crystal] = None,
  ksampling: Optional[KSampling] = None,
):
  crystal = create_crystal(config) if crystal is None else crystal
  grid_sizes = proper_grid_size(config.basis.grid_sizes)
  g_vector_grid = g_vectors(crystal.cell_vectors, grid_sizes)
  r_vector_grid = r_vectors(crystal.cell_vectors, grid_sizes)
  if ksampling is None:
    k_grid_sizes = proper_grid_size(config.ksampling.k_grid_sizes)
    kpts, k_weights = k_vectors(
      crystal.cell_vectors,
      k_grid_sizes,
      symmetry_reduction=config.ksampling.symmetry_reduction,
      scaled_positions=crystal.scaled_positions,
      charges=crystal.charges,
    )
    ksampling = KSampling(mode="mesh", kpts=kpts, weights=k_weights)
  return g_vector_grid, r_vector_grid, ksampling


def create_optimizer(config: JrystalConfigDict) -> optax.GradientTransformation:
  optimizer_config = dict(config.solver.direct_opt.optimizer)
  optimizer_name = optimizer_config.pop("name")
  stage_line("DirectOpt", f"optimizer={optimizer_name}")
  opt = getattr(alias, optimizer_name, None)
  config_dict = dict(optimizer_config)
  lr = config_dict.pop("learning_rate")
  stage_line("DirectOpt", f"learning_rate={lr}")
  if config.solver.direct_opt.scheduler:
    raise NotImplementedError("Scheduler is not implemented yet.")

  # TODO: Add scheduler

  if opt:
    optimizer = opt(learning_rate=lr, **config_dict)
  else:
    raise NotImplementedError(f'"{optimizer_name}" is not found in optax.')
  return optimizer


def get_ewald_coulomb_repulsion(
  config: JrystalConfigDict,
  crystal: Optional[Crystal] = None,
  g_vector_grid=None,
):
  crystal = create_crystal(config) if crystal is None else crystal
  ewald_grid = translation_vectors(crystal.cell_vectors, config.ewald.cutoff)
  if g_vector_grid is None:
    g_vector_grid, _, _ = create_grids(config, crystal=crystal)
  ew = ewald_coulomb_repulsion(
    crystal.positions,
    crystal.charges,
    g_vector_grid,
    crystal.vol,
    ewald_eta=config.ewald.eta,
    ewald_grid=ewald_grid
  )
  return ew


def save_beta_sbt(output, filename=None):
  if filename is None:
    cache_dir = os.path.join(get_pkg_path(), "_cache")
    filename = f"{cache_dir}/beta_sbt.npz"
  if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
  np.savez(filename, *output)
