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

from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from ml_collections import ConfigDict


class JrystalConfigDict(ConfigDict):
  crystal: Optional[str]
  crystal_file_path_path: Optional[str]
  spin: int
  save_dir: Optional[str]
  xc: str
  use_pseudopotential: bool
  pseudopotential_file_dir: Optional[str]
  freq_mask_method: str
  cutoff_energy: float
  grid_sizes: Union[int, List[int], Tuple[int]]
  k_grid_sizes: Union[int, List[int], Tuple[int]]
  occupation: str
  smearing: float
  empty_bands: int
  spin_restricted: bool
  ewald_args: Dict[str, float]
  epoch: int
  optimizer: str
  optimizer_args: Dict[str, Any]
  scheduler: Optional[str]
  convergence_window_size: int
  convergence_condition: float
  band_structure_empty_bands: Optional[int]
  k_path_special_points: Optional[str]
  num_kpoints: Optional[int]
  k_path_file: Optional[str]
  band_structure_epoch: int
  k_path_fine_tuning: bool
  k_path_fine_tuning_epoch: int
  seed: int
  parallel_over_k_mesh: bool
  parallel_over_k_path: bool
  xla_preallocate: bool
  jax_enable_x64: bool
  jax_debug_nans: bool
  verbose: bool
  eps: float


default_config = {
  "crystal": "diamond",
  "crystal_file_path_path": None,
  "save_dir": None,
  "spin": 0,
  "xc": "lda_x",
  "use_pseudopotential": False,
  "pseudopotential_file_dir": None,
  "freq_mask_method": "spherical",
  "cutoff_energy": 100,
  "grid_sizes": 64,
  "k_grid_sizes": 3,
  "occupation": "uniform",
  "smearing": 0.001,
  "empty_bands": 8,
  "spin_restricted": True,
  "ewald_args": {
    'ewald_eta': 0.1, 'ewald_cutoff': 2e4
  },
  "epoch": 5000,
  "optimizer": "adam",
  "optimizer_args": {
    "learning_rate": 0.01,
    "b1": 0.9,
    "b2": 0.99
  },
  "scheduler": None,
  "convergence_window_size": 20,
  "convergence_condition": 1e-4,
  "band_structure_empty_bands": 8,
  "k_path_special_points": None,
  "num_kpoints": 60,
  "k_path_file": None,
  "band_structure_epoch": 5000,
  "k_path_fine_tuning": True,
  "k_path_fine_tuning_epoch": 300,
  "seed": 123,
  "parallel_over_k_mesh": False,
  "parallel_over_k_path": True,
  "xla_preallocate": True,
  "jax_enable_x64": True,
  "jax_debug_nans": False,
  "verbose": True,
  "eps": 1e-8,
}


def get_config(config_file: Optional[str] = None) -> JrystalConfigDict:
  if config_file is not None:
    with open(config_file, 'r') as file:
      config = yaml.safe_load(file)
    config = JrystalConfigDict(config)

  else:
    config = JrystalConfigDict(default_config)

  if config.band_structure_empty_bands is None:
    config.band_structure_empty_bands = config.empty_bands

  return config
