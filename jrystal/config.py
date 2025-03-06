from typing import Dict, Union, List, Tuple, Optional
from ml_collections import ConfigDict
import yaml


class JrystalConfigDict(ConfigDict):
  crystal: Optional[str]
  crystal_file_path_path: Optional[str]
  use_pseudopotential: bool
  pseudopotential_file_dir: Optional[str]
  g_grid_mask_method: str
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
  optimizer_args: Dict[str, float]
  scheduler: Optional[str]
  convergence_condition: float
  band_structure_empty_bands: Optional[int]
  k_path_special_points: Optional[str]
  num_kpoints: Optional[int]
  k_path_file: Optional[str]
  band_structure_epoch: int
  k_path_fine_tuning: bool
  k_path_fine_tuning_epoch: int
  seed: int
  xla_preallocate: bool
  jax_enable_x64: bool
  verbose: bool
  eps: float


default_config = {
  "crystal": "diamond",
  "crystal_file_path_path": None,
  "xc": "lda_x",
  "use_pseudopotential": False,
  "pseudopotential_file_dir": None,
  "g_grid_mask_method": "spherical",
  "cutoff_energy": 100,
  "grid_sizes": 64,
  "k_grid_sizes": 3,
  "occupation": "fermi-dirac",
  "smearing": 0.0001,
  "empty_bands": 8,
  "spin_restricted": True,
  "ewald_args": {
    'ewald_eta': 0.1, 'ewald_cutoff': 2e4
  },
  "epoch": 5000,
  "optimizer": "adam",
  "optimizer_args": {
    "learning_rate": 1e-2
  },
  "scheduler": "adam",
  "convergence_condition": 1e-8,
  "band_structure_empty_bands": 8,
  "k_path_special_points": None,
  "num_kpoints": 60,
  "k_path_file": None,
  "band_structure_epoch": 5000,
  "k_path_fine_tuning": True,
  "k_path_fine_tuning_epoch": 300,
  "seed": 123,
  "xla_preallocate": True,
  "jax_enable_x64": True,
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
