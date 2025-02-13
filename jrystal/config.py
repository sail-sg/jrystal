from typing import Dict, Union, List, Tuple, Optional
from ml_collections import ConfigDict
import yaml


class JrystalConfigDict(ConfigDict):
  crystal: Optional[str]
  crystal_xyz_file_path: Optional[str]
  use_pseudopotential: bool
  pseudopotential_file_dir: Optional[str]
  g_grid_mask_method: str
  cutoff_energy: float
  grid_sizes: Union[int, List[int], Tuple[int]]
  k_grid_sizes: Union[int, List[int], Tuple[int]]
  occupation: str
  smearing: float
  empty_bands: int
  restricted: bool
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


def get_config(config_file: Optional[str] = None) -> JrystalConfigDict:
  if config_file is not None:
    with open(config_file, 'r') as file:
      config = yaml.safe_load(file)
    return JrystalConfigDict(config)

  config = {

    ################################################
    #             Crystal Setting                  #
    ################################################

    # Code will load the "$CRYSTAL.xyz" file in the geometry directory if
    # crystal name is provided, otherwise the code will read the from the path
    # specified by "crystal_xyz_file_path".
    "crystal": "diamond",
    "crystal_xyz_file_path": None,

    ################################################
    #         Exchange-Correlation Setting         #
    ################################################
    # xc functional
    "xc": "lda_x",

    ################################################
    #           Pseudopotential Setting            #
    ################################################
    "use_pseudopotential": False,
    "pseudopotential_file_dir": None,
    # The pseudopotential file directory. If None, the code will use the default
    # pseudopotential file directory.

    ################################################
    #       Planewave Hyper-parameters             #
    ################################################

    "g_grid_mask_method": "spherical",
    # g_grid_mask_method can be either "cubic" or "spherical".
    # For "cubic", the cutoff cannot be assigned, but will be estimated.
    # Only for "spherical" method, the cutoff_energy argument is valid.

    # cutoff energy in Hartree
    "cutoff_energy": 100,
    # grid sizes for the FFT grid
    "grid_sizes": 64,
    # k grid sizes for the Brillouin zone sampling using Monkhorst-Pack scheme
    "k_grid_sizes": 3,
    # Occupation method
    "occupation": "fermi-dirac",
    # occupation support {"fermi-dirac", "gamma", "uniform"}

    "smearing": 0.0001,  # smearing factor for the Fermi-Dirac distribution
    "empty_bands": 10,  # the total number of unoccupapied bands
    "restricted": True,
    # whether the calculation is restricted or not, ie., spin-up and spin-down
    # electrons share the same spatial orbitals.

    # Hyperparameters for the Ewald sum.
    "ewald_args": {
      'ewald_eta': 0.1,
      'ewald_cutoff': 2e4
    },

    ################################################
    #           Optimizizer Setting                #
    ################################################
    "epoch": 5000,
    "optimizer": "adam",
    "optimizer_args": {
      "learning_rate": 1e-2
    },
    # arguments for the optimizer. Please refer to optax documentation.
    # https://optax.readthedocs.io/en/latest/api/optimizers.html
    # WARNING: the learning_rate is sensitve to the number of G vectors and
    # cutoff energy. It is suggested to tune the learning rate for different
    # systems.

    # "scheduler": "piecewise_constant_schedule",
    "scheduler": None,
    # optax scheduler. None if no scheduler is used.

    "convergence_condition": 1e-8,
    # Convergence criterion for optimization.
    # The code monitors the variance of the objective values (e.g., energy)
    # over the last 50 steps (if available). If the computed variance meets
    # the convergence condition, the calculation will terminate.

    ################################################
    # Band Structure Calculation Hyper-parameters. #
    ################################################

    # NOTE: The following configs are only valid for band structure calculation.
    # Need to either indicate the k_path or the k_path file.
    "band_structure_empty_bands": None,
    "k_path_special_points": None,
    # If `k_path_special_points` is None, the code will use the default
    # special points for the given unit cell, as defined in ASE.
    # For more details, refer to:
    # https://wiki.fysik.dtu.dk/ase/ase/dft/kpoints.html

    # "k_path_special_points": "GXMG",
    # "k_path_special_points": "LGXL",
    "num_kpoints": 60,

    # Alternatively, the user can provide a `k_path` file containing k-path
    # vectors. The file must be in `.npy` format and contain a 2D array,
    # where each row represents the relative coordinates of a k-point in
    # reciprocal space.
    "k_path_file": None,
    "band_structure_epoch": 5000,

    # Flag indicating whether fine-tuning is enabled.
    # If True, the optimization will start with the first k-point,
    # using its parameters as the initialization for subsequent k-points.
    # This approach is recommended for large crystal systems.
    "k_path_fine_tuning": True,
    "k_path_fine_tuning_epoch": 300,

    ################################################
    #              System Setting                  #
    ################################################
    # Random seed
    "seed": 123,

    # Environment setting
    "xla_preallocate": True,
    "jax_enable_x64": True,
    "verbose": True,
    "eps": 1e-8,

  }

  return JrystalConfigDict(config)
