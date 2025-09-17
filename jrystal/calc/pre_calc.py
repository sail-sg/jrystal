import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from ..pseudopotential.beta import _beta_sbt_single_atom
from .opt_utils import save_beta_sbt


def _to_map(args):
  return _beta_sbt_single_atom(*args)


def pre_calc_beta_sbt(pseudopot, g_vector_grid, kpts, save_cache=False):
  """Pre-calculate the beta functions for the given configuration.

  Args:
    pseudopot: The pseudopotential object.
    g_vector_grid: The reciprocal space grid.
    kpts: The k-points grid.

  Returns:
    The concatenated beta functions in reciprocal space.
  """
  mp.set_start_method("spawn", force=True)

  # Prepare arguments for multiprocessing
  args_list = []
  for r, b, l in zip(pseudopot.r_grid, pseudopot.nonlocal_beta_grid,
                     pseudopot.nonlocal_angular_momentum):
    args_list.append((r, b, l, g_vector_grid, kpts))

  # Use multiprocessing Pool to parallelize computation
  mp.set_start_method("spawn", force=True)
  with ProcessPoolExecutor(max_workers=mp.cpu_count()//2) as exe:
    output = list(exe.map(_to_map, args_list))

  # Create cache directory if it doesn't exist
  if save_cache:
    save_beta_sbt(output)

  return output
