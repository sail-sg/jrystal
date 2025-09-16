import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from ..pseudopotential.beta import _beta_sbt_single_atom, beta_sbt_grid
from .opt_utils import save_beta_sbt


def _to_map(args):
  return _beta_sbt_single_atom(*args)


def pre_calc_beta_sbt(
  pseudopot, g_vector_grid, kpts, save_cache=False,
  sbt_method="sbt"
):
  """Pre-calculate the beta functions for the given configuration.

  Args:
    pseudopot: The pseudopotential object.
    g_vector_grid: The reciprocal space grid.
    kpts: The k-points grid.
    sbt_method: The method to use for the spherical bessel transform.
    save_cache: Whether to save the beta functions.

  Returns:
    The concatenated beta functions in reciprocal space.
  """

  # Prepare arguments for multiprocessing

  assert sbt_method in ["numerical", "sbt"], \
    "Invalid SBT method. Only 'numerical', 'sbt' are supported."

  if sbt_method == "sbt":
    mp.set_start_method("spawn", force=True)

    args_list = []
    for r, b, l, rab in zip(
      pseudopot.r_grid, pseudopot.nonlocal_beta_grid,
      pseudopot.nonlocal_angular_momentum, pseudopot.r_ab
    ):
      args_list.append((r, b, l, g_vector_grid, kpts, rab, "sbt"))

    # Use multiprocessing Pool to parallelize computation
    mp.set_start_method("spawn", force=True)
    with ProcessPoolExecutor(max_workers=mp.cpu_count()//2) as exe:
      output = list(exe.map(_to_map, args_list))

  elif sbt_method == "numerical":
    return beta_sbt_grid(
      r_grid=pseudopot.r_grid,
      nonlocal_beta_grid=pseudopot.nonlocal_beta_grid,
      nonlocal_angular_momentum=pseudopot.nonlocal_angular_momentum,
      g_vector_grid=g_vector_grid,
      kpts=kpts,
      r_ab=pseudopot.r_ab,
      method="numerical"
    )

  # Create cache directory if it doesn't exist
  if save_cache:
    save_beta_sbt(output)

  return output
