"""Numerical Sherical Bessel Transform.

  S(f) = int_0^\infty f(r) j_l(r) r^2 dr

"""
from typing import Union, Optional, Sequence, Tuple
import numpy as np
from scipy.special import spherical_jn as jn
from jaxtyping import Array, Float
from einops import einsum


def sbt(
  r_grid: Float[Array, "r"],
  f_grid: Float[Array, "f r"],
  l: Union[int, Sequence[int]],
  kmax: float = None,
  delta_r: Optional[Float[Array, "r"]] = None,
) -> Tuple[Float[Array, "g"], Float[Array, "f g"]]:
  """Numerical Sherical Bessel Transform.

  This function is used to compute the numerical sherical bessel transform of
  the function f(r) on the grid r_grid.

  .. warning::
    Currently this function is not differentiable, as the spherical bessel
    function is not implemented in JAX.

    S(f) = int_0^\infty f(r) j_l(r) r^2 dr

  Args:
    r_grid (Float[Array, "r"]): The grid of the radial coordinate.
    g_grid (Float[Array, "g"]): The grid of the momentum coordinate.
    f_grid (Float[Array, "r"]): The function values on the grid r_grid.
    l (int): The angular momentum. l can be a list of ints, only if the length
    of batch dimension of f_grid is the same as the length of l. For example,
    if f_grid.shape[0] == 2, l can be [0, 1] or 0.
    delta_r (Optional[Float[Array, "r"]]): The grid spacing of the radial
    coordinate. If None, the grid spacing is calculated from the grid r_grid.

  Returns:
    Float[Array, "g"]: The sherical bessel transform of the function f.
  """
  if delta_r is None:
    delta_r = np.zeros_like(r_grid)
    delta_r[:-1] = r_grid[1:] - r_grid[:-1]

  g_max = kmax
  g_min = 0.0001
  g_grid = np.linspace(g_min, g_max, len(r_grid)*2)
  gr = einsum(g_grid, r_grid, "g, r -> g r")   # shape [g_batch* r]

  if hasattr(l, "__len__") and hasattr(l, "__getitem__"):

    assert f_grid.ndim == 2 and f_grid.shape[0] == len(l), \
      "The length of l must be the same as the batch dimension of f_grid"

    jn_gr = np.empty((len(l), *gr.shape))  # [l g r]

    for i in range(len(l)):
      jn_gr[i] = jn(l[i], gr)   # [l g r]

    output = einsum(
      f_grid, r_grid**2, jn_gr, delta_r, "l r, r, l g r, r -> l g"
    )

  elif isinstance(l, int):
    if f_grid.ndim == 1:
      f_grid = np.expand_dims(f_grid, axis=0)

    jn_gr = jn(l, gr)  # shape: [g r]

    output = einsum(
      f_grid, r_grid**2, jn_gr, delta_r,
      "f r, r, g r, r -> f g"
    )

  else:
    raise ValueError("\'l\' must be a integer or a list of int")

  return g_grid, output
