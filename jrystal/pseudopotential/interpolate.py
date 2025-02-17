"""Interpolation Functions"""
import numpy as np
from jaxtyping import Float, Array
from scipy.interpolate import CubicSpline


def cubic_spline(
  x: Float[Array, "num_grid"],
  y: Float[Array, "*batch_y num_grid"],
  new_x: Float[Array, "*batch_new_x num_grid"]
) -> Float[Array, "*batch_y_and_batch_new_x num_grid"]:
  """ Interpolate a function evaluated at point x with new inputs using cubic
    spline. Both y and new_x can have batch dimensions. Return the interpolated
    values, which have the same batch dimensions as y and new_x.  The function
    is not differentiable.

  Warning: Cubic spline interpolation is not implemented in JAX. This function
    uses NumPy and is not differentiable.

  Args:
    x (Float[Array, "num_grid"]): x values.
    y (Float[Array, "*batch_y num_grid"]): y values.
    new_x (Float[Array, "*batch_new_x num_grid"]): new x values.

  Returns:
    Float[Array, "*batch_y *batch_new_x num_grid"]: interpolated y values.
  """

  cs = CubicSpline(x, y, axis=-1)
  new_y = np.apply_along_axis(cs, -1, new_x)
  y_batch_dim = y.ndim - 1
  new_y = np.moveaxis(new_y, range(-y_batch_dim - 1, -1), range(0, y_batch_dim))
  return new_y
