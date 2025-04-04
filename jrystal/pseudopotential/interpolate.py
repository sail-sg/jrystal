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
"""Interpolation Functions"""
import numpy as np
from jaxtyping import Float, Array
from scipy.interpolate import CubicSpline 
# from interpax import CubicSpline


def cubic_spline(
  x: Float[Array, "x"],
  y: Float[Array, "*batch_y x"],
  new_x: Float[Array, "*batch_new_x x"]
) -> Float[Array, "*batch_y_and_batch_new_x x"]:
  """Interpolate a function evaluated at point :math:`x` with new inputs using cubic spline. Both :math:`y` and new :math:`x` can have batch dimensions. Return the interpolated values, which have the same batch dimensions as :math:`y` and :math:`new_x`. The function is not differentiable.

  .. warning::
    Cubic spline interpolation is not implemented in JAX. This function uses ``NumPy`` and is not differentiable.

  Args:
    x (Float[Array, "x"]): x values.
    y (Float[Array, "*batch_y x"]): y values.
    new_x (Float[Array, "*batch_new_x x"]): new x values.

  Returns:
    Float[Array, "*batch_y *batch_new_x x"]: interpolated y values.
  """

  cs = CubicSpline(x, y, axis=-1)
  new_y = np.apply_along_axis(cs, -1, new_x)
  y_batch_dim = y.ndim - 1
  new_y = np.moveaxis(
    new_y, range(-y_batch_dim - 1, -1), range(0, y_batch_dim)
  )
  return new_y
