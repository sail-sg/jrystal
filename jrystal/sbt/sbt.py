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
"""Spherical Bessel Transform. """
import jax.numpy as jnp
from typing import Tuple, Sequence, Union
from jaxtyping import Array, Float
import numpy as np
import scipy

from .pysbt import pyNumSBT


def sbt(
  r_grid: Float[Array, "num_r"],
  f_grid: Float[Array, "num_r"],
  l: int = 0,
  kmax: float = 100,
  norm: bool = False
) -> Tuple[Float[Array, "num_r"], Float[Array, "num_r"]]:
  """Spherical Bessel Transform.

  ..math::
    g(k) = int_0^\infty f(r) j_l(r) r^2 dr

  Warning:
    This function is a wrapper of pysbt. It is not differentiable.

  Args:
    r_grid (Float[Array, "num_r"]): the r grid corresponding to the function.
    f_grid (Float[Array, "num_r"]): the function values.
    l (int, optional): Angular momentum. Defaults to 0.
    kmax (float, optional): Maximum k value. Defaults to 100.
    norm (bool, optional): Whether to normalize the output. Defaults to False.

  Returns:
    Tuple[Float[Array, "num_r"], Float[Array, "num_r"]]: A tuple of (k_grid,
      transformed_f_grid). k_grid is a 1d-array, and transformed_f_grid is the
      transformed function values.
  """
  r_grid = r_grid.astype(jnp.result_type(float, r_grid.dtype))
  f_grid = f_grid.astype(jnp.result_type(float, f_grid.dtype))

  def _sbt(r_grid, f_grid, l=l, kmax=kmax, norm=norm):  # noqa
    ss = pyNumSBT(r_grid, kmax=kmax)
    return (
      jnp.asarray(ss.kk, dtype=jnp.result_type(float, r_grid.dtype)),
      jnp.asarray(
        ss.run(f_grid, l=l, direction=1, norm=norm),
        dtype=jnp.result_type(float, r_grid.dtype)
      )
    )

  return _sbt(r_grid, f_grid, l=l, kmax=kmax, norm=norm)


def batch_sbt(
  r_grid: Float[Array, "ngrid"],
  f_grid: Float[Array, "nbatch ngrid"],
  l: Union[int, Sequence[int]],
  kmax: float = 100,
  norm: bool = False
) -> Tuple:
  """batched spherical bessel transform for multiple functions.

    sbt: g(k) = int_0^\infty f(r) j_l(r) r^2 dr

  Args:
    r_grid (Float[Array, "ngrid"]): the r grid corresponding to the function.
    f_grid (Float[Array, "nbatch ngrid"]): a batch of the function values.
    l (Union[int, List[int]]): Angular momentum. If l is a list, the length
      must be the same as the number of batches.
    kmax (float, optional): Maximum k value. Defaults to 100.
    norm (bool, optional): Whether to normalize the output. Defaults to False.

  Returns:
    Tuple: A tuple of (k_grid, batched_transformed_f_grid). k_grid is a
    1d-array, and batched_transformed_f_grid is BxN shaped, where B is the
    number of batches and N is the number of grid point.
  """
  output = []
  if hasattr(l, "__len__") and hasattr(l, "__getitem__"):
    for f, li in zip(f_grid, l):
      k, g = sbt(r_grid, f, l=li, kmax=kmax, norm=norm)
      output.append(g)
  elif isinstance(l, int):
    for f in f_grid:
      k, g = sbt(r_grid, f, l=l, kmax=kmax, norm=norm)
      output.append(g)
  else:
    raise ValueError("\'l\' must be a integer or a list of int")
  return k, jnp.vstack(output)


def _sbt_hankel(r_grid, f_grid, l: int = 0, kmax: int = 100):  # noqa
  """spherical bessel transform via Hankel transform.

  NOTE: this function is less stable than pysbt. Use `sbt` instead.
  """
  dln = np.mean(np.diff(np.log(r_grid)))
  initial = np.log(kmax) - (np.log(r_grid[-1]) - np.log(r_grid[0]))
  offset = scipy.fft.fhtoffset(dln, mu=l + 0.5, initial=initial)
  fht = scipy.fft.fht(
    f_grid * (r_grid**1.5), dln, mu=l + 0.5, offset=offset + np.log(r_grid[-1])
  )
  k = np.exp(offset) * np.exp(np.arange(r_grid.shape[0]) * dln)
  return fht / (k**1.5), k
