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
"""Hessian for Complex-Valued Functions. """
from typing import Callable

import jax
import jax.numpy as jnp


def complex_hessian(
  f: Callable[[jnp.ndarray], jnp.ndarray],
  primal: jnp.ndarray,
) -> jnp.ndarray:
  """Compute the Hessian of a complex-valued function at a point.

  .. warning::

    Only for :math:`x` being a vector is tested.

  Args:
    f: A function that takes a complex-valued array and returns a complex-valued array.
    primal: The point at which to compute the Hessian.

  Returns:
    The Hessian of the function at the point.
  """
  dtype = primal.dtype
  dim = primal.shape[-1]

  def grad_f(x):
    value, vjp_fn = jax.vjp(f, x)
    return vjp_fn(jnp.ones(value.shape, dtype=dtype))[0]

  def hessian_vec_prod(x, v):
    _, vjp_fn = jax.vjp(grad_f, x)
    return vjp_fn(v)[0]

  # x = jnp.ones(dim, dtype=dtype)
  return jnp.array(
    [
      hessian_vec_prod(primal, jnp.eye(dim, dtype=dtype)[i])
      for i in range(dim)
    ]
  )
