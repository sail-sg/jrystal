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
import operator
import numpy as np

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
    f: A function that takes a complex-valued array and returns a
    complex-valued array.
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

  return jnp.array(
    [
      hessian_vec_prod(primal, jnp.eye(dim, dtype=dtype)[i])
      for i in range(dim)
    ]
  )


def hvp(f, x, v):
  """compute the the Hessian-vector product.

  Args:
    f: A function that takes a complex-valued vector and returns a
    real-valued scalar.
    x: The primal at which to compute the Hessian-vector product.
    v: The vector at which to compute the Hessian-vector product.  It must have
    the same structure as the output of f.

  Returns:
    The Hessian-vector product of the function .
  """

  _, vjp_fun = jax.vjp(jax.grad(f), x)

  # the vjp_fun takes an object that has the same shape as the output of the
  # function f, same structure if the return is a pytree.

  return vjp_fun(v)[0].T.conj()  # here assume that the output is a vector.


def hessian_diag_vector(f: Callable) -> Callable:
  """Compute the diagonal elements of the Hessian matrix of a function
  evalueated at the primal value x.

  This function only applies to the case where the input is a vector.

  Args:
    f (Callable): A function that takes a vector and returns a real-valued
    scalar.

  Returns:
    Callable: A function that takes a vector and returns the diagonal elements
    of the Hessian matrix of the function at the point.
  """

  def _hess_diag_vec(x):
    n = x.shape[0]
    # basis = jnp.eye(n, dtype=x.dtype)

    def g(carry, xx):
      e = jnp.zeros(n, dtype=x.dtype)
      e = e.at[xx].set(1.).astype(x.dtype)
      # Each diag entry is e_i^T H e_i
      y = hvp(f, x, e).dot(e)     # trade time for space.
      return None, y

    return jax.lax.scan(g, None, jnp.arange(n))[1]

  return _hess_diag_vec


def hessian_diag_pytree(f: Callable) -> Callable:
  """Compute the diagonal elements of the Hessian matrix of a function.

  Args:
    f (Callable): A function that takes a PyTree and returns a real-valued
    scalar.

  Returns:
    Callable:A function that takes a PyTree and returns the diagonal elements
    of the Hessian matrix of the function.
  """
  grad_fn = jax.grad(f)

  def hvp(x, v):
    v = jax.tree.map(lambda a, b: b.astype(a.dtype), x, v)
    """Hessian-vector product"""
    return jax.jvp(grad_fn, (x,), (v,))[1]
    # the output and input of grad_fn have same structure, therefore jvp and
    # vjp are equally effective.

  def _diag_hessian(x):
    flat_x, tree_def = jax.tree_util.tree_flatten(x)

    result = []
    dtype = jnp.int32 if flat_x[0].dtype == jnp.complex64 else jnp.int64
    flat_x_zeros = jax.tree.map(
      lambda x: jnp.zeros(x.shape, dtype=dtype), flat_x
    )

    for leaf_idx, leaf in enumerate(flat_x):
      leaf_zeros = jnp.zeros(leaf.shape, dtype=leaf.dtype)

      def f_scan(carry, idx):
        _result, leaf, flat_x_zeros = carry
        mask = jnp.zeros(leaf.shape, dtype=jnp.bool)
        mask = mask.at[idx].set(True)
        e = flat_x_zeros.copy()
        e[leaf_idx] = mask
        e = jax.tree.unflatten(tree_def, e)
        ehv = hvp(x, e)
        ehve = jax.tree.map(lambda x, y: x * y, ehv, e)
        _result = _result + jax.tree.flatten(ehve)[0][leaf_idx]

        return (_result, leaf, flat_x_zeros), None

      # breakpoint()
      scan_carry, _ = jax.lax.scan(
        f_scan, (leaf_zeros, leaf, flat_x_zeros),
        np.array([idx for idx in np.ndindex(leaf.shape)])
      )
      result.append(scan_carry[0])

    return jax.tree.unflatten(tree_def, result)
  return _diag_hessian
