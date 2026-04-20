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
"""Utilities for generating unitary (or orthogonal) matrices."""
from typing import Dict, List, Optional, Union

import jax
import jax.numpy as jnp
from jax.sharding import Sharding
from jaxtyping import Array

from .uniform import uniform


def unitary_matrix(
  params: Dict[str, Array],
  complex: bool = False,
  method: str = "qr",
  sharding: Optional[Sharding] = None
) -> Array:
  """Build a unitary/orthogonal matrix from learnable parameters.

  Args:
    params (Dict[str, Array]): Parameter dictionary from
      :func:`unitary_matrix_param_init`.
    complex (bool): If ``True``, build a complex unitary matrix. If ``False``,
      build a real orthogonal matrix.
    method (str): Orthogonalization method. Supported values are ``'qr'``,
      ``'cholesky'``, and ``'householder'``.
    sharding (Optional[Sharding]): Optional sharding for JIT execution.

  Returns:
    Array: Matrix with orthonormal columns.
  """
  weight_real = params['w_re']
  if complex:
    weight_imaginary = 1.j * params['w_im']
  else:
    weight_imaginary = 0.

  weight = weight_real + weight_imaginary

  def _cholesky(x):
    m = x.shape[1]
    overlap = x.T @ x
    _l = jnp.linalg.cholesky(overlap)
    _l_inv = jax.scipy.linalg.solve_triangular(_l.T, jnp.eye(m))
    x = jnp.matmul(x, _l_inv)
    return x

  def _householder(x):
    dim1, dim2 = x.shape
    a = jnp.tril(x)
    a = a.at[jnp.arange(dim2), jnp.arange(dim2)].set(1.)
    tau = jnp.array(2. / (jnp.linalg.norm(a, axis=0)**2), dtype=x.dtype)
    hp = jax.lax.linalg.householder_product(a, tau)
    # differentiation rule is not implemented.
    return hp

  for _ in range(weight.ndim - 2):
    _cholesky = jax.vmap(_cholesky)
    _householder = jax.vmap(_householder)

  if method == 'qr':

    def ortho_fn(w):
      return jnp.linalg.qr(w, mode='reduced')[0]
  elif method == 'cholesky':
    ortho_fn = _cholesky
  elif method == 'householder':
    ortho_fn = _householder

  if sharding:
    ortho_fn = jax.jit(ortho_fn, in_shardings=sharding)

  orthogonal_columns = ortho_fn(weight)
  return orthogonal_columns


def unitary_matrix_param_init(
  key: Array,
  shape: Union[tuple, List[int]],
  complex: bool = True,
  sharding: Optional[Sharding] = None
) -> Dict[str, Array]:
  """Initialize parameters used by :func:`unitary_matrix`.

  Args:
    key (Array): JAX PRNG key.
    shape (Union[tuple, List[int]]): Matrix shape.
    complex (bool): If ``True``, initialize both real and imaginary parts.
    sharding (Optional[Sharding]): Optional output sharding.

  Returns:
    Dict[str, Array]: Dictionary with keys ``'w_re'`` and ``'w_im'``.
  """
  key_re, key_im = jax.random.split(key)
  weight_real = uniform(key_re, shape, out_sharding=sharding)

  if complex:
    weight_imaginary = uniform(key_im, shape, out_sharding=sharding)
  else:
    weight_imaginary = 0.

  return {'w_re': weight_real, 'w_im': weight_imaginary}
