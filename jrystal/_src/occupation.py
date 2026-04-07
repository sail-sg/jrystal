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
"""Occupation-number parameterizations and projection utilities."""
from typing import Any, Optional

import einops
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .unitary_module import unitary_matrix, unitary_matrix_param_init
from .utils import check_spin_number


def __idempotent_param_init(
  key: Array,
  num_bands: int,
  num_electrons: int,
  num_kpts: int,
  spin: int = 0,
  spin_restricted: bool = True,
) -> dict:
  r"""Initialize parameters for idempotent occupation optimization.

  Args:
    key (Array): Random key used for initialization.
    num_bands (int): Number of bands.
    num_electrons (int): Number of electrons.
    num_kpts (int): Number of :math:`k` points.
    spin (int): Number of unpaired electrons.
    spin_restricted (bool): If ``True``, share parameters across spin channels.

  Returns:
    dict: Parameter dictionary for :func:`idempotent`.
  """

  check_spin_number(num_electrons, spin)

  num_elec_up = (num_electrons + spin) // 2
  num_elec_down = (num_electrons - spin) // 2

  param_up = unitary_matrix_param_init(
    key, [num_bands * num_kpts, num_elec_up * num_kpts], complex=False
  )
  if spin_restricted:
    return {"param_up": param_up, "param_down": param_up}

  else:
    param_down = unitary_matrix_param_init(
      key, [num_bands * num_kpts, num_elec_down * num_kpts], complex=False
    )
    return {"param_up": param_up, "param_down": param_down}


def __idempotent(
  params: dict,
  num_kpts: int,
  spin_restricted: bool = True,
) -> Float[Array, 'spin kpt band']:
  r"""Compute occupations from idempotent parameters.

  Args:
    params (dict): Parameters from :func:`idempotent_param_init`.
    num_kpts (int): Number of :math:`k` points.
    spin_restricted (bool): If ``True``, return one combined spin channel.

  Returns:
    Float[Array, 'spin kpt band']: Occupation tensor.
  """
  param_up = params["param_up"]
  param_down = params["param_down"]

  num_bands = param_up["w_re"].shape[0] // num_kpts

  def o(params):
    u = unitary_matrix(params, False)
    # [num_bands * num_kpts, num_elec_up * num_kpts]
    occ = einops.einsum(u, u.T, "nk ik, ik nk -> nk")
    return occ.reshape([num_kpts, num_bands])

  occ_up = o(param_up)
  occ_down = o(param_down)

  if spin_restricted:
    return jnp.expand_dims(occ_up + occ_down, axis=0) / num_kpts
  else:
    return jnp.stack([occ_up, occ_down], axis=0) / num_kpts


def _get_fixed_occupation(
  num_k: int,
  num_electrons: int,
  spin: int = 0,
  num_bands: Optional[int] = None,
  spin_restricted: bool = True
) -> Float[Array, 'spin kpt band']:
  """Create fixed occupations across all :math:`k` points.

  Warning: This function is only physical for insulator. It assumes the number
  of occupied bands are the same for each k-point.

  Args:
    num_k (int): Number of :math:`k` points.
    num_electrons (int): Number of electrons.
    spin (int): Number of unpaired electrons.
    num_bands (Optional[int]): Number of bands. Defaults to ``num_electrons``.
    spin_restricted (bool): If ``True``, merge spin channels.

  Returns:
    Float[Array, 'spin kpt band']: Occupation array.
  """
  check_spin_number(num_electrons, spin)
  num_bands = num_electrons if num_bands is None else num_bands

  occ = jnp.zeros([2, num_k, num_bands])
  occ = occ.at[0, :, :(num_electrons + spin) // 2].set(1.)
  occ = occ.at[1, :, :(num_electrons - spin) // 2].set(1.)

  if spin_restricted:
    return jnp.sum(occ, axis=0, keepdims=True)

  return occ


def _simplex_projector_init(
  num_bands: int,
  num_kpts: int,
) -> dict:
  n = num_bands * num_kpts
  params_up = (jnp.arange(n) - n // 2) * 0.1
  params_up = params_up.reshape([num_kpts, num_bands])

  params_down = (jnp.arange(n) - n // 2) * 0.1
  params_down = params_down.reshape([num_kpts, num_bands])

  return {"param_up": params_up, "param_down": params_down}


def __proj(x: jnp.array, sum: jnp.array):
  """Project a vector onto the bounded simplex.

  The projected vector satisfies ``0 <= x_i <= 1`` and
  ``jnp.sum(x) == sum``.

  Args:
    x (jnp.array): One-dimensional input vector.
    sum (jnp.array): Target sum after projection.

  Returns:
    jnp.array: Projected vector.
  """
  n = x.shape[0]

  def pushdown(x):
    x_sorted = jnp.sort(x)
    x_sum = jnp.sum(x)
    x_cumsum = jnp.cumsum(x_sorted)
    _rho = (x_sum - sum - x_cumsum) / (n - jnp.arange(n) - 1)
    k = jnp.argmax(x_sorted > _rho)
    _lambda = jax.lax.select(
      k == 0, (x_sum - sum) / n,
      (x_sum - sum - x_cumsum.at[k - 1].get()) / (n - k)
    )
    return jnp.maximum(x - _lambda, 0.)

  def pushup(x):
    x_sorted = jnp.sort(x)
    x_sum = jnp.sum(x)
    x_sorted = x_sorted.at[::-1].get()
    x_cumsum = jnp.cumsum(x_sorted)
    _rho = (x_sum - sum +
            (jnp.arange(n) + 1 - x_cumsum)) / (n - jnp.arange(n) - 1)
    k = jnp.argmax(x_sorted - _rho < 1.)
    _lambda = jax.lax.select(
      k == 0, (sum - x_sum) / n,
      (sum - x_sum - (k - x_cumsum.at[k - 1].get())) / (n - k)
    )
    return jnp.minimum(x + _lambda, 1.)

  output = jax.lax.cond(jnp.sum(x) <= sum, pushup, pushdown, x)
  return output


def _simplex_projector(
  params: dict,
  num_electrons: int,
  spin: int = 0,
  spin_restricted: bool = True,
) -> Float[Array, 'spin kpt band']:
  num_kpts = params["param_up"].shape[0]
  num_bands = params["param_up"].shape[1]

  params_up = jax.nn.sigmoid(params["param_up"])
  params_up = jnp.ravel(params_up)
  m_up = (num_electrons + spin) // 2 * num_kpts
  m_down = (num_electrons - spin) // 2 * num_kpts
  occ_up = __proj(params_up, m_up)
  occ_up = occ_up.reshape([num_kpts, num_bands])

  params_down = jax.nn.sigmoid(params["param_down"])
  params_down = jnp.ravel(params_down)
  occ_down = __proj(params_down, m_down)
  occ_down = occ_down.reshape([num_kpts, num_bands])
  occ = jnp.stack([occ_up, occ_down], axis=0)

  if spin_restricted:
    return jnp.sum(occ, axis=0, keepdims=True)
  else:
    return occ


def get_occupation_fn(
  num_electrons: int,
  spin: int = 0,
  spin_restricted: bool = True,
  *,
  method: str = "simplex-projector",
  **kwargs: Any,
):
  """Get a function that computes occupations from parameters.

  Example:
  >>> occ_fn = get_occupation_fn(num_electrons=20, spin=0, spin_restricted=True)
  >>> params = params_init(num_bands=20, num_kpts=10)
  >>> occ = occ_fn(params)
  >>> print(occ)
  >>> print(np.sum(occ)/10)  # should be 20

  Args:
    num_electrons (int): Total number of electrons in the system.
    spin (int): Number of unpaired electrons.
    spin_restricted (bool): If ``True``, merge spin channels.
    method (str): Occupation method. "simplex-projector" is supported for now.
    **kwargs: Additional keyword arguments.

  Returns:
    Callable[[dict], Float[Array, 'spin kpt band']]: Function that computes
    occupations from parameters.
  """
  check_spin_number(num_electrons, spin)
  if method == "simplex-projector":

    def fn(params: dict) -> Float[Array, 'spin kpt band']:
      return _simplex_projector(params, num_electrons, spin, spin_restricted)

    return fn
  else:
    raise ValueError(
      f"Occupation method {method} is not implemented. "
      "Only simplex-projector is supported for now."
    )


def params_init(
  num_bands: int,
  num_kpts: int,
  *,
  method: str = "simplex-projector",
  **kwargs: Any,
) -> dict:
  """Initialize parameters for occupation computation.

  Example:
  >>> params = params_init(num_bands=20, num_kpts=10)
  >>> print(params["param_up"].shape)
  (1, 10, 20)

  Args:
    num_bands (int): Number of bands.
    num_kpts (int): Number of :math:`k` points.
    method (str): Occupation method. "simplex-projector" is supported for now.
    **kwargs: Additional keyword arguments.

  Returns:
    dict: Parameters for occupation computation.
  """
  if method == "simplex-projector":
    return _simplex_projector_init(num_bands, num_kpts)
  else:
    raise ValueError(
      f"Occupation method {method} is not implemented. "
      "Only simplex-projector is supported for now."
    )


__all__ = ["get_occupation_fn", "params_init"]
