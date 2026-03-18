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
"""Potential terms used in electronic-structure calculations."""
from typing import Tuple, Union

import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from jaxtyping import Array, Complex, Float

from . import xc as _xc


def hartree_reciprocal(
  density_grid_reciprocal: Complex[Array, 'spin x y z'],
  g_vector_grid: Float[Array, 'x y z 3'],
  kohn_sham: bool = False
) -> Complex[Array, 'x y z']:
  r"""Compute the Hartree potential in reciprocal space.

  Args:
    density_grid_reciprocal (Complex[Array, 'spin x y z']): Spin-resolved
      reciprocal-space density.
    g_vector_grid (Float[Array, 'x y z 3']): Reciprocal-space G-vector grid.
    kohn_sham (bool): Whether to use the Kohn-Sham potential convention.

  Returns:
    Complex[Array, 'x y z']: Reciprocal-space Hartree potential.
  """
  dim = g_vector_grid.shape[-1]
  assert density_grid_reciprocal.ndim == dim + 1, (
    'density_grid_reciprocal must contains spin axis'
  )
  density_grid_reciprocal = jnp.sum(density_grid_reciprocal, axis=0)

  g_vec_square = jnp.sum(g_vector_grid**2, axis=-1)  # [x y z]
  g_vec_square = g_vec_square.at[(0,) * dim].set(1)

  if kohn_sham:
    density_grid_reciprocal = stop_gradient(density_grid_reciprocal)

  output = density_grid_reciprocal / g_vec_square
  output = output.at[(0,) * dim].set(0)
  output = output * 4 * jnp.pi

  if not kohn_sham:
    output /= 2

  return output


def hartree(
  density_grid_reciprocal: Complex[Array, 'spin x y z'],
  g_vector_grid: Float[Array, 'x y z 3'],
  kohn_sham: bool = False
) -> Complex[Array, 'x y z']:
  r"""Compute the Hartree potential in real space via inverse FFT.

  Args:
    density_grid_reciprocal (Complex[Array, 'spin x y z']): Spin-resolved
      reciprocal-space density.
    g_vector_grid (Float[Array, 'x y z 3']): Reciprocal-space G-vector grid.
    kohn_sham (bool): Whether to use the Kohn-Sham potential convention.

  Returns:
    Complex[Array, 'x y z']: Real-space Hartree potential.
  """
  assert density_grid_reciprocal.ndim == 4, (      # noqa: PLR2004
    'density_grid_reciprocal must contains spin axis'
  )
  density_grid_reciprocal = jnp.sum(density_grid_reciprocal, axis=0)
  har_pot_grid_rcprl = hartree_reciprocal(
    density_grid_reciprocal, g_vector_grid, kohn_sham
  )
  return jnp.fft.ifftn(har_pot_grid_rcprl, axes=range(-3, 0))


def external_reciprocal(
  position: Float[Array, 'atom 3'],
  charge: Float[Array, ' atom'],
  g_vector_grid: Float[Array, 'x y z 3'],
  vol: Float,
) -> Complex[Array, 'x y z']:
  r"""Compute the electron-ion external potential in reciprocal space.

  Args:
    position (Float[Array, 'atom 3']): Atomic positions.
    charge (Float[Array, 'atom']): Atomic charges.
    g_vector_grid (Float[Array, 'x y z 3']): Reciprocal-space G-vector grid.
    vol (Float): Unit-cell volume.

  Returns:
    Complex[Array, 'x y z']: Reciprocal-space external potential.
  """
  dim = position.shape[-1]
  g_norm_square = jnp.sum(g_vector_grid**2, axis=-1)
  si = jnp.exp(-1.j * jnp.matmul(g_vector_grid, position.transpose()))
  num_grids = jnp.prod(jnp.array(g_vector_grid.shape[:-1]))
  # num_grids is to cancel the parseval factor in ``reciprocal_braket``

  charge = jnp.expand_dims(charge, range(3))
  g_norm_square = jnp.expand_dims(g_norm_square, -1)
  vi = charge / (g_norm_square + 1e-10)
  vi = vi.at[(0,) * dim].set(0)
  vi *= 4 * jnp.pi

  output = jnp.sum(vi * si, axis=-1)
  return -output * num_grids / vol


def external(
  position: Float[Array, 'atom 3'],
  charge: Float[Array, ' atom'],
  g_vector_grid: Float[Array, 'x y z 3'],
  vol: Float,
) -> Complex[Array, 'x y z']:
  r"""Compute the electron-ion external potential in real space.

  Args:
    position (Float[Array, 'atom 3']): Atomic positions.
    charge (Float[Array, 'atom']): Atomic charges.
    g_vector_grid (Float[Array, 'x y z 3']): Reciprocal-space G-vector grid.
    vol (Float): Unit-cell volume.

  Returns:
    Complex[Array, 'x y z']: Real-space external potential.
  """
  ext_pot_grid_rcprl = external_reciprocal(position, charge, g_vector_grid, vol)
  return jnp.fft.ifftn(ext_pot_grid_rcprl, axes=range(-3, 0))


def effective(
  density_grid: Float[Array, 'spin x y z'],
  position: Float[Array, "num_atom 3"],
  charge: Float[Array, " num_atom"],
  g_vector_grid: Float[Array, 'x y z 3'],
  vol: Float,
  split: bool = False,
  xc_type: str = "lda_x",
  kohn_sham: bool = False,
) -> Union[Tuple[Float[Array, '... x y z'],
                 Float[Array, '... x y z'],
                 Float[Array, '... x y z']],
           Float[Array, '... x y z']]:
  r"""Compute the effective potential :math:`V_\mathrm{eff}`.

  Args:
    density_grid (Float[Array, 'spin x y z']): Real-space density grid
      (optionally without explicit spin axis).
    position (Float[Array, "num_atom 3"]): Atomic positions.
    charge (Float[Array, "num_atom"]): Atomic charges.
    g_vector_grid (Float[Array, 'x y z 3']): Reciprocal-space G-vector grid.
    vol (Float): Unit-cell volume.
    split (bool): If ``True``, return ``(V_H, V_ext, V_xc)``.
    xc_type (str): XC functional specification.
    kohn_sham (bool): Whether to use Kohn-Sham convention.

  Returns:
    Union[Tuple[Array, Array, Array], Array]: Effective potential components or
    their sum.
  """
  dim = position.shape[-1]
  assert density_grid.ndim in [dim, dim + 1]

  if density_grid.ndim == dim:
    density_grid = jnp.expand_dims(density_grid, 0)

  polarized = density_grid.shape[0] == 2
  if kohn_sham:
    # In Kohn-Sham matrix construction, V_eff is treated as fixed with respect
    # to orbital variation; stopping here avoids nested AD through jxc(vxc).
    density_grid = stop_gradient(density_grid)

  density_grid_reciprocal = jnp.fft.fftn(density_grid, axes=range(-dim, 0))
  # reciprocal space:
  v_hartree = hartree_reciprocal(
    density_grid_reciprocal, g_vector_grid, kohn_sham
  )
  v_external = external_reciprocal(position, charge, g_vector_grid, vol)

  # XC potential via jxc
  level = _xc.xc_level(xc_type)
  rho = density_grid if polarized else density_grid[0]

  sigma = None
  if level in ('gga', 'mgga'):
    sigma = _xc.compute_sigma(density_grid, g_vector_grid)

  vxc_dict = _xc.xc_potential(rho, xc_type, polarized, sigma=sigma)

  if level == 'lda':
    vrho = vxc_dict['vrho']
    if not polarized:
      v_xc = vrho[None, ...]  # (1, x, y, z)
    else:
      v_xc = jnp.moveaxis(vrho, -1, 0)  # (2, x, y, z)
  else:
    # GGA (and MGGA local part): vrho − 2∇·(vsigma ∇ρ)
    v_xc = _xc.gga_xc_potential(
      vxc_dict['vrho'], vxc_dict['vsigma'],
      density_grid, g_vector_grid,
    )

  v_xc = stop_gradient(v_xc)

  # transform to real space
  v_hartree = jnp.fft.ifftn(v_hartree, axes=range(-dim, 0))
  v_external = jnp.fft.ifftn(v_external, axes=range(-dim, 0))

  if split:
    return v_hartree, v_external, v_xc
  else:
    return v_hartree[None, ...] + v_external[None, ...] + v_xc
