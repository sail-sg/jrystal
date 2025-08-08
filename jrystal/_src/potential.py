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
"""Potentials."""
from typing import Tuple, Union

import jax.numpy as jnp
from jax.lax import stop_gradient
from jaxtyping import Array, Complex, Float

from .xc import xc_density


def hartree_reciprocal(
  density_grid_reciprocal: Complex[Array, 'spin x y z'],
  g_vector_grid: Float[Array, 'x y z 3'],
  kohn_sham: bool = False
) -> Complex[Array, 'x y z']:
  r"""Calculate the Hartree potential in reciprocal space.

  The Hartree potential represents the classical electrostatic interaction between electrons.
  In reciprocal space, it obeys the following equation:

  .. math::
      \hat{V}_H(\mathbf{G}) = 4\pi \frac{\hat{n}(\mathbf{G})}{|\mathbf{G}|^2},
      \quad \hat{V}_H(\mathbf{0}) = 0

  where:

  - :math:`\hat{V}_H(\mathbf{G})` is the Hartree potential in reciprocal space
  - :math:`\hat{n}(\mathbf{G})` is the electron density in reciprocal space
  - :math:`\mathbf{G}` is the reciprocal lattice vector

  Please also refer to the tutorial :doc:`Total Energy Minimization <../tutorial/total_energy>`
  for more details.

  Args:
    density_grid_reciprocal (Complex[Array, 'spin x y z']): Electron density in
      reciprocal space. The input density must contains spin axis.
    g_vector_grid (Float[Array, 'x y z 3']): Grid of G-vectors in reciprocal
      space.
    kohn_sham (bool, optional): If True, use Kohn-Sham formalism. Defaults to
      False.

  Returns:
    Complex[Array, 'x y z']: Hartree potential in reciprocal space. If  density_grid_reciprocal has batch axes, they are preserved in the output.
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
  r"""Calculate the Hartree potential in real space.

  Computes the Hartree potential by applying inverse Fourier transform to the
  reciprocal space potential. The real-space Hartree potential represents the
  classical electrostatic interaction between electrons:

  .. math::
      V_H(\mathbf{r}) = \mathcal{F}^{-1}[\hat{V}_H(\mathbf{G})]

  where:

  - :math:`V_H(\mathbf{r})` is the Hartree potential in real space
  - :math:`\hat{V}_H(\mathbf{G})` is the Hartree potential in reciprocal space
  - :math:`\mathcal{F}^{-1}` denotes the inverse Fourier transform

  Args:
    density_grid_reciprocal (Complex[Array, 'spin x y z']): Electron density in
      reciprocal space. The input density must contains spin axis.
    g_vector_grid (Float[Array, 'x y z 3']): Grid of G-vectors in reciprocal
      space.
    kohn_sham (bool, optional): If True, use Kohn-Sham formalism. Defaults to
      False.

  Returns:
    Complex[Array, 'x y z']: Hartree potential in real space. If density_grid_reciprocal has batch axes, they are preserved in the output.
  """
  assert density_grid_reciprocal.ndim == 4, (
    'density_grid_reciprocal must contains spin axis'
  )
  density_grid_reciprocal = jnp.sum(density_grid_reciprocal, axis=0)
  har_pot_grid_rcprl = hartree_reciprocal(
    density_grid_reciprocal, g_vector_grid, kohn_sham
  )
  return jnp.fft.ifftn(har_pot_grid_rcprl, axes=range(-3, 0))


def external_reciprocal(
  position: Float[Array, 'atom 3'],
  charge: Float[Array, 'atom'],
  g_vector_grid: Float[Array, 'x y z 3'],
  vol: Float,
) -> Complex[Array, 'x y z']:
  r"""Calculate the external potential in reciprocal space.

  The external potential represents the Coulomb interaction between electrons
  and nuclei. In reciprocal space, it is computed as a sum over atomic
  contributions:

  .. math::
      \hat{V}_{\text{ext}}(\mathbf{G}) = \sum_{\alpha} Z_{\alpha}
      e^{-i\mathbf{G}\cdot\mathbf{R}_{\alpha}} v(\mathbf{G})

  where:

  - :math:`Z_{\alpha}` is the nuclear charge of atom :math:`\alpha`
  - :math:`\mathbf{R}_{\alpha}` is the position of atom :math:`\alpha`
  - :math:`v(\mathbf{G}) = -4\pi/|\mathbf{G}|^2` is the Coulomb potential in reciprocal space
  - :math:`\mathbf{G}` is the reciprocal lattice vector

  Args:
    position (Float[Array, 'atom 3']): Atomic positions in the unit cell.
    charge (Float[Array, 'atom']): Nuclear charges.
    g_vector_grid (Float[Array, 'x y z 3']): Grid of G-vectors in reciprocal space.
    vol (Float): Volume of the unit cell.

  Returns:
    Complex[Array, 'x y z']: External potential in reciprocal space.
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
  charge: Float[Array, 'atom'],
  g_vector_grid: Float[Array, 'x y z 3'],
  vol: Float,
) -> Complex[Array, 'x y z']:
  r"""Calculate the external potential in real space.

  Computes the external potential by applying inverse Fourier transform to the
  reciprocal space potential. The real-space external potential represents the
  Coulomb interaction between electrons and nuclei:

  .. math::
      V_{\text{ext}}(\mathbf{r}) = \mathcal{F}^{-1}[\hat{V}_{\text{ext}}(\mathbf{G})]

  where:

  - :math:`V_{\text{ext}}(\mathbf{r})` is the external potential in real space
  - :math:`\hat{V}_{\text{ext}}(\mathbf{G})` is the external potential in reciprocal space
  - :math:`\mathcal{F}^{-1}` denotes the inverse Fourier transform

  Args:
    position (Float[Array, 'atom 3']): Atomic positions in the unit cell.
    charge (Float[Array, 'atom']): Nuclear charges.
    g_vector_grid (Float[Array, 'x y z 3']): Grid of G-vectors in reciprocal space.
    vol (Float): Volume of the unit cell.

  Returns:
    Complex[Array, 'x y z']: External potential in real space.
  """
  ext_pot_grid_rcprl = external_reciprocal(position, charge, g_vector_grid, vol)
  return jnp.fft.ifftn(ext_pot_grid_rcprl, axes=range(-3, 0))


def effective(
  density_grid: Float[Array, 'spin x y z'],
  position: Float[Array, "num_atom 3"],
  charge: Float[Array, "num_atom"],
  g_vector_grid: Float[Array, 'x y z 3'],
  vol: Float,
  split: bool = False,
  xc_type: str = "lda_x",
  kohn_sham: bool = False,
) -> Union[Tuple[Float[Array, '... x y z'],
                 Float[Array, '... x y z'],
                 Float[Array, '... x y z']],
           Float[Array, '... x y z']]:
  r"""Calculate the effective potential for electronic structure calculations.

  The effective potential is the sum of three contributions:

  .. math::
      V_{\text{eff}}(\mathbf{r}) = V_H(\mathbf{r}) + V_{\text{ext}}(\mathbf{r})
      + V_{xc}(\mathbf{r})

  where:

  - :math:`V_H(\mathbf{r})` is the Hartree potential
  - :math:`V_{\text{ext}}(\mathbf{r})` is the external (nuclear) potential
  - :math:`V_{xc}(\mathbf{r})` is the exchange-correlation potential

  .. warning::
    Currently supports only LDA exchange-correlation functional.

  Args:
    density_grid (Union[Float[Array, 'x y z'], Float[Array, 'spin x y z']]):
      Real-space electron density. The dimension of the density grid can be
      either 3 or 4, where 4 dimension is for spin-polarized calculation (first
      dimension is spin channel).
    position (Float[Array, "num_atom 3"]): Atomic positions in the unit cell.
    charge (Float[Array, "num_atom"]): Nuclear charges.
    g_vector_grid (Float[Array, 'x y z 3']): Grid of G-vectors in reciprocal
      space.
    vol (Float): Volume of the unit cell.
    split (bool, optional): If True, return individual potential components.
      Defaults to False.
    xc (str, optional): Exchange-correlation functional type. Only "lda_x" is
      currently supported. Defaults to "lda_x".
    kohn_sham (bool, optional): If True, use Kohn-Sham formalism. Defaults to
      False.
    spin_restricted (bool, optional): If True, use spin-restricted calculation.
      Defaults to True.

  Returns:
    Union[Tuple[Array, Array, Array], Array]: If split is True, returns (Hartree, external, xc) potentials. Otherwise returns their sum as the total effective potential. For spin-polarized calculation, the shape of the output is [2, x, y, z], else [x, y, z].
  """
  dim = position.shape[-1]
  assert density_grid.ndim in [dim, dim + 1]

  if density_grid.ndim == dim:
    density_grid = jnp.expand_dims(density_grid, 0)

  density_grid_reciprocal = jnp.fft.fftn(density_grid, axes=range(-dim, 0))
  # reciprocal space:
  v_hartree = hartree_reciprocal(
    density_grid_reciprocal, g_vector_grid, kohn_sham
  )
  v_external = external_reciprocal(position, charge, g_vector_grid, vol)

  # real space:
  v_xc: Float[Array, 's x y z']
  v_xc = xc_density(density_grid, g_vector_grid, kohn_sham, xc_type)

  # transform to real space
  v_hartree = jnp.fft.ifftn(v_hartree, axes=range(-dim, 0))
  v_external = jnp.fft.ifftn(v_external, axes=range(-dim, 0))

  if split:
    return v_hartree, v_external, v_xc
  else:
    return v_hartree[None, ...] + v_external[None, ...] + v_xc
