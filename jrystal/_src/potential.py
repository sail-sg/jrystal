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

import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from jax_xc.impl.gga_x_pbe import unpol as pbe_jac_xc
from jax_xc.utils import get_p
from jaxtyping import Array, Complex, Float

from .utils import absolute_square


def hartree_reciprocal(
  density_grid_reciprocal: Complex[Array, 'x y z'],
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
    density_grid_reciprocal (Complex[Array, 'x y z']): Electron density in
      reciprocal space.
    g_vector_grid (Float[Array, 'x y z 3']): Grid of G-vectors in reciprocal space.
    kohn_sham (bool, optional): If True, use Kohn-Sham formalism. Defaults to False.

  Returns:
    Complex[Array, 'x y z']: Hartree potential in reciprocal space. If  density_grid_reciprocal has batch axes, they are preserved in the output.
  """
  dim = g_vector_grid.shape[-1]
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
  density_grid_reciprocal: Complex[Array, 'x y z'],
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
    density_grid_reciprocal (Complex[Array, 'x y z']): Electron density in
      reciprocal space.
    g_vector_grid (Float[Array, 'x y z 3']): Grid of G-vectors in reciprocal space.
    kohn_sham (bool, optional): If True, use Kohn-Sham formalism. Defaults to False.

  Returns:
    Complex[Array, 'x y z']: Hartree potential in real space. If density_grid_reciprocal has batch axes, they are preserved in the output.
  """
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


def _lda_x(density_grid: Float[Array, '*d x y z']) -> Float[Array, '*d x y z']:
  r"""Calculate the LDA exchange-correlation energy density.

  Implements the Local Density Approximation (LDA) for the exchange-correlation
  energy density in the spin-unpolarized case:

  .. math::
      \varepsilon_{xc}^{\text{LDA}}[n] = -\frac{3}{4}
      \left(\frac{3}{\pi}\right)^{1/3} n(\mathbf{r})^{1/3}

  where:

  - :math:`n(\mathbf{r})` is the electron density
  - :math:`\varepsilon_{xc}^{\text{LDA}}` is the exchange-correlation energy density

  Args:
    density_grid (Float[Array, '*d x y z']): Real-space electron density.
      May include batch dimensions.

  Returns:
    Float[Array, '*d x y z']: LDA exchange-correlation energy density.
      Preserves any batch dimensions from the input.
  """
  t3 = 3**(0.1e1 / 0.3e1)
  t4 = jnp.pi**(0.1e1 / 0.3e1)
  t8 = 2.220446049250313e-16**(0.1e1 / 0.3e1)
  t10 = jnp.where(0.1e1 <= 2.22044604925e-16, t8 * 2.22044604925e-16, 1)
  t11 = density_grid**(0.1e1 / 0.3e1)
  t15 = jnp.where(
    density_grid / 0.2e1 <= 1e-15, 0, -0.3e1 / 0.8e1 * t3 / t4 * t10 * t11
  )
  res = 0.2e1 * 1. * t15
  return res


def vxc_gga_recp(exc, rho_r, rho_r_grad_norm, gs):
  rho_G = jnp.fft.fftn(rho_r)
  rho_r_flat = rho_r.reshape(-1)
  rho_r_grad_norm_flat = rho_r_grad_norm.reshape(-1)
  dexc_drho_flat = jax.vmap(jax.grad(exc))(rho_r_flat, rho_r_grad_norm_flat)
  dexc_drho_grad_norm_flat = jax.vmap(jax.grad(exc, argnums=1)
                                     )(rho_r_flat, rho_r_grad_norm_flat)
  lapl_rho_G = -1 * (gs**2).sum(-1) * rho_G
  lapl_rho_r = jnp.fft.ifftn(lapl_rho_G)

  grid_sizes = gs.shape[:-1]
  t = dexc_drho_grad_norm_flat.reshape(grid_sizes) * rho_r
  t = jnp.where(rho_r_grad_norm > 0, t, 0)
  # # HACK
  # t = t.at[0, 0, 0].set(0)

  g_axes = list(range(gs.ndim - 1))

  t1 = dexc_drho_flat.reshape(grid_sizes) * rho_r
  t2 = exc(rho_r, rho_r_grad_norm)
  t3 = (
    jnp.fft.ifftn(1j * gs * jnp.fft.fftn(t)[..., None], axes=g_axes) *
    jnp.fft.ifftn(1j * gs * rho_G[..., None], axes=g_axes)
  ).sum(-1)
  t4 = t * lapl_rho_r
  integrand = t1 + t2 - 2 * (t3 + t4)
  vxc_G = jnp.fft.fftn(integrand)
  return vxc_G


def xc_lda(density_grid: Float[Array, 'x y z'],
           kohn_sham: bool = False) -> Float[Array, 'x y z']:
  r"""Calculate the LDA exchange-correlation potential.

  Implements the Local Density Approximation (LDA) for the exchange-correlation
  potential in the spin-unpolarized case. The potential is obtained as the
  functional derivative of the LDA energy:

  .. math::
      v_{xc}^{\text{LDA}}(\mathbf{r}) = -\left(\frac{3\rho(\mathbf{r})}{\pi}
      \right)^{1/3}

  where:

  - :math:`\rho(\mathbf{r})` is the electron density
  - :math:`v_{xc}^{\text{LDA}}` is the exchange-correlation potential

  Args:
    density_grid (Float[Array, 'x y z']): Real-space electron density.
    kohn_sham (bool, optional): If True, use Kohn-Sham formalism. Defaults to False.

  Returns:
    Float[Array, 'x y z']: LDA exchange-correlation potential.
  """
  dim = density_grid.ndim
  if dim > 3:
    density_grid = jnp.sum(density_grid, axis=range(0, dim - 3))

  if kohn_sham:
    density_grid = stop_gradient(density_grid)
    output = -(density_grid * 3. / jnp.pi)**(1 / 3)
  else:
    return _lda_x(density_grid)

  return output


def _pbe_x(rho_r, rho_r_grad_norm):
  """wrapper to low level API of jax_xc"""
  p = get_p("gga_x_pbe", 1)
  grid_shape = rho_r.shape
  rho_r_flat = rho_r.reshape(-1)
  rho_r_grad_norm_flat = rho_r_grad_norm.reshape(-1)
  out = jax.vmap(pbe_jac_xc, (None, 0, 0),
                 0)(p, rho_r_flat, rho_r_grad_norm_flat)
  return out.reshape(grid_shape)

  # kappa, mu = 0.804, 0.21951
  # kf = (3 * jnp.pi**2 * rho_r)**(1 / 3)
  # # reduced_density_gradient
  # div_kf = jnp.where(kf > 0, 1 / kf, 0)
  # drho = jnp.sqrt(rho_r_grad_norm)
  # s = jnp.where(rho_r > 0, drho * div_kf / 2 / rho_r, 0)
  # # enhancement factor
  # e_f = 1 + kappa - kappa / (1 + mu * s**2 / kappa)
  # return _lda_x(rho_r) * e_f


def rho_r_grad_norm_fn(density_grid, gs):
  rho_G = jnp.fft.fftn(density_grid)
  grad_rho_G = rho_G[..., None] * 1j * gs
  grad_rho_r = jnp.fft.ifftn(grad_rho_G, axes=(0, 1, 2))
  return absolute_square(grad_rho_r).sum(-1)


def xc_pbe(
  density_grid: Float[Array, 'x y z'],
  g_vector_grid,
  kohn_sham: bool = False
) -> Float[Array, 'x y z']:
  """
  Args:
    kohn_sham: if True, calculate vxc
  """
  dim = density_grid.ndim
  if dim > 3:
    density_grid = jnp.sum(density_grid, axis=range(0, dim - 3))

  grad_norm = rho_r_grad_norm_fn(density_grid, g_vector_grid)

  if kohn_sham:  # return vxc
    # NOTE: v_eff can be calculated in reciprocal space
    vxc_G = vxc_gga_recp(_pbe_x, density_grid, grad_norm, g_vector_grid)
    vxc_r = jnp.fft.ifftn(vxc_G)
    return vxc_r
  else:
    return _pbe_x(density_grid, grad_norm)


def xc_density(
  density_grid: Float[Array, 'x y z'],
  g_vector_grid,
  kohn_sham: bool = False,
  xc: str = "lda_x"
):
  # TODO: support lda_c etc.
  if xc == 'lda_x':
    return xc_lda(density_grid, kohn_sham)
  elif xc == 'pbe':
    return xc_pbe(density_grid, g_vector_grid, kohn_sham)
  else:
    raise NotImplementedError(f"XC {xc} not supported.")


def effective(
  density_grid: Float[Array, 'x y z'],
  position: Float[Array, "num_atom 3"],
  charge: Float[Array, "num_atom"],
  g_vector_grid: Float[Array, 'x y z 3'],
  vol: Float,
  split: bool = False,
  xc: str = "lda",
  kohn_sham: bool = False,
  spin_restricted: bool = True,
) -> Union[Tuple[
  Float[Array, 'x y z'], Float[Array, 'x y z'], Float[Array, 'x y z']],
           Float[Array, 'x y z']]:
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
    density_grid (Float[Array, 'x y z']): Real-space electron density.
    position (Float[Array, "num_atom 3"]): Atomic positions in the unit cell.
    charge (Float[Array, "num_atom"]): Nuclear charges.
    g_vector_grid (Float[Array, 'x y z 3']): Grid of G-vectors in reciprocal space.
    vol (Float): Volume of the unit cell.
    split (bool, optional): If True, return individual potential components.
      Defaults to False.
    xc (str, optional): Exchange-correlation functional type. Only "lda" is
      currently supported. Defaults to "lda".
    kohn_sham (bool, optional): If True, use Kohn-Sham formalism. Defaults to False.
    spin_restricted (bool, optional): If True, use spin-restricted calculation.
      Defaults to True.

  Returns:
    Union[Tuple[Array, Array, Array], Array]: If split is True, returns (Hartree, external, exchange-correlation) potentials. Otherwise returns their sum as the total effective potential.
  """
  dim = position.shape[-1]

  assert density_grid.ndim in [dim, dim + 1]  # w/w\o spin channel

  grid_size = g_vector_grid.shape[0]
  # make sure for both spin-polarized (unpolarized) version the shape is
  # consistent
  density_grid = density_grid.reshape(-1, grid_size, grid_size, grid_size)
  density_grid_reciprocal = jnp.fft.fftn(density_grid, axes=range(-dim, 0))
  # reciprocal space:
  v_hartree = hartree_reciprocal(
    jnp.sum(density_grid_reciprocal, axis=0), g_vector_grid, kohn_sham
  )
  v_external = external_reciprocal(position, charge, g_vector_grid, vol)

  # real space:
  if spin_restricted:
    v_xc = xc_density(density_grid, g_vector_grid, kohn_sham, xc)
  else:
    v_xp = xc_density(2 * density_grid[0], g_vector_grid, kohn_sham, xc)
    v_xn = xc_density(2 * density_grid[1], g_vector_grid, kohn_sham, xc)
    v_xc = jnp.vstack([[v_xp, v_xn]])

  # transform to real space
  v_hartree = jnp.fft.ifftn(v_hartree, axes=range(-dim, 0))
  v_external = jnp.fft.ifftn(v_external, axes=range(-dim, 0))

  if split:
    return v_hartree, v_external, v_xc
  else:
    return v_hartree[None, ...] + v_external[None, ...] + v_xc
