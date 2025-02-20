"""Potentials."""
from typing import Tuple, Union

import jax.numpy as jnp
from jax.lax import stop_gradient
from jaxtyping import Array, Complex, Float

from .typing import ScalarGrid, VectorGrid


def hartree_reciprocal(
  density_grid_reciprocal: ScalarGrid[Complex, 3],
  g_vector_grid: VectorGrid[Float, 3],
  kohn_sham: bool = False
) -> ScalarGrid[Complex, 3]:
  r"""
  Compute the Hartree potential in reciprocal space.

  This function calculates the Hartree potential evaluated at reciprocal grid
  from the electron density in reciprocal space and reciprocal grid vectors.

  The Hartree potential induced by the electron density $n(r)$ follows
  the following Poisson equation:

  .. math::
    \Delta V(r) = -4 \pi n(r)

  with periodic boundary condition. Applying Fourier transform to both sides,
  one obtains their relation in reciprocal space:

  .. math::
    \hat{V}(G) = 4 \pi \dfrac{\hat{n}(G)}{\|G\|^2},
    \hat{V}(0) = 0.

  TODO: we may need to explain kohn_sham?

  Args:
    density_grid_reciprocal (ScalarGrid[Complex, 3]): the electron density on
    reciprocal space lattice.
    g_vector_grid (VectorGrid[Float, 3]): reciprocal lattice vector.
    kohn_sham (bool, optional): If True, use Kohn-Sham potential. Defaults to
    False.

  Returns:
    ScalarGrid[Complex, 3]: Hartree potential evaluated at reciprocal lattice
    vector. If density_grid_reciprocal has batch axes, the output will
    keep them.

  """
  dim = g_vector_grid.shape[-1]
  g_vec_square = jnp.sum(g_vector_grid**2, axis=-1)  # [N1, N2, N3]
  g_vec_square = g_vec_square.at[(0,) * dim].set(1e-16)

  if kohn_sham:
    density_grid_reciprocal = stop_gradient(density_grid_reciprocal)

  output = density_grid_reciprocal / g_vec_square
  output = output.at[(0,) * dim].set(0)
  output = output * 4 * jnp.pi

  if not kohn_sham:
    output /= 2

  return output


def hartree(
  density_grid_reciprocal: ScalarGrid[Complex, 3],
  g_vector_grid: VectorGrid[Float, 3],
  kohn_sham: bool = False
) -> ScalarGrid[Complex, 3]:
  r"""
  Compute the Hartree potential in real space.

  This function applies inverse Fourier transform to the output of
  :py:func:`hartree_reciprocal` to get the Hartree potential evaluated at real
  space grid points.

  Args:
    density_grid_reciprocal (ScalarGrid[Complex, 3]): the electron density on
    reciprocal space lattice.
    g_vector_grid (VectorGrid[Float, 3]): reciprocal lattice vector.
    kohn_sham (bool, optional): If True, use Kohn-Sham potential. Defaults to
    False.


  Returns:
    ScalarGrid[Complex, 3]: Hartree potential evaluated at real space grid
    points. If density_grid_reciprocal has batch axes, the output will
    keep them.

  """
  har_pot_grid_rcprl = hartree_reciprocal(
    density_grid_reciprocal, g_vector_grid, kohn_sham
  )
  return jnp.fft.ifftn(har_pot_grid_rcprl, axes=range(-3, 0))


def external_reciprocal(
  position: Float[Array, 'num_atoms 3'],
  charge: Float[Array, 'num_atoms'],
  g_vector_grid: VectorGrid[Float, 3],
  vol: Float,
) -> ScalarGrid[Complex, 3]:
  r"""
    Compute the external potential in reciprocal space.

    This function calculates the external potential evaluated at reciprocal
    grid from the nucleus position, charge, reciprocal lattice vectors, and
    crystal volume.

    The external potential induced by the nucleus charge can be evaluated in
    reciprocal space as:

    .. math::
        V(G) = \sum_k s_k(G) v_k(G)

    The summation is over all atoms in the unit cell. $s_k(G)$ and $v_k(G)$ are
    structure factor and form factor respectively and are defined as:

    .. math::
        s_k(G) = exp(iG\tau_k)
        v_k(G) = - \dfrac{4 \pi Z_k e^2}{\|G\|^2}

    where $\tau_k$ is the position of atom $k$ in the unit cell, $Z_k$ is the
    charge of atom $k$, $G$ is the reciprocal lattice vector, and $e$ is the
    unit charge of the electron.

    Args:
      position (VectorGrid[Float, 3]): Coordinates of atoms in a unit cell.
      charge (VectorGrid[Float, 1]): Charges of atoms.
      g_vector_grid (VectorGrid[Float, 3]): reciprocal lattice vector.
      vol (Float): the volume of unit cell.

    Returns:
        ScalarGrid[Complex, 3]: external potential evaluated at reciprocal
        grid points

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
  position: Float[Array, 'num_atoms 3'],
  charge: Float[Array, 'num_atoms'],
  g_vector_grid: VectorGrid[Float, 3],
  vol: Float,
) -> ScalarGrid[Complex, 3]:
  r"""
  Compute the external potential in real space.

  This function applies inverse Fourier transform to the output of
  :py:func:`external_reciprocal` to get the external potential evaluated at
  real space grid points.

  Args:
      position (VectorGrid[Float, 3]): Coordinates of atoms in a unit cell.
      charge (VectorGrid[Float, 1]): Charges of atoms.
      g_vector_grid (VectorGrid[Float, 3]): reciprocal lattice vector.
      vol (Float): the volume of unit cell.

  Returns:
      ScalarGrid[Complex, 3]: external potential evaluated at reciprocal
      grid points

  """
  ext_pot_grid_rcprl = external_reciprocal(position, charge, g_vector_grid, vol)
  return jnp.fft.ifftn(ext_pot_grid_rcprl, axes=range(-3, 0))


def lda_density(density_grid: ScalarGrid[Float, 3]) -> ScalarGrid[Float, 3]:
  """Compute the energy density of the LDA exchange-correlation functional.

  .. math::
    \epsilon_{xc} = -\dfrac{3}{4} \left(\dfrac{3}{\pi}\right)^{1/3} n(r)^{1/3}

  Args:
      density_grid (ScalarGrid[Float, 3]): the electron density on real space
      grid.

  NOTE: I do not understand why the code is written in this way.

  Returns:
    ScalarGrid[Float, 3]: lda energy density on real space grid
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


def xc_lda(
  density_grid: ScalarGrid[Float, 3],
  kohn_sham: bool = False
) -> ScalarGrid[Float, 3]:
  r"""local density approximation potential.

  See Eq. (7.4.9) Robert G. Parr, Yang Weitao 1994

  NOTE: this is a non-polarized lda potential

  .. math::
    v_lda = - (3 * n(r) / \pi )^{\frac 1/3 }

  Args:
      density_grid (ScalarGrid[Float, 3]): the electron density on real space
      grid.
      vol (Float): the volume of unit cell.

  Returns:
      ScalarGrid[Float, 3]: the variation of the lda energy with respect to
      the density.

  """
  dim = density_grid.ndim
  if dim > 3:
    density_grid = jnp.sum(density_grid, axis=range(0, dim - 3))

  if kohn_sham:
    output = -(density_grid * 3. / jnp.pi)**(1 / 3)
  else:
    return lda_density(density_grid)

  return output


def effective(
  density_grid: ScalarGrid[Float, 3],
  position: Float[Array, "num_atom 3"],
  charge: Float[Array, "num_atom"],
  g_vector_grid: VectorGrid[Float, 3],
  vol: Float,
  split: bool = False,
  xc: str = "lda",
  kohn_sham: bool = False
) -> Union[Tuple[Array, Array, Array], Array]:
  """
  Compute the effective potentials in real space.

  .. math::
    \hat V|\psi>
    \hat V_eff = \hat V_\{hartree} + \hat V_\{ext} + \hat V_\{xc}

  Args:
    density_grid (ScalarGrid[Float, 3]): The Hamiltonian density grid.
    position (Array): Atomic positions.
    charge (Array): Atomic charges.
    g_vector_grid (VectorGrid[Float, 3]): The G vector grid.
    vol (Float): Volume of the system.
    split (bool): If True, return split potentials
    [V_hartree, V_external, V_xc], else return the sum. Defaults to False.
    xc (str): Exchange-correlation functional. Defaults to "lda".
    kohn_sham (bool): If True, use Kohn-Sham potential. Defaults to False.


  Returns:
    Tuple[Array, Array, Array] | Array : Hartree, external, and
      exchange-correlation potentials. If split is false, then will return
      the sum of hartree, external and xc potentials.

  Note:
    Now only support lda potential.

  TODO:
    report all xc potentials using jax-xc.
  """

  dim = position.shape[-1]

  assert density_grid.ndim in [dim, dim + 1]  # w/w\o spin channel
  if density_grid.ndim == dim + 1:
    density_grid = jnp.sum(density_grid, axis=0)

  density_grid_reciprocal = jnp.fft.fftn(density_grid, axes=range(-dim, 0))
  # reciprocal space:
  v_hartree = hartree_reciprocal(
    density_grid_reciprocal, g_vector_grid, kohn_sham
  )
  v_external = external_reciprocal(position, charge, g_vector_grid, vol)

  # real space:
  if xc.strip() in ["lda", "lda_x"]:
    v_xc = xc_lda(density_grid, kohn_sham)
  else:
    raise NotImplementedError("XC only support lda for now.")

  # transform to real space
  v_hartree = jnp.fft.ifftn(v_hartree, axes=range(-dim, 0))
  v_external = jnp.fft.ifftn(v_external, axes=range(-dim, 0))

  if split:
    return v_hartree, v_external, v_xc
  else:
    return v_hartree + v_external + v_xc
