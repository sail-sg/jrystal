"""Energy functions. """

import numpy as np
import jax
import jax.numpy as jnp
import jax_xc

from jrystal._src.utils import complex_norm_square, vmapstack
from jrystal._src import potential
from jrystal._src import xc_density

from typing import Callable, Union
from jaxtyping import Float, Array
from jrystal._src.jrystal_typing import ComplexGrid, RealVecterGrid, RealScalar
from jrystal._src.jrystal_typing import RealGrid


def reciprocal_braket(
  potential_grids: ComplexGrid, density_grids: ComplexGrid, vol: RealScalar
) -> RealScalar:
  r"""This function calculate the inner product of <f|g> in reciprocal space

  .. math::
    <f|g> \approx \sum_g f^*(g)r(g) * vol / N / N

  where N is the number of grid size.

  NOTE: in this project, hartree and external energy integral is calculated
  in reciprocal space.

  Args:
      potential_grids (ComplexGrid): potential in reciprocal space.
      density_grids (ComplexGrid): density in reciprocal space.
      vol (RealScalar): the volume of unit cell.

  Returns:
      RealScalar: the value of the inner product.
  """
  if potential_grids.shape != density_grids.shape:
    raise ValueError(
      f"potential and density shape are not aligned. Got "
      f"{potential_grids.shape} and {density_grids.shape}."
    )

  num_grids = np.prod(np.array(potential_grids.shape))
  discretize_factor = vol / num_grids / num_grids
  product = jnp.sum(
    jnp.conj(potential_grids) * density_grids
  ) * discretize_factor

  return product.real


def real_braket(
  potential_grids: ComplexGrid, density_grids: ComplexGrid, vol: RealScalar
) -> RealScalar:
  r"""This function calculate the inner product of <f|g> in real space

  .. math::
    <f|g> \approx \sum_r f^*(r)r(r) * vol / N

  where N is the number of grid size.

  NOTE: in this project, exchange-correlation energy integral is calculated
  in real space.

  Args:
      potential_grids (ComplexGrid): potential in real space.
      density_grids (ComplexGrid): density in real space.
      vol (RealScalar): the volume of unit cell.

  Returns:
      RealScalar: the value of the inner product.
  """
  if potential_grids.shape != density_grids.shape:
    raise ValueError(
      f"potential and density shape are not aligned. Got "
      f"{potential_grids.shape} and {density_grids.shape}."
    )

  num_grids = np.prod(np.array(potential_grids.shape))
  discretize_factor = vol / num_grids
  product = jnp.sum(potential_grids * density_grids) * discretize_factor
  return product


def hartree(
  reciprocal_density_grid: ComplexGrid,
  g_vector_grid: RealVecterGrid,
  vol: RealScalar
) -> RealScalar:
  r"""Hartree energy for plane wave orbitals on reciprocal space.

  .. math::
    E = 2\pi \sum_i \sum_k \sum_G \dfrac{n(G)^2}{\|G\|^2}

  Args:
    reciprocal_density_grid (nd-array): the density of grid points in
      reciprocal space. Shape: [_, N1, N2, N3]
    g_vector_grid (4D array): g vector grid. Shape: [N1, N2, N3, 3]
    vol: scalar

  Return:
    Hartree energy: float.
  """
  dim = g_vector_grid.shape[-1]

  if reciprocal_density_grid.ndim == dim + 1:
    reciprocal_density_grid = jnp.sum(reciprocal_density_grid, axis=0)

  v_hartree_reciprocal = potential.hartree_reciprocal(
    reciprocal_density_grid, g_vector_grid
  )

  hartree_energy = reciprocal_braket(
    v_hartree_reciprocal, reciprocal_density_grid, vol
  ) / 2

  return hartree_energy.real


def external(
  reciprocal_density_grid: ComplexGrid,
  positions: Float[Array, 'num_atoms d'],
  charges: Float[Array, 'num_atoms'],
  g_vector_grid: RealVecterGrid,
  vol: RealScalar
) -> RealScalar:
  r"""

    Externel energy for plane waves

    .. math::
        V = \sum_G \sum_i s_i(G) v_i(G)
        E = \int V(r) \rho(r) dr

    where

    .. math::
        s_i(G) = exp(jG\tau_i)
        v_i(G) = -4 \pi z_i / \Vert G \Vert^2

    Args:
      reciprocal_density_grid (ComplexGrid): the density of grid points in
        reciprocal space.
      positions (Array): Coordinates of atoms in a unit cell.
        Shape: [num_atoms d].
      charges (Array): Charges of atoms. Shape: [num_atoms].
      g_vector_grid (RealVecterGrid): G vector grid.

    Return:
      RealScalar: External energy.

  """
  dim = g_vector_grid.shape[-1]
  if reciprocal_density_grid.ndim == dim + 1:
    reciprocal_density_grid = jnp.sum(reciprocal_density_grid, axis=0)

  v_externel_reciprocal = potential.externel_reciprocal(
    positions, charges, g_vector_grid, vol
  )
  externel_energy = reciprocal_braket(
    v_externel_reciprocal, reciprocal_density_grid, vol
  )

  return externel_energy.real


def kinetic(
  g_vector_grid: RealVecterGrid,
  k_vector_grid: RealVecterGrid,
  coeff_grid: ComplexGrid,
  occupation=None
) -> Union[RealScalar, Float[Array, "num_spin num_k num_bands"]]:
  r"""Kinetic energy.

  .. math::
      E = 1/2 \sum_{G} |k + G|^2 c_{i,k,G}^2

  Args:
      g_vector_grid (RealVecterGrid):  G vector grid.
      k_vector_grid (RealVecterGrid):  k vector grid.
      coeff_grid (ComplexGrid): Plane wave coefficient.
      occupation (Array, optional): occupation array. If provided, then the
        function will be reduced by applying occupation number. If not provided,
        then the function will return the kinetic energy of all the orbitals.

  Returns:
      RealScalar or Float[Array, "num_spin num_k num_bands"]]: kinetic energy.

  """

  dim = g_vector_grid.shape[-1]

  _g = jnp.expand_dims(g_vector_grid, axis=range(3))
  _k = jnp.expand_dims(
    k_vector_grid, axis=[0] + [i + 2 for i in range(dim + 1)]
  )
  e_kin = jnp.sum((_g + _k)**2, axis=-1)  # [1, nk, ni, N1, N2, N3]
  e_kin = jnp.sum(
    e_kin * complex_norm_square(coeff_grid), axis=range(3, dim + 3)
  )

  if occupation is not None:
    e_kin = jnp.sum(e_kin * occupation) / 2
  else:
    e_kin /= 2

  return e_kin


def xc(
  density_fn: Callable,
  r_vector_grid: RealVecterGrid,
  vol: RealScalar,
  xc: str = 'lda_x'
) -> Float:
  epsilon_xc = getattr(jax_xc, xc, None)
  if epsilon_xc:
    epsilon_xc = epsilon_xc(polarized=True)

  num_grid = jnp.prod(jnp.array(r_vector_grid.shape))
  map_dim = r_vector_grid.ndim - 1

  def _integrad(r):
    return epsilon_xc(density_fn, r) * density_fn(r)

  _eps_den_grid = vmapstack(map_dim)(_integrad)(r_vector_grid)
  return jnp.sum(_eps_den_grid) * vol / num_grid


def xc_lda(density_grid: RealGrid, vol: RealScalar):
  r"""local density approximation potential.

  NOTE: this is a non-polarized lda potential

  .. math::
      E_{\rm x}^{\mathrm{LDA}}[\rho] = - \frac{3}{4}\left( \frac{3}{\pi} \right)^{1/3}\int\rho(\mathbf{r})^{4/3}  # noqa: E501

  Args:
      density_grid (RealGrid): the density of grid points in
        real space.
      vol (RealScalar): the volume of unit cell.

  Returns:
      RealGrid: the variation of the lda energy with respect to the density.

  """

  if density_grid.ndim == 4:  # have spin channel
    density_grid = jnp.sum(density_grid, axis=0)

  num_grid = jnp.prod(jnp.array(density_grid.shape))
  e_lda = jnp.sum(xc_density.lda_x(density_grid) * density_grid)

  return e_lda * vol / num_grid


def ewald_coulomb_repulsion(
  positions: Float[Array, 'num_atoms d'],
  charges: Float[Array, 'num_atoms'],
  g_vector_grid: RealVecterGrid,
  vol: RealScalar,
  ewald_eta: float,
  ewald_grid: Float[Array, 'num_translations d']
) -> RealScalar:
  """
  Ewald summation.

  Ref: Martin, R. M. (2020). Electronic structure: basic theory and practical
  methods. Cambridge university press. (Appendix F.2)


  Args:
      positions (Float[Array, &#39;num_atoms d&#39;]): _description_
      charges (Float[Array, &#39;num_atoms&#39;]): _description_
      cell_vector (Float[Array, &#39;d d&#39;]): _description_
      reciprocal_density_grid (ComplexGrid): _description_
      vol (RealScalar): _description_
      ewald_eta (float): _description_
      ewald_grid (RealGrid): a grid for ewald sum.
        Can be generated by jrystal._src.grid.translation_vectors

  Returns:
      RealScalar: ewald sum of coulomb repulsion energy.

  """
  dim = positions.shape[-1]

  tau = jnp.expand_dims(positions, 0) - jnp.expand_dims(positions, 1)
  tau_t = jnp.expand_dims(tau, 2) - jnp.expand_dims(ewald_grid, axis=(0, 1))
  # [na, na, nt, 3]
  tau_t_norm = jnp.sqrt(jnp.sum(tau_t**2, axis=-1) + 1e-20)  # [na, na, nt]
  tau_t_norm = jnp.where(tau_t_norm <= 1e-9, 1e20, tau_t_norm)

  #  atom-atom part:
  ew_ovlp = jnp.sum(
    jax.scipy.special.erfc(ewald_eta * tau_t_norm) / tau_t_norm, axis=2
  )

  # the reciprocal space part:
  gvec_norm_sq = jnp.sum(g_vector_grid**2, axis=3)  # [N1, N2, N3]
  gvec_norm_sq = gvec_norm_sq.at[(0,) * dim].set(1e16)

  ew_rprcl = jnp.exp(-gvec_norm_sq / 4 / ewald_eta**2) / gvec_norm_sq
  ew_rprcl1 = jnp.expand_dims(ew_rprcl, range(3, 3 + dim))
  ew_rprcl2 = jnp.cos(
    jnp.expand_dims(g_vector_grid, axis=(-2, -3)) *
    jnp.expand_dims(tau, range(dim))
  )
  ew_rprcl = jnp.sum(ew_rprcl1 * ew_rprcl2, axis=-1)  # [N1, N2, N3, na, na]

  ew_rprcl = jnp.sum(
    ew_rprcl.at[(0,) * dim].set(0), axis=range(dim)
  )  # [na, na]
  ew_rprcl = ew_rprcl * 4 * jnp.pi / vol
  ew_aa = jnp.einsum('i,ij->j', charges, ew_ovlp + ew_rprcl)
  ew_aa = jnp.dot(ew_aa, charges) / 2

  # single atom part
  ew_a = -jnp.sum(charges**2) * 2 * ewald_eta / jnp.sqrt(jnp.pi) / 2
  ew_a -= jnp.sum(charges)**2 * jnp.pi / ewald_eta**2 / vol / 2

  return ew_aa + ew_a
