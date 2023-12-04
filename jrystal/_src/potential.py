"""Potential modules."""
import jax.numpy as jnp
from jaxtyping import Float, Array

from jrystal._src.jrystal_typing import ComplexGrid, RealGrid, RealVecterGrid


def hartree_reciprocal(
  reciprocal_density_grid: ComplexGrid,
  g_vector_grid: RealVecterGrid,
) -> ComplexGrid:
  r"""Hartree potential for planewaves on reciprocal space.

  .. math::
    V = 4 \pi \sum_i \sum_k \sum_G \dfrac{n(G)}{\|G\|^2}

  Args:
    reciprocal_density_grid (ComplexGrid): the density of grid points in
      reciprocal space.
    g_vector_grid (RealVecterGrid): G vector grid.

  Returns:
    ComplexGrid: Hartree potential evaluated at reciprocal grid points

  """
  dim = g_vector_grid.shape[-1]
  if reciprocal_density_grid.ndim == dim + 1:  # have spin channel
    reciprocal_density_grid = jnp.sum(reciprocal_density_grid, axis=0)

  g_vec_square = jnp.sum(g_vector_grid**2, axis=-1)  # [N1, N2, N3]
  g_vec_square = g_vec_square.at[(0,) * dim].set(1e-16)
  output = reciprocal_density_grid / g_vec_square
  output = output.at[(0,) * dim].set(0)
  output = output * 4 * jnp.pi
  return output


def externel_reciprocal(
  positions: Float[Array, 'num_atoms d'],
  charges: Float[Array, 'num_atoms'],
  g_vector_grid: RealVecterGrid,
  vol: Float[Array, '']
) -> ComplexGrid:
  r"""
    Externel potential.

    .. math::
        V = \sum_G \sum_i s_i(G) v_i(G)

    where

    .. math::
        s_i(G) = exp(jG\tau_i)
        v_i(G) = -4 \pi z_i / \Vert G \Vert^2

    Args:
      positions (Array): Coordinates of atoms in a unit cell.
        Shape: [num_atoms d].
      charges (Array): Charges of atoms. Shape: [num_atoms].
      g_vector_grid (RealVecterGrid): G vector grid.
      vol (RealScalar): the volume of unit cell.

    Returns:
        ComplexGrid: external potential evaluated at reciprocal grid points

  """
  dim = positions.shape[-1]
  g_norm_square_grid = jnp.sum(g_vector_grid**2, axis=-1)
  si = jnp.exp(1.j * jnp.matmul(g_vector_grid, positions.transpose()))
  num_grids = jnp.prod(jnp.array(g_vector_grid.shape[:-1]))

  charges = jnp.expand_dims(charges, range(3))
  g_norm_square_grid = jnp.expand_dims(g_norm_square_grid, -1)
  vi = charges / g_norm_square_grid
  vi = vi.at[(0,) * dim].set(0)
  vi *= 4 * jnp.pi

  output = jnp.sum(vi * si, axis=-1)
  return -output * num_grids / vol


def xc_lda(density_grid: RealGrid) -> RealGrid:
  r"""local density approximation potential.

  NOTE: this is a non-polarized lda potential

  .. math::
    v_lda = - (3 * n(r) / \pi )^{\frac 1/3 }

  Args:
      density_grid (RealGrid): the density of grid points in
        real space.
      vol (RealScalar): the volume of unit cell.

  Returns:
      RealGrid: the variation of the lda energy with respect to the density.

  """

  if density_grid.ndim == 4:
    density_grid = jnp.sum(density_grid, axis=0)

  output = -(density_grid * 3. / jnp.pi)**(1 / 3)

  if density_grid.ndim == 4:
    output = jnp.expand_dims(output / 2, axis=0)

  return output / 2
