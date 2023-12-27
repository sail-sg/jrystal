import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from .grid import r_vectors, g_vectors
import jax.core as core
import jax.extend.linear_util as lu
from .bloch import u

from autofd.operators import def_operator
from autofd.general_array import operator, function, Spec, Ret

hartree_potential_p = core.Primitive("hartree_potential")


def hartree_potential(density, *, cell_vectors, grid_sizes, force_fft=True):
  return hartree_potential_p.bind(
    density,
    cell_vectors=cell_vectors,
    grid_sizes=grid_sizes,
    force_fft=force_fft,
  )


@operator
@lu.wrap_init
def hartree_potential_impl(density, *, cell_vectors, grid_sizes, force_fft):
  ndim = cell_vectors.shape[-1]
  g_vector_grid = g_vectors(cell_vectors, grid_sizes)
  r_vector_grid = r_vectors(cell_vectors, grid_sizes)
  g_vec_square = jnp.sum(g_vector_grid**2, axis=-1)
  g_vec_square = g_vec_square.at[(0,) * ndim].set(1e-16)

  r_vec = jnp.reshape(r_vector_grid, (-1, ndim))
  density_grid = jax.vmap(density)(r_vec).reshape(r_vector_grid.shape[:-1])
  reciprocal_density_grid = jnp.fft.fftn(density_grid)
  reciprocal_potential_grid = reciprocal_density_grid / g_vec_square
  reciprocal_potential_grid = (
    4 * jnp.pi * reciprocal_potential_grid.at[(0,) * ndim].set(0)
  )

  @function
  @lu.wrap_init
  def potential(r):
    return u(
      cell_vectors,
      reciprocal_potential_grid,
      r,
      force_fft=force_fft,
    ) / np.prod(grid_sizes)

  return potential


def hartree_potential_spec(density, *, cell_vectors, grid_sizes, force_fft):
  real_z = jnp.zeros((), dtype=density.ret.spec.dtype)
  dtype = real_z.astype(complex).dtype
  return (Ret(Spec((), dtype)), density.shape[1]), None


def hartree_potential_grid(density, *, cell_vectors, grid_sizes, force_fft):
  return density.grid


def_operator(
  hartree_potential_p,
  hartree_potential_impl,
  hartree_potential_spec,
  hartree_potential_grid,
)


def _hartree_potential_transpose_rule(
  t, density, *, cell_vectors, grid_sizes, force_fft
):
  assert ad.is_undefined_primal(density)
  return (
    hartree_potential_p.bind(
      t,
      cell_vectors=cell_vectors,
      grid_sizes=grid_sizes,
      force_fft=force_fft,
    ),
  )


jax.interpreters.ad.deflinear2(
  hartree_potential_p, _hartree_potential_transpose_rule
)
jax.interpreters.ad.primitive_transposes[hartree_potential_p] = (
  _hartree_potential_transpose_rule
)
