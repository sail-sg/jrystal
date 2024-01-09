import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from .grid import r_vectors, g_vectors
import jax.core as core
import jax.extend.linear_util as lu
from .bloch import u

from autofd.operators import def_operator
from autofd.general_array import operator, function, Spec, Ret, Arg, Grid

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
    # TODO: we're reusing the u function here, but it is not optimal
    # because in the case of density which is a real function,
    # the rfft can be used for saving some computation/memory.
    # TODO: the division of np.prod(grid_sizes) will cancel with the
    # multiplication in the u function, but XLA may not cancel them cleanly.
    return jnp.real(
      u(
        cell_vectors,
        reciprocal_potential_grid,
        r,
        force_fft=force_fft,
      )
    ) / np.prod(grid_sizes)

  return potential


def hartree_potential_spec(density, *, cell_vectors, grid_sizes, force_fft):
  return density.shape, None


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

external_potential_p = core.Primitive("external_potential")


def external_potential(
  *,
  positions,
  charges,
  cell_vectors,
  grid_sizes,
  force_fft=True,
  real=True,
):
  return external_potential_p.bind(
    positions=positions,
    charges=charges,
    cell_vectors=cell_vectors,
    grid_sizes=grid_sizes,
    force_fft=force_fft,
    real=real,
  )


@operator
@lu.wrap_init
def external_potential_impl(
  *,
  positions,
  charges,
  cell_vectors,
  grid_sizes,
  force_fft,
  real,
):
  """External potential for plane waves.

  Args:
    positions (Array): Coordinates of atoms in a unit cell.
      Shape: [num_atoms d].
    charges (Array): Charges of atoms. Shape: [num_atoms].
    cell_vectors (Array): Vectors of unit cell.
      Shape: [d d].
    grid_sizes (Array): Number of grid points in each direction.
  """
  ndim = cell_vectors.shape[-1]
  g_vector_grid = g_vectors(cell_vectors, grid_sizes)
  g_vec_square = jnp.sum(g_vector_grid**2, axis=-1)
  si = jnp.exp(1.j * jnp.matmul(g_vector_grid, jnp.transpose(positions)))
  vi = charges / jnp.expand_dims(g_vec_square, -1)
  vi = vi.at[(0,) * ndim].set(0)
  vi *= 4 * jnp.pi
  vol = np.linalg.det(cell_vectors)
  reciprocal_potential_grid = jnp.sum(vi * si, axis=-1) / vol

  @function
  @lu.wrap_init
  def potential(r):
    ret = -u(
      cell_vectors,
      reciprocal_potential_grid,
      r,
      force_fft=force_fft,
    )
    return jnp.real(ret) if real else ret

  return potential


def external_potential_spec(
  *,
  positions,
  charges,
  cell_vectors,
  grid_sizes,
  force_fft,
  real,
):
  dtype = (
    positions.dtype
    if real else jnp.zeros((), dtype=positions.dtype).astype(complex).dtype
  )
  return (
    (Ret(Spec((), dtype=dtype)), Arg(Spec((3,), dtype=positions.dtype))), None
  )


def external_potential_grid(
  *,
  positions,
  charges,
  cell_vectors,
  grid_sizes,
  force_fft,
  real,
):
  r_vector_grid = r_vectors(cell_vectors, grid_sizes)
  vol = jnp.linalg.det(cell_vectors)
  nodes = jnp.reshape(r_vector_grid, (-1, 3))
  weights = vol / np.prod(grid_sizes)
  return Grid((nodes,), (weights,))


def_operator(
  external_potential_p,
  external_potential_impl,
  external_potential_spec,
  external_potential_grid,
)
