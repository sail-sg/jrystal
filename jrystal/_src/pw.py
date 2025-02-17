"""Planewave module."""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple
from jaxtyping import Array, Complex, Bool, Float

from .typing import OccupationArray, ScalarGrid
from .utils import reshape_coefficient, absolute_square, volume
from .unitary_module import unitary_matrix, unitary_matrix_param_init
from .grid import g_vectors


def param_init(
  key,
  num_bands: int,
  num_kpts: int,
  freq_mask: ScalarGrid[Bool, 3],
  restricted: bool = True,
):
  num_spin = 1 if restricted else 2
  num_g = np.sum(freq_mask).item()
  shape = (num_spin, num_kpts, num_g, num_bands)
  return unitary_matrix_param_init(key, shape, complex=True)


def coeff(
  pw_param: Array | Tuple, freq_mask: ScalarGrid[Bool, 3]
) -> Complex[Array, "spin kpts band n1 n2 n3"]:
  r"""Reshape the plane wave coefficients to the shape of the frequency mask.

  Args:
      pw_param (Array): plane wave coefficients.
      freq_mask (ScalarGrid[Bool, 3]): frequency mask.

  Returns:
      Complex[Array, "spin kpts band n1 n2 n3"]: reshaped plane wave
      coefficients.
  """
  coeff = unitary_matrix(pw_param, complex=True)
  coeff = jnp.swapaxes(coeff, -1, -2)

  return reshape_coefficient(coeff, freq_mask)


def wave_grid(
  coeff: Complex[Array, "spin kpts band n1 n2 n3"],
  vol: float | Array,
):
  r"""Construct the wave function in real space evaluated at fft grid from
  plane wave coefficients.

  Args:
      coeff (Complex[Array, "spin kpts band n1 n2 n3"]): plane wave
        coefficients.
      vol (float | Array): volume of the unit cell.

  Returns:
      Complex[Array, "spin kpts band n1 n2 n3"]: wave grid.
  """
  grid_sizes = coeff.shape[-3:]
  wave_grid = jnp.fft.ifftn(coeff, axes=range(-3, 0))
  wave_grid *= np.prod(grid_sizes) / jnp.sqrt(vol)
  return wave_grid


def density_grid(
  coeff: Complex[Array, "spin kpts band n1 n2 n3"],
  vol: float | Array,
  occupation: Optional[OccupationArray] = None
) -> ScalarGrid[Complex, 3]:
  r"""Construct the density grid from plane wave coefficients.

  Args:
      coeff (Complex[Array, "spin kpts band n1 n2 n3"]): plane wave
        coefficients.
      vol (float | Array): volume of the unit cell.
      occpation (Optional[OccupationArray]): occupation numbers.

  Returns:
      ScalarGrid[Complex, 3]: density grid.
  """
  wave_grid_arr = wave_grid(coeff, vol)
  dens = absolute_square(wave_grid_arr)

  if occupation is not None:
    try:
      occupation = jnp.expand_dims(occupation, range(-3, 0))
      dens = jnp.sum(dens * occupation, axis=range(3))
    except:
      raise ValueError(
        f"wave_grid's shape ({wave_grid.shape}) and occupation's shape "
        f"({occupation.shape}) cannot align."
      )
  return dens


def density_grid_reciprocal(
  coeff: Complex[Array, "spin kpts band n1 n2 n3"],
  vol: float | Array,
  occupation: Optional[OccupationArray] = None
) -> ScalarGrid[Complex, 3]:
  r"""Construct the density grid from plane wave coefficients in reciprocal
  space.

  Args:
      coeff (Complex[Array, "spin kpts band n1 n2 n3"]): plane wave
        coefficients.
      vol (float | Array): volume of the unit cell.
      occpation (Optional[OccupationArray]): occupation numbers.

  Returns:
      ScalarGrid[Complex, 3]: density grid.
  """
  dens = density_grid(coeff, vol, occupation)
  return jnp.fft.fftn(dens, axes=range(-3, 0))


def wave_r(
  r: Float[Array, "3"],
  coeff: Complex[Array, "spin kpts band n1 n2 n3"],
  cell_vectors: Float[Array, "3 3"],
  g_vector_grid: Optional[ScalarGrid[Float, 3]] = None,
) -> Complex[Array, "spin kpts band *b"]:
  """Evaluate plane wave functions at location r.

  .. math::
    \psi_i(r) = \sum_G c_{i, G} exp(iGr)

  Args:
    r (Array): locations. Shape: [*b 3].
    coeff (Array): plane wave coefficients, which is the output of the
      `pw_coeff` function. Shape: [spin kpts band n1 n2 n3].
    vol (float | Array): volume of the unit cell.

  Returns:
    Complex[Array, "spin kpts band"]: wave functions at location r.
  """

  vol = volume(cell_vectors)
  _, num_kpts, num_bands, n1, n2, n3 = coeff.shape

  if g_vector_grid is None:
    g_vector_grid = g_vectors(cell_vectors, [n1, n2, n3])

  gr = jnp.dot(g_vector_grid, r)
  output = jnp.exp(1j * gr)
  output = jnp.einsum("skbxyz,xyz->skb", coeff, output)
  return output / jnp.sqrt(vol)


def density_r(
  r: Float[Array, "3"],
  coeff: Complex[Array, "spin kpts band n1 n2 n3"],
  cell_vectors: Float[Array, "3 3"],
  g_vector_grid: Optional[ScalarGrid[Float, 3]] = None,
  occupation: Optional[OccupationArray] = None,
):
  density = absolute_square(wave_r(r, coeff, cell_vectors, g_vector_grid))
  if occupation is not None:
    density = jnp.sum(density * occupation)

  return density


def nabla_density_r(
  r: Float[Array, "3"],
  coeff: Complex[Array, "spin kpts band n1 n2 n3"],
  cell_vectors: Float[Array, "3 3"],
  g_vector_grid: Optional[ScalarGrid[Float, 3]] = None,
  occupation: Optional[OccupationArray] = None,
):

  def den(r):
    return density_r(r, coeff, cell_vectors, g_vector_grid, occupation)

  return jax.grad(den)(r)


def nabla_density_grid(
  r: Float[Array, "3"],
  coeff: Complex[Array, "spin kpts band n1 n2 n3"],
  cell_vectors: Float[Array, "3 3"],
  g_vector_grid: Optional[ScalarGrid[Float, 3]] = None,
  occupation: Optional[OccupationArray] = None,
) -> ScalarGrid[Float, 3]:
  r = jnp.reshape(r, (-1))

  if r.shape[0] != 3:
    raise ValueError("r must have shape (3,)")

  vol = volume(cell_vectors)
  n1, n2, n3 = coeff.shape[-3:]

  if g_vector_grid is None:
    g_vector_grid = g_vectors(cell_vectors, [n1, n2, n3])

  gr = jnp.dot(g_vector_grid, r)

  cosgr = jnp.cos(gr)
  singr = jnp.sin(gr)
  cr = jnp.real(coeff)
  ci = jnp.imag(coeff)

  rcos = jnp.einsum("skbxyz,xyz->skbxyz", cr, cosgr)
  rsin = jnp.einsum("skbxyz,xyz->skbxyz", cr, singr)
  icos = jnp.einsum("skbxyz,xyz->skbxyz", ci, cosgr)
  isin = jnp.einsum("skbxyz,xyz->skbxyz", ci, singr)

  o1 = jnp.sum(rcos - isin, axis=range(-3, 0))
  o1 = jnp.expand_dims(o1, -1)
  o2 = -jnp.einsum("skbxyz,xyzd->skbd", rsin + icos, g_vector_grid)
  o3 = jnp.sum(rsin + icos, axis=range(-3, 0))
  o3 = jnp.expand_dims(o3, -1)
  o4 = jnp.einsum("skbxyz,xyzd->skbd", rcos - isin, g_vector_grid)

  output = 2 * (o1 * o2 + o3 * o4)

  if occupation is not None:
    occupation = jnp.expand_dims(occupation, -1)
    output = jnp.sum(output * occupation, axis=range(0, 3)) / vol

  return output
