"""Planewave module."""
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Complex, Float

from .grid import g_vectors
from .typing import OccupationArray, ScalarGrid
from .unitary_module import unitary_matrix, unitary_matrix_param_init
from .utils import absolute_square, reshape_coefficient, volume


def param_init(
  key,
  num_bands: int,
  num_kpts: int,
  freq_mask: ScalarGrid[Bool, 3],
  restricted: bool = True,
):
  r"""Initialize the raw parameters.

    In planewave based calculation, a wave function is represented as a
    linear combination of the fourier series in 3D. Therefore to create one
    wave function we need a 3D shaped tensor to represent the mixing
    coefficients on each of the frequency component (denoted as `G`).
    `freq_mask` provides a 3D mask to decide which frequency components
    are selected, the number of selected components is denoted as `num_g`.

    The `num_bands` & `num_kpts` is a bit hard to explain, intuitively,
    the wave functions consists of high frequency components that has a period
    smaller than the unit cell (denoted $G$) and components that has a period
    larger than the unit cell (denoted $k$). However, with the contraints that
    the resulting system is periodic, we don't need to linearly combine over all
    the components, but instead just linearly combine over the $G$ components
    with different coefficients for each $k$.

    Extension reads:
    1. Why and how to mask the frequency components.
    2. Bloch theorum.

    As far as this function is concerned, it simply just returns a randomly
    initialized parameter of shape `(num_spin, num_kpts, num_g, num_bands)`.
    The input arguments to this function is only used to determine the shape.

    Note that this function returns the raw parameter that can not be used
    directly to weight the frequency components, as in quantum chemistry we
    require the wave functions to be orthogonal to each other.
    check :py:func:`coeff` for converting the raw parameter into an unitary
    tensor.

    Args:
      key: ranodm key for initializing the parameters
      num_bands: the number of bands.
      num_kpts: the number of k points.
      freq_mask: a 3D mask that denotes which frequency components are selected.
      restricted: if `True`, `num_spin=2` else `num_spin=1`.

    Returns:
      A complex type raw parameter of shape
      `(num_spin, num_kpts, num_g, num_bands)`.
    """
  num_spin = 1 if restricted else 2
  num_g = np.sum(freq_mask).item()
  shape = (num_spin, num_kpts, num_g, num_bands)
  return unitary_matrix_param_init(key, shape, complex=True)


def coeff(
  pw_param: Union[Array, Tuple], freq_mask: ScalarGrid[Bool, 3]
) -> Complex[Array, "spin kpts band n1 n2 n3"]:
  r"""Create the linear coefficients to combine the frequency components.

    The `pw_param` should be created from :py:func:`param_init`, and the same
    `freq_mask` used in :py:func:`param_init` should be used here. As mentioned
    in :py:func:`param_init`, we use linear combination over 3D fourier
    components for creating wave functions. Some extra requirements are

    1. The wave functions that has the same spin and same k component needs
       to be orthogonal to each other.
    2. We only activate some of the frequency components with the `freq_mask`.

    As the raw parameter returned from :py:func:`param_init` has the shape
    `(num_spin, num_kpts, num_g, num_bands)`, where `num_g` is the number of
    activated frequencies flattend from the activated entries in the `freq_mask`
    This function first orthogonalize over the last two dimensions and
    reorganize the orthogonalized parameter into a 3D grid the same shape of
    the frequency mask.

    Extension reads:
    1. Why and how to mask the frequency components.
    2. Bloch theorum.

    Args:
      pw_param: the raw parameter, maybe created from
      :py:func:`param_init`.
      freq_mask: a 3D mask to select the frequency components.

    Returns:
      Complex array of shape `(num_spin, num_kpts, num_band, n1, n2, n3)`.
      It satisfies the unitary constraint that for any `i,j`
      `einsum('kabc,labc->kl', ret[i, j], ret[i, j])` is an identity matrix.
    """
  coeff = unitary_matrix(pw_param, complex=True)
  coeff = jnp.swapaxes(coeff, -1, -2)
  return reshape_coefficient(coeff, freq_mask)


def wave_grid(
  coeff: Complex[Array, "spin kpts band n1 n2 n3"],
  vol: Union[float, Array],
):
  r"""Wave function evaluated at a grid of spacial locations.

    Our wave functions lives in the 3D space, and we use linear combination of
    3D fourier components to parameterize them, the parameters are the
    linear coeffcients. A single wave function look like

    $$
    \psi(r)=\frac{1}{\sqrt{V}} \sum_G c_{G} e^{iG^\top r}
    $$

    $G$ is the 3D frequency components, $V$ is the volume of the crystal
    unit cell, which is to make sure the wave function is normalized within
    the cell.

    where $c$ is the linear coefficient. It combines over different $G$
    components that is generated with :py:func:`jrystal.grid.g_vectors`.
    We can evaluate the wave function at any spatial location $r$ which takes
    $O(|G|)$ computation. However, if we evaluate this function on a specific
    spatial grid of size $|G|$, we can be faster than $O(|G|^2)$ by using
    fourier transform. IFFT gives us an $O(|G|\log(|G|))$ implementation of the
    above equation. The $G$ and $R$ grid can be obtained from
    :py:func:`jrystal.grid.g_vectors` and :py:func:`jrystal.grid.r_vectors`
    correspondingly.

    ```python
    G = jrystal.grid.g_vectors(*args)  # (n1, n2, n3, 3)
    R = jrystal.grid.r_vectors(*args)  # (n1, n2, n3, 3)
    coefficients = ...  # (n1, n2, n3)
    vol = ...

    def wave_function(r):
      return (coefficients * jnp.exp(1j * G @ r)).sum() / jnp.sqrt(vol)

    # The following is O(|G|^2)
    wave_at_R_naive = jax.vmap(jax.vmap(jax.vmap(wave_function)))(R)
    # The following is O(|G|log|G|)
    wave_at_R_fft = wave_grid(coefficients, vol)
    ```

    As IFFT implements

    $$
    x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k e^{i\frac{2\pi}{N}kn}
    $$

    It is a bit different from the definition of the wave function,
    if you check the code, we do two things to align them,

    1. we multiply back the $N$ to cancel the $\frac{1}{N}$
    factor in the IFFT (in 3D the `np.prod(grid_sizes)`).
    2. we divide by the $\sqrt{V}$.

    The `coeff` passed to this function has shape `(..., n1, n2, n3)`,
    it can have any leading dimension.
    It can be created using :py:func:`param_init` and :py:func:`coeff`.
    :py:func:`param_init` creates a raw parameter and :py:func:`coeff` converts
    that parameter into coefficients that are used to linearly weight the 3D
    fourier components.

    Args:
      coeff: linear combination coefficients over the 3D fourier components.
        shape is `(..., n1, n2, n3)` where `(n1, n2, n3)` is the shape of the
        3D frequency components generated
        from :py:func:`jrystal.grid.g_vectors`.
      vol: volume of the unit cell.
    Returns:
      The result of evaluating the wave function on the specific spatial grid.
    """
  grid_sizes = coeff.shape[-3:]
  wave_grid = jnp.fft.ifftn(coeff, axes=range(-3, 0))
  wave_grid *= np.prod(grid_sizes) / jnp.sqrt(vol)
  return wave_grid


def density_grid(
  coeff: Complex[Array, "spin kpts band n1 n2 n3"],
  vol: Union[float, Array],
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
  vol: Union[float, Array],
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
