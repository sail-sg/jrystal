import jax
import jax.numpy as jnp
import flax.linen as nn

from jrystal._src.bloch import bloch_wave
from jrystal._src.wave_ops import coeff_expand
from jrystal._src.initializer import normal

from typing import Union, List
from jaxtyping import Int, Array
from jrystal._src.jrystal_typing import MaskGrid, CellVector
from jrystal._src.jrystal_typing import ComplexVecterGrid, RealVecterGrid


class PlaneWave(nn.Module):
  """The plane wave module.

  It maps position r with shape [..., d] to [nspin, nk, ni, ...]

  Attributes:
    shape (List): the shape of coeffients.
      A typical shape is [2, num_k, num_g, num_bands]
      TODO: num_bands is a bit confusing
      where num_g must be greater than num_bands.
    mask (MaskGrid): The mask for g vector grid cut-off.
    cell_vectors (CellVectors): cell vectors.
    k_vector_grid (RealVecterGrid): k vector grid.

  Args:
    r (jax.array): the position in real space. Can also be a vector grid, eg,
      input shape [n1, n2, n3, 3].

  Returns:
    ComplexVecterGrid: planewaves evaluated at r.
      Shape[nspin, num_k, num_band, ..., 3]

  Ref. https://en.wikipedia.org/wiki/Bloch%27s_theorem

  """
  # TODO: shape maybe derived from mask?
  # maybe we can specify num_k, num_electron, mask
  # and derive num_g = sum(mask)
  shape: Union[Int[Array, 'nspin num_k num_g num_bands'], List]
  mask: MaskGrid
  cell_vectors: CellVector
  k_vector_grid: RealVecterGrid

  @nn.compact
  def __call__(self, r) -> ComplexVecterGrid:
    coeff_dense = QRDecomp(self.shape)()
    coeff_dense = jnp.swapaxes(coeff_dense, -1, -2)
    coeff_grid = coeff_expand(coeff_dense, self.mask)
    return bloch_wave(self.cell_vectors, coeff_grid, self.k_vector_grid)(r)


class QRDecomp(nn.Module):
  """the QR decomposition will map over the first to last two dimension.
  The input is a batched tall and skinny matrix with shape (..., M, K).
  Returns: a batch of matrices with orthonormal columns (..., M, K)
  where M >= K.

  Example:

  >>> key = jax.random.PRNGKey(123)
  >>> shape = [2, 6, 4]
  >>> qr = QRDecomp(shape)
  >>> params = qr.init(key)
  >>> cg = qr.apply(params)

  cg is the orthogonal coeffiencts which has the same shape of input arg shape.

  """

  shape: Int[Array, '*batch num_g num_bands']
  polarize: bool = True
  complex_weights: bool = True

  @nn.compact
  def __call__(self) -> jax.Array:

    # if self.shape[-1] > self.shape[-2]:
    #   raise errors.InitiateQRDecompShapeError(self.shape)

    weight_real = self.param('w_re', normal(), self.shape)
    if self.complex_weights:
      weight_imaginary = 1.j * self.param('w_im', normal(), self.shape)
    else:
      weight_imaginary = 0.

    weight = weight_real + weight_imaginary
    coeff_dense = jnp.linalg.qr(weight, mode='reduced')[0]
    return coeff_dense
