import jax
import jax.numpy as jnp
import flax.linen as nn

from jrystal._src.bloch import bloch_wave
from jrystal._src.utils import vmapstack
from jrystal._src.functional import coeff_expand, batched_ifft
from jrystal._src.initializer import normal

from typing import Union, List
from jaxtyping import Int, Float, Array
from jrystal._src.jrystal_typing import MaskGrid, CellVector
from jrystal._src.jrystal_typing import ComplexVecterGrid, RealVecterGrid


class PlaneWave(nn.Module):
  """The plane wave module. 
  
  It maps position r with shape [..., d] to [nspin, nk, ni, ...]

  Attributes:
    shape (List): the shape of coeffients. 
      A typical shape is [2, num_k, num_g, num_bands] 
      where num_g must be greater than num_bands.
    mask (MaskGrid): The mask for g vector grid cut-off.
    cell_vectors (CellVectors): cell vectors.
    k_vector_grid (RealVecterGrid): k vector grid.
  
  Args:
    r (jax.array): the position in real space. Can also be a vector grid, eg, 
      input shape [n1, n2, n3, 3]. 

  Returns:
    ComplexVecterGrid: planewaves evaluated at r. 

  Ref. https://en.wikipedia.org/wiki/Bloch%27s_theorem

  """
  shape: Union[Int[Array, 'nspin num_k num_g num_bands'], List]
  mask: MaskGrid
  cell_vectors: CellVector
  k_vector_grid: RealVecterGrid

  @nn.compact
  def __call__(self, r) -> ComplexVecterGrid:
    coeff_dense = QRDecomp(self.shape)()
    coeff_dense = jnp.swapaxes(coeff_dense, -1, -2)
    coeff_grid = coeff_expand(coeff_dense, self.mask)
    wave = BatchedBlochWave(self.cell_vectors,
                            self.k_vector_grid)(r, coeff_grid)
    return wave


class _PlaneWaveFFT(nn.Module):
  # This module is just for testing purpose. It won't be exposed to the user.
  shape: Union[Int[Array, 'nspin num_k num_g num_bands'], List]
  mask: MaskGrid
  k_vector_grid: RealVecterGrid

  @nn.compact
  def __call__(self):
    coeff_dense = QRDecomp(self.shape)()
    coeff_dense = jnp.swapaxes(coeff_dense, -1, -2)
    coeff_grid = coeff_expand(coeff_dense, self.mask)
    dim = self.k_vector_grid.shape[-1]
    N = jnp.prod(jnp.array(coeff_grid.shape[-dim:]))
    wave = batched_ifft(coeff_grid, self.k_vector_grid.shape[-1]) * N
    return wave


# I prefer not use Crystal object as input as I want this API to be decoupled
# with other modules.


class BatchedBlochWave(nn.Module):
  # There are three reasons that I still keep this module.

  # First, this function contains at least 4 arguments, and half of them are
  # crystal-relavent, which won't change during the training, therefore
  # define an object that maintains the frozen crystal related parameters can
  # help produce more clean API.

  # Second, in the computation we may want to access to the intermediate
  # output, such as computating kinetic energy requires the direct output of
  # QR, therefore it would be troublesome if we wrap the QR layer inside this
  # module, and wrapped by the PlaneWaveDensity module.

  # Third, The purpose of this module is conceptually simple; however, for
  # practical implementation, it contains many sub-operations. It would be more
  # efficient to wrap these as a single unit, which is more clean than
  # add funtools.partial()

  # TODO: we may just need the u function in bloch wave for feature usage.
  """Batched bloch wave module. 
  
    Attributes:
      cell_vectors (CellVector): the cell vectors.
      k_vector_grid: the k vector grid.

    Args:
      r (jax.array): the position in real space. Can also be a vector grid, eg, 
        input shape [n1, n2, n3, 3]. 
      coeff_grid (ComplexVecterGrid): The Hermitian coeffecient grid. A typical
        shape is [num_spin, num_k, num_bands, n1, n2, n3]

    Returns:
      ComplexVecterGrid: the bloch wave function evaluated at r.

    Ref. https://en.wikipedia.org/wiki/Bloch%27s_theorem
    
  """

  cell_vectors: CellVector
  k_vector_grid: RealVecterGrid

  @nn.compact
  def __call__(
    self, r: Float[Array, '*batches r'], coeff_grid: ComplexVecterGrid
  ) -> ComplexVecterGrid:
    dim = r.shape[-1]
    wavefun = bloch_wave(self.cell_vectors, coeff_grid, self.k_vector_grid)  #
    if r.ndim == 1:
      wave = wavefun(r)
    else:
      wave = vmapstack(coeff_grid.ndim - dim)(wavefun)(r)
      wave = jnp.moveaxis(wave, range(dim), range(-dim, 0))
    return wave


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
