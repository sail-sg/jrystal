import jax
import jax.numpy as jnp
import flax.linen as nn

from jrystal._src.bloch import bloch_wave
from jrystal._src.utils import vmapstack
from jrystal._src.functional import coeff_expand, batched_fft, batched_ifft
from jrystal._src.occupation import OccGamma, OccUniform
from jrystal._src.utils import complex_norm_square
from jrystal._src.initializer import normal
from jrystal._src import energy
from jrystal._src.grid import r_vectors, g_vectors
from jrystal._src.crystal import Crystal

from typing import Union, List
from jaxtyping import Int, Float, Array, ArrayLike
from jrystal._src.jrystal_typing import MaskGrid, CellVector
from jrystal._src.jrystal_typing import ComplexVecterGrid, RealVecterGrid
from jrystal._src import errors


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
class PlaneWaveDensity(nn.Module):
  shape: Union[Int[Array, "..."], List]  # [nspin, nk, ng, ni]
  mask: MaskGrid
  cell_vectors: CellVector
  k_vector_grid: RealVecterGrid
  spin: Int
  vol: float
  occupation_method: str = 'gamma'

  def setup(self):
    _, nk, _, ni = self.shape
    self.qr = QRDecomp(self.shape)
    self.bloch = BatchedBlochWave(self.cell_vectors, self.k_vector_grid)

    if self.occupation_method == 'gamma':
      self.occupation = OccGamma(nk, ni, self.spin)

    elif self.occupation_method == 'uniform':
      self.occupation = OccUniform(nk, ni, self.spin)

  def __call__(self, r) -> jax.Array:
    return self.density(r, reduce=True)

  def density(self, r, reduce=True) -> jax.Array:
    occ = self.occupation()
    coeff_dense = self.qr()
    coeff_dense = jnp.swapaxes(coeff_dense, -1, -2)
    coeff_grid = coeff_expand(coeff_dense, self.mask)
    wave = self.bloch(r, coeff_grid)

    if (wave.ndim < occ.ndim or wave.shape[:occ.ndim] != occ.shape):
      raise errors.WavevecOccupationMismatchError(wave.shape, occ.shape)

    density = complex_norm_square(wave)
    occ = jnp.expand_dims(occ, range(occ.ndim, wave.ndim))

    if reduce:  # reduce over k, i
      density = jnp.sum(occ * density, axis=(1, 2)) / self.vol
    else:
      density = occ * density / self.vol

    return density

  def density_reciprocal(self, r, reduce=True):
    density = self.density(r, reduce=reduce)
    num_grids = jnp.prod(jnp.array(self.mask.shape))
    dim = self.cell_vectors.shape[0]

    if reduce:  # reduce over k, i
      dim = density.ndim - 1
      density_fft = batched_fft(density, dim) / num_grids * self.vol
    else:
      dim = density.ndim - 3
      density_fft = batched_fft(density, dim) / num_grids * self.vol

    return density_fft

  def get_coefficient(self):
    coeff_dense = self.qr()
    coeff_dense = jnp.swapaxes(coeff_dense, -1, -2)
    coeff_grid = coeff_expand(coeff_dense, self.mask)
    return coeff_grid

  def get_occupation(self):
    return self.occupation()

  def hartree(self):
    r_vector_grid = r_vectors(self.cell_vectors, self.mask.shape)
    g_vector_grid = g_vectors(self.cell_vectors, self.mask.shape)
    reciprocal_density_grid = self.density_reciprocal(
      r_vector_grid, reduce=True
    )
    return energy.hartree(reciprocal_density_grid, g_vector_grid, self.vol)

  def external(self, crystal: Crystal):
    r_vector_grid = r_vectors(self.cell_vectors, self.mask.shape)
    g_vector_grid = g_vectors(self.cell_vectors, self.mask.shape)
    reciprocal_density_grid = self.density_reciprocal(
      r_vector_grid, reduce=True
    )
    positions = crystal.positions
    charges = crystal.charges
    return energy.external(
      reciprocal_density_grid, positions, charges, g_vector_grid, self.vol
    )

  def kinetic(self):
    g_vector_grid = g_vectors(self.cell_vectors, self.mask.shape)
    coeff_grid = self.get_coefficient()
    occupation = self.get_occupation()
    return energy.kinetic(
      g_vector_grid, self.k_vector_grid, coeff_grid, occupation
    )

  def xc(self, xc='lda_x'):
    r_vector_grid = r_vectors(self.cell_vectors, self.mask.shape)
    density = self.density(r_vector_grid, True)
    return energy.xc_lda(density, self.vol)


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

  shape: Int[ArrayLike, '*batch ng ni']
  polarize: bool = True
  complex_weights: bool = True

  @nn.compact
  def __call__(self) -> jax.Array:

    # if self.shape[-1] > self.shape[-2]:
    #   raise errors.InitiateQRDecompShapeError(self.shape)

    weight_real = self.param('w_re', normal(), self.shape)
    if self.complex_weights:
      weight_imaginary = self.param('w_im', normal(), self.shape)
    else:
      weight_imaginary = 0

    weight = weight_real + 1.j * weight_imaginary
    coeff_dense = jnp.linalg.qr(weight, mode='reduced')[0]
    return coeff_dense
