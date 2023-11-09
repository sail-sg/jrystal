import jax
import jax.numpy as jnp
import flax.linen as nn
from jaxtyping import Int, Float, Array
from jrystal._src.bloch import bloch_wave
from jrystal._src.utils import vmapstack

# TODO: we don't use functions like `_coeff_compress` from another python file.
# functions with underscore prefix are private to the file it is defined.
# if this function has some generality and should be part of our open API,
# consider remove the underscore prefix and document it well.

from jrystal._src.pw import coeff_compress, coeff_expand, get_cg
# TODO: _complex_norm_square should be defined without underscore prefix.
# and moved to files like utils.py than pw.py, as it is general.
from jrystal._src.pw import complex_norm_square
from jrystal._src.initializer import normal
from jrystal._src import errors
from jrystal._src.occupation import OccGamma, OccUniform


class PlaneWave(nn.Module):
  """The plane wave module.

  it maps r with shape [..., d] to [nspin, nk, ni, ...]

  Returns:
      _type_: _description_
  """
  shape: Int[Array, 'nspin nk ng ni']
  mask: jax.Array
  A: jax.Array
  k_grid: jax.Array

  @nn.compact
  def __call__(self, r) -> jax.Array:
    cg = QRDecomp(self.shape)()
    cg = jnp.swapaxes(cg, -1, -2)
    cg = ExpandCoeff(self.mask)(cg)
    wave = BatchedBlochWave(self.A, self.k_grid)(r, cg)
    return wave


class _PlaneWaveFFT(nn.Module):
  shape: Int  # [nspin, nk, ng, ni]
  mask: jax.Array
  k_grid: jax.Array

  @nn.compact
  def __call__(self):
    cg = QRDecomp(self.shape)()
    cg = jnp.swapaxes(cg, -1, -2)
    # TODO: This is no easier than calling _expand_coeff(cg, self.mask).
    # so we don't need the extra ExpandCoeff module.
    cg = ExpandCoeff(self.mask)(cg)
    dim = self.k_grid.shape[-1]
    N = jnp.prod(jnp.array(cg.shape[-dim:]))
    wave = BatchedInverseFFT(self.k_grid.shape[-1])(cg) * N
    return wave


class PlaneWaveDensity(nn.Module):
  shape: Int  # [nspin, nk, ng, ni]
  mask: jax.Array
  A: jax.Array
  k_grid: jax.Array
  spin: Int
  vol: float
  occ: str = 'gamma'

  def setup(self):
    self.pw = PlaneWave(self.shape, self.mask, self.A, self.k_grid)
    _, nk, _, ni = self.shape

    if self.occ == 'gamma':
      self.occ_mask = OccGamma(nk, ni, self.spin)
    elif self.occ == 'uniform':
      self.occ_mask = OccUniform(nk, ni, self.spin)

  @nn.compact
  def __call__(self, r, reduce=True) -> jax.Array:
    w = self.pw(r)
    dim = self.A.shape[0]
    occ = self.occ_mask()

    if (w.ndim < occ.ndim or w.shape[:occ.ndim] != occ.shape):
      raise errors.WavevecOccupationMismatchError(w.shape, occ.shape)

    w = complex_norm_square(w)
    occ = jnp.expand_dims(occ, range(occ.ndim, w.ndim))
    N = jnp.prod(jnp.array(self.mask.shape))

    if reduce:  # reduce over k, i
      density = jnp.sum(occ * w, axis=(1, 2)) / self.vol
      dim = density.ndim - 1
      density_fft = BatchedFFT(dim)(density) / N * self.vol
    else:
      density = occ * w / self.vol
      dim = density.ndim - 3
      density_fft = BatchedFFT(dim)(density) / N * self.vol

    return density, density_fft

  def get_cg(self, params):
    w_re = params['pw']['QRDecomp_0']['w_re']
    w_im = params['pw']['QRDecomp_0']['w_im']
    w = w_re + 1.j * w_im
    return get_cg(w, self.mask)

  def get_occ(self, params):
    m = self.bind(params)
    return m.occ_mask()


class BatchedBlochWave(nn.Module):
  """Bloch Wave function with fixed lattice vector, and kmesh
      The input cg is an expanded form (in 3D with a shape
      [*batches, n1, n2, n3, 3])

      Example:
      bw = BatchedBlochWave(a, k_grid, cg)

      bw: [n1, n2, n3, 3] -> [2, nk, ni, n1, n2, n3]

  """

  A: Float[Array, "dim1 dim2"]
  k_grid: Float[Array, "nk 3"]
  cg: Float = None
  rd: int = 3

  @nn.compact
  def __call__(self, r: Float[Array, '*batches r'], cg=None) -> jax.Array:
    # TODO: if cg is passed as an argument,
    # then we don't need this module at all, we just need to call the original
    # bloch_wave function.
    # To make sense of this module, we should make this module initialize
    # cg by itself instead of passing it as an argument.
    cg = nn.merge_param('cg', self.cg, cg)
    wave = bloch_wave(self.A, cg, self.k_grid)  #
    wave = vmapstack(self.rd)(wave)(r)  # [n1, n2, n3, 2, ni, nk]
    wave = jnp.moveaxis(wave, range(3), range(-3, 0))
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

  shape: Int[Array, '*batch ng ni']
  polarize: bool = True
  complex_weights: bool = True

  # def __post_init__(self):
  #   super().__post_init__()
  #   if self.shape[-1] > self.shape[-2]:
  #     raise errors.InitiateQRDecompShapeError(self.shape)

  @nn.compact
  def __call__(self) -> jax.Array:
    shape = self.shape  # [*batch, self.ng, self.ni]
    weights_re = self.param('w_re', normal(), shape)
    if self.complex_weights:
      weights_im = self.param('w_im', normal(), shape)
    else:
      weights_im = 0

    weights = weights_re + 1.j * weights_im
    cg = jnp.linalg.qr(weights, mode='reduced')[0]
    return cg


class BatchedInverseFFT(nn.Module):
  # TODO: prefer using ifftn directly, this class is not necessary.
  """Batched inverse Fourier transform module"""
  fft_dim: Int = 3

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    x_dim = len(x.shape)
    if x_dim < self.fft_dim:
      raise errors.ApplyFFTShapeError(self.fft_dim, x.shape)

    ifft = vmapstack(x_dim - self.fft_dim)(jnp.fft.ifftn)
    return ifft(x)


class BatchedFFT(nn.Module):
  # TODO: prefer using fftn directly, this class is not necessary.
  """Batched fourier transform module"""
  fft_dim: Int = 3

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    x_dim = len(x.shape)
    if x_dim < self.fft_dim:
      raise errors.ApplyFFTShapeError(self.fft_dim, x.shape)

    fft = vmapstack(x_dim - self.fft_dim)(jnp.fft.fftn)
    return fft(x)


class ExpandCoeff(nn.Module):
  # TODO: remove this module since it doesn't manage any parameters.
  mask: Int[Array, "*nd"]

  def __call__(self, x):
    return coeff_expand(x, self.mask)


class CompressCoeff(nn.Module):
  # TODO: remove this module since it doesn't manage any parameters.
  mask: Int[Array, "*nd"]

  def __call__(self, x):
    return coeff_compress(x, self.mask)
