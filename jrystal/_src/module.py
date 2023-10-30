import jax
import jax.numpy as jnp
import flax.linen as nn
from jaxtyping import Int, Float, Array
from jrystal._src.bloch import bloch_wave
from jrystal._src.utils import vmapstack
from jrystal._src.pw import _coeff_compress, _coeff_expand
from jrystal._src.pw import _complex_norm_square
from jrystal._src.initializer import normal
from jrystal import errors
from jrystal.occupations import OccSimple


class PlaneWave(nn.Module):
  """The plane wave module.
  
  it maps r with shape [..., d] to [nspin, nk, ni, ...]

  Returns:
      _type_: _description_
  """
  shape: Int    
  mask: jax.Array
  a: jax.Array
  k_grid: jax.Array

  @nn.compact
  def __call__(self, r) -> jax.Array:
    cg = QRdecomp(self.shape)()
    cg = jnp.swapaxes(cg, -1, -2)
    cg = ExpandCoeff(self.mask)(cg)
    wave = BatchedBlochWave(self.a, self.k_grid)(r, cg)
    return wave 


class _PlaneWaveFFT(nn.Module):
  shape: Int  # [nspin, nk, ng, ni]
  mask: jax.Array
  a: jax.Array
  k_grid: jax.Array

  @nn.compact
  def __call__(self):
    cg = QRdecomp(self.shape)()
    cg = jnp.swapaxes(cg, -1, -2)
    cg = ExpandCoeff(self.mask)(cg)
    dim = self.k_grid.shape[-1]
    N = jnp.prod(jnp.array(cg.shape[-dim:]))
    wave = BatchedInverseFFT(self.k_grid.shape[-1])(cg) * N
    return wave


class PlaneWaveDensity(nn.Module):
  shape: Int        # [nspin, nk, ng, ni]
  mask: jax.Array   
  a: jax.Array
  k_grid: jax.Array
  spin: Int
  
  @nn.compact
  def __call__(self, r, reduce=True) -> jax.Array:
    w = PlaneWave(self.shape, self.mask, self.a, self.k_grid)(r)
    _, nk, _, ni = self.shape
    dim = self.a.shape[0]
    occ = OccSimple(nk, ni, self.spin)()
    if (w.ndim < occ.ndim or w.shape[:occ.ndim] != occ.shape):
      raise errors.WavevecOccupationMismatchError(w.shape, occ.shape)
    
    w = _complex_norm_square(w)
    occ = jnp.expand_dims(occ, range(occ.ndim, w.ndim))
    N = jnp.prod(jnp.array(self.mask.shape))
    
    if reduce:
      density = jnp.sum(occ * w, axis=(1, 2))  # reduce over k, i
      dim = density.ndim-1
      density_fft = BatchedFFT(dim)(density) * N
    else:
      density = occ * w
      dim = density.ndim-3
      density_fft = BatchedFFT(dim)(density) * N
    
    return density, density_fft
    

class BatchedBlochWave(nn.Module):
  """Bloch Wave function with fixed lattice vector, and kmesh
      The input cg is an expanded form (in 3D with a shape 
      [*batches, n1, n2, n3, 3])
      
      Example:
      bw = BatchedBlochWave(a, k_grid, cg)
      
      bw: [n1, n2, n3, 3] -> [2, ni, nk, n1, n2, n3]
      
  """

  a: Float[Array, "dim1 dim2"]
  k_grid: Float[Array, "nk 3"]
  cg: Float = None

  @nn.compact
  def __call__(self, r: Float[Array, '*batches r'], cg=None) -> jax.Array:
    cg = nn.merge_param('cg', self.cg, cg)
    rd = r.ndim - 1  # （N1, N2, N3, 3)
    wave = bloch_wave(self.a, cg, self.k_grid)  #
    wave = vmapstack(rd)(wave)(r)  # [n1, n2, n3, 2, ni, nk]
    wave = jnp.moveaxis(wave, jnp.arange(rd), jnp.arange(rd) - rd)
    return wave


class QRdecomp(nn.Module):
  """the QR decomposition will map over the first to last two dimension. 
  The input is a batched tall and skinny matrix with shape (..., M, K). 
  Returns: a batch of matrices with orthonormal columns (..., M, K) 
  where M >= K. 
  
  Example:
  
  >>> key = jax.random.PRNGKey(123)
  >>> shape = [2, 6, 4]
  >>> qr = QRdecomp(shape)
  >>> params = qr.init(key)
  
  >>> cg = qr.apply(params)
  
  cg is the orthogonal coeffiencts which has the same shape of input arg shape.
  
  """
  
  shape: Int[Array, '*batch ng ni']
  polarize: bool = True
  complex_weights: bool = True
  
  def __post_init__(self):
    super().__post_init__()
    if self.shape[-1] > self.shape[-2]:
      raise errors.InitiateQRdecompShapeError(self.shape)

  @nn.compact
  def __call__(self, *args) -> jax.Array:
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
  mask: Int[Array, "*nd"]

  def __call__(self, x):
    return _coeff_expand(x, self.mask)


class CompressCoeff(nn.Module):
  mask: Int[Array, "*nd"]

  def __call__(self, x):
    return _coeff_compress(x, self.mask)
