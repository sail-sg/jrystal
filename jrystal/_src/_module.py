import jax
import jax.numpy as jnp
import flax.linen as nn
from jaxtyping import Int, Float, Array, Complex
from typing import List
from jrystal._src._bloch import bloch_wave
from jrystal._src._utils import vmapstack
from jrystal._src._pw import _coeff_compress, _coeff_expand
from jrystal._src._initialize import normal_init


class PlaneWave(nn.Module):
  shape: List  # [self.nspin, self.nk, self.ng, self.ni]
  mask: jax.Array
  a: jax.Array
  k_grid: jax.Array
  
  @nn.compact
  def __call__(self, r) -> jax.Array:
    cg = QRdecomp(self.shape)()
    cg = ExpandCoeff(self.mask)(cg)
    wave = BatchedBlochWave(self.a, self.k_grid)(r, cg)
    return wave


class BatchedBlochWave(nn.Module):
  """Bloch Wave function with fixed lattice vector, and kmesh
      The input cg is an expanded form (in 3D with a shape [*batches, n1, n2, n3, 3])
  """

  a: Float[Array, "dim1 dim2"]
  k_grid: Float[Array, "nk 3"]
  cg: Float=None
  
  @nn.compact
  def __call__(self, r:Float[Array, 'd *batches'], cg=None) -> jax.Array:
    cg = nn.merge_param('cg', self.cg, cg)
    rd = r.ndim - 1
    wave = bloch_wave(self.a, cg, self.k_grid)  # cg: an expanded form
    wave = vmapstack(rd)(wave)
    return wave(r)    


class QRdecomp(nn.Module):
  """the QR decomposition will map over the first to last two dimension. 
  The input is a batched tall and skinny matrix with shape (..., M, K). 
  Returns: a matrix with orthonormal columns (..., M, K) where M >= K. 
  """ 
  shape: Int[Array, '*batch ng ni']
  polarize: bool = True
  complex_weights: bool = True
  
  @nn.compact
  def __call__(self, *args) -> jax.Array:
    shape = self.shape  # [*batch, self.ng, self.ni]
    weights_re = self.param('w_re', normal_init(shape))
    weights_im = self.param('w_im', normal_init(shape)) if self.complex_weights else 0
    weights = weights_re + 1.j * weights_im
    cg = jnp.linalg.qr(weights, mode='reduced')[0]  
    cg = jnp.swapaxes(cg, -1, -2)  # [*batch, self.ni, self.ng]
    return cg


class BatchedInverseFFT(nn.Module):
  """Batched inverse Fourier transform module"""
  fft_dim: Int=3
  
  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    x_dim = len(x.shape)
    if x_dim < self.fft_dim:
      raise ValueError(f'Input x must have higher dimension than ndim'
                       f'({self.fft_dim})')
    
    ifft = vmapstack(x_dim-self.fft_dim)(jnp.fft.ifftn)
    return ifft(x)


class BatchedFFT(nn.Module):
  """Batched fourier transform module"""
  fft_dim: Int=3
  
  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    x_dim = len(x.shape)
    if x_dim < self.fft_dim:
      raise ValueError(f'Input x must have higher dimension than ndim'
                       f'({self.fft_dim})')
    
    fft = vmapstack(x_dim-self.fft_dim)(jnp.fft.fftn)
    return fft(x)


class ComplexNormSquare(nn.Module):
  """Compute the Square of the norm of a complex number
  """
  def __call__(self, x) -> jax.Array:
    return jnp.abs(jnp.conj(x) * x)
  

class ExpandCoeff(nn.Module):
  mask: Int[Array, "*nd"]
  
  def __call__(self, x):
    return _coeff_expand(x, self.mask)


class CompressCoeff(nn.Module):
  mask: Int[Array, "*nd"]
  
  def __call__(self, x):
    return _coeff_compress(x, self.mask)
