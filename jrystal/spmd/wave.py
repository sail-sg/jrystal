"""Wave function modules with customized sharding ops. """

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
import flax.linen as nn
import numpy as np
from typing import List, Union
from jaxtyping import Int, Array
from jax.sharding import Sharding
from .._src.grid import (
  r_vectors,
  g_vectors,
  half_frequency_shape,
  half_frequency_pad_to,
)
from .._src.initializer import normal
from .fft import ifftn
from .. import energy

from functools import partial


def qr(x):
  return jax.numpy.linalg.qr(x)[0]


class QR(nn.Module):
  shape: Int[Array, '*batch num_g num_bands']
  complex_weights: bool = True

  @nn.compact
  def __call__(self, sharding=None) -> jax.Array:

    # if self.shape[-1] > self.shape[-2]:
    #   raise errors.InitiateQRDecompShapeError(self.shape)

    weight_real = self.param('w_re', normal(), self.shape)
    if self.complex_weights:
      weight_imaginary = 1.j * self.param('w_im', normal(), self.shape)
    else:
      weight_imaginary = 0.

    weight = weight_real + weight_imaginary
    coeff_dense = pjit(qr, sharding, sharding)(weight)
    return coeff_dense


class PlaneWave(nn.Module):
  num_electrons: int
  grid_sizes: List | Array
  k_grid_sizes: List | Array
  spin: Union[Int, None] = None
  polarize: bool = True

  def setup(self):
    self.half_shape = half_frequency_shape(self.grid_sizes)
    num_g = np.prod(self.half_shape)
    num_k = np.prod(self.k_grid_sizes).item()
    num_s = 2 if self.polarize else 1
    param_shape = (num_s, num_k, num_g, self.num_electrons)
    self.qr = QR(param_shape)

  def r_vector_grid(self, cell_vectors):
    return r_vectors(cell_vectors, self.grid_sizes)

  def g_vector_grid(self, cell_vectors):
    return g_vectors(cell_vectors, self.grid_sizes)

  def k_vectors(self, cell_vectors):
    k_vector_grid = r_vectors(jnp.linalg.inv(cell_vectors), self.k_grid_sizes)
    return k_vector_grid.reshape((-1, 3))

  def coefficient(self, sharding=None) -> jax.Array:
    coeff_dense = self.qr()
    coeff_dense = jnp.swapaxes(coeff_dense, -1, -2)
    num_k = np.prod(np.array(self.k_grid_sizes)).item()
    ns = 2 if self.polarize else 1
    coeff_dense = jnp.reshape(
      coeff_dense,
      (ns, num_k, self.num_electrons, *self.half_shape),
    )

    def pad(coeff_dense):
      return half_frequency_pad_to(coeff_dense, self.grid_sizes)

    coeff = pjit(pad, in_shardings=None, out_shardings=sharding)(coeff_dense)
    return coeff

  def wave(self, sharding: Sharding = None):
    coeff = self.coefficient(sharding)
    if sharding:
      coeff = jax.device_put(coeff, sharding)
    grid_sizes = coeff.shape[-3:]
    wave = pjit(ifftn, in_shardings=sharding, out_shardings=sharding)(coeff)
    wave = wave * np.prod(grid_sizes)
    return wave

  def density(self, cell_vectors, occupation, sharding: Sharding = None):
    vol = jnp.linalg.det(cell_vectors)
    wave = self.wave(sharding=sharding)
    dens = jnp.abs(wave)**2 / vol
    return jnp.einsum("ski...,ski->s...", dens, occupation)

  def reciprocal_density(self, cell_vectors, sharding: Sharding = None):
    density_grid = self.density(cell_vectors, sharding=sharding)
    return jnp.fft.fftn(density_grid, axes=range(-3, 0))

  def __call__(self, *args):
    return self.wave(*args)

  def kinetic(self, cell_vectors, kpts, occupation, sharding: Sharding = None):
    g_vector_grid = g_vectors(cell_vectors, self.grid_sizes)

    @partial(pjit, in_shardings=sharding)
    def kin(coeff):
      return energy.kinetic(g_vector_grid, kpts, coeff, occupation)

    return kin(self.coefficient(sharding))
