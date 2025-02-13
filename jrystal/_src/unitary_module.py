import jax
import jax.numpy as jnp
from chex import dataclass
from jaxtyping import Int, Array


@dataclass
class UnitaryMatrix():
  """parameterization of unitary matrix."""
  shape: Int[Array, '*batch num_g num_bands']
  complex: bool = True

  def __call__(self, params):
    return unitary_matrix(params, self.complex)

  def init(self, key):
    return unitary_matrix_param_init(key, self.shape, self.complex)


def unitary_matrix(params, complex: bool = True):
  weight_real = params['w_re']
  if complex:
    weight_imaginary = 1.j * params['w_im']
  else:
    weight_imaginary = 0.

  weight = weight_real + weight_imaginary
  orthogonal_columns = jnp.linalg.qr(weight, mode='reduced')[0]
  return orthogonal_columns


def unitary_matrix_param_init(key, shape, complex: bool = True):
  key_re, key_im = jax.random.split(key)
  weight_real = jax.random.uniform(key_re, shape)
  if complex:
    weight_imaginary = jax.random.uniform(key_im, shape)
  else:
    weight_imaginary = 0.
  return {'w_re': weight_real, 'w_im': weight_imaginary}
