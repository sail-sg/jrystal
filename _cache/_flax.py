"""Some flax modules."""

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import uniform
from jaxtyping import Array, Int


class UnitaryMatrix(nn.Module):
  """the QR decomposition will map over the first to last two dimension.
  The input is a batch tall and skinny matrix with shape (..., M, K).
  Returns: a batch of matrices with orthonormal columns (..., M, K)
  where M >= K.

  .. code-block:: python
    key = jax.random.PRNGKey(123)
    shape = [2, 6, 4]
    qr = UnitaryMatrix(shape)
    params = qr.init(key)
    cg = qr.apply(params)

  cg is the orthogonal coeffiencts which has the same shape of input arg shape.

  """

  shape: Int[Array, '*batch num_g num_bands']
  complex: bool = True

  @nn.compact
  def __call__(self) -> jax.Array:

    weight_real = self.param('w_re', uniform(), self.shape)
    if self.complex:
      weight_imaginary = 1.j * self.param('w_im', uniform(), self.shape)
    else:
      weight_imaginary = 0.

    weight = weight_real + weight_imaginary
    orthogonal_columns = jnp.linalg.qr(weight, mode='reduced')[0]
    return orthogonal_columns
