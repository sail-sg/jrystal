"""
  Parameter initializers.
  
  
  Format for defining initializer:
  
  def normal(shape):

    def init(rng):
      return jax.random.normal(rng, shape) / jnp.sqrt(shape[-1])

    return init
  
"""
from flax.linen.initializers import constant as constant
from flax.linen.initializers import normal as normal
from flax.linen.initializers import orthogonal as orthogonal
from flax.linen.initializers import delta_orthogonal as delta_orthogonal
from flax.linen.initializers import uniform as uniform
from flax.linen.initializers import xavier_normal as xavier_normal
from flax.linen.initializers import xavier_uniform as xavier_uniform

__all__ = (
  "constant", 
  "normal", 
  "orthogonal", 
  "delta_orthogonal",
  'uniform',
  'xavier_normal',
  'xavier_uniform',
  )

# TODO: initialize parameters by solving atomic systems.
