"""
  Parameter initialization modules.
"""
import jax
import jax.numpy as jnp

def normal_init(shape):
  def init(rng):
    return jax.random.normal(rng, shape) / jnp.sqrt(shape[-1])
  return init


#TODO: initialize parameters by solving atomic systems. 