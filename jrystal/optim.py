"""Optimization 
  
  This module is for optimizing the parameters.
"""
import jax
import jax.numpy as jnp
from typing import NamedTuple
import haiku as hk
import optax


class TrainState(NamedTuple):
  params: hk.Params
  avg_params: hk.Params
  opt_state: optax.OptState
