import jax.numpy as jnp

from jrystal.calc.types import KSampling


def test_ksampling_keeps_labels_and_segments():
  sampling = KSampling(
    mode="path",
    kpts=jnp.zeros((3, 3), dtype=jnp.float32),
    weights=jnp.ones((3,), dtype=jnp.float32),
    labels=["G", "X", "M"],
    segments=[(0, 1), (1, 2)],
  )
  assert sampling.mode == "path"
  assert sampling.labels == ["G", "X", "M"]
  assert sampling.segments == [(0, 1), (1, 2)]
