import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jrystal._src.grid import (
  translation_vectors,
  grid_1d,
  half_frequency_shape,
  half_frequency_pad_to,
  half_frequency_mask,
)


class _Test_grid(parameterized.TestCase):

  def setUp(self):

    def translation_vectors(a, ew_cut=1e4):
      n = jnp.ceil(ew_cut / jnp.linalg.norm(jnp.sum(a, axis=0))**2)
      n = jnp.arange(n) - n // 2
      n = n[:, None, None, None] * a[None, None, None, 0] + \
        n[None, :, None, None] * a[None, None, None, 1] + \
        n[None, None, :, None] * a[None, None, None, 2]
      return jnp.reshape(n, [-1, 3])

    self.translation_vectors = translation_vectors

  def test_grid(self):
    key = jax.random.PRNGKey(123)
    a = jax.random.uniform(key, [3, 3])
    ec = 1e2
    l1 = self.translation_vectors(a, ec)
    l2 = translation_vectors(a, ec)

    np.testing.assert_allclose(l1.shape, l2.shape, 1e-8)

  def test_update_mask_vs_update_slice_speed(self):
    grid_sizes = (64,) * 3
    shape = half_frequency_shape(grid_sizes)
    batch_dims = (2, 1, 12)
    raw_param = jax.random.normal(
      jax.random.PRNGKey(123),
      shape=batch_dims + shape,
      dtype=jnp.complex64,
    )
    mask = half_frequency_mask(grid_sizes)

    @jax.jit
    def pad1(raw_param):
      return half_frequency_pad_to(raw_param, grid_sizes)

    @jax.jit
    def pad2(raw_param):
      return jnp.zeros(
        batch_dims + grid_sizes, dtype=raw_param.dtype
      ).at[..., mask].set(raw_param.reshape(batch_dims + (-1,)))

    # warmup
    pad1(raw_param)
    pad2(raw_param)
    import time
    t0 = time.time()
    for _ in range(100):
      pad1(raw_param)
    t1 = time.time()
    for _ in range(100):
      pad2(raw_param)
    t2 = time.time()
    print(f"lax.dynamic_update_slice takes {t1 - t0}s")
    print(f"at[mask].set(update) takes {t2 - t1}s")

  def test_update_mask_vs_update_slice(self):
    grid_sizes = [10, 10, 10]
    shape = half_frequency_shape(grid_sizes)
    batch_dims = (2, 1, 12)
    raw_param = jax.random.normal(
      jax.random.PRNGKey(123),
      shape=batch_dims + shape,
      dtype=jnp.complex64,
    )
    res1 = half_frequency_pad_to(raw_param, grid_sizes)
    mask = half_frequency_mask(grid_sizes)
    res2 = jnp.zeros_like(res1).at[..., mask].set(
      raw_param.reshape(batch_dims + (-1,))
    )
    np.testing.assert_array_equal(res1, res2)

  def test_fftfreq(self):
    np.testing.assert_array_equal(
      grid_1d(9, normalize=True), jnp.fft.fftfreq(9, 1)
    )
    np.testing.assert_array_equal(
      grid_1d(9, normalize=False), jnp.fft.fftfreq(9, 1 / 9)
    )
    np.testing.assert_array_equal(
      grid_1d(10, normalize=True), jnp.fft.fftfreq(10, 1)
    )
    np.testing.assert_array_equal(
      grid_1d(10, normalize=False), jnp.fft.fftfreq(10, 1 / 10)
    )


if __name__ == '__main__':
  absltest.main()
