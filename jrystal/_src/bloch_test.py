import jax
import jax.numpy as jnp
from jax.config import config
from absl.testing import absltest, parameterized
import numpy as np
import os
from jrystal._src.bloch import u, _u_impl, bloch_wave, r_vectors
from jrystal.utils import view_hlo

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
config.update("jax_enable_x64", True)


class _TestBlochWave(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # hyper params
    ni, nk, nd = 1, 2, 3
    grid_sizes = (8,) * nd
    key = jax.random.PRNGKey(42)
    # random lattice
    key, subkey = jax.random.split(key)
    self.a, *_ = jnp.linalg.qr(jax.random.normal(subkey, (nd, nd)))
    # random cg
    key, subkey = jax.random.split(key)
    self.cg = jax.random.normal(subkey, (ni, nk, *grid_sizes))
    # random k_vec
    key, subkey = jax.random.split(key)
    self.k_vec = jax.random.normal(subkey, (nk, nd))
    random_r = jax.random.normal(key, (*grid_sizes, nd))
    self.batch_r = jnp.reshape(random_r, (-1, nd))
    self.batch3d_r = random_r
    self.r_vec = r_vectors(self.a, grid_sizes)
    self.r_vec_1d = jnp.reshape(self.r_vec, (-1, nd))
    self.ni = ni
    self.nk = nk
    self.nd = nd
    self.grid_sizes = grid_sizes

  def test_u(self):
    a, cg, batch_r = self.a, self.cg, self.batch_r
    out0 = jax.jit(u)(a, cg, batch_r[0])
    self.assertEqual(out0.shape, cg.shape[:-self.nd])
    out = jax.jit(u)(a, cg, batch_r)
    self.assertEqual(out.shape, (*cg.shape[:-self.nd], *batch_r.shape[:-1]))

  def test_u_vmap(self):
    a, cg, batch_r, batch3d_r = self.a, self.cg, self.batch_r, self.batch3d_r
    ndim = a.shape[-1]
    out1 = jax.jit(jax.vmap(u, in_axes=(None, 0, None)))(a, cg, batch_r)
    vmap_r = lambda u: jax.vmap(
      u, in_axes=(None, None, 0), out_axes=(cg.ndim - ndim)
    )
    out2 = jax.jit(vmap_r(u))(a, cg, batch_r)
    out3 = jax.jit(vmap_r(vmap_r(vmap_r(u))))(a, cg, batch3d_r)
    np.testing.assert_array_equal(out1, out2)
    np.testing.assert_array_equal(out1, out3.reshape(out1.shape))

  def test_u_fft(self):
    a, cg, r_vec, r_vec_1d = self.a, self.cg, self.r_vec, self.r_vec_1d
    ndim = a.shape[-1]
    vmap_r = lambda u: jax.vmap(
      u, in_axes=(None, None, 0), out_axes=(cg.ndim - ndim)
    )
    out1 = jax.jit(vmap_r(u))(a, cg, r_vec_1d)
    out2 = jax.jit(vmap_r(vmap_r(vmap_r(u))))(a, cg, r_vec)
    np.testing.assert_array_equal(out1, out2.reshape(out1.shape))
    out3 = jnp.roll(
      jax.jit(vmap_r(u))(a, cg, jnp.roll(r_vec_1d, 1, axis=0)), -1, axis=-1
    )
    # result from fft and direct calculation should not be equal
    with np.testing.assert_raises(AssertionError):
      np.testing.assert_array_equal(out1, out3.reshape(out1.shape))
    # result from fft and direct calculation should be close
    if config.jax_enable_x64:
      np.testing.assert_array_almost_equal(
        out1, out3.reshape(out1.shape), decimal=12
      )
    else:
      np.testing.assert_array_almost_equal(
        out1, out3.reshape(out1.shape), decimal=5
      )

  @parameterized.parameters((0, "a"), (1, "cg"), (2, "r"))
  def test_u_gradient(self, argnums, name):
    a, cg, r_vec_1d = self.a, self.cg, self.r_vec_1d

    def impl_loss(a, cg, r):
      return jnp.abs(jnp.sum(_u_impl(a, cg, r)))

    def loss(a, cg, r):
      return jnp.abs(jnp.sum(u(a, cg, r)))

    roll_r_vec_1d = jnp.roll(r_vec_1d, 1, axis=0)
    g_impl = jax.jit(jax.grad(impl_loss, argnums=argnums))(a, cg, roll_r_vec_1d)
    g_custom = jax.jit(jax.grad(loss, argnums=argnums))(a, cg, roll_r_vec_1d)
    if argnums == 2:
      g_impl = jnp.roll(g_impl, -1, axis=0)
      g_custom = jnp.roll(g_custom, -1, axis=0)
    g_custom_fft = jax.grad(loss, argnums=argnums)(a, cg, r_vec_1d)
    if config.jax_enable_x64:
      np.testing.assert_array_almost_equal(g_impl, g_custom, decimal=11)
      np.testing.assert_array_almost_equal(g_impl, g_custom_fft, decimal=11)
      np.testing.assert_array_almost_equal(g_custom, g_custom_fft, decimal=11)
    else:
      np.testing.assert_array_almost_equal(g_impl, g_custom, decimal=5)
      np.testing.assert_array_almost_equal(g_impl, g_custom_fft, decimal=5)
      np.testing.assert_array_almost_equal(g_custom, g_custom_fft, decimal=5)

  def test_oom_fft(self):
    ni, nk, nd = 10, 1, 3
    grid_sizes = (32,) * nd
    key = jax.random.PRNGKey(42)
    # random lattice
    key, subkey = jax.random.split(key)
    a, *_ = jnp.linalg.qr(jax.random.normal(subkey, (nd, nd)))
    # random cg
    key, subkey = jax.random.split(key)
    cg = jax.random.normal(subkey, (ni, nk, *grid_sizes))
    # random k_vec
    key, subkey = jax.random.split(key)
    k_vec = jax.random.normal(subkey, (nk, nd))
    r_vec = r_vectors(a, grid_sizes)

    # r_vec_1d = jnp.reshape(r_vec, (-1, nd))
    # ndim = a.shape[-1]
    # vmap_r = lambda u: jax.vmap(u, in_axes=0, out_axes=(cg.ndim - ndim))

    def loss(cg):
      wave = bloch_wave(a, cg, k_vec)
      return jnp.abs(jnp.sum(wave(r_vec)))

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    value, grad = value_and_grad(cg)

  def test_bloch_wave(self):
    a = self.a
    cg = self.cg
    r_vec = self.r_vec
    r_vec_1d = self.r_vec_1d
    k_vec = self.k_vec

    ndim = a.shape[-1]
    wave = bloch_wave(a, cg, k_vec)
    vmap_r = lambda u: jax.vmap(u, in_axes=0, out_axes=(cg.ndim - ndim))
    out1 = jax.jit(vmap_r(wave))(r_vec_1d)
    out2 = jax.jit(vmap_r(vmap_r(vmap_r(wave))))(r_vec)
    np.testing.assert_array_equal(out1, out2.reshape(out1.shape))
    out3 = jnp.roll(
      jax.jit(vmap_r(wave))(jnp.roll(r_vec_1d, 1, axis=0)), -1, axis=-1
    )
    # result from fft and direct calculation should not be equal
    with np.testing.assert_raises(AssertionError):
      np.testing.assert_array_equal(out1, out3.reshape(out1.shape))
    # result from fft and direct calculation should be close
    if config.jax_enable_x64:
      np.testing.assert_array_almost_equal(
        out1, out3.reshape(out1.shape), decimal=12
      )
    else:
      np.testing.assert_array_almost_equal(
        out1, out3.reshape(out1.shape), decimal=5
      )

  def test_view_hlo(self):
    a = self.a
    cg = self.cg
    r_vec = self.r_vec
    r_vec_1d = self.r_vec_1d
    k_vec = self.k_vec

    ndim = a.shape[-1]
    wave = lambda r: bloch_wave(a, cg, k_vec)(r, force_fft=True)
    vmap_r = lambda u: jax.vmap(u, in_axes=0, out_axes=(cg.ndim - ndim))

    @jax.jit
    def compute_wave():
      return vmap_r(wave)(r_vec_1d)

    view_hlo(compute_wave)()

    def density(r):
      return (wave(r).conj() * wave(r)).real

    @jax.jit
    def compute_density():
      return vmap_r(density)(r_vec_1d)

    view_hlo(compute_density)()

    @jax.jit
    def compute_reciprocal_wave():
      wave_grid = vmap_r(vmap_r(vmap_r(wave)))(r_vec)
      ret = jnp.fft.fftn(wave_grid, axes=tuple(range(-3, 0)))
      return ret

    view_hlo(compute_reciprocal_wave)()


if __name__ == "__main__":
  absltest.main()
