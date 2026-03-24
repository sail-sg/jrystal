# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for fftn.py."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .fft import (
  _fftn_infer_sharding_from_operands,
  _normalize_axes,
  fftn,
  ifftn,
)

jax.config.update("jax_enable_x64", True)


class _TestModules(parameterized.TestCase):

  def setUp(self):
    self.key = jax.random.PRNGKey(0)
    self.x_real = jax.random.normal(self.key, (4, 6, 8), dtype=jnp.float64)
    self.x_complex = (
      self.x_real +
      1j * jax.random.normal(self.key, (4, 6, 8), dtype=jnp.float64)
    )

  def test_normalize_axes(self):
    self.assertEqual(_normalize_axes(4, (-1, -2), None), (2, 3))
    self.assertEqual(_normalize_axes(4, None, (6, 8)), (2, 3))
    self.assertEqual(_normalize_axes(3, None, None), (0, 1, 2))

  def test_normalize_axes_validation(self):
    with self.assertRaisesRegex(ValueError, "must be unique"):
      _normalize_axes(3, (1, 1), None)
    with self.assertRaisesRegex(ValueError, "out of range"):
      _normalize_axes(3, (3,), None)
    with self.assertRaisesRegex(ValueError, "len\\(s\\) must be <="):
      _normalize_axes(2, None, (2, 2, 2))

  def test_public_signatures_match_jax_fft(self):
    sig = inspect.signature(fftn)
    sig_ref = inspect.signature(jnp.fft.fftn)
    self.assertEqual(tuple(sig.parameters.keys()), tuple(sig_ref.parameters.keys()))
    for name in sig.parameters:
      self.assertEqual(sig.parameters[name].default, sig_ref.parameters[name].default)

    isig = inspect.signature(ifftn)
    isig_ref = inspect.signature(jnp.fft.ifftn)
    self.assertEqual(
      tuple(isig.parameters.keys()), tuple(isig_ref.parameters.keys())
    )
    for name in isig.parameters:
      self.assertEqual(
        isig.parameters[name].default, isig_ref.parameters[name].default
      )

  def test_fftn_matches_jnp(self):
    y = fftn(self.x_complex, axes=(-2, -1), norm="backward")
    y_ref = jnp.fft.fftn(self.x_complex, axes=(-2, -1), norm="backward")
    np.testing.assert_allclose(y, y_ref, atol=1e-10, rtol=1e-10)

  def test_ifftn_matches_jnp(self):
    y = ifftn(self.x_complex, axes=(-2, -1), norm="backward")
    y_ref = jnp.fft.ifftn(self.x_complex, axes=(-2, -1), norm="backward")
    np.testing.assert_allclose(y, y_ref, atol=1e-10, rtol=1e-10)

  def test_grad_through_fftn(self):

    def loss_fn(x):
      y = fftn(x, axes=(-2, -1))
      return jnp.real(jnp.sum(y * jnp.conj(y)))

    def loss_ref(x):
      y = jnp.fft.fftn(x, axes=(-2, -1))
      return jnp.real(jnp.sum(y * jnp.conj(y)))

    g = jax.grad(loss_fn)(self.x_complex)
    g_ref = jax.grad(loss_ref)(self.x_complex)
    np.testing.assert_allclose(g, g_ref, atol=1e-9, rtol=1e-9)

  def test_sharding_mask_keeps_non_fft_axes(self):
    devices = np.array(jax.devices()[:1]).reshape((1,))
    mesh = Mesh(devices, ("d",))
    in_sharding = NamedSharding(mesh, P("d", None, None))
    x = jax.device_put(self.x_complex, in_sharding)

    with mesh:
      y = jax.jit(lambda z: fftn(z, axes=(-2, -1)))(x)

    self.assertIsInstance(y.sharding, NamedSharding)
    self.assertEqual(y.sharding.spec, P("d", None, None))

  def test_sharding_mask_replicates_fft_axis(self):
    devices = np.array(jax.devices()[:1]).reshape((1,))
    mesh = Mesh(devices, ("d",))
    in_sharding = NamedSharding(mesh, P("d", None, None))
    arg_shape = type(
      "ArgShape",
      (),
      {
        "shape": self.x_complex.shape, "sharding": in_sharding
      },
    )()
    out_sharding = _fftn_infer_sharding_from_operands(
      None, (0,), None, mesh, [arg_shape], None
    )
    self.assertIsInstance(out_sharding, NamedSharding)
    self.assertEqual(out_sharding.spec, P(None, None, None))


if __name__ == "__main__":
  absltest.main()
