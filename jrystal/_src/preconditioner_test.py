"""Test the preconditioner."""
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jrystal._src.preconditioner import (
  preconditioner_neumann
)
from typing import Sequence
from jaxtyping import Array


jax.config.update("jax_enable_x64", True)


class _TestPreconditioner(parameterized.TestCase):

  def setUp(self):
    self.key = jax.random.PRNGKey(12341)
    self.dim = 100
    A = jax.random.normal(
      self.key, shape=(self.dim, self.dim)
    ) / jnp.sqrt(self.dim)
    self.A = A @ A.T + jnp.eye(self.dim) * 0.2

    print(f"max eigval: {np.linalg.eigvalsh(self.A).max()}")
    print(f"min eigval: {np.linalg.eigvalsh(self.A).min()}")
    self.b = jax.random.uniform(self.key, shape=(self.dim,))

    def f(x: Sequence[Array]):
      x = x["test"]
      return 0.5 * x.T @ self.A @ x - self.b.T @ x

    self.fun = f

  def test_preconditioner_neumann_pytree(self):
    x = jax.random.normal(self.key, shape=(self.dim,)) / jnp.sqrt(self.dim)
    x = {"test": x}
    preconditioner = preconditioner_neumann(self.fun, x, max_iter=400)
    np.testing.assert_allclose(
      preconditioner({"test": self.b})["test"],
      jnp.linalg.solve(self.A, self.b),
      atol=1e-6
    )
