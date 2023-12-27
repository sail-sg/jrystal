import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jrystal._src.operators import hartree_potential
from jrystal._src.grid import r_vectors, g_vectors
from jrystal._src.bloch import u
from jrystal._src.energy import hartree
from autofd import function
import autofd.operators as o
from jaxtyping import Float32, Complex64, Array
from jrystal.utils import view_hlo


class _TestOperators(parameterized.TestCase):

  def test_hartree_potential(self):
    key = jax.random.PRNGKey(123)
    cell_vectors = jax.random.uniform(key, [3, 3])
    grid_sizes = (64,) * 3

    @function
    def density(r: Float32[Array, "3"]) -> Float32[Array, ""]:
      return jnp.sum(jnp.sin(r), axis=-1)

    vhar = hartree_potential(
      density, cell_vectors=cell_vectors, grid_sizes=grid_sizes
    )
    integrand = 0.5 * vhar * density
    r_vector_grid = r_vectors(cell_vectors, grid_sizes)
    r_vec = jnp.reshape(r_vector_grid, (-1, 3))
    vol = jnp.linalg.det(cell_vectors)

    @jax.jit
    def energy(r_vec):
      return jax.vmap(integrand)(r_vec).sum() * vol / np.prod(grid_sizes)

    out1 = energy(r_vec)

    reciprocal_density_grid = jnp.fft.fftn(
      jax.vmap(density)(r_vec).reshape(grid_sizes)
    )
    g_vector_grid = g_vectors(cell_vectors, grid_sizes)
    out2 = hartree(reciprocal_density_grid, g_vector_grid, vol)
    print(out1, out2)

  def test_functional_gradient(self):
    key = jax.random.PRNGKey(123)
    cell_vectors = jax.random.uniform(key, [3, 3])
    grid_sizes = (48,) * 3

    @view_hlo
    @jax.jit
    def potential_grid():

      @function
      def density(r: Float32[Array, "3"]) -> Complex64[Array, ""]:
        return jnp.sum(jnp.sin(r), axis=-1) + 0.j

      def hartree_energy(density):
        vhar = hartree_potential(
          density, cell_vectors=cell_vectors, grid_sizes=grid_sizes
        )
        return o.braket(density, vhar, real=True)

      with jax.ensure_compile_time_eval():
        d = jax.grad(hartree_energy)(density)
      r_vector_grid = r_vectors(cell_vectors, grid_sizes)
      r_vec = jnp.reshape(r_vector_grid, (-1, 3))
      return jax.vmap(d)(r_vec)

    potential_grid()


if __name__ == '__main__':
  absltest.main()
