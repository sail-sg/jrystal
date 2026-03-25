import time

import jax
import jax.numpy as jnp
import jrystal as jr

jax.config.update("jax_enable_x64", True)

key = jax.random.PRNGKey(123)
diamond_file_path = "../geometry/diamond.xyz"
crystal = jr.Crystal.create_from_file(diamond_file_path)
num_bands = crystal.num_electron
grid_size = [32] * 3

kpts = jr.grid.k_vectors(crystal.A, [1, 1, 1])
g_vecs = jr.grid.g_vectors(crystal.A, grid_size)
r_vecs = jr.grid.r_vectors(crystal.A, grid_size)
freq_mask = jr.grid.cubic_mask(grid_size)
pw_param = jr.pw.pw_param_init(key, num_bands, 1, freq_mask)

coeff = jr.pw.pw_coeff(pw_param, freq_mask)
occupation = jr.occupation.gamma(1, crystal.num_electron, num_bands=num_bands)


@jr.utils.vmapstack(3)
def nabla_n1(r):
  return jr.pw.nabla_density_grid(r, coeff, crystal.A, g_vecs, occupation)


@jr.utils.vmapstack(3)
def nabla_n2(r):

  def f(r):
    return jr.pw.density_r(r, coeff, crystal.A, g_vecs, occupation)

  return jax.grad(f)(r)


@jr.utils.vmapstack(3)
def laplacian(r):

  def den(r):
    return jr.pw.density_r(r, coeff, crystal.A, g_vecs, occupation)

  def _lap(index):

    def den_i(x):
      _r = r.at[index].set(x)
      return den(_r)

    return jax.grad(jax.grad(den_i))(r[index])

  lap_x = _lap(0)
  lap_y = _lap(1)
  lap_z = _lap(2)

  return lap_x + lap_y + lap_z


# n1 = nabla_n1(r_vecs)
# n2 = nabla_n2(r_vecs)
l = laplacian(r_vecs)

# print(jnp.mean(jnp.abs(n1 - n2)))

###############################################
# start = time.time()
# # nabla_n1(r_vecs)
# for i in range(10):
#   jax.block_until_ready(nabla_n1(r_vecs))
# print(f"Analytic: {time.time() - start}")

# start = time.time()
# for i in range(10):
#   jax.block_until_ready(nabla_n2(r_vecs))
# print(f"nabla n(ad): {time.time() - start}")

start = time.time()
for i in range(10):
  jax.block_until_ready(laplacian(r_vecs))
print(f"Laplacian: {time.time() - start}")
