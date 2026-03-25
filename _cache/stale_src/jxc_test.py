import jax.numpy as jnp
import jxc

r = jnp.ones([2, 3])
sigma = jnp.ones([3, 3])

# xc = jxc.get_xc_functional("mgga_x_scan", polarized=True, order="exc")
xc = jxc.get_xc_functional("gga_x_pbe", polarized=True, order="exc")


# xc(r[0], sigma=sigma[0], tau=r[0])
print(xc(r, sigma=sigma))
