import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from einops import einsum
from scipy.special import spherical_jn as jn

import jrystal as jr
from jrystal import sbt
from jrystal.pseudopotential.dataclass import UltrasoftPseudopotential as USPP

pseudopotential_path = "/home/aiops/litb/projects/jrystal/pseudopotential/ultrasoft/"
crystal_path = "/home/aiops/litb/projects/jrystal/geometry/si.xyz"
ang_mom = 1


crystal = jr.Crystal.create_from_file(crystal_path)
uspp_data = USPP.create(crystal, pseudopotential_path)

beta_r = uspp_data.nonlocal_beta_grid[0]
r_ab = uspp_data.r_ab[0]
r = uspp_data.r_grid[0]


gg, beta_g_sbt = sbt.batch_sbt(r, beta_r, ang_mom, kmax=100)
gr = jnp.outer(gg, r)

beta_g_num = einsum(
  beta_r, r**2, jn(ang_mom, gr), r_ab, "i r, r, g r, r -> i g"
)

print(jnp.allclose(beta_g_sbt, beta_g_num, atol=1e-5))
print(jnp.mean(jnp.abs(beta_g_sbt - beta_g_num)))
