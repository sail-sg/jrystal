"""LDA exchange-correlation functional."""
import jax.numpy as jnp
from jax.lax import stop_gradient
from jaxtyping import Array, Float


def lda_x_density(
  density_grid: Float[Array, '*d x y z']
) -> Float[Array, '*d x y z']:
  r"""Calculate the LDA exchange-correlation energy density.

  Implements the Local Density Approximation (LDA) for the exchange-correlation
  energy density in the spin-unpolarized case:

  .. math::
      \varepsilon_{xc}^{\text{LDA}}[n] = -\frac{3}{4}
      \left(\frac{3}{\pi}\right)^{1/3} n(\mathbf{r})^{1/3}

  where:

  - :math:`n(\mathbf{r})` is the electron density
  - :math:`\varepsilon_{xc}^{\text{LDA}}` is the exchange-correlation energy density

  Args:
    density_grid (Float[Array, '*d x y z']): Real-space electron density.
      May include batch dimensions.

  Returns:
    Float[Array, '*d x y z']: LDA exchange-correlation energy density.
      Preserves any batch dimensions from the input.
  """
  t3 = 3**(0.1e1 / 0.3e1)
  t4 = jnp.pi**(0.1e1 / 0.3e1)
  t8 = 2.220446049250313e-16**(0.1e1 / 0.3e1)
  t10 = jnp.where(0.1e1 <= 2.22044604925e-16, t8 * 2.22044604925e-16, 1)
  t11 = density_grid**(0.1e1 / 0.3e1)
  t15 = jnp.where(
    density_grid / 0.2e1 <= 1e-15, 0, -0.3e1 / 0.8e1 * t3 / t4 * t10 * t11
  )
  res = 0.2e1 * 1. * t15
  return res


def lda_x(density_grid: Float[Array, 'spin x y z'],
          kohn_sham: bool = False) -> Float[Array, 'spin x y z']:
  r"""Calculate the LDA exchange-correlation potential.

  Implements the Local Density Approximation (LDA) for the exchange-correlation
  potential in the spin-unpolarized case. The potential is obtained as the
  functional derivative of the LDA energy:

  .. math::
      v_{xc}^{\text{LDA}}(\mathbf{r}) = -\left(\frac{3\rho(\mathbf{r})}{\pi}
      \right)^{1/3}

  where:

  - :math:`\rho(\mathbf{r})` is the electron density
  - :math:`v_{xc}^{\text{LDA}}` is the exchange-correlation potential

  Args:
    density_grid (Float[Array, 'spin x y z']): Real-space electron density.
      The input density must contains spin axis.
    kohn_sham (bool, optional): If True, use Kohn-Sham formalism. Defaults to False.

  Returns:
    Float[Array, 'spin x y z']: LDA exchange-correlation potential.
  """
  assert density_grid.ndim == 4, ('density_grid must contains spin axis')
  density_grid = jnp.sum(density_grid, axis=0)

  if kohn_sham:
    density_grid = stop_gradient(density_grid)
    output = -(density_grid * 3. / jnp.pi)**(1 / 3)
  else:
    output = lda_x_density(density_grid)

  return jnp.expand_dims(output, axis=0)
