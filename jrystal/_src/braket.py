"""Integration operations. """

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Complex
import einops
from .typing import ScalarGrid


def reciprocal_braket(
  bra: ScalarGrid[Complex, 3] | ScalarGrid[Float, 3],
  ket: ScalarGrid[Complex, 3] | ScalarGrid[Float, 3],
  vol: Float,
) -> Float:
  r"""This function calculate the inner product of <f|g> in reciprocal space

  .. math::
    <f|g> \approx \sum_g f^*(g)r(g) * vol / N / N

  where N is the number of grid size.

  NOTE: in this project, hartree and external energy integral is calculated
  in reciprocal space.

  Args:
      bra (ScalarGrid[Complex, 3] | ScalarGrid[Float, 3]): bra in reciprocal
        space.
      ket (ScalarGrid[Complex, 3] | ScalarGrid[Float, 3]): ket in reciprocal
        space.
      vol (Float): the volume of unit cell.

  Returns:
      Float: the value of the inner product.
  """
  if bra.shape != ket.shape:
    raise ValueError(
      f"bra and ket shape are not aligned. Got "
      f"{bra.shape} and {ket.shape}."
    )

  num_grids = np.prod(np.array(bra.shape))
  # Parseval's theorem
  parseval_factor = 1 / num_grids
  # numerical integration weights
  numerical_integral_weight = vol / num_grids
  product = jnp.sum(
    jnp.conj(bra) * ket
  ) * parseval_factor * numerical_integral_weight
  return product.real


def real_braket(
  bra: ScalarGrid[Complex, 3] | ScalarGrid[Float, 3],
  ket: ScalarGrid[Complex, 3] | ScalarGrid[Float, 3],
  vol: Float,
) -> Float:
  r"""This function calculate the inner product of <f|g> in real space

  .. math::
    <f|g> \approx \sum_r f^*(r)r(r) * vol / N

  where N is the number of grid size.

  NOTE: in this project, exchange-correlation energy integral is calculated
  in real space.

  Args:
      bra (ScalarGrid[Complex, 3] | ScalarGrid[Float, 3]): bra in reciprocal
        space.
      ket (ScalarGrid[Complex, 3] | ScalarGrid[Float, 3]): ket in reciprocal
        space.
      vol (Float): the volume of unit cell.

  Returns:
      Float: the value of the inner product.
  """
  if bra.shape != ket.shape:
    raise ValueError(
      f"bra and ket shape are not aligned. Got "
      f"{bra.shape} and {ket.shape}."
    )

  num_grids = np.prod(np.array(bra.shape))
  numerical_integral_weight = vol / num_grids
  product = jnp.sum(bra * ket) * numerical_integral_weight
  return product


def expectation(
  bra: ScalarGrid[Complex, 3] | ScalarGrid[Float, 3],
  hamiltonian: Float[Array, "nk n1 n2 n3"] | Float[Array, "n1 n2 n3"],
  vol: Float,
  ket: ScalarGrid[Complex, 3] | ScalarGrid[Float, 3] | None = None,
  diagonal: bool = False,
  mode: str = 'real'
) -> Array:
  """calculate the expectation of a hamiltonian operator in real space.

  See: https://en.wikipedia.org/wiki/Expectation_value_(quantum_mechanics)

  .. math::
    expectation_ij = <bra_i | hamil | ket_j>

  Args:
      bra (ScalarGrid): _description_
      hamiltonian (ComplexGrid): _description_
      vol (Float): volume of unit cell.
      ket (ComplexGrid, optional): the . Defaults to None.
      diagonal (bool, optional): if true, only calculate the diagonal elements.
      Defaults to False.
      mode (string, optional): options are 'real', 'reciprocal', 'kinetic'.
        There will be respective integral factors for different mode.

  Returns:
      Float: _description_
  """
  ket = bra if ket is None else ket
  assert bra.ndim == 6
  assert hamiltonian.ndim in [3, 4]
  num_grids = np.prod(bra.shape[-3:])

  if mode == 'reciprocal':
    # Parseval's theorem
    parseval_factor = 1 / num_grids
    integral_factor = vol / num_grids * parseval_factor

  elif mode == 'real':
    integral_factor = vol / num_grids

  elif mode == 'kinetic':
    integral_factor = 1.

  else:
    raise ValueError(
      'Argument \'mode\' must be one of \'real\', \'reciprocal\', or',
      f'\'kinetic\'. Got {mode}'
    )

  if hamiltonian.ndim == 3:
    if diagonal is False:
      p = "a nk ni1 n1 n2 n3, n1 n2 n3, a nk ni2 n1 n2 n3 -> a nk ni1 ni2"
    else:
      p = "a nk ni n1 n2 n3, n1 n2 n3, a nk ni n1 n2 n3 -> a nk ni"

  elif hamiltonian.ndim == 4:
    if diagonal is False:
      p = "a nk ni1 n1 n2 n3, nk n1 n2 n3, a nk ni2 n1 n2 n3 -> a nk ni1 ni2"
    else:
      p = "a nk ni n1 n2 n3, nk n1 n2 n3, a nk ni n1 n2 n3 -> a nk ni"

  else:
    raise ValueError(
      "Hamitonian array must have 3 or 4 dimensions",
      f"(with k-point channel). Given {hamiltonian.ndim} dimensions."
    )
  output = einops.einsum(jnp.conj(bra), hamiltonian, ket, p) * integral_factor
  return output
