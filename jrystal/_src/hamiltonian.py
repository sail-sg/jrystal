# Copyright 2025 Garena Online Private Limited
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
"""Hamiltonian matrix construction in a plane-wave basis."""

from typing import Union, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Int

from . import braket, potential, pw, utils
from .hessian import complex_hessian
from .kinetic import kinetic_operator


def _hamiltonian_matrix(
  coefficient: Complex[Array, "spin kpt band x y z"],
  positions: Float[Array, "atom 3"],
  charges: Int[Array, " atom"],
  effictive_density_grid: Union[Float[Array, "x y z"],
                                Float[Array, "spin x y z"]],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  vol: Float,
  xc: str = 'lda_x',
  kohn_sham: bool = False
) -> Float[Array, "spin kpt band band"]:
  r"""Compute Hamiltonian matrix elements :math:`H_{ij}`.

  Args:
    coefficient (Complex[Array, "spin kpt band x y z"]): Plane-wave
      coefficients.
    positions (Float[Array, "atom 3"]): Atomic positions.
    charges (Int[Array, "atom"]): Atomic charges.
    effictive_density_grid (Union[Float[Array, "x y z"], Float[Array, "spin x y z"]]):
      Density used to build the effective potential.
    g_vector_grid (Float[Array, "x y z 3"]): Reciprocal-space G-vector grid.
    kpts (Float[Array, "kpt 3"]): :math:`k` points.
    vol (Float): Unit-cell volume.
    xc (str): XC functional specification.
    kohn_sham (bool): Whether to use Kohn-Sham form.

  Returns:
    Float[Array, "spin kpt band band"]: Hamiltonian matrices.
  """
  if effictive_density_grid.ndim == 4:  # has spin axis, sum over spin axis
    effictive_density_grid = jnp.sum(effictive_density_grid, axis=0)

  v_eff = potential.effective(
    effictive_density_grid,
    positions,
    charges,
    g_vector_grid,
    vol,
    split=False,
    xc_type=xc,
    kohn_sham=kohn_sham
  )  # [x y z]
  wave_grid = pw.wave_grid(coefficient, vol)  # [spin kpt band x y z]
  f_eff = braket.expectation(wave_grid, v_eff, vol, diagonal=False, mode="real")

  t_kin = kinetic_operator(g_vector_grid, kpts)
  f_kin = braket.expectation(
    coefficient, t_kin, vol, diagonal=False, mode='kinetic'
  )

  return (f_eff + f_kin)


def hamiltonian_matrix_trace(
  band_coefficient: Complex[Array, "spin kpt band x y z"],
  positions: Float[Array, "atom 3"],
  charges: Int[Array, " atom"],
  effictive_density_grid: Union[Float[Array, "x y z"],
                                Float[Array, "spin x y z"]],
  vol: Float,
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  kpts_weights: Optional[Float[Array, " kpt"]] = None,
  xc: str = 'lda_x',
  *,
  kohn_sham: bool = True,
  keep_spin_axis: bool = False,
) -> Union[Float[Array, " spin"], Float[Array, "spin kpt"]]:
  r"""Compute the trace of the Hamiltonian matrix.

  Args:
    band_coefficient (Complex[Array, "spin kpt band x y z"]): Plane-wave
      coefficients.
    positions (Float[Array, "atom 3"]): Atomic positions.
    charges (Int[Array, "atom"]): Atomic charges.
    effictive_density_grid (Union[Float[Array, "x y z"], Float[Array, "spin x y z"]]):
      Density used to build the effective potential.
    g_vector_grid (Float[Array, "x y z 3"]): Reciprocal-space G-vector grid.
    kpts (Float[Array, "kpt 3"]): :math:`k` points.
    vol (Float): Unit-cell volume.
    xc (str): XC functional specification.
    kohn_sham (bool): Whether to use Kohn-Sham form.
    keep_spin_axis (bool): If ``True``, return one value per spin channel.

  Returns:
    Union[Float[Array, "spin"], Float[Array, "spin kpt"]]: Trace values.
  """
  if kohn_sham:
    effictive_density_grid = jax.lax.stop_gradient(effictive_density_grid)

  v_eff = potential.effective(
    effictive_density_grid,
    positions,
    charges,
    g_vector_grid,
    vol,
    split=False,
    xc_type=xc,
    kohn_sham=kohn_sham,
  )
  wave_grid = pw.wave_grid(band_coefficient, vol)
  f_eff = braket.expectation(wave_grid, v_eff, vol, diagonal=True, mode="real")

  t_kin = kinetic_operator(g_vector_grid, kpts)
  f_kin = braket.expectation(
    band_coefficient, t_kin, vol, diagonal=True, mode='kinetic'
  )  # [spin, kpt, band]
  hamil_trace = (f_eff + f_kin).real
  if kpts_weights is not None:
    hamil_trace = hamil_trace * kpts_weights[None, :, None]

  if keep_spin_axis:
    return jnp.sum(hamil_trace, axis=(1, 2))
  else:
    return jnp.sum(hamil_trace, axis=(0, 1, 2))


def hamiltonian_matrix(
  band_coefficient: Complex[Array, "spin kpt band x y z"],
  positions: Float[Array, "atom 3"],
  charges: Int[Array, " atom"],
  effictive_density_grid: Union[Float[Array, "x y z"],
                                Float[Array, "spin x y z"]],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  vol: Float,
  xc: str = 'lda_x',
  kohn_sham: bool = True,
) -> Complex[Array, "spin kpt band band"]:
  r"""Compute the full Hamiltonian matrix in the orbital basis.

  Args:
    band_coefficient (Complex[Array, "spin kpt band x y z"]): Plane-wave
      coefficients.
    positions (Float[Array, "atom 3"]): Atomic positions.
    charges (Int[Array, "atom"]): Atomic charges.
    effictive_density_grid (Union[Float[Array, "x y z"], Float[Array, "spin x y z"]]):
      Density used to build the effective potential.
    g_vector_grid (Float[Array, "x y z 3"]): Reciprocal-space G-vector grid.
    kpts (Float[Array, "kpt 3"]): :math:`k` points.
    vol (Float): Unit-cell volume.
    xc (str): XC functional specification.
    kohn_sham (bool): Whether to use Kohn-Sham form.

  Returns:
    Complex[Array, "spin kpt band band"]: Hamiltonian matrices for each spin
    and :math:`k` point.
  """
  assert band_coefficient.ndim == 6, "band_coefficient must have 6 dimensions"
  num_bands = band_coefficient.shape[-4]

  def hamil_k(k, coeff_k):
    k = jnp.reshape(k, [-1, 3])

    def efun(u):
      _coeff = jnp.einsum("i,ixyz->xyz", u, coeff_k)
      _coeff = jnp.expand_dims(_coeff, axis=range(3))

      band_energies = hamiltonian_matrix_trace(
        _coeff,
        positions,
        charges,
        effictive_density_grid,
        g_vector_grid,
        k,
        vol,
        xc,
        kohn_sham=kohn_sham,
        keep_spin_axis=False,
      )
      return 0.5 * jnp.sum(band_energies).astype(band_coefficient.dtype)

    x = jnp.ones(num_bands, dtype=band_coefficient.dtype)
    return complex_hessian(efun, x)

  h = jax.vmap(
    lambda coeff: jax.vmap(hamil_k, in_axes=(0, 0), out_axes=0)(kpts, coeff)
  )(
    band_coefficient
  )

  return h


def _hamiltonian_matrix_basis(
  freq_mask: Int[Array, "x y z"],
  positions: Float[Array, "atom 3"],
  charges: Int[Array, "atom"],
  effictive_density_grid: Float[Array, "x y z"],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  vol: Float,
  xc: str = 'lda_x',
  kohn_sham: bool = True,
) -> Complex[Array, "kpt band band"]:
  r"""Compute Hamiltonian matrices in the masked plane-wave basis.

  Args:
    freq_mask (Int[Array, "x y z"]): Basis mask for selected G vectors.
    positions (Float[Array, "atom 3"]): Atomic positions.
    charges (Int[Array, "atom"]): Atomic charges.
    effictive_density_grid (Float[Array, "x y z"]): Density used for the
      effective potential.
    g_vector_grid (Float[Array, "x y z 3"]): Reciprocal-space G-vector grid.
    kpts (Float[Array, "kpt 3"]): :math:`k` points.
    vol (Float): Unit-cell volume.
    xc (str): XC functional specification.
    kohn_sham (bool): Whether to use Kohn-Sham form.

  Returns:
    Complex[Array, "kpt band band"]: Hamiltonian matrices per :math:`k` point.
  """

  num_basis = jnp.sum(freq_mask)

  if g_vector_grid.dtype == jnp.float64:
    _dtype = jnp.complex128
  elif g_vector_grid.dtype == jnp.float32:
    _dtype = jnp.complex64
  else:
    raise ValueError(f"Unsupported dtype: {g_vector_grid.dtype}")

  def hamil_k(k):
    k = k.reshape([-1, 3])

    def efun(u):
      u = jnp.expand_dims(u, axis=(0, 1, 3))  # add spin, kpt, and band axes

      _coeff = utils.expand_coefficient(u, freq_mask)
      hamil_trace = hamiltonian_matrix_trace(
        _coeff,
        positions,
        charges,
        effictive_density_grid,
        g_vector_grid,
        k,
        vol,
        xc,
        kohn_sham,
      )
      return 0.5 * jnp.sum(hamil_trace).astype(_dtype)

    x = jnp.ones(num_basis, dtype=_dtype)
    return complex_hessian(efun, x)

  hamil_basis = jax.vmap(hamil_k)(kpts)
  return hamil_basis
