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
"""Energy terms for plane-wave electronic-structure calculations."""
from typing import Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float, Int

from . import braket, potential, pw
from . import xc as _xc
from .ewald import ewald_coulomb_repulsion
from .grid import translation_vectors
from .utils import (
  absolute_square,
  safe_real,
  wave_to_density,
  wave_to_density_reciprocal,
)


def hartree(
  density_grid_reciprocal: Complex[Array, 'spin x y z'],
  g_vector_grid: Float[Array, 'x y z 3'],
  vol: Float,
  kohn_sham: bool = False
) -> Float:
  r"""Compute the Hartree energy from reciprocal-space density.

  Args:
    density_grid_reciprocal (Complex[Array, 'spin x y z']): Spin-resolved
      reciprocal-space density.
    g_vector_grid (Float[Array, 'x y z 3']): Reciprocal-space G-vector grid.
    vol (Float): Unit-cell volume.
    kohn_sham (bool): Whether to use the Kohn-Sham potential convention.

  Returns:
    Float: Hartree energy.
  """
  dim = g_vector_grid.shape[-1]

  assert density_grid_reciprocal.ndim == dim + 1, (
    'density_grid_reciprocal must contains spin axis'
  )

  v_hartree_reciprocal = potential.hartree_reciprocal(
    density_grid_reciprocal, g_vector_grid, kohn_sham
  )
  v_hartree_reciprocal = jnp.expand_dims(v_hartree_reciprocal, axis=0)
  hartree_energy = braket.reciprocal_braket(
    v_hartree_reciprocal, density_grid_reciprocal, vol
  )

  return safe_real(hartree_energy)


def external(
  density_grid_reciprocal: Complex[Array, 'spin x y z'],
  position: Float[Array, 'atom 3'],
  charge: Float[Array, ' atom'],
  g_vector_grid: Float[Array, 'x y z 3'],
  vol: Float
) -> Float:
  r"""Compute the electron-ion external potential energy.

  Args:
    density_grid_reciprocal (Complex[Array, 'spin x y z']): Spin-resolved
      reciprocal-space density.
    position (Float[Array, 'atom 3']): Atomic positions.
    charge (Float[Array, 'atom']): Atomic charges.
    g_vector_grid (Float[Array, 'x y z 3']): Reciprocal-space G-vector grid.
    vol (Float): Unit-cell volume.

  Returns:
    Float: External potential energy.
  """
  dim = g_vector_grid.shape[-1]

  assert density_grid_reciprocal.ndim == dim + 1, (
    'density_grid_reciprocal must contains spin axis'
  )

  v_external_reciprocal = potential.external_reciprocal(
    position, charge, g_vector_grid, vol
  )
  v_external_reciprocal = jnp.expand_dims(v_external_reciprocal, axis=0)
  external_energy = braket.reciprocal_braket(
    v_external_reciprocal, density_grid_reciprocal, vol
  )

  return safe_real(external_energy)


def kinetic(
  coeff_grid: Complex[Array, 'spin kpt band x y z'],
  g_vector_grid: Float[Array, 'x y z 3'],
  kpts: Float[Array, 'kpt 3'],
  kpts_weights: Optional[Float[Array, ' kpt']] = None,
  occupation: Optional[Float[Array, 'spin kpt band']] = None,
) -> Union[Float, Float[Array, "spin kpt band"]]:
  r"""Compute kinetic energy from plane-wave coefficients.

  Args:
    coeff_grid (Complex[Array, 'spin kpt band x y z']): Plane-wave coefficients.
    g_vector_grid (Float[Array, 'x y z 3']): Reciprocal-space G-vector grid.
    kpts (Float[Array, 'kpt 3']): :math:`k`-point coordinates.
    kpts_weights (Float[Array, ' kpt']): :math:`k`-point weights.
    occupation (Optional[Float[Array, 'spin kpt band']]): Occupation numbers.

  Returns:
    Union[Float, Float[Array, "spin kpt band"]]: Total kinetic energy when
    ``occupation`` is provided; otherwise per-state kinetic energies.
  """

  dim = g_vector_grid.shape[-1]

  _g = jnp.expand_dims(g_vector_grid, axis=range(3))
  _k = jnp.expand_dims(kpts, axis=[0] + [i + 2 for i in range(dim + 1)])
  e_kin = jnp.sum((_g + _k)**2, axis=-1)  # [1, nk, ni, x y z]
  e_kin = jnp.sum(e_kin * absolute_square(coeff_grid), axis=range(3, dim + 3))

  if occupation is not None:
    if kpts_weights is not None:
      assert occupation.shape[1] == kpts_weights.shape[0], (
        f"occupation.shape[1] ({occupation.shape[1]}) must be equal to "
        f"kpts_weights.shape[0] ({kpts_weights.shape[0]})."
      )
      occupation = occupation * kpts_weights[None, :, None]
    e_kin = jnp.sum(e_kin * occupation) / 2
  else:
    e_kin /= 2

  return safe_real(e_kin)


def xc_energy(
  density_grid: Float[Array, 'spin x y z'],
  g_vector_grid: Float[Array, 'x y z 3'],
  vol: Float,
  xc_type: str,
  kohn_sham: bool = False,
  tau: Optional[Float[Array, 'spin x y z']] = None,
) -> Float:
  r"""Compute exchange-correlation energy for a real-space density.

  Supports LDA, GGA, and MGGA functionals, including compound specifications
  such as ``'gga_x_pbe+gga_c_pbe'``.

  Args:
    density_grid (Float[Array, 'spin x y z']): Spin-resolved real-space
      density.
    g_vector_grid (Float[Array, 'x y z 3']): Reciprocal-space G-vector grid.
    vol (Float): Unit-cell volume.
    xc_type (str): XC functional specification.
    kohn_sham (bool): Whether to compute the Kohn-Sham XC potential form.
    tau (Optional[Float[Array, 'spin x y z']]): Kinetic energy density.
      Required for MGGA functionals.

  Returns:
    Float: Exchange-correlation energy.
  """

  assert density_grid.ndim == 4, ('density_grid must contains spin axis')

  num_grid = jnp.prod(jnp.array(density_grid.shape[-3:]))
  polarized = density_grid.shape[0] == 2

  if kohn_sham:
    raise NotImplementedError

  level = _xc.xc_level(xc_type)

  # Prepare jxc inputs: strip the spin axis for unpolarized
  rho = density_grid if polarized else density_grid[0]

  sigma = None
  if level in ('gga', 'mgga'):
    sigma = _xc.compute_sigma(density_grid, g_vector_grid)

  tau_arg = None
  if level == 'mgga':
    if tau is None:
      raise ValueError(
        f"MGGA functional '{xc_type}' requires tau (kinetic energy density)."
      )
    tau_arg = tau if polarized else tau[0]

  exc = _xc.xc_energy_density(rho, xc_type, polarized, sigma, tau_arg)
  # exc shape: (x, y, z); density_grid shape: (spin, x, y, z)
  # Broadcasting gives sum over spins automatically.
  e_xc = jnp.sum(exc * density_grid)
  e_xc = safe_real(e_xc)
  return e_xc * vol / num_grid


def nuclear_repulsion(
  position: Float[Array, 'atom 3'],
  charge: Float[Array, ' atom'],
  cell_vectors: Float[Array, '3 3'],
  g_vector_grid: Float[Array, 'x y z 3'],
  vol: Float,
  ewald_eta: float,
  ewald_cutoff: float,
) -> Float:
  r"""Compute ion-ion repulsion energy with Ewald summation.

  Args:
    position (Float[Array, 'atom 3']): Atomic positions.
    charge (Float[Array, 'atom']): Atomic charges.
    cell_vectors (Float[Array, '3 3']): Real-space cell vectors.
    g_vector_grid (Float[Array, 'x y z 3']): Reciprocal-space G-vector grid.
    vol (Float): Unit-cell volume.
    ewald_eta (float): Ewald splitting parameter.
    ewald_cutoff (float): Real-space cutoff.

  Returns:
    Float: Nuclear repulsion energy.
  """
  ewald_grid = translation_vectors(cell_vectors, ewald_cutoff)
  return ewald_coulomb_repulsion(
    position, charge, g_vector_grid, vol, ewald_eta, ewald_grid
  )


def total_energy(
  coefficient: Complex[Array, "spin kpts band x y z"],
  position: Float[Array, "atom 3"],
  charge: Int[Array, " atom"],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  vol: Float,
  kpts_weights: Optional[Float[Array, " kpt"]] = None,
  occupation: Optional[Float[Array, "spin kpt band"]] = None,
  xc: str = 'lda_x',
  *,
  kohn_sham: bool = False,
  split: bool = False,
) -> Union[Float, Tuple[Float, Float, Float, Float]]:
  r"""Compute total electronic energy.

  The returned total includes kinetic, external, Hartree, and XC terms.

  Args:
    coefficient (Complex[Array, "spin kpts band x y z"]): Plane-wave
      coefficients.
    position (Float[Array, "atom 3"]): Atomic positions.
    charge (Int[Array, "atom"]): Atomic charges.
    g_vector_grid (Float[Array, "x y z 3"]): Reciprocal-space G-vector grid.
    kpts (Float[Array, "kpt 3"]): :math:`k` points.
    kpts_weights (Float[Array, " kpt"]): :math:`k`-point weights.
    vol (Float): Unit-cell volume.
    occupation (Optional[Float[Array, "spin kpt band"]]): Occupation numbers.
    kohn_sham (bool): Whether to use Kohn-Sham formalism.
    xc (str): XC functional specification.
    split (bool): If ``True``, return component energies.

  Returns:
    Union[Float, Tuple[Float, Float, Float, Float]]: Total energy or
    ``(E_kin, E_ext, E_har, E_xc)``.

  .. warning::
    This function does not include ion-ion Ewald energy.
  """

  wave_grid_arr = pw.wave_grid(coefficient, vol)

  occupation = jnp.ones(
    shape=coefficient.shape[:3]
  ) if occupation is None else occupation

  density_grid = wave_to_density(wave_grid_arr, occupation)
  density_grid_rec = wave_to_density_reciprocal(wave_grid_arr, occupation)

  # Compute tau for MGGA functionals
  tau = None
  if _xc.xc_level(xc) == 'mgga':
    tau = pw.tau_grid(coefficient, vol, g_vector_grid, kpts, occupation)

  e_kin = kinetic(coefficient, g_vector_grid, kpts, kpts_weights, occupation)
  e_ext = external(density_grid_rec, position, charge, g_vector_grid, vol)
  e_har = hartree(density_grid_rec, g_vector_grid, vol, kohn_sham)
  e_xc = xc_energy(density_grid, g_vector_grid, vol, xc, kohn_sham, tau)

  if split:
    return e_kin, e_ext, e_har, e_xc

  return e_kin + e_ext + e_har + e_xc


def band_energy(*args, **kwargs):
  raise DeprecationWarning(
    "band_energy is deprecated. Use `hamiltonian_matrix_diagonal(...)` instead."
  )


def hamiltonian_matrix_diagonal(
  coefficient: Complex[Array, "spin kpt band x y z"],
  position: Float[Array, "atom 3"],
  charge: Int[Array, " atom"],
  vol: Float,
  g_vector_grid: Float[Array, "x y z 3"],
  occupation: Float[Array, "spin kpt band"],
  kpts: Float[Array, "kpt 3"],
  kpts_weights: Optional[Float[Array, " kpt"]] = None,
  kohn_sham: bool = False,
  xc_type: str = "lda_x"
) -> Float[Array, "spin kpt band"]:
  r"""Compute the diagonal elements of the Hamiltonian matrix.

  Args:
    coefficient (Complex[Array, "spin kpt band x y z"]): Plane-wave
      coefficients.
    position (Float[Array, "atom 3"]): Atomic positions.
    charge (Int[Array, "atom"]): Atomic charges.
    g_vector_grid (Float[Array, "x y z 3"]): Reciprocal-space G-vector grid.
    kpts (Float[Array, "kpt 3"]): :math:`k` points.
    kpts_weights (Float[Array, " kpt"]): :math:`k`-point weights.
    vol (Float): Unit-cell volume.
    occupation (Float[Array, "spin kpt band"]): Occupation numbers.
    kohn_sham (bool): Whether to use Kohn-Sham formalism.
    xc_type (str): XC functional specification.

  Returns:
    Float[Array, "spin kpt band"]: Band energies.
  """

  density_grid_sum = pw.density_grid(coefficient, vol, occupation)
  density_per_band = pw.density_grid(coefficient, vol)
  v_eff = potential.effective(
    density_grid_sum,
    position,
    charge,
    g_vector_grid,
    vol,
    split=False,
    xc_type=xc_type,
    kohn_sham=kohn_sham
  )
  e_eff = braket.real_braket(density_per_band, v_eff, vol)

  e_kin = kinetic(coefficient, g_vector_grid, kpts, kpts_weights)
  assert np.array_equal(e_kin.shape, e_eff.shape)
  return safe_real(e_eff + e_kin)
