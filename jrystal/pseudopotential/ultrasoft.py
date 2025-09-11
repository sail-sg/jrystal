from typing import Callable, List

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from einops import einsum
from interpax import CubicSpline

# from scipy.interpolate import CubicSpline
from jaxtyping import Array, Complex, Float

from .._src import pw
from .._src.utils import expand_coefficient
from ..grid import r2g_vector_grid
from .beta import beta_sbt_grid
from .clebsch_gordan import batch_gaunt, batch_wigner_3j, batch_clebsch_gordan
from .local import potential_local_reciprocal
from .nloc import energy_local
from .nloc import energy_nonlocal as _energy_nonlocal
from .nloc import potential_nonlocal_psi_reciprocal
from .spherical import (
  batch_sph_harm, cartesian_to_spherical, batch_sph_harm_real,
)
from .utils import map_over_atoms

__all__ = [
  'get_ultrasoft_coeff_fun',
  'density_grid',
  'density_grid_reciprocal',
  'potential_local_reciprocal',
  'potential_nonlocal_psi_reciprocal',
  'beta_sbt_grid',
  'energy_nonlocal',
  'energy_local',
]


def energy_nonlocal(
  us_coeff: Complex[Array, "s kpt band x y z"],
  potential_nl: Float[Array, "kpt beta x y z"],
  vol: float,
  occupation: Float[Array, "s kpt band"],
) -> Float[Array, "s kpt band"]:
  return _energy_nonlocal(us_coeff, potential_nl, vol, occupation)


def get_ultrasoft_coeff_fun(
  position: Float[Array, "atom 3"],
  kpts: Float[Array, "kpt 3"],
  g_vector_grid: Float[Array, "x y z 3"],
  freq_mask: Float[Array, "x y z"],
  vol: float,
  r_grid: List[Float[Array, "r"]],
  nonlocal_beta_grid: List[Float[Array, "beta r"]],
  nonlocal_angular_momentum: List[List[int]],
  nonlocal_q_matrix: List[Float[Array, "beta beta"]],
  beta_gk: Float[Array, "kpt beta x y z"],
) -> Callable[[Float[Array, "band x y z"]], Float[Array, "kpt beta x y z"]]:
  """

    S^{-1/2} | G + k > · C = S^{-1/2} psi(C)

  return a callable function that can transform the coefficients of the wave
  functions into the overlap sqrt.

    f: C -> S^{-1/2} psi(C)

  The reason why we need this functional is because it is usually intractable
  to calculate the sqrt of the overlap operator on plane wave basis, but we can
  calcualte the product of that with the coefficients of the wave functions.

  This output f function transform the orthogonal coefficients into the
  ultrasoft coefficient such that the new coefficients C' satisfy

    < C' | S | C' > = 1

  where S is the overlap operator in ultrasoft pseudopotential.
  """

  const = jnp.sqrt(4*jnp.pi)
  nonlocal_q_matrix = [q * const for q in nonlocal_q_matrix]

  _iden = [jnp.eye(q.shape[0]) for q in nonlocal_q_matrix]

  psi_G = potential_nonlocal_psi_reciprocal(
    position,
    g_vector_grid,
    kpts,
    r_grid,
    nonlocal_beta_grid,
    nonlocal_angular_momentum,
    _iden,
    beta_gk,
    concat=False
  )
  m = [p.shape[2] for p in psi_G]   # [k beta m x y z]

  def _process_psi_G(_pg):
    _pg = _pg.at[..., freq_mask].get()  # [k beta m g]
    _pg = jnp.reshape(_pg, (_pg.shape[0], -1, _pg.shape[-1]))  # [kpt i g]
    _pg = einsum(_pg, "k i g -> k g i")
    return _pg

  psi_G = jnp.concatenate([_process_psi_G(_pg) for _pg in psi_G], axis=-1)
  psi_G = psi_G / jnp.sqrt(vol)
  # [kpt g atom_i]

  q_mat = [
    scipy.linalg.block_diag(*([q] * _m)) for q, _m in zip(nonlocal_q_matrix, m)
  ]
  q_mat = scipy.linalg.block_diag(*q_mat) * vol / np.prod(freq_mask.shape)

  def _get_s_sqrt(B, x):
    U, R = jnp.linalg.qr(B)
    Sigma, V = jnp.linalg.eigh(R @ q_mat @ R.T.conj())
    # assert jnp.min(Sigma) >= -1, "The operator is not positive semi-definite."
    y = U.conj().T @ x
    y = V.conj().T @ y
    y = ((Sigma+1)**(-0.5)-1) * y
    y = V @ y
    y = U @ y
    y = y + x
    return y

  def f(
    coeff: Complex[Array, "s k band x y z"],
  ) -> Complex[Array, "s k band x y z"]:
    coeff = coeff.at[..., freq_mask].get()
    # coeff = jnp.reshape(coeff, (coeff.shape[:3], -1))  # [s k b G]
    coeff = einsum(coeff, "s k b g -> s b k g")  # [s band k g]

    # _get_s_sqrt requires the input to be [g, atom_m] and [g], but what we have
    # is psi_G as B [k g atom_m] and coeff as x [s b k g]
    _s_sqrt = jax.vmap(_get_s_sqrt, in_axes=(0, 0))  # map over kpt
    _s_sqrt = jax.vmap(
      jax.vmap(_s_sqrt, in_axes=(None, 0)), in_axes=(None, 0)
    )  # map over s and band
    output = _s_sqrt(psi_G, coeff.conj())  # shape = [s b k g]

    output = einsum(output, "s b k g -> s k g b")
    output = expand_coefficient(output, freq_mask)
    return output

  return f


def _get_ultrasoft_coeff_fun(
  position: Float[Array, "atom 3"],
  kpts: Float[Array, "kpt 3"],
  g_vector_grid: Float[Array, "x y z 3"],
  freq_mask: Float[Array, "x y z"],
  vol: float,
  r_grid: List[Float[Array, "r"]],
  nonlocal_beta_grid: List[Float[Array, "beta r"]],
  nonlocal_angular_momentum: List[List[int]],
  nonlocal_q_matrix: List[Float[Array, "beta beta"]],
  beta_gk: Float[Array, "kpt beta x y z"],
) -> Callable[[Float[Array, "band x y z"]], Float[Array, "kpt beta x y z"]]:
  """

    S^{-1/2} | G + k > · C = S^{-1/2} psi(C)

  return a callable function that can transform the coefficients of the wave
  functions into the overlap sqrt.

    f: C -> S^{-1/2} psi(C)

  The reason why we need this functional is because it is usually intractable
  to calculate the sqrt of the overlap operator on plane wave basis, but we can
  calcualte the product of that with the coefficients of the wave functions.

  This output f function transform the orthogonal coefficients into the
  ultrasoft coefficient such that the new coefficients C' satisfy

    < C' | S | C' > = 1

  where S is the overlap operator in ultrasoft pseudopotential.
  """
  # transform the q_ij matrix (1D radial) to the real space. The difference is
  # sizes = g_vector_grid.shape[:3]
  const = jnp.sqrt(4*jnp.pi)
  nonlocal_q_matrix = [-q * const for q in nonlocal_q_matrix]
  # The minus sign is because the q_ij matrix is negative semi-definite.
  # Need to make it positive semi-definite.

  psi_G = potential_nonlocal_psi_reciprocal(
    position,
    g_vector_grid,
    kpts,
    r_grid,
    nonlocal_beta_grid,
    nonlocal_angular_momentum,
    nonlocal_q_matrix,
    beta_gk,
    concat=False
  )
  # [kpt beta m x y z]  <G | beta>, the integral is analytically done,
  # therefore no factor is needed.

  # grid_sizes = g_vector_grid.shape[:3]
  # psi_G = [p * jnp.sqrt(vol / np.prod(grid_sizes)) for p in psi_G]

  def _process_psi_G(_pg):
    _pg = _pg.at[..., freq_mask].get()  # [k beta m g]
    _pg = jnp.reshape(_pg, (_pg.shape[0], -1, _pg.shape[-1]))  # [kpt i g]
    _pg = einsum(_pg, "k i g -> k g i")
    return _pg

  psi_G = jnp.concatenate([_process_psi_G(_pg) for _pg in psi_G], axis=-1)
  # [kpt g atom_i]

  def _get_s_sqrt(B, x):   # (B: [G atom_m], x: [G]) -> [G]
    # this function returns S^{-1/2} x, where S = I - B B^H
    # B is a matrix of shape [G, atom_m], and x is a vec of size G
    assert B.shape[0] == x.shape[0]
    # get the sqrt of the S matrix. S is written as S = I - B B^T
    U, L, _ = jnp.linalg.svd(B, full_matrices=False)  # U shape: [G atom_m]
    y = (U.conj().T@x.conj())  # [atom_m]
    # diag = (1. - L**2/(1+L**2))**0.5-1.
    # output = x + einsum(U.conj(), diag, y, "g b, b, b -> g")
    diag = (1. + L**2/(1+0.j-L**2))**0.5 - 1.
    output = x.conj() + einsum(U, diag.conj(), y, "g b, b, b -> g")
    return output   # output is a vec of size g

  def f(
    coeff: Complex[Array, "s k band x y z"],
  ) -> Complex[Array, "s k band x y z"]:
    coeff = coeff.at[..., freq_mask].get()
    # coeff = jnp.reshape(coeff, (coeff.shape[:3], -1))  # [s k b G]
    coeff = einsum(coeff, "s k b g -> s b k g")  # [s band k g]

    # _get_s_sqrt requires the input to be [g, atom_m] and [g], but what we have
    # is psi_G as B [k g atom_m] and coeff as x [s b k g]
    _s_sqrt = jax.vmap(_get_s_sqrt, in_axes=(0, 0))  # map over kpt
    _s_sqrt = jax.vmap(
      jax.vmap(_s_sqrt, in_axes=(None, 0)), in_axes=(None, 0)
    )  # map over s and band
    output = _s_sqrt(psi_G, coeff.conj())  # shape = [s b k g]

    output = einsum(output, "s b k g -> s k g b")
    output = expand_coefficient(output, freq_mask)
    return output

  return f


def _Q_ij_real_grid(
  r_vector_grid: Float[Array, "x y z 3"],
  position: Float[Array, "3"],
  r_grid: List[Float[Array, "r"]],
  nonlocal_augmentation_qij: List[Float[Array, "i j l x y z"]],
) -> List[Float[Array, "i j l m x y z"]]:
  """
  Calculate the Q_ij matrix: (q_with_l = False)

    q_ij(r) = q_ij(r_radius-R) * Y_lm(r_theta, r_phi)

  """
  def _Q_ij_single_atom(
    r_grid: Float[Array, "r"],
    qij: Float[Array, "i j l r"],
    pos: Float[Array, "3"],
  ) -> Float[Array, "i j x y z"]:
    l_max = qij.shape[2] - 1

    # calculate the spherical harmonic at the grid points.
    r_pos = r_vector_grid - jnp.expand_dims(pos, (0, 1, 2))  # [x y z 3]
    r_sph = cartesian_to_spherical(r_pos)
    r_norm, r_theta, r_phi = r_sph[..., 0], r_sph[..., 1], r_sph[..., 2]
    cs = CubicSpline(r_grid, qij, axis=-1)
    q_ij_radius = cs(r_norm)   # [i j l x y z]

    output = jnp.zeros(
      (qij.shape[0], qij.shape[1], l_max + 1, 2*l_max + 1,
        r_theta.shape[0], r_theta.shape[1], r_theta.shape[2])  # [i j l m x y z]
    ) + 0.j

    for _l in range(l_max + 1):
      sph_harm_grid = batch_sph_harm_real(_l, r_theta, r_phi)  # [x y z m]
      _q_ij_lm = einsum(
        q_ij_radius[:, :, _l, ...], sph_harm_grid,
        "i j x y z, x y z m -> i j m x y z"
      )
      m = sph_harm_grid.shape[-1]
      m_start = l_max - (m//2)
      m_end = m_start + m
      output = output.at[..., _l, m_start:m_end, :, :, :].set(
        _q_ij_lm
      )    # [i j l m x y z]  m = 0 is at the center.

    return output

  output = [
    _Q_ij_single_atom(r, q, p) for r, q, p in zip(
      r_grid, nonlocal_augmentation_qij, position
    )
  ]

  return output  # list of [i j l m x y z]


def _rho_ij(
  ultra_coeff: Complex[Array, "s kpt band x y z"],
  freq_mask: Float[Array, "x y z"],
  occupation: Float[Array, "s kpt band"],
  position: Float[Array, "atom 3"],
  vol: float,
  kpts: Float[Array, "kpt 3"],
  g_vector_grid: Float[Array, "x y z 3"],
  r_grid: List[Float[Array, "r"]],
  nonlocal_beta_grid: List[Float[Array, "beta r"]],
  nonlocal_angular_momentum: List[List[int]],
  beta_gk: List[Float[Array, "kpt beta x y z"]],
) -> List[Float[Array, "beta1 beta2"]]:
  """

  rho_ij = \sum_{s, k} \sum_{beta}
      f_{i, k} < \psi_{i, k} | beta > < beta | \psi_{i, k} >

  """
  _iden = [jnp.eye(b.shape[0]) for b in nonlocal_beta_grid]

  psi_g = potential_nonlocal_psi_reciprocal(
    position, g_vector_grid, kpts, r_grid, nonlocal_beta_grid,
    nonlocal_angular_momentum,
    _iden, beta_gk, concat=False
  )   # list of [kpt beta m x y z]
  ultra_coeff_dens = ultra_coeff.at[..., freq_mask].get()

  @map_over_atoms
  def _fun(psi_g, angular_momentum):
    _psi_g = psi_g.at[..., freq_mask].get()
    coeff_rho_sqrt = einsum(
      ultra_coeff_dens.conj(), _psi_g, "s k b g, k beta m g -> s k b beta m"
    )
    rho_i = einsum(
      coeff_rho_sqrt, occupation, coeff_rho_sqrt.conj(),
      's k b beta1 m, s k b, s k b beta2 m -> beta1 beta2'
    ) / np.prod(freq_mask.shape)
    # mask = angular_momentum[:, None] == angular_momentum[None, :]
    return rho_i

  output = _fun(psi_g, nonlocal_angular_momentum)
  return output


def _augmentation_density(
  q_ij: List[Float[Array, "i j l m x y z"]],
  rho_ij: List[Float[Array, "beta1 beta2"]],
  nonlocal_angular_momentum: List[List[int]],
  nonlocal_augmentation_q_with_l: List[bool],
) -> Float[Array, "s x y z"]:

  # @map_over_atoms
  def _fun(q, rho):
    return einsum(q, rho, "beta1 beta2 l m x y z, beta1 beta2 -> x y z")

  output = [
    _fun(q, rho) for q, rho in zip(q_ij, rho_ij)
  ]
  return jnp.sum(jnp.stack(output), axis=0)


def __augmentation_density(
  q_ij: List[Float[Array, "i j l m x y z"]],
  rho_ij: List[Float[Array, "beta1 beta2"]],
  nonlocal_angular_momentum: List[List[int]],
  nonlocal_augmentation_q_with_l: List[bool],
) -> Float[Array, "s x y z"]:
  """
  Calculate the augmentation density:

    n_aug(r) = \sum_ij CG_coeff Q_ij(r) * \rho_ij(r)

  q_ij is the real space Q_ij(r), which can be calculated by `_Q_ij_real_grid`.
  rho_ij is the real space density matrix, which can be calculated by _rho_ij.
  nonlocal_augmentation_q_with_l is the flag to indicate whether the Q_ij(r)
  is with angular momentum.

  """
  @map_over_atoms
  def _fun(q, rho, q_with_l, proj_l):
    l_q_max = q.shape[2] - 1
    l_proj_max = np.max(proj_l)

    m_proj = np.arange(-l_proj_max, l_proj_max+1)
    m_q = np.arange(-l_q_max, l_q_max + 1)

    if q_with_l:
      _cg = batch_gaunt(
        proj_l, proj_l, np.arange(l_q_max + 1), m_proj, m_proj, m_q
      )  # [beta, beta, q_l, m, m, q_m]
      _cg = einsum(
        _cg, (-1)**proj_l,  "b1 b2 l m1 m2 m, b2-> b1 b2 l m"
      )

      _cg *= jnp.sqrt(4*jnp.pi)

      output = einsum(
        _cg, q,
        "beta1 beta2 l m, beta1 beta2 l m x y z -> beta1 beta2 x y z"
      )
      # output = einsum(q, "beta1 beta2 l m x y z -> beta1 beta2 x y z")
      output = einsum(
        output, rho, "beta1 beta2 x y z, beta1 beta2 -> x y z"
      )  # the minus sign is because the q_ij matrix is negative semi-definite.
      return output

    else:
      raise NotImplementedError("q_with_l is not supported.")
      # return einsum(q, rho, "beta1 beta2 l m x y z, beta1 beta2 -> x y z")

  output = _fun(
    q_ij, rho_ij, nonlocal_augmentation_q_with_l, nonlocal_angular_momentum
  )
  return jnp.sum(jnp.stack(output), axis=0)


def density_grid(
  ultrasoft_coeff: Complex[Array, "s kpt band x y z"],
  freq_mask: Float[Array, "x y z"],
  occupation: Float[Array, "s kpt band"],
  kpts: Float[Array, "kpt 3"],
  r_vector_grid: List[Float[Array, "x y z 3"]],
  position: Float[Array, "atom 3"],
  vol: float,
  r_grid: List[Float[Array, "r"]],
  nonlocal_beta_grid: List[Float[Array, "beta r"]],
  nonlocal_angular_momentum: List[List[int]],
  nonlocal_augmentation_qij: List[Float[Array, "i j m x y z"]],
  nonlocal_augmentation_q_with_l: List[bool],
  beta_gk: List[Float[Array, "kpt beta x y z"]],
  _q_ij_real_grid: List[Float[Array, "i j l m x y z"]],
) -> Float[Array, "s x y z"]:
  """
    n(r) = n_pw(r) + \sum_ij Q_ij(r) * \rho_ij(r)
  """
  # g_vector_grid = g2r_vector_grid(g_vector_grid)
  g_vector_grid = r2g_vector_grid(r_vector_grid)
  density1 = pw.density_grid(ultrasoft_coeff, vol, occupation)

  # _q_ij_list = _Q_ij_real_grid(
  #   r_vector_grid, position, r_grid, nonlocal_augmentation_qij
  # )  # list of [beta1 beta2 L M x y z]

  _rho_ij_list = _rho_ij(
    ultrasoft_coeff, freq_mask, occupation, position, vol, kpts, g_vector_grid,
    r_grid, nonlocal_beta_grid, nonlocal_angular_momentum, beta_gk
  )  # list of [beta1 beta2]

  # the negative sign is because the q_ij matrix is negative semi-definite.
  density2 = _augmentation_density(
    _q_ij_real_grid, _rho_ij_list, nonlocal_angular_momentum,
    nonlocal_augmentation_q_with_l
  ) / vol * np.prod(freq_mask.shape)

  assert occupation.shape[0] == 1   # TODO: support spin polarization
  density2 = density2[None, ...]

  return density1, density2, _rho_ij_list
  # return density1 + density2


def density_grid_reciprocal(
  ultrasoft_coeff: Complex[Array, "s kpt band x y z"],
  occupation: Float[Array, "s kpt band"],
  kpts: Float[Array, "kpt 3"],
  r_vector_grid: List[Float[Array, "x y z 3"]],
  position: Float[Array, "atom 3"],
  vol: float,
  r_grid: List[Float[Array, "r"]],
  nonlocal_beta_grid: List[Float[Array, "beta r"]],
  nonlocal_angular_momentum: List[List[int]],
  nonlocal_augmentation_qij: List[Float[Array, "i j m x y z"]],
  nonlocal_augmentation_q_with_l: bool,
  beta_gk: List[Float[Array, "kpt beta x y z"]],
) -> Float[Array, "s x y z"]:
  density = density_grid(
    ultrasoft_coeff, occupation, kpts, r_vector_grid, position, vol,
    r_grid, nonlocal_beta_grid, nonlocal_angular_momentum,
    nonlocal_augmentation_qij, nonlocal_augmentation_q_with_l, beta_gk
  )

  return jnp.fft.fftn(density, axes=(1, 2, 3))
