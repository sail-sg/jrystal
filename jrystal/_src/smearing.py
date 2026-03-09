"""Smearing-based occupations from Kohn-Sham eigenvalues.

All methods return occupations with the same shape as ``eigenvalues``:
``[spin, kpts, bands]``.
"""

import math
from typing import Optional, Union

import jax
import jax.numpy as jnp
from jax.scipy.special import erfc
from jaxtyping import Array, Float


def _reduced_energy(
  eigenvalues: Float[Array, "spin kpts bands"],
  chemical_potential: Union[float, Float[Array, ""]],
  smearing: Union[float, Float[Array, ""]],
) -> Float[Array, "spin kpts bands"]:
  """Return ``x = (eigenvalues - chemical_potential) / smearing``."""
  eigenvalues = jnp.asarray(eigenvalues)
  if eigenvalues.ndim != 3:
    raise ValueError(
      "eigenvalues must have shape [spin, kpts, bands], "
      f"got shape {eigenvalues.shape}."
    )

  dtype = jnp.result_type(
    eigenvalues, jnp.asarray(chemical_potential), jnp.asarray(smearing),
    jnp.float32
  )
  eigenvalues = eigenvalues.astype(dtype)
  chemical_potential = jnp.asarray(chemical_potential, dtype=dtype)
  smearing = jnp.asarray(smearing, dtype=dtype)

  # Keep JIT compatibility while preventing division by zero.
  sigma_min = jnp.finfo(dtype).tiny
  smearing = jnp.maximum(smearing, sigma_min)
  return (eigenvalues - chemical_potential) / smearing


def _physicists_hermite(n: int, x: Array) -> Array:
  """Evaluate physicists' Hermite polynomial H_n(x)."""
  if n < 0:
    raise ValueError(f"n must be non-negative, got {n}.")
  if n == 0:
    return jnp.ones_like(x)
  if n == 1:
    return 2.0 * x

  h_nm2 = jnp.ones_like(x)
  h_nm1 = 2.0 * x
  for k in range(2, n + 1):
    h_n = 2.0 * x * h_nm1 - 2.0 * (k - 1) * h_nm2
    h_nm2 = h_nm1
    h_nm1 = h_n
  return h_nm1


def fermi_dirac(
  eigenvalues: Float[Array, "spin kpts bands"],
  chemical_potential: Union[float, Float[Array, ""]] = 0.0,
  smearing: Union[float, Float[Array, ""]] = 1e-3,
) -> Float[Array, "spin kpts bands"]:
  """Fermi-Dirac occupation."""
  x = _reduced_energy(eigenvalues, chemical_potential, smearing)
  return 1.0 / (jnp.exp(x) + 1.0)


def gaussian(
  eigenvalues: Float[Array, "spin kpts bands"],
  chemical_potential: Union[float, Float[Array, ""]] = 0.0,
  smearing: Union[float, Float[Array, ""]] = 1e-3,
) -> Float[Array, "spin kpts bands"]:
  """Gaussian (order-0 Methfessel-Paxton) occupation."""
  x = _reduced_energy(eigenvalues, chemical_potential, smearing)
  return 0.5 * erfc(x)


def marzari_vanderbilt(
  eigenvalues: Float[Array, "spin kpts bands"],
  chemical_potential: Union[float, Float[Array, ""]] = 0.0,
  smearing: Union[float, Float[Array, ""]] = 1e-3,
) -> Float[Array, "spin kpts bands"]:
  r"""Marzari-Vanderbilt (cold smearing) occupation.

  Uses the same convention as Quantum ESPRESSO cold smearing in terms of
  reduced energy :math:`x = (\varepsilon - \mu)/\sigma`.
  """
  x = _reduced_energy(eigenvalues, chemical_potential, smearing)
  shift = 1.0 / jnp.sqrt(2.0)
  y = x + shift
  return 0.5 * erfc(y) + jnp.exp(-y * y) / jnp.sqrt(2.0 * jnp.pi)


def methfessel_paxton(
  eigenvalues: Float[Array, "spin kpts bands"],
  chemical_potential: Union[float, Float[Array, ""]] = 0.0,
  smearing: Union[float, Float[Array, ""]] = 1e-3,
  order: int = 1,
) -> Float[Array, "spin kpts bands"]:
  r"""Methfessel-Paxton occupation of arbitrary order.

  ``order=0`` recovers :func:`gaussian`.
  """
  if order < 0:
    raise ValueError(f"order must be >= 0, got {order}.")

  x = _reduced_energy(eigenvalues, chemical_potential, smearing)
  occupation = 0.5 * erfc(x)
  if order == 0:
    return occupation

  correction = jnp.zeros_like(x)
  inv_sqrt_pi = 1.0 / jnp.sqrt(jnp.pi)
  for m in range(1, order + 1):
    coeff = ((-1.0)**m) * inv_sqrt_pi / (float(math.factorial(m)) * (4.0**m))
    correction = correction + coeff * _physicists_hermite(2 * m - 1, x)

  return occupation + jnp.exp(-x * x) * correction


def _occupation_from_method(
  eigenvalues: Float[Array, "spin kpts bands"],
  chemical_potential: Union[float, Float[Array, ""]],
  smearing: Union[float, Float[Array, ""]],
  method: str,
  methfessel_paxton_order: int = 1,
) -> Float[Array, "spin kpts bands"]:
  """Compute occupations for a selected smearing method."""
  if not isinstance(method, str):
    raise TypeError(f"method must be a string, got {type(method)}.")
  method_normalized = method.lower().replace("_", "-")
  if method_normalized in ("fermi-dirac", "fermi", "fd"):
    return fermi_dirac(eigenvalues, chemical_potential, smearing)
  if method_normalized in ("gaussian", "gauss"):
    return gaussian(eigenvalues, chemical_potential, smearing)
  if method_normalized in ("marzari-vanderbilt", "marzari", "mv", "cold"):
    return marzari_vanderbilt(eigenvalues, chemical_potential, smearing)
  if method_normalized in ("methfessel-paxton", "methfessel", "mp"):
    return methfessel_paxton(
      eigenvalues,
      chemical_potential,
      smearing,
      order=methfessel_paxton_order,
    )
  raise ValueError(
    "Unknown smearing method "
    f"'{method}'. Expected one of "
    "['fermi-dirac', 'gaussian', 'marzari-vanderbilt', 'methfessel-paxton']."
  )


def _bisect_root(
  residual_fn,
  lower: Array,
  upper: Array,
  max_iter: int,
) -> Array:
  """Find a scalar root by bracketed bisection."""

  def body(_, state):
    lo, hi = state
    mid = 0.5 * (lo + hi)
    f_mid = residual_fn(mid)
    lo = jnp.where(f_mid < 0.0, mid, lo)
    hi = jnp.where(f_mid < 0.0, hi, mid)
    return lo, hi

  lo, hi = jax.lax.fori_loop(0, max_iter, body, (lower, upper))
  return 0.5 * (lo + hi)


def find_chemical_potential(
  eigenvalues: Float[Array, "spin kpts bands"],
  num_electrons: Union[float, Float[Array, ""]],
  smearing: Union[float, Float[Array, ""]] = 1e-3,
  *,
  method: str = "fermi-dirac",
  methfessel_paxton_order: int = 1,
  k_weights: Optional[Float[Array, " kpts"]] = None,
  max_iter: int = 200,
  bracket_scale: float = 20.0,
  use_custom_root: bool = False,
) -> Float[Array, ""]:
  r"""Solve for chemical potential from charge neutrality.

  Finds :math:`\mu` such that weighted occupations match ``num_electrons``:

  .. math::
    N_e = g_s \sum_{s,k,b} w_k \, f(\varepsilon_{skb}; \mu, \sigma),

  where ``g_s`` is spin degeneracy (``2`` if ``spin_restricted=True``,
  otherwise ``1``), and ``w_k`` are normalized ``k``-point weights.

  Args:
    eigenvalues: Kohn-Sham eigenvalues with shape ``[spin, kpts, bands]``.
    num_electrons: Target total electron number.
    smearing: Smearing width (same unit as eigenvalues).
    method: Smearing method name.
    methfessel_paxton_order: Order used only for ``method='methfessel-paxton'``.
    k_weights: Optional :math:`k`-point weights of shape ``[kpts]``.
      If ``None``, uniform weights are used.
    max_iter: Number of bisection iterations.
    bracket_scale: Bracket size multiplier relative to spectral spread.
    use_custom_root: If ``True``, wrap bisection in ``jax.lax.custom_root``.

  Returns:
    Float[Array, ""]: Chemical potential.
  """
  eigenvalues = jnp.asarray(eigenvalues)
  if eigenvalues.ndim != 3:
    raise ValueError(
      "eigenvalues must have shape [spin, kpts, bands], "
      f"got shape {eigenvalues.shape}."
    )
  if max_iter <= 0:
    raise ValueError(f"max_iter must be positive, got {max_iter}.")
  if bracket_scale <= 0.0:
    raise ValueError(f"bracket_scale must be positive, got {bracket_scale}.")

  dtype = jnp.result_type(
    eigenvalues,
    jnp.asarray(num_electrons),
    jnp.asarray(smearing),
    jnp.float32,
  )
  eigenvalues = eigenvalues.astype(dtype)
  num_electrons = jnp.asarray(num_electrons, dtype=dtype)
  smearing = jnp.asarray(smearing, dtype=dtype)

  num_spin, num_kpts, _ = eigenvalues.shape
  if k_weights is None:
    k_weights = jnp.ones((num_kpts,), dtype=dtype) / num_kpts
  else:
    k_weights = jnp.asarray(k_weights, dtype=dtype)
    if k_weights.shape != (num_kpts,):
      raise ValueError(
        f"k_weights must have shape ({num_kpts},), got {k_weights.shape}."
      )
    weight_sum = jnp.sum(k_weights)
    k_weights = k_weights / jnp.maximum(weight_sum, jnp.finfo(dtype).tiny)

  spin_degeneracy = 2.0 if num_spin == 1 else 1.0
  spin_degeneracy = jnp.asarray(spin_degeneracy, dtype=dtype)

  def electron_count(mu):
    occ = _occupation_from_method(
      eigenvalues,
      mu,
      smearing,
      method,
      methfessel_paxton_order,
    )
    weighted_occ = occ * k_weights[None, :, None]
    return spin_degeneracy * jnp.sum(weighted_occ)

  def residual(mu):
    return electron_count(mu) - num_electrons

  e_min = jnp.min(eigenvalues)
  e_max = jnp.max(eigenvalues)
  spread = jnp.maximum(e_max - e_min, smearing)
  pad = bracket_scale * spread + 10.0 * smearing
  lower = e_min - pad
  upper = e_max + pad

  res_lower = residual(lower)
  res_upper = residual(upper)
  if (
    not isinstance(res_lower, jax.core.Tracer) and
    not isinstance(res_upper, jax.core.Tracer)
  ):
    res_lower_f = float(res_lower)
    res_upper_f = float(res_upper)
    if res_lower_f > 0.0 or res_upper_f < 0.0:
      raise ValueError(
        "Unable to bracket chemical potential. "
        f"Residual at lower bound is {res_lower_f:.6e}, "
        f"at upper bound is {res_upper_f:.6e}. "
        "Check num_electrons, k_weights, or bracket_scale."
      )

  if use_custom_root:

    def solve_fn(f, initial_guess):
      del initial_guess
      return _bisect_root(f, lower, upper, max_iter)

    def tangent_solve(g, y):
      one = jnp.asarray(1.0, dtype=y.dtype)
      return y / g(one)

    initial_guess = 0.5 * (lower + upper)
    return jax.lax.custom_root(
      residual,
      initial_guess,
      solve_fn,
      tangent_solve,
    )

  return _bisect_root(residual, lower, upper, max_iter)


__all__ = [
  "fermi_dirac",
  "gaussian",
  "marzari_vanderbilt",
  "methfessel_paxton",
  "find_chemical_potential",
]
