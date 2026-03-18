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


def _resolve_spin_polorized(
  eigenvalues: Float[Array, "spin kpts bands"],
  spin_polorized: Optional[bool],
) -> bool:
  """Resolve spin mode; infer from eigenvalue shape when not provided."""
  if spin_polorized is None:
    # Convention requested by user:
    # spin=1 -> non-polarized occupations in [0, 2]
    return eigenvalues.shape[0] != 1
  if not isinstance(spin_polorized, bool):
    raise TypeError(
      "spin_polorized must be bool or None. "
      f"Got {type(spin_polorized)}."
    )
  return spin_polorized


def _occupation_bounds(spin_polorized: bool) -> tuple[float, float]:
  """Return lower/upper occupation bounds for spin mode."""
  return 0.0, 1.0 if spin_polorized else 2.0


def _apply_spin_mode(
  occupation: Float[Array, "spin kpts bands"],
  spin_polorized: bool,
) -> Float[Array, "spin kpts bands"]:
  """Scale/clip occupations according to spin convention."""
  lo, hi = _occupation_bounds(spin_polorized)
  occ = occupation if spin_polorized else 2.0 * occupation
  return jnp.clip(occ, lo, hi)


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
  spin_polorized: Optional[bool] = None,
) -> Float[Array, "spin kpts bands"]:
  """Fermi-Dirac occupation.

  Args:
    spin_polorized (bool): If ``True``, occupations are in ``[0, 1]``.
      If ``False``, occupations are in ``[0, 2]``.
  """
  x = _reduced_energy(eigenvalues, chemical_potential, smearing)
  spin_polorized = _resolve_spin_polorized(x, spin_polorized)
  occ = 1.0 / (jnp.exp(x) + 1.0)
  return _apply_spin_mode(occ, spin_polorized)


def gaussian(
  eigenvalues: Float[Array, "spin kpts bands"],
  chemical_potential: Union[float, Float[Array, ""]] = 0.0,
  smearing: Union[float, Float[Array, ""]] = 1e-3,
  spin_polorized: Optional[bool] = None,
) -> Float[Array, "spin kpts bands"]:
  """Gaussian (order-0 Methfessel-Paxton) occupation."""
  x = _reduced_energy(eigenvalues, chemical_potential, smearing)
  spin_polorized = _resolve_spin_polorized(x, spin_polorized)
  occ = 0.5 * erfc(x)
  return _apply_spin_mode(occ, spin_polorized)


def marzari_vanderbilt(
  eigenvalues: Float[Array, "spin kpts bands"],
  chemical_potential: Union[float, Float[Array, ""]] = 0.0,
  smearing: Union[float, Float[Array, ""]] = 1e-3,
  spin_polorized: Optional[bool] = None,
) -> Float[Array, "spin kpts bands"]:
  r"""Marzari-Vanderbilt (cold smearing) occupation.

  Uses the same convention as Quantum ESPRESSO cold smearing in terms of
  reduced energy :math:`x = (\varepsilon - \mu)/\sigma`.
  """
  x = _reduced_energy(eigenvalues, chemical_potential, smearing)
  spin_polorized = _resolve_spin_polorized(x, spin_polorized)
  shift = 1.0 / jnp.sqrt(2.0)
  y = x + shift
  occ = 0.5 * erfc(y) + jnp.exp(-y * y) / jnp.sqrt(2.0 * jnp.pi)
  return _apply_spin_mode(occ, spin_polorized)


def methfessel_paxton(
  eigenvalues: Float[Array, "spin kpts bands"],
  chemical_potential: Union[float, Float[Array, ""]] = 0.0,
  smearing: Union[float, Float[Array, ""]] = 1e-3,
  order: int = 1,
  spin_polorized: Optional[bool] = None,
) -> Float[Array, "spin kpts bands"]:
  r"""Methfessel-Paxton occupation of arbitrary order.

  ``order=0`` recovers :func:`gaussian`.
  """
  if order < 0:
    raise ValueError(f"order must be >= 0, got {order}.")

  x = _reduced_energy(eigenvalues, chemical_potential, smearing)
  spin_polorized = _resolve_spin_polorized(x, spin_polorized)
  occupation = 0.5 * erfc(x)
  if order == 0:
    return _apply_spin_mode(occupation, spin_polorized)

  correction = jnp.zeros_like(x)
  inv_sqrt_pi = 1.0 / jnp.sqrt(jnp.pi)
  for m in range(1, order + 1):
    coeff = ((-1.0)**m) * inv_sqrt_pi / (float(math.factorial(m)) * (4.0**m))
    correction = correction + coeff * _physicists_hermite(2 * m - 1, x)

  occ = occupation + jnp.exp(-x * x) * correction
  return _apply_spin_mode(occ, spin_polorized)


def _occupation_from_method(
  eigenvalues: Float[Array, "spin kpts bands"],
  chemical_potential: Union[float, Float[Array, ""]],
  smearing: Union[float, Float[Array, ""]],
  method: str,
  methfessel_paxton_order: int = 1,
  spin_polorized: Optional[bool] = None,
) -> Float[Array, "spin kpts bands"]:
  """Compute occupations for a selected smearing method."""
  if not isinstance(method, str):
    raise TypeError(f"method must be a string, got {type(method)}.")
  method_normalized = method.lower().replace("_", "-")
  if method_normalized in ("fermi-dirac", "fermi", "fd"):
    return fermi_dirac(
      eigenvalues, chemical_potential, smearing, spin_polorized
    )
  if method_normalized in ("gaussian", "gauss"):
    return gaussian(
      eigenvalues, chemical_potential, smearing, spin_polorized
    )
  if method_normalized in ("marzari-vanderbilt", "marzari", "mv", "cold"):
    return marzari_vanderbilt(
      eigenvalues, chemical_potential, smearing, spin_polorized
    )
  if method_normalized in ("methfessel-paxton", "methfessel", "mp"):
    return methfessel_paxton(
      eigenvalues,
      chemical_potential,
      smearing,
      order=methfessel_paxton_order,
      spin_polorized=spin_polorized,
    )
  raise ValueError(
    "Unknown smearing method "
    f"'{method}'. Expected one of "
    "['fermi-dirac', 'gaussian', 'marzari-vanderbilt', 'methfessel-paxton']."
  )


def _normalize_k_weights(
  k_weights: Optional[Float[Array, " kpts"]],
  num_kpts: int,
  dtype,
) -> Float[Array, " kpts"]:
  """Return normalized k-point weights with shape ``[kpts]``."""
  if k_weights is None:
    return jnp.ones((num_kpts,), dtype=dtype) / num_kpts
  k_weights = jnp.asarray(k_weights, dtype=dtype)
  if k_weights.shape != (num_kpts,):
    raise ValueError(
      f"k_weights must have shape ({num_kpts},), got {k_weights.shape}."
    )
  weight_sum = jnp.sum(k_weights)
  return k_weights / jnp.maximum(weight_sum, jnp.finfo(dtype).tiny)


def _sorted_fill_occupation(
  eigenvalues: Float[Array, "spin kpts bands"],
  num_electrons: Union[float, Float[Array, ""]],
  *,
  spin_polorized: Optional[bool] = None,
  k_weights: Optional[Float[Array, " kpts"]] = None,
  electron_tolerance: float = 1e-10,
) -> tuple[Float[Array, ""], Float[Array, "spin kpts bands"]]:
  """Zero-temperature occupations from sorted eigenvalues.

  Fills states in ascending eigenvalue order using normalized k-point weights.
  Returns ``(chemical_potential, occupation)``.
  """
  eigenvalues = jnp.asarray(eigenvalues)
  if eigenvalues.ndim != 3:
    raise ValueError(
      "eigenvalues must have shape [spin, kpts, bands], "
      f"got shape {eigenvalues.shape}."
    )
  if isinstance(num_electrons, jax.core.Tracer):
    raise ValueError(
      "num_electrons must be concrete for sorted zero-smearing occupation."
    )

  dtype = jnp.result_type(eigenvalues, jnp.asarray(num_electrons), jnp.float32)
  eigenvalues = eigenvalues.astype(dtype)
  spin_polorized = _resolve_spin_polorized(eigenvalues, spin_polorized)
  _, occ_max = _occupation_bounds(spin_polorized)
  occ_max = float(occ_max)

  _, num_kpts, _ = eigenvalues.shape
  k_weights = _normalize_k_weights(k_weights, num_kpts, dtype)

  flat_e = eigenvalues.reshape(-1)
  num_states = int(flat_e.size)
  if num_states == 0:
    raise ValueError("eigenvalues must contain at least one state.")

  state_weights = jnp.broadcast_to(
    k_weights[None, :, None], eigenvalues.shape
  ).reshape(-1)
  order = jnp.argsort(flat_e)
  e_sorted = flat_e[order]
  w_sorted = state_weights[order]

  full_state_electrons = occ_max * w_sorted
  cumulative = jnp.cumsum(full_state_electrons)
  total_electrons = float(cumulative[-1])
  target_electrons = min(max(float(num_electrons), 0.0), total_electrons)

  n_full = int(jnp.sum(cumulative <= target_electrons + electron_tolerance))
  n_full = min(max(n_full, 0), num_states)

  occ_sorted = jnp.zeros_like(e_sorted)
  if n_full > 0:
    occ_sorted = occ_sorted.at[:n_full].set(occ_max)

  partial_occ = 0.0
  if n_full < num_states:
    filled_electrons = float(cumulative[n_full - 1]) if n_full > 0 else 0.0
    remaining = target_electrons - filled_electrons
    if remaining > electron_tolerance:
      wk = max(float(w_sorted[n_full]), float(jnp.finfo(dtype).tiny))
      partial_occ = min(max(remaining / wk, 0.0), occ_max)
      occ_sorted = occ_sorted.at[n_full].set(partial_occ)

  inverse_order = jnp.argsort(order)
  occ_flat = occ_sorted[inverse_order]
  occupation = occ_flat.reshape(eigenvalues.shape)

  spread = float(jnp.maximum(jnp.max(flat_e) - jnp.min(flat_e), 1.0))
  if n_full == 0 and partial_occ <= electron_tolerance:
    mu = float(e_sorted[0]) - spread
  elif n_full == num_states:
    mu = float(e_sorted[-1]) + spread
  elif (
    partial_occ > electron_tolerance and
    partial_occ < occ_max - electron_tolerance
  ):
    mu = float(e_sorted[n_full])
  else:
    lower_idx = max(n_full - 1, 0)
    upper_idx = min(n_full, num_states - 1)
    mu = 0.5 * (float(e_sorted[lower_idx]) + float(e_sorted[upper_idx]))

  return jnp.asarray(mu, dtype=dtype), occupation


def occupations_from_eigenvalues(
  eigenvalues: Float[Array, "spin kpts bands"],
  num_electrons: Union[float, Float[Array, ""]],
  smearing: Union[float, Float[Array, ""]] = 1e-3,
  *,
  method: str = "fermi-dirac",
  methfessel_paxton_order: int = 1,
  spin_polorized: Optional[bool] = None,
  k_weights: Optional[Float[Array, " kpts"]] = None,
  small_smearing_ratio: float = 1e-6,
) -> tuple[Float[Array, ""], Float[Array, "spin kpts bands"]]:
  """Return ``(chemical_potential, occupation)`` with adaptive filling.

  Uses sorted zero-temperature filling when smearing is zero or very small
  relative to the eigenvalue spread. Otherwise uses the requested smearing
  method with root finding.
  """
  eigenvalues = jnp.asarray(eigenvalues)
  if eigenvalues.ndim != 3:
    raise ValueError(
      "eigenvalues must have shape [spin, kpts, bands], "
      f"got shape {eigenvalues.shape}."
    )
  if small_smearing_ratio < 0.0:
    raise ValueError(
      "small_smearing_ratio must be non-negative, "
      f"got {small_smearing_ratio}."
    )

  dtype = jnp.result_type(
    eigenvalues,
    jnp.asarray(num_electrons),
    jnp.asarray(smearing),
    jnp.float32,
  )
  eigenvalues = eigenvalues.astype(dtype)
  smearing_arr = jnp.asarray(smearing, dtype=dtype)
  spread = jnp.maximum(jnp.max(eigenvalues) - jnp.min(eigenvalues), 1.0)

  if (
    not isinstance(smearing_arr, jax.core.Tracer) and
    float(smearing_arr) <= small_smearing_ratio * float(spread)
  ):
    return _sorted_fill_occupation(
      eigenvalues,
      num_electrons,
      spin_polorized=spin_polorized,
      k_weights=k_weights,
    )

  mu = find_chemical_potential(
    eigenvalues,
    num_electrons,
    smearing=smearing_arr,
    method=method,
    methfessel_paxton_order=methfessel_paxton_order,
    spin_polorized=spin_polorized,
    k_weights=k_weights,
  )
  occ = _occupation_from_method(
    eigenvalues,
    mu,
    smearing_arr,
    method,
    methfessel_paxton_order,
    spin_polorized,
  )
  return mu, occ


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
  spin_polorized: Optional[bool] = None,
  k_weights: Optional[Float[Array, " kpts"]] = None,
  max_iter: int = 200,
  bracket_scale: float = 20.0,
  use_custom_root: bool = False,
) -> Float[Array, ""]:
  r"""Solve for chemical potential from charge neutrality.

  Finds :math:`\mu` such that weighted occupations match ``num_electrons``:

  .. math::
    N_e = \sum_{s,k,b} w_k \, f(\varepsilon_{skb}; \mu, \sigma),

  where ``f`` follows ``spin_polorized``:
  ``[0, 1]`` if ``True`` and ``[0, 2]`` if ``False``.

  Args:
    eigenvalues: Kohn-Sham eigenvalues with shape ``[spin, kpts, bands]``.
    num_electrons: Target total electron number.
    smearing: Smearing width (same unit as eigenvalues).
    method: Smearing method name.
    methfessel_paxton_order: Order used only for ``method='methfessel-paxton'``.
    spin_polorized: If provided, explicitly choose occupation range.
      If ``None``, infer from ``eigenvalues.shape[0]``:
      ``1 -> [0, 2]`` (non-polarized), otherwise ``[0, 1]``.
    k_weights: Optional :math:`k`-point weights of shape ``[kpts]``.
      If ``None``, uniform weights are used.
    max_iter: Number of bisection iterations.
    bracket_scale: Bracket size multiplier relative to spectral spread.
    use_custom_root: If ``True``, wrap bisection in ``jax.lax.custom_root``.

  Returns:
    Float[Array, ""]: Chemical potential.

  Notes:
    When ``smearing`` is effectively zero (<= ``1e-6 * spectral_spread``),
    this uses sorted zero-temperature filling instead of root finding.
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

  spin_polorized = _resolve_spin_polorized(eigenvalues, spin_polorized)
  _occupation_bounds(spin_polorized)
  _, num_kpts, _ = eigenvalues.shape
  k_weights = _normalize_k_weights(k_weights, num_kpts, dtype)

  spread = jnp.maximum(jnp.max(eigenvalues) - jnp.min(eigenvalues), 1.0)
  if (
    not isinstance(smearing, jax.core.Tracer) and
    float(smearing) <= 1e-6 * float(spread)
  ):
    mu_zero, _ = _sorted_fill_occupation(
      eigenvalues,
      num_electrons,
      spin_polorized=spin_polorized,
      k_weights=k_weights,
    )
    return mu_zero

  def electron_count(mu):
    occ = _occupation_from_method(
      eigenvalues,
      mu,
      smearing,
      method,
      methfessel_paxton_order,
      spin_polorized,
    )
    weighted_occ = occ * k_weights[None, :, None]
    return jnp.sum(weighted_occ)

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
  "occupations_from_eigenvalues",
]
