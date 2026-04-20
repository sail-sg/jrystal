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
"""Non-self-consistent (NSCF) band-structure solver.

Given a converged ground-state density, computes eigenvalues along
a k-path by minimising the Hamiltonian trace at each k-point.

Two internal code-paths exist:
  * **All-electron**: uses ``_src.hamiltonian`` directly.
  * **Norm-conserving**: builds the nonlocal potential per k-point
    from the SBT cache stored in ``ctx.potential_nonlocal``.

Both paths share the same outer loop (optimise first k-point, then
fine-tune along the path, then diagonalise).
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from functools import partial
from math import ceil
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .._src import hamiltonian as _hamiltonian
from .._src import pw as _pw
from .._src.utils import expand_coefficient, squeeze_coefficient
from ..pseudopotential import normcons as _normcons
from ..pseudopotential import ultrasoft as _ultrasoft
from ..pseudopotential.kernel import UltrasoftPathCache
from ..terminal_ui import stage_line
from .opt_utils import create_optimizer
from .timer import PhaseTimer
from .types import BandStructureResult
from .workflow_logging import log_timing_breakdown

if TYPE_CHECKING:
  from ..config import JrystalConfigDict
  from .backend import (
    AllElectronBackend,
    NormConservingBackend,
    UltrasoftBackend,
  )
  from .runtime import RuntimeContext
  from .types import GroundStateResult

# ---------------------------------------------------------------------------
# All-electron NSCF internals
# ---------------------------------------------------------------------------


def _block_tree(value):
  """Synchronize a pytree of JAX arrays."""
  return jax.tree.map(
    lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
    value,
  )


def _occupation_max(spin_restricted: bool) -> float:
  """Maximum occupation per state for the requested spin treatment."""
  return 2.0 if spin_restricted else 1.0


def _kinetic_preconditioner(
  g_vec,
  freq_mask,
  kpt,
  *,
  shift: float = 1.0,
):
  """Return a masked-G kinetic preconditioner ``1 / (|G+k|^2 / 2 + shift)``."""
  eff_g = jnp.asarray(g_vec)[freq_mask]
  kpt = jnp.asarray(kpt, dtype=eff_g.dtype)
  if kpt.ndim == 2:
    if kpt.shape[0] != 1:
      raise ValueError(
        "Expected one k-point for the kinetic preconditioner, "
        f"got shape {kpt.shape}."
      )
    kpt = kpt[0]
  elif kpt.ndim != 1:
    raise ValueError(
      "Expected a rank-1 or rank-2 k-point array, "
      f"got shape {kpt.shape}."
    )
  kinetic = 0.5 * jnp.sum((eff_g + kpt[None, :])**2, axis=-1)
  shift = jnp.asarray(shift, dtype=kinetic.dtype)
  floor = jnp.maximum(shift, jnp.finfo(kinetic.dtype).eps)
  return jnp.reciprocal(jnp.maximum(kinetic + shift, floor))


def _run_nscf_ae(
  config,
  ctx,
  density,
  num_bands,
  *,
  phase_timer: PhaseTimer | None = None,
):
  """Band structure for all-electron backend."""
  key = jax.random.PRNGKey(config.execution.seed)
  crystal = ctx.crystal
  g_vec = ctx.g_vec
  freq_mask = ctx.basis.freq_mask
  ksampling = ctx.ksampling
  xc = config.method.xc

  num_devices = ctx.execution.num_devices
  num_kpts = int(ksampling.kpts.shape[0])
  util_devices = (
    max(1, min(num_devices, num_kpts)) if ctx.execution.parallel_over_k else 1
  )

  optimizer = create_optimizer(config)
  params_pw = _pw.param_init(
    key,
    num_bands,
    1,
    freq_mask,
    spin_restricted=config.system.spin_restricted,
  )
  opt_state = optimizer.init(params_pw)

  def hamiltonian_trace(params, kpts, g_vec_grid):
    coeff = _pw.coeff(params, freq_mask)
    output = _hamiltonian.hamiltonian_matrix_trace(
      coeff,
      crystal.positions,
      crystal.charges,
      density,
      crystal.vol,
      g_vec_grid,
      kpts,
      xc=xc,
      kohn_sham=True,
    )
    return jnp.sum(output)

  @jax.jit
  def update(params, opt_state, kpts, g_vec_grid):
    val, grad = jax.value_and_grad(hamiltonian_trace)(
      params, kpts, g_vec_grid,
    )
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, val

  def update_scan(carry, _):
    params, opt_state, kpts = carry
    params, opt_state, _ = update(params, opt_state, kpts, g_vec)
    return (params, opt_state, kpts), None

  @partial(jax.pmap, in_axes=(0, 0, 0), devices=jax.devices()[:util_devices])
  def optimize_first_kpoint(kpts, params_pw, opt_state):

    carry, _ = jax.lax.scan(
      update_scan, (params_pw, opt_state, kpts[0:1]),
      length=config.band.epoch, unroll=1,
    )
    params_first, opt_state, _ = carry
    return params_first, opt_state

  @partial(jax.pmap, in_axes=(0, 0, 0), devices=jax.devices()[:util_devices])
  def fine_tune_path(kpts, params_first, opt_state):

    def finetune(carry, kpt):
      kpt = jnp.expand_dims(kpt, 0)
      params, opt_state = carry
      carry, _ = jax.lax.scan(
        update_scan, (params, opt_state, kpt),
        length=config.band.fine_tuning_epoch, unroll=1,
      )
      params, opt_state, _ = carry
      return (params, opt_state), params

    _, params_rest = jax.lax.scan(
      finetune, (params_first, opt_state), kpts[1:],
    )
    return params_rest

  @partial(jax.pmap, in_axes=(0, 0, 0), devices=jax.devices()[:util_devices])
  def diagonalize_path(kpts, params_first, params_rest):

    def eig_fn(param, kpt):
      coeff = _pw.coeff(param, freq_mask)
      hmat = _hamiltonian.hamiltonian_matrix(
        coeff,
        crystal.positions,
        crystal.charges,
        density,
        g_vec,
        kpt,
        crystal.vol,
        xc=xc,
        kohn_sham=True,
      )
      return jnp.linalg.eigvalsh(hmat)

    eig_first = eig_fn(params_first, kpts[0:1])

    def eig_scan(_, x):
      kpt, prm = x
      return None, eig_fn(prm, jnp.expand_dims(kpt, 0))

    _, eig_rest = jax.lax.scan(
      eig_scan, None, (kpts[1:], params_rest),
    )
    return jnp.concatenate([eig_first[None, ...], eig_rest], axis=0)

  k_path, num_kpts, util_devices = _chunk_kpoint_axis(
    ksampling.kpts, util_devices,
  )
  params_pw = jax.tree.map(
    lambda x: jnp.stack([x] * util_devices, axis=0),
    params_pw,
  )
  opt_state = jax.tree.map(
    lambda x: jnp.stack([x] * util_devices, axis=0),
    opt_state,
  )

  first_phase = (
    "first_call_overhead"
    if phase_timer is not None and phase_timer.enabled else "optimize_first_k"
  )
  with (
    phase_timer.phase(first_phase) if phase_timer is not None else nullcontext()
  ):
    params_first, opt_state = optimize_first_kpoint(k_path, params_pw, opt_state)
    if phase_timer is not None and phase_timer.enabled:
      params_first, opt_state = _block_tree((params_first, opt_state))

  with (
    phase_timer.phase("fine_tune_path")
    if phase_timer is not None else nullcontext()
  ):
    params_rest = fine_tune_path(k_path, params_first, opt_state)
    if phase_timer is not None and phase_timer.enabled:
      params_rest = _block_tree(params_rest)

  with (
    phase_timer.phase("diagonalize_path")
    if phase_timer is not None else nullcontext()
  ):
    eigen_values = diagonalize_path(k_path, params_first, params_rest)
    if phase_timer is not None and phase_timer.enabled:
      eigen_values = _block_tree(eigen_values)

  return _reshape_eigenvalues(eigen_values, num_kpts, num_bands)


# ---------------------------------------------------------------------------
# Norm-conserving NSCF internals
# ---------------------------------------------------------------------------


def _run_nscf_nc(
  config,
  ctx,
  density,
  num_bands,
  *,
  phase_timer: PhaseTimer | None = None,
):
  """Band structure for norm-conserving pseudopotential backend."""
  key = jax.random.PRNGKey(config.execution.seed)
  crystal = ctx.crystal
  g_vec = ctx.g_vec
  freq_mask = ctx.basis.freq_mask
  ksampling = ctx.ksampling
  pseudo_cache = ctx.pseudo_cache
  potential_loc = ctx.potential_local
  beta_gk = ctx.potential_nonlocal  # species-level SBT cache for path mode
  xc = config.method.xc

  atom_positions = pseudo_cache.atom_species_map.positions
  species_index = tuple(
    int(i) for i in pseudo_cache.atom_species_map.species_index
  )
  atom_setups = tuple(pseudo_cache.species_setups[i] for i in species_index)
  atom_r_grid = [setup.radial.r_g for setup in atom_setups]
  atom_beta_grid = [setup.projectors.beta_jr for setup in atom_setups]
  atom_l = [setup.projectors.l_j for setup in atom_setups]
  atom_d = [setup.projectors.d_jj for setup in atom_setups]

  num_devices = ctx.execution.num_devices
  num_kpts = int(ksampling.kpts.shape[0])
  util_devices = (
    max(1, min(num_devices, num_kpts)) if ctx.execution.parallel_over_k else 1
  )

  optimizer = create_optimizer(config)
  params_pw = _pw.param_init(
    key,
    num_bands,
    1,
    freq_mask,
    spin_restricted=config.system.spin_restricted,
  )
  opt_state = optimizer.init(params_pw)

  def _select_beta(beta_gk, idx):
    species_beta = [beta.at[idx:idx + 1].get() for beta in beta_gk]
    return [species_beta[i] for i in species_index]

  def _get_nl(kpt, bgk):
    return _normcons.potential_nonlocal_psi_reciprocal(
      atom_positions,
      g_vec,
      kpt,
      atom_r_grid,
      atom_beta_grid,
      atom_l,
      atom_d,
      bgk,
    )

  def hamiltonian_trace(params, kpts, g_vec_grid, potential_nl):
    coeff = _pw.coeff(params, freq_mask)
    return _normcons.hamiltonian_trace(
      coeff,
      density,
      potential_loc,
      potential_nl,
      g_vec_grid,
      kpts,
      crystal.vol,
      xc=xc,
      kohn_sham=True,
    )

  @jax.jit
  def update(params, opt_state, kpts, g_vec_grid, potential_nl):
    val, grad = jax.value_and_grad(hamiltonian_trace)(
      params, kpts, g_vec_grid, potential_nl,
    )
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, val

  def update_scan(carry, _):
    params, opt_state, nl, kpt = carry
    params, opt_state, _ = update(params, opt_state, kpt, g_vec, nl)
    return (params, opt_state, nl, kpt), None

  @partial(
    jax.pmap,
    in_axes=(0, 0, 0, 0),
    devices=jax.devices()[:util_devices],
  )
  def optimize_first_kpoint(kpts, beta_gk, params_pw, opt_state):

    nl_first = _get_nl(kpts[0:1], _select_beta(beta_gk, 0))
    carry, _ = jax.lax.scan(
      update_scan, (params_pw, opt_state, nl_first, kpts[0:1]),
      length=config.band.epoch, unroll=1,
    )
    params_first, opt_state, _, _ = carry
    return params_first, opt_state

  @partial(
    jax.pmap,
    in_axes=(0, 0, 0, 0),
    devices=jax.devices()[:util_devices],
  )
  def fine_tune_path(kpts, beta_gk, params_first, opt_state):

    def finetune(carry, x):
      kpt, bgk = x
      kpt = jnp.expand_dims(kpt, 0)
      bgk = [jnp.expand_dims(b, 0) for b in bgk]
      bgk = [bgk[i] for i in species_index]
      params, opt_state = carry
      nl = _get_nl(kpt, bgk)
      carry, _ = jax.lax.scan(
        update_scan, (params, opt_state, nl, kpt),
        length=config.band.fine_tuning_epoch, unroll=1,
      )
      params, opt_state, _, _ = carry
      return (params, opt_state), params

    _, params_rest = jax.lax.scan(
      finetune, (params_first, opt_state),
      (kpts[1:], [b[1:] for b in beta_gk]),
    )
    return params_rest

  @partial(
    jax.pmap,
    in_axes=(0, 0, 0, 0),
    devices=jax.devices()[:util_devices],
  )
  def diagonalize_path(kpts, beta_gk, params_first, params_rest):

    def eig_fn(param, kpt, nl):
      coeff = _pw.coeff(param, freq_mask)
      hmat = _normcons.hamiltonian_matrix(
        coeff,
        density,
        potential_loc,
        nl,
        g_vec,
        kpt,
        crystal.vol,
        xc,
        kohn_sham=True,
      )
      return jnp.linalg.eigvalsh(hmat)

    eig_first = eig_fn(
      params_first,
      kpts[0:1],
      _get_nl(kpts[0:1], _select_beta(beta_gk, 0)),
    )

    def eig_scan(_, x):
      kpt, bgk, prm = x
      kpt = jnp.expand_dims(kpt, 0)
      bgk = [jnp.expand_dims(b, 0) for b in bgk]
      bgk = [bgk[i] for i in species_index]
      nl = _get_nl(kpt, bgk)
      return None, eig_fn(prm, kpt, nl)

    _, eig_rest = jax.lax.scan(
      eig_scan, None,
      (kpts[1:], [b[1:] for b in beta_gk], params_rest),
    )
    return jnp.concatenate([eig_first[None, ...], eig_rest], axis=0)

  k_path, num_kpts, util_devices = _chunk_kpoint_axis(
    ksampling.kpts, util_devices,
  )
  beta_gk_reshaped, _, _ = _chunk_beta_sbt(beta_gk, util_devices)
  params_pw = jax.tree.map(
    lambda x: jnp.stack([x] * util_devices, axis=0),
    params_pw,
  )
  opt_state = jax.tree.map(
    lambda x: jnp.stack([x] * util_devices, axis=0),
    opt_state,
  )

  first_phase = (
    "first_call_overhead"
    if phase_timer is not None and phase_timer.enabled else "optimize_first_k"
  )
  with (
    phase_timer.phase(first_phase) if phase_timer is not None else nullcontext()
  ):
    params_first, opt_state = optimize_first_kpoint(
      k_path,
      beta_gk_reshaped,
      params_pw,
      opt_state,
    )
    if phase_timer is not None and phase_timer.enabled:
      params_first, opt_state = _block_tree((params_first, opt_state))

  with (
    phase_timer.phase("fine_tune_path")
    if phase_timer is not None else nullcontext()
  ):
    params_rest = fine_tune_path(
      k_path,
      beta_gk_reshaped,
      params_first,
      opt_state,
    )
    if phase_timer is not None and phase_timer.enabled:
      params_rest = _block_tree(params_rest)

  with (
    phase_timer.phase("diagonalize_path")
    if phase_timer is not None else nullcontext()
  ):
    eigen_values = diagonalize_path(
      k_path,
      beta_gk_reshaped,
      params_first,
      params_rest,
    )
    if phase_timer is not None and phase_timer.enabled:
      eigen_values = _block_tree(eigen_values)

  return _reshape_eigenvalues(eigen_values, num_kpts, num_bands)


# ---------------------------------------------------------------------------
# Ultrasoft NSCF internals
# ---------------------------------------------------------------------------


def _run_nscf_us(
  config,
  ctx,
  density,
  num_bands,
  backend,
  source_coeff=None,
  *,
  phase_timer: PhaseTimer | None = None,
):
  """Band structure for ultrasoft pseudopotential backend."""
  if not isinstance(ctx.pseudo_cache, UltrasoftPathCache):
    raise TypeError("USPP band calculation requires an UltrasoftPathCache.")

  key = jax.random.PRNGKey(config.execution.seed)
  del source_coeff
  g_vec = ctx.g_vec
  freq_mask = ctx.basis.freq_mask
  ksampling = ctx.ksampling
  vol = ctx.crystal.vol
  num_kpts = int(ksampling.kpts.shape[0])
  nscf_state = backend.prepare_nscf(density, ctx)
  optimizer = create_optimizer(config)
  params_pw = _pw.param_init(
    key,
    num_bands,
    1,
    freq_mask,
    spin_restricted=config.system.spin_restricted,
  )
  opt_state = optimizer.init(params_pw)

  def _normalize_channel_mask(channel_mask, channel_qii):
    if channel_mask is None:
      return jnp.ones(channel_qii.shape[:2], dtype=channel_qii.dtype)
    return channel_mask

  bundles = []
  with (
    phase_timer.phase("operator_bundle")
    if phase_timer is not None else nullcontext()
  ):
    for k_idx in range(num_kpts):
      bundle = backend.build_kpoint_operator(k_idx, nscf_state, ctx)
      bundles.append(
        (
          bundle.kpt,
          bundle.local_potential_r,
          bundle.projector_channels_g,
          bundle.channel_qii,
          bundle.channel_dii_eff,
          _normalize_channel_mask(bundle.channel_mask, bundle.channel_qii),
        )
      )

  def _projected_trace(
    params,
    kpt,
    local_potential_r,
    projector_channels_g,
    channel_qii,
    channel_dii_eff,
    channel_mask,
  ):
    coeff = _pw.coeff(params, freq_mask)
    coeff_compact = squeeze_coefficient(coeff, freq_mask)
    coeff_canon = _canonicalize_uspp_subspace_arrays(
      coeff_compact,
      projector_channels_g=projector_channels_g,
      channel_qii=channel_qii,
      channel_mask=channel_mask,
      freq_mask=freq_mask,
      vol=vol,
    )
    h_coeff = _uspp_h_apply_compact_arrays(
      coeff_canon,
      kpt=kpt,
      local_potential_r=local_potential_r,
      projector_channels_g=projector_channels_g,
      channel_dii_eff=channel_dii_eff,
      channel_mask=channel_mask,
      g_vec=g_vec,
      freq_mask=freq_mask,
      vol=vol,
    )
    hmat = jnp.einsum(
      "...gb,...gc->...bc",
      jnp.conj(coeff_canon),
      h_coeff,
    )
    hmat = 0.5 * (hmat + jnp.swapaxes(jnp.conj(hmat), -1, -2))
    return jnp.real(jnp.trace(hmat, axis1=-2, axis2=-1)).sum()

  @jax.jit
  def _update(
    params,
    opt_state,
    kpt,
    local_potential_r,
    projector_channels_g,
    channel_qii,
    channel_dii_eff,
    channel_mask,
  ):
    val, grad = jax.value_and_grad(_projected_trace)(
      params,
      kpt,
      local_potential_r,
      projector_channels_g,
      channel_qii,
      channel_dii_eff,
      channel_mask,
    )
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, val

  def _make_optimize_kpoint(num_steps):

    @jax.jit
    def _optimize(
      params,
      opt_state,
      kpt,
      local_potential_r,
      projector_channels_g,
      channel_qii,
      channel_dii_eff,
      channel_mask,
    ):

      def _scan(carry, _):
        params, opt_state = carry
        params, opt_state, _ = _update(
          params,
          opt_state,
          kpt,
          local_potential_r,
          projector_channels_g,
          channel_qii,
          channel_dii_eff,
          channel_mask,
        )
        return (params, opt_state), None

      (params, opt_state), _ = jax.lax.scan(
        _scan,
        (params, opt_state),
        xs=None,
        length=num_steps,
        unroll=1,
      )
      return params, opt_state

    return _optimize

  @jax.jit
  def _diagonalize_subspace(
    params,
    kpt,
    local_potential_r,
    projector_channels_g,
    channel_qii,
    channel_dii_eff,
    channel_mask,
  ):
    coeff = _pw.coeff(params, freq_mask)
    coeff_compact = squeeze_coefficient(coeff, freq_mask)
    coeff_canon = _canonicalize_uspp_subspace_arrays(
      coeff_compact,
      projector_channels_g=projector_channels_g,
      channel_qii=channel_qii,
      channel_mask=channel_mask,
      freq_mask=freq_mask,
      vol=vol,
    )
    h_coeff = _uspp_h_apply_compact_arrays(
      coeff_canon,
      kpt=kpt,
      local_potential_r=local_potential_r,
      projector_channels_g=projector_channels_g,
      channel_dii_eff=channel_dii_eff,
      channel_mask=channel_mask,
      g_vec=g_vec,
      freq_mask=freq_mask,
      vol=vol,
    )
    hmat = jnp.einsum(
      "...gb,...gc->...bc",
      jnp.conj(coeff_canon),
      h_coeff,
    )
    hmat = 0.5 * (hmat + jnp.swapaxes(jnp.conj(hmat), -1, -2))
    return jnp.linalg.eigvalsh(hmat)

  optimize_first = _make_optimize_kpoint(config.band.epoch)
  optimize_finetune = _make_optimize_kpoint(config.band.fine_tuning_epoch)

  first_phase = (
    "first_call_overhead"
    if phase_timer is not None and phase_timer.enabled else "optimize_first_k"
  )
  with (
    phase_timer.phase(first_phase) if phase_timer is not None else nullcontext()
  ):
    params_first, opt_state = optimize_first(
      params_pw,
      opt_state,
      *bundles[0],
    )
    if phase_timer is not None and phase_timer.enabled:
      params_first, opt_state = _block_tree((params_first, opt_state))

  params_path = [params_first]
  params_curr = params_first
  opt_state_curr = opt_state
  with (
    phase_timer.phase("fine_tune_path")
    if phase_timer is not None else nullcontext()
  ):
    for bundle_arrays in bundles[1:]:
      params_curr, opt_state_curr = optimize_finetune(
        params_curr,
        opt_state_curr,
        *bundle_arrays,
      )
      params_path.append(params_curr)
    if phase_timer is not None and phase_timer.enabled:
      params_path = _block_tree(params_path)

  eigenvalues = []
  with (
    phase_timer.phase("diagonalize_path")
    if phase_timer is not None else nullcontext()
  ):
    for params_k, bundle_arrays in zip(params_path, bundles):
      eigenvalues.append(_diagonalize_subspace(params_k, *bundle_arrays))
    if phase_timer is not None and phase_timer.enabled:
      eigenvalues = _block_tree(eigenvalues)

  return jnp.concatenate(eigenvalues, axis=1)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _initial_band_guess(
  key,
  num_bands,
  freq_mask,
  source_coeff=None,
  *,
  spin_restricted=True,
):
  """Return a compact masked-G initial guess for a band solve."""
  if source_coeff is not None:
    source_coeff = jnp.asarray(source_coeff)
    if source_coeff.shape[-1] >= num_bands:
      return source_coeff[..., :num_bands]

  params = _pw.param_init(
    key,
    num_bands,
    1,
    freq_mask,
    spin_restricted=spin_restricted,
  )
  return squeeze_coefficient(_pw.coeff(params, freq_mask), freq_mask)


def _hermitian_inverse_sqrt(matrix):
  """Return a numerically safe inverse square root of a Hermitian matrix."""
  matrix = 0.5 * (matrix + jnp.swapaxes(jnp.conj(matrix), -1, -2))
  eigvals, eigvecs = jnp.linalg.eigh(matrix)
  real_dtype = jnp.real(matrix).dtype
  eps = jnp.finfo(real_dtype).eps
  scale = jnp.max(jnp.abs(eigvals), axis=-1, keepdims=True)
  floor = jnp.maximum(scale * eps * matrix.shape[-1], eps)
  inv_sqrt = jnp.reciprocal(jnp.sqrt(jnp.maximum(eigvals, floor)))
  return jnp.einsum(
    "...ik,...k,...jk->...ij",
    eigvecs,
    inv_sqrt,
    jnp.conj(eigvecs),
  )


def _solve_dense_generalized(h_dense, s_dense, k):
  """Solve a dense Hermitian generalized eigenproblem for the lowest k roots."""
  h_dense = 0.5 * (h_dense + jnp.swapaxes(jnp.conj(h_dense), -1, -2))
  s_dense = 0.5 * (s_dense + jnp.swapaxes(jnp.conj(s_dense), -1, -2))
  s_inv_sqrt = _hermitian_inverse_sqrt(s_dense)
  h_whitened = s_inv_sqrt.conj().T @ h_dense @ s_inv_sqrt
  h_whitened = 0.5 * (h_whitened + h_whitened.conj().T)
  evals, vecs_white = jnp.linalg.eigh(h_whitened)
  order = jnp.argsort(jnp.real(evals))[:k]
  evals = jnp.take(evals, order, axis=-1)
  vecs_white = jnp.take(vecs_white, order, axis=-1)
  vecs = s_inv_sqrt @ vecs_white
  return jnp.real(evals), vecs


def _uspp_h_apply_compact_arrays(
  coeff_compact,
  *,
  kpt,
  local_potential_r,
  projector_channels_g,
  channel_dii_eff,
  channel_mask,
  g_vec,
  freq_mask,
  vol,
):
  """Apply the fixed-density USPP Hamiltonian using data-only operator arrays."""
  coeff_compact = jnp.asarray(coeff_compact)
  if coeff_compact.ndim == 3:
    coeff_compact = coeff_compact[:, None, ...]
  coeff_full = expand_coefficient(coeff_compact.conj(), freq_mask)
  kinetic = _ultrasoft.kinetic_apply(coeff_full, g_vec, kpt)
  local = _ultrasoft.local_potential_apply(
    coeff_full,
    local_potential_r,
    vol,
  )
  nonlocal_term = _ultrasoft.channel_nonlocal_apply(
    coeff_full,
    projector_channels_g,
    channel_dii_eff,
    vol,
    channel_mask=channel_mask,
  )
  hpsi_full = jnp.conj(kinetic + local + nonlocal_term)
  return squeeze_coefficient(hpsi_full, freq_mask)


def _uspp_s_apply_compact_arrays(
  coeff_compact,
  *,
  projector_channels_g,
  channel_qii,
  channel_mask,
  freq_mask,
  vol,
):
  """Apply the USPP overlap operator using data-only operator arrays."""
  coeff_compact = jnp.asarray(coeff_compact)
  if coeff_compact.ndim == 3:
    coeff_compact = coeff_compact[:, None, ...]
  coeff_full = expand_coefficient(coeff_compact.conj(), freq_mask)
  spsi_full = _ultrasoft.overlap_apply(
    coeff_full,
    projector_channels_g,
    channel_qii,
    vol,
    channel_mask=channel_mask,
  )
  return squeeze_coefficient(spsi_full.conj(), freq_mask)


def _uspp_h_apply_compact(coeff_compact, bundle, g_vec, freq_mask, vol):
  """Apply the fixed-density USPP Hamiltonian to compact masked-G coefficients."""
  return _uspp_h_apply_compact_arrays(
    coeff_compact,
    kpt=bundle.kpt,
    local_potential_r=bundle.local_potential_r,
    projector_channels_g=bundle.projector_channels_g,
    channel_dii_eff=bundle.channel_dii_eff,
    channel_mask=bundle.channel_mask,
    g_vec=g_vec,
    freq_mask=freq_mask,
    vol=vol,
  )


def _uspp_s_apply_compact(coeff_compact, bundle, freq_mask, vol):
  """Apply the USPP overlap operator to compact masked-G coefficients."""
  return _uspp_s_apply_compact_arrays(
    coeff_compact,
    projector_channels_g=bundle.projector_channels_g,
    channel_qii=bundle.channel_qii,
    channel_mask=bundle.channel_mask,
    freq_mask=freq_mask,
    vol=vol,
  )


def _canonicalize_uspp_subspace_arrays(
  coeff_compact,
  *,
  projector_channels_g,
  channel_qii,
  channel_mask,
  freq_mask,
  vol,
):
  """Canonicalize compact coefficients using data-only USPP overlap arrays."""
  coeff_compact = jnp.asarray(coeff_compact)
  if coeff_compact.ndim == 3:
    coeff_compact = coeff_compact[:, None, ...]
  s_coeff = _uspp_s_apply_compact_arrays(
    coeff_compact,
    projector_channels_g=projector_channels_g,
    channel_qii=channel_qii,
    channel_mask=channel_mask,
    freq_mask=freq_mask,
    vol=vol,
  )
  overlap = jnp.einsum(
    "...gb,...gc->...bc",
    jnp.conj(coeff_compact),
    s_coeff,
  )
  inv_sqrt = _hermitian_inverse_sqrt(overlap)
  return jnp.einsum("...gb,...bc->...gc", coeff_compact, inv_sqrt)


def _canonicalize_uspp_subspace(coeff_compact, bundle, g_vec, freq_mask, vol):
  """Canonicalize a compact coefficient subspace with respect to the bundle S(k)."""
  coeff_compact = jnp.asarray(coeff_compact)
  if coeff_compact.ndim == 3:
    coeff_compact = coeff_compact[:, None, ...]
  s_coeff = _uspp_s_apply_compact(coeff_compact, bundle, freq_mask, vol)
  overlap = jnp.einsum(
    "...gb,...gc->...bc",
    jnp.conj(coeff_compact),
    s_coeff,
  )
  inv_sqrt = _hermitian_inverse_sqrt(overlap)
  return jnp.einsum("...gb,...bc->...gc", coeff_compact, inv_sqrt)


def _build_dense_uspp_kpoint_operators(bundle, g_vec, freq_mask, vol):
  """Materialize dense H/S matrices for one USPP k-point operator bundle."""
  g_dim = int(np.sum(np.asarray(freq_mask)))
  eye = jnp.eye(g_dim, dtype=jnp.complex64).reshape(1, 1, g_dim, g_dim)
  h_dense = _uspp_h_apply_compact(eye, bundle, g_vec, freq_mask, vol)[0, 0]
  s_dense = _uspp_s_apply_compact(eye, bundle, freq_mask, vol)[0, 0]
  return h_dense, s_dense


def _chunk_kpoint_axis(array, num_devices):
  """Pad and reshape a leading k-point axis for device-parallel NSCF."""
  num_kpts = int(array.shape[0])
  util_devices = max(1, min(num_devices, num_kpts))
  chunk_size = ceil(num_kpts / util_devices)
  padded_kpts = chunk_size * util_devices
  pad_count = padded_kpts - num_kpts

  if pad_count > 0:
    pad_block = jnp.repeat(array[-1:], pad_count, axis=0)
    array = jnp.concatenate([array, pad_block], axis=0)

  return (
    jnp.reshape(array, (util_devices, chunk_size, *array.shape[1:])),
    num_kpts,
    util_devices,
  )


def _chunk_beta_sbt(beta_gk, num_devices):
  """Pad and reshape SBT caches consistently with the k-path chunks."""
  chunked_beta = []
  num_kpts = None
  util_devices = None
  for beta in beta_gk:
    chunked, valid_kpts, chunk_devices = _chunk_kpoint_axis(beta, num_devices)
    chunked_beta.append(chunked)
    if num_kpts is None:
      num_kpts = valid_kpts
      util_devices = chunk_devices
  return chunked_beta, num_kpts, util_devices


def _reshape_eigenvalues(eigen_values, num_kpts, num_bands):
  """Stack pmap outputs into a trimmed ``(spin, kpt, band)`` array."""
  eigen_values = jnp.asarray(eigen_values)
  if eigen_values.ndim == 1:
    eigen_values = jnp.stack(eigen_values)
  if eigen_values.ndim == 5:
    eigen_values = jnp.squeeze(eigen_values, axis=3)
  elif eigen_values.ndim != 4:
    raise ValueError(
      "Unexpected eigenvalue tensor rank. "
      f"Expected 4 or 5, got {eigen_values.ndim}."
    )

  num_spin = eigen_values.shape[2]
  eigen_values = jnp.reshape(
    eigen_values,
    (-1, num_spin, num_bands),
    order="F",
  )
  eigen_values = eigen_values[:num_kpts]
  return jnp.transpose(eigen_values, (1, 0, 2))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_nscf(
  config: JrystalConfigDict,
  ctx: RuntimeContext,
  backend: AllElectronBackend | NormConservingBackend | UltrasoftBackend,
  ground_state_result: GroundStateResult,
) -> BandStructureResult:
  """Run a non-self-consistent band-structure calculation.

  Uses the converged density from *ground_state_result* to compute
  eigenvalues along the k-path in *ctx.ksampling*.

  Args:
    config: Jrystal configuration.
    ctx: Runtime context built with ``mode="path"``.
    backend: Electronic-structure backend.
    ground_state_result: Converged ground-state result.

  Returns:
    Band-structure result with eigenvalues along the k-path.
  """
  from .backend import NormConservingBackend as _NCBackend
  from .backend import UltrasoftBackend as _USBackend

  density = ground_state_result.density
  num_electrons = backend.num_electrons(ctx)
  empty_bands = config.band.empty_bands
  num_bands = ceil(
    num_electrons / _occupation_max(config.system.spin_restricted)
  ) + empty_bands

  num_kpts = ctx.ksampling.kpts.shape[0]
  stage_line(
    "Band",
    f"Band structure: {num_kpts} k-points, {num_bands} bands",
  )
  band_start = time.time()
  phase_timer = PhaseTimer(enabled=config.execution.profile)

  if isinstance(backend, _NCBackend):
    eigenvalues = _run_nscf_nc(
      config,
      ctx,
      density,
      num_bands,
      phase_timer=phase_timer,
    )
  elif isinstance(backend, _USBackend):
    source_coeff = None
    if getattr(ground_state_result, "coefficients", None) is not None:
      coeff_dict = ground_state_result.coefficients
      if isinstance(
        coeff_dict, dict
      ) and "w_re" in coeff_dict and "w_im" in coeff_dict:
        source_coeff = (
          jnp.asarray(coeff_dict["w_re"]) +
          1.0j * jnp.asarray(coeff_dict["w_im"])
        )
    eigenvalues = _run_nscf_us(
      config,
      ctx,
      density,
      num_bands,
      backend,
      source_coeff=source_coeff,
      phase_timer=phase_timer,
    )
  else:
    eigenvalues = _run_nscf_ae(
      config,
      ctx,
      density,
      num_bands,
      phase_timer=phase_timer,
    )

  wall_time = time.time() - band_start
  profiling = {}
  stage_line("Band", f"Band calculation done. ({wall_time:.2f}s)")
  if config.execution.profile:
    profiling = {
      "kind": "band",
      "enabled": True,
      "wall_time_sec": wall_time,
      "phases": phase_timer.summary(total_wall_time=wall_time),
    }
    log_timing_breakdown(
      "Band",
      profiling["phases"],
      total_wall_time=wall_time,
    )

  return BandStructureResult(
    config=config,
    crystal=ctx.crystal,
    kpath=ctx.ksampling,
    eigenvalues=eigenvalues,
    ground_state_energy=ground_state_result.total_energy,
    reference_energy=ground_state_result.fermi_energy,
    wall_time=wall_time,
    profiling=profiling,
  )
