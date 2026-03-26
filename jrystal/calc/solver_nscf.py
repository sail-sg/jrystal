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
from functools import partial
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .._src import hamiltonian as _hamiltonian
from .._src import pw as _pw
from ..pseudopotential import normcons as _normcons
from ..terminal_ui import stage_line
from .opt_utils import create_optimizer
from .types import BandStructureResult

if TYPE_CHECKING:
  from ..config import JrystalConfigDict
  from .backend import AllElectronBackend, NormConservingBackend
  from .runtime import RuntimeContext
  from .types import GroundStateResult


# ---------------------------------------------------------------------------
# All-electron NSCF internals
# ---------------------------------------------------------------------------

def _run_nscf_ae(config, ctx, density, num_bands):
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
    max(1, min(num_devices, num_kpts))
    if ctx.execution.parallel_over_k else 1
  )

  optimizer = create_optimizer(config)
  params_pw = _pw.param_init(
    key, num_bands, 1, freq_mask,
    spin_restricted=config.system.spin_restricted,
  )
  opt_state = optimizer.init(params_pw)

  def hamiltonian_trace(params, kpts, g_vec_grid):
    coeff = _pw.coeff(params, freq_mask)
    output = _hamiltonian.hamiltonian_matrix_trace(
      coeff, crystal.positions, crystal.charges,
      density, crystal.vol, g_vec_grid, kpts,
      xc=xc, kohn_sham=True,
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

  @partial(
    jax.pmap, in_axes=(0, 0, 0), devices=jax.devices()[:util_devices],
  )
  def optimize_eigenvalues(kpts, params_pw, opt_state):

    def update_scan(carry, _):
      params, opt_state, kpts = carry
      params, opt_state, _ = update(params, opt_state, kpts, g_vec)
      return (params, opt_state, kpts), None

    carry, _ = jax.lax.scan(
      update_scan, (params_pw, opt_state, kpts[0:1]),
      length=config.band.epoch, unroll=1,
    )
    params_first, opt_state, _ = carry

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

    def eig_fn(param, kpt):
      coeff = _pw.coeff(param, freq_mask)

      def _trace(c):
        return _hamiltonian.hamiltonian_matrix_trace(
          c,
          crystal.positions,
          crystal.charges,
          density,
          crystal.vol,
          g_vec,
          kpt,
          xc=xc,
          kohn_sham=True,
          keep_spin_axis=False,
        )

      hpsi = jax.grad(_trace)(coeff.conj()) / 2.0
      num_grids = np.prod(coeff.shape[-3:])
      hmat = jnp.einsum(
        "skixyz,skjxyz->skij", jnp.conj(coeff), hpsi,
      ) * (crystal.vol / (num_grids**2))
      return jax.vmap(jnp.linalg.eigvalsh)(hmat)

    eig_first = eig_fn(params_first, kpts[0:1])

    def eig_scan(_, x):
      kpt, prm = x
      return None, eig_fn(prm, jnp.expand_dims(kpt, 0))

    _, eig_rest = jax.lax.scan(
      eig_scan, None, (kpts[1:], params_rest),
    )
    return [eig_first] + list(eig_rest)

  k_path, num_kpts, util_devices = _chunk_kpoint_axis(
    ksampling.kpts, util_devices,
  )
  params_pw = jax.tree.map(
    lambda x: jnp.stack([x] * util_devices, axis=0), params_pw,
  )
  opt_state = jax.tree.map(
    lambda x: jnp.stack([x] * util_devices, axis=0), opt_state,
  )

  t0 = time.time()
  eigen_values = optimize_eigenvalues(k_path, params_pw, opt_state)
  dt = time.time() - t0
  stage_line("Band", f"Band calculation done. ({dt:.2f}s)")

  return _reshape_eigenvalues(eigen_values, num_kpts, num_bands)


# ---------------------------------------------------------------------------
# Norm-conserving NSCF internals
# ---------------------------------------------------------------------------

def _run_nscf_nc(config, ctx, density, num_bands):
  """Band structure for norm-conserving pseudopotential backend."""
  key = jax.random.PRNGKey(config.execution.seed)
  crystal = ctx.crystal
  g_vec = ctx.g_vec
  freq_mask = ctx.basis.freq_mask
  ksampling = ctx.ksampling
  pseudopot = ctx.pseudopotential
  potential_loc = ctx.potential_local
  beta_gk = ctx.potential_nonlocal  # SBT cache for path mode
  xc = config.method.xc

  num_devices = ctx.execution.num_devices
  num_kpts = int(ksampling.kpts.shape[0])
  util_devices = (
    max(1, min(num_devices, num_kpts))
    if ctx.execution.parallel_over_k else 1
  )

  optimizer = create_optimizer(config)
  params_pw = _pw.param_init(key, num_bands, 1, freq_mask)
  opt_state = optimizer.init(params_pw)

  def _select_beta(beta_gk, idx):
    return jax.tree.map(lambda x: x.at[idx:idx + 1].get(), beta_gk)

  def _get_nl(kpt, bgk):
    return _normcons.potential_nonlocal_psi_reciprocal(
      crystal.positions, g_vec, kpt,
      pseudopot.r_grid, pseudopot.nonlocal_beta_grid,
      pseudopot.nonlocal_angular_momentum,
      pseudopot.nonlocal_d_matrix, bgk,
    )

  def hamiltonian_trace(params, kpts, g_vec_grid, potential_nl):
    coeff = _pw.coeff(params, freq_mask)
    return _normcons.hamiltonian_trace(
      coeff, density, potential_loc, potential_nl,
      g_vec_grid, kpts, crystal.vol, xc=xc, kohn_sham=True,
    )

  @jax.jit
  def update(params, opt_state, kpts, g_vec_grid, potential_nl):
    val, grad = jax.value_and_grad(hamiltonian_trace)(
      params, kpts, g_vec_grid, potential_nl,
    )
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, val

  @partial(
    jax.pmap, in_axes=(0, 0, 0, 0), devices=jax.devices()[:util_devices],
  )
  def optimize_eigenvalues(kpts, beta_gk, params_pw, opt_state):

    def update_scan(carry, _):
      params, opt_state, nl, kpt = carry
      params, opt_state, _ = update(params, opt_state, kpt, g_vec, nl)
      return (params, opt_state, nl, kpt), None

    nl_first = _get_nl(kpts[0:1], _select_beta(beta_gk, 0))
    carry, _ = jax.lax.scan(
      update_scan, (params_pw, opt_state, nl_first, kpts[0:1]),
      length=config.band.epoch, unroll=1,
    )
    params_first, opt_state, _, _ = carry

    def finetune(carry, x):
      kpt, bgk = x
      kpt = jnp.expand_dims(kpt, 0)
      bgk = [jnp.expand_dims(b, 0) for b in bgk]
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

    def eig_fn(param, kpt, nl):
      coeff = _pw.coeff(param, freq_mask)
      hmat = _normcons.hamiltonian_matrix(
        coeff, density, potential_loc, nl,
        g_vec, kpt, crystal.vol, xc, kohn_sham=True,
      )
      return jnp.linalg.eigvalsh(hmat[0])

    eig_first = eig_fn(
      params_first, kpts[0:1],
      _get_nl(kpts[0:1], _select_beta(beta_gk, 0)),
    )

    def eig_scan(_, x):
      kpt, bgk, prm = x
      kpt = jnp.expand_dims(kpt, 0)
      bgk = [jnp.expand_dims(b, 0) for b in bgk]
      nl = _get_nl(kpt, bgk)
      return None, eig_fn(prm, kpt, nl)

    _, eig_rest = jax.lax.scan(
      eig_scan, None,
      (kpts[1:], [b[1:] for b in beta_gk], params_rest),
    )
    return [eig_first] + list(eig_rest)

  k_path, num_kpts, util_devices = _chunk_kpoint_axis(
    ksampling.kpts, util_devices,
  )
  beta_gk_reshaped, _, _ = _chunk_beta_sbt(beta_gk, util_devices)
  params_pw = jax.tree.map(
    lambda x: jnp.stack([x] * util_devices, axis=0), params_pw,
  )
  opt_state = jax.tree.map(
    lambda x: jnp.stack([x] * util_devices, axis=0), opt_state,
  )

  t0 = time.time()
  eigen_values = optimize_eigenvalues(
    k_path, beta_gk_reshaped, params_pw, opt_state,
  )
  dt = time.time() - t0
  stage_line("Band", f"Band calculation done. ({dt:.2f}s)")

  return _reshape_eigenvalues(eigen_values, num_kpts, num_bands)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

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
    eigen_values, (-1, num_spin, num_bands), order="F",
  )
  eigen_values = eigen_values[:num_kpts]
  return jnp.transpose(eigen_values, (1, 0, 2))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_nscf(
  config: JrystalConfigDict,
  ctx: RuntimeContext,
  backend: AllElectronBackend | NormConservingBackend,
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

  density = ground_state_result.density
  num_electrons = backend.num_electrons(ctx)
  num_bands = ceil(num_electrons / 2) + config.band.empty_bands

  stage_line(
    "Band",
    f"Band structure: {ctx.ksampling.kpts.shape[0]} k-points, {num_bands} bands",
  )

  if isinstance(backend, _NCBackend):
    eigenvalues = _run_nscf_nc(config, ctx, density, num_bands)
  else:
    eigenvalues = _run_nscf_ae(config, ctx, density, num_bands)

  save_file = "".join(ctx.crystal.symbols) + "_band_structure.npy"
  save_dir = Path(config.io.save_dir) if config.io.save_dir else Path.cwd()
  save_dir.mkdir(parents=True, exist_ok=True)
  output_path = save_dir / save_file
  np.save(output_path, np.asarray(eigenvalues))
  stage_line("Band", f"Results saved in {output_path}")

  return BandStructureResult(
    config=config,
    crystal=ctx.crystal,
    kpath=ctx.ksampling,
    eigenvalues=eigenvalues,
    ground_state_energy=ground_state_result.total_energy,
  )
