"""Reproduce negative eigenvalues in PAW overlap correction."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

from jrystal.calc.calc_paw import calc_paw, setup_gpaw
from jrystal.calc.opt_utils import (
  create_crystal,
  create_freq_mask,
  create_grids,
  set_env_params,
)
from jrystal.config import get_config
from jrystal.pseudopotential.beta import beta_sbt_grid
from jrystal.pseudopotential.nloc import potential_nonlocal_psi_reciprocal


def _build_paw_pseudopotential(crystal, xc_name: str):
  pseudopot = SimpleNamespace(
    r_grid=[],
    nonlocal_beta_grid=[],
    nonlocal_angular_momentum=[],
    nonlocal_d_matrix=[],
  )

  for symbol in crystal.symbols:
    setup_data = setup_gpaw(symbol, xc_name)
    results = calc_paw(setup_data)

    n_proj = setup_data['pt_jg'].shape[0]
    tmp_mat = np.zeros((n_proj, n_proj))
    tmp_mat[np.triu_indices(n_proj)] = results['Delta_lq'][0]
    tmp_mat = tmp_mat + tmp_mat.T - np.diag(np.diag(tmp_mat))
    tmp_mat = tmp_mat / np.sqrt(4 * np.pi)

    pseudopot.r_grid.append(setup_data['r_g'])
    pseudopot.nonlocal_beta_grid.append(setup_data['pt_jg'])
    pseudopot.nonlocal_angular_momentum.append(setup_data['l_j'])
    pseudopot.nonlocal_d_matrix.append(tmp_mat)

  return pseudopot


def _process_psi_g(psi_g, freq_mask):
  psi_g = psi_g.at[..., freq_mask].get()
  psi_g = jnp.reshape(psi_g, (psi_g.shape[0], -1, psi_g.shape[-1]))
  return jnp.swapaxes(psi_g, 1, 2)


def _expand_q_matrix(q, l_list, l_max_global):
  q = np.array(q)
  l_list = np.array(l_list, dtype=int)
  n_m = 2 * l_max_global + 1
  size = int(len(l_list) * n_m)
  q_full = np.zeros((size, size), dtype=q.dtype)

  for b1, l1 in enumerate(l_list):
    for b2, l2 in enumerate(l_list):
      if l1 != l2:
        continue
      q_val = q[b1, b2]
      m_start = l_max_global - l1
      m_end = l_max_global + l1 + 1
      for m_idx in range(m_start, m_end):
        i = b1 * n_m + m_idx
        j = b2 * n_m + m_idx
        q_full[i, j] = q_val
  return q_full


def _assemble_q_matrix(nonlocal_q_matrix, l_lists):
  l_max_global = int(np.max(np.hstack(l_lists)))
  q_blocks = []
  for q, l_list in zip(nonlocal_q_matrix, l_lists):
    q_blocks.append(_expand_q_matrix(q, l_list, l_max_global))
  q_mat = scipy.linalg.block_diag(*q_blocks)
  return jnp.array(q_mat)


def _check_overlap_psd(B, q_mat, tol, show_all):
  bad = False
  min_eig = 1.0e9
  min_eig_plus = 1.0e9

  for k_index in range(B.shape[0]):
    B_k = B[k_index]
    _, R = jnp.linalg.qr(B_k)
    rq = R @ q_mat @ R.T.conj()
    rq = (rq + rq.T.conj()) / 2
    eigvals = jnp.linalg.eigvalsh(rq).real
    eigvals_plus = eigvals + 1.0
    k_min = float(jnp.min(eigvals))
    k_min_plus = float(jnp.min(eigvals_plus))

    if show_all or k_min_plus < -tol:
      print(
        f"k={k_index}: min eig(RQR^H)={k_min:.6e}, "
        f"min eig(I+RQR^H)={k_min_plus:.6e}"
      )

    min_eig = min(min_eig, k_min)
    min_eig_plus = min(min_eig_plus, k_min_plus)
    if k_min_plus < -tol:
      bad = True

  return bad, min_eig, min_eig_plus


def main():
  parser = argparse.ArgumentParser(
    description="Check PAW overlap correction eigenvalues."
  )
  parser.add_argument(
    "--config",
    default="config.yaml",
    help="Path to config.yaml",
  )
  parser.add_argument(
    "--tol",
    type=float,
    default=1.0e-8,
    help="Tolerance for negative eigenvalues.",
  )
  parser.add_argument(
    "--show-all",
    action="store_true",
    help="Print eigenvalue minima for every k-point.",
  )
  parser.add_argument(
    "--allow-negative",
    action="store_true",
    help="Do not raise if I+RQR^H is not positive definite.",
  )
  args = parser.parse_args()

  config = get_config(args.config)
  set_env_params(config)
  jax.config.update("jax_enable_x64", True)

  crystal = create_crystal(config)
  g_vec, _, kpts = create_grids(config)
  freq_mask = jnp.array(create_freq_mask(config))

  xc_name = "LDA" if "lda" in config.xc.lower() else "PBE"
  pseudopot = _build_paw_pseudopotential(crystal, xc_name)

  beta_gk = beta_sbt_grid(
    pseudopot.r_grid,
    pseudopot.nonlocal_beta_grid,
    pseudopot.nonlocal_angular_momentum,
    np.array(g_vec),
    np.array(kpts),
  )
  beta_gk = [jnp.array(b) for b in beta_gk]

  nonlocal_q_matrix = [
    jnp.array(q) * jnp.sqrt(4 * jnp.pi)
    for q in pseudopot.nonlocal_d_matrix
  ]
  iden = [jnp.eye(q.shape[0]) for q in nonlocal_q_matrix]

  psi_G = potential_nonlocal_psi_reciprocal(
    crystal.positions,
    g_vec,
    kpts,
    pseudopot.r_grid,
    pseudopot.nonlocal_beta_grid,
    pseudopot.nonlocal_angular_momentum,
    iden,
    beta_gk,
    concat=False,
  )

  breakpoint()
  psi_G = jnp.concatenate(
    [_process_psi_g(p, freq_mask) for p in psi_G], axis=-1
  )
  psi_G = psi_G / jnp.sqrt(crystal.vol)
  breakpoint()

  q_mat = _assemble_q_matrix(
    nonlocal_q_matrix,
    [np.array(l) for l in pseudopot.nonlocal_angular_momentum],
  )

  print(f"symbols: {crystal.symbols}")
  print(f"grid_sizes: {config.grid_sizes}")
  print(f"k_grid_sizes: {config.k_grid_sizes}")
  print(f"g-points (mask): {int(jnp.sum(freq_mask))}")
  print(f"q_mat shape: {q_mat.shape}")

  bad, min_eig, min_eig_plus = _check_overlap_psd(
    psi_G, q_mat, args.tol, args.show_all
  )
  print(f"global min eig(RQR^H): {min_eig:.6e}")
  print(f"global min eig(I+RQR^H): {min_eig_plus:.6e}")

  if bad and not args.allow_negative:
    raise AssertionError(
      "I + R @ Q @ R^H is not positive definite. "
      f"min eig = {min_eig_plus:.6e}"
    )


if __name__ == "__main__":
  main()
