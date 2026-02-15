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

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from typing import Optional, Tuple

from .paw_calc import calc_paw
from .dataclass import PawPseudopotential, PawSetupBundle
from .load_gpaw import find_gpaw_setup, parse_paw_setup
from .load_qe import parse_upf
from .beta import _beta_sbt_single_atom
from .utils import map_over_atoms, pack


def _expand_paw_matrix(data: list, n_proj, l_j) -> jnp.ndarray:
  """Expand a radial matrix to the projector (l,m) basis."""
  data = jnp.array(data).reshape((len(l_j), len(l_j)))
  expanded_data = jnp.zeros((n_proj, n_proj))
  i1 = 0
  for j1, l1 in enumerate(l_j):
    for m1 in range(2 * l1 + 1):
      i2 = 0
      for j2, l2 in enumerate(l_j):
        for m2 in range(2 * l2 + 1):
          if l1 == l2 and m1 == m2:
            expanded_data = expanded_data.at[i1, i2].set(data[j1, j2])
          i2 += 1
      i1 += 1
  return expanded_data


def build_paw_setup(crystal, xc_name: str) -> PawSetupBundle:
  """Build PAW setup data and a minimal pseudopotential container."""
  atom_symbols = list(crystal.symbols)
  atoms_list = [f"{sym}{i + 1}" for i, sym in enumerate(atom_symbols)]
  atom_symbol_map = {label: sym for label, sym in zip(atoms_list, atom_symbols)}
  atom_index_map = {label: i for i, label in enumerate(atoms_list)}

  K_p = {}
  K_c = {}
  M = {}
  M_p = {}
  M_pp = {}
  MB = {}
  MB_p = {}
  n_qg = {}
  nt_qg = {}
  nc_g = {}
  nct_g = {}
  g_lg = {}
  Delta_pL = {}
  Delta0 = {}
  lmax = {}
  e_xc0 = {}
  r_g = {}
  dr_g = {}
  vbar_g = {}
  T_Lqp = {}

  r_grids = []
  dr_grids = []
  nonlocal_beta_grid = []
  nonlocal_angular_momentum = []
  nonlocal_d_matrix = []
  paw_valence_charges = []

  for a in atoms_list:
    setup_data = setup_gpaw(atom_symbol_map[a], xc_name)
    paw_valence_charges.append(int(round(setup_data.get('valence', 0))))

    r_grids.append(setup_data['r_g'])
    dr_grids.append(setup_data['dr_g'])
    nonlocal_beta_grid.append(setup_data['pt_jg'])
    nonlocal_angular_momentum.append(setup_data['l_j'])

    results = calc_paw(setup_data)
    n_proj = setup_data['pt_jg'].shape[0]
    tmp_mat = np.zeros((n_proj, n_proj))
    tmp_mat[np.triu_indices(n_proj)] = results['Delta_lq'][0]
    tmp_mat = tmp_mat + tmp_mat.T - np.diag(np.diag(tmp_mat))
    nonlocal_d_matrix.append(tmp_mat / np.sqrt(4 * np.pi))

    n_proj_m = int(np.sum(2 * setup_data['l_j'] + 1))
    K_p[a] = _expand_paw_matrix(setup_data['K_p'], n_proj_m, setup_data['l_j'])
    K_c[a] = setup_data['K_c']
    M[a] = results["M"]
    M_p[a] = results["M_p"]
    M_pp[a] = results["M_pp"]
    MB[a] = results["MB"]
    MB_p[a] = results["MB_p"]
    n_qg[a] = results["n_qg"]
    nt_qg[a] = results["nt_qg"]
    T_Lqp[a] = results["T_Lqp"]
    g_lg[a] = results["g_lg"]
    Delta_pL[a] = results["Delta_pL"]
    Delta0[a] = results["Delta0"]

    nc_g[a] = setup_data["nc_g"]
    nct_g[a] = setup_data["nct_g"]
    lmax[a] = int(setup_data["lmax"])
    e_xc0[a] = setup_data["e_xc"]
    r_g[a] = setup_data["r_g"]
    dr_g[a] = setup_data["dr_g"]
    vbar_g[a] = setup_data["vbar_g"]

  pseudopot = PawPseudopotential(
    num_atom=len(atom_symbols),
    positions=crystal.positions,
    charges=crystal.charges,
    atomic_symbols=atom_symbols,
    valence_charges=paw_valence_charges,
    r_grid=r_grids,
    dr_grid=dr_grids,
    nonlocal_beta_grid=nonlocal_beta_grid,
    nonlocal_angular_momentum=nonlocal_angular_momentum,
    nonlocal_d_matrix=nonlocal_d_matrix
  )

  # Build projector indices per atom for D_p construction
  l_max_global = int(np.max(np.hstack(nonlocal_angular_momentum)))
  beta_counts = [len(l_list) for l_list in nonlocal_angular_momentum]
  beta_offsets = np.cumsum([0] + beta_counts[:-1])
  index_map = {}
  for a, offset, l_list in zip(
    atoms_list, beta_offsets, nonlocal_angular_momentum
  ):
    beta_idx = []
    phi_idx = []
    for b, ell in enumerate(l_list):
      ell = int(ell)
      for m in range(-ell, ell + 1):
        beta_idx.append(offset + b)
        phi_idx.append(l_max_global + m)
    index_map[a] = (jnp.array(beta_idx), jnp.array(phi_idx))

  valence_charges = np.sum(pseudopot.valence_charges)

  return PawSetupBundle(
    pseudopot=pseudopot,
    atoms_list=atoms_list,
    atom_symbol_map=atom_symbol_map,
    atom_index_map=atom_index_map,
    index_map=index_map,
    valence_charges=valence_charges,
    K_p=K_p,
    K_c=K_c,
    M=M,
    M_p=M_p,
    M_pp=M_pp,
    MB=MB,
    MB_p=MB_p,
    n_qg=n_qg,
    nt_qg=nt_qg,
    nc_g=nc_g,
    nct_g=nct_g,
    g_lg=g_lg,
    Delta_pL=Delta_pL,
    Delta0=Delta0,
    lmax=lmax,
    e_xc0=e_xc0,
    r_g=r_g,
    dr_g=dr_g,
    vbar_g=vbar_g,
    T_Lqp=T_Lqp
  )


def _build_paw_precompute(paw, crystal, g_vec):
  """CPU-boundary precompute for PAW terms (NumPy + one-time JAX transfer)."""
  from .spherical_harmonics import Yarr

  g_vec_np = np.asarray(g_vec)
  num_grids = int(np.prod(g_vec_np.shape[:-1]))

  atoms_list = paw.atoms_list
  positions = [
    np.asarray(crystal.positions[paw.atom_index_map[atom]])
    for atom in atoms_list
  ]
  g_lg_list = [np.asarray(paw.g_lg[atom]) for atom in atoms_list]
  r_g_list = [np.asarray(paw.r_g[atom]) for atom in atoms_list]
  dr_g_list = [np.asarray(paw.dr_g[atom]) for atom in atoms_list]
  lmax_list = [paw.lmax[atom] for atom in atoms_list]
  nct_g_list = [np.asarray(paw.nct_g[atom]) for atom in atoms_list]
  vbar_g_list = [np.asarray(paw.vbar_g[atom]) for atom in atoms_list]

  def precompute_ghat_LG(g_vec_grid, g_lg_radial, r_radial, dr_radial, lmax_val):
    g_norm = np.linalg.norm(g_vec_grid, axis=-1)
    g_hat = np.zeros_like(g_vec_np)
    mask = g_norm > 0
    g_hat[mask] = g_vec_grid[mask] / g_norm[mask][..., None]
    Lmax_val = (lmax_val + 1) ** 2
    Y_LG = Yarr(list(range(Lmax_val)), g_hat)

    l_list = np.arange(lmax_val + 1, dtype=int)
    radial_int_lG = _beta_sbt_single_atom(
      r_radial,
      dr_radial,
      g_lg_radial,
      l_list,
      g_vec_grid,
      None,
    )[0]

    ghat_LG = np.zeros((Lmax_val, *g_norm.shape), dtype=np.complex128)
    for L in range(Lmax_val):
      ell = int(np.floor(np.sqrt(L)))
      ghat_LG[L] = 4 * np.pi * (-1j) ** ell * radial_int_lG[ell] * Y_LG[L]
    # Match FFT normalization used by density_grid_reciprocal
    ghat_LG *= num_grids / crystal.vol
    return ghat_LG

  @map_over_atoms
  def _precompute_atom(position, g_lg, r_g, dr_g, lmax, nct_g, vbar_g):
    ghat_LG = precompute_ghat_LG(g_vec_np, g_lg, r_g, dr_g, lmax)
    phase_G = np.exp(
      -1j * np.einsum("xyzc,c->xyz", g_vec_np, position)
    )
    nct_G_atom = _beta_sbt_single_atom(
      r_g,
      dr_g,
      nct_g[None, :],
      np.array([0]),
      g_vec_np,
      None,
    )[0, 0] * 4 * np.pi * num_grids / crystal.vol
    vbar_G_atom = _beta_sbt_single_atom(
      r_g,
      dr_g,
      vbar_g[None, :] / np.sqrt(4 * np.pi),
      np.array([0]),
      g_vec_np,
      None,
    )[0, 0] * 4 * np.pi * num_grids / crystal.vol
    e_zero0_atom = np.sum(
      nct_g * np.sqrt(4 * np.pi) * vbar_g * r_g**2 * dr_g
    )
    return (
      ghat_LG,
      phase_G,
      phase_G * nct_G_atom,
      phase_G * vbar_G_atom,
      e_zero0_atom,
    )

  outputs = _precompute_atom(
    positions, g_lg_list, r_g_list, dr_g_list, lmax_list, nct_g_list,
    vbar_g_list
  )
  if outputs:
    ghat_list, phase_list, nct_terms, vbar_terms, e_zero_terms = zip(
      *outputs
    )
  else:
    ghat_list = []
    phase_list = []
    nct_terms = []
    vbar_terms = []
    e_zero_terms = []

  ghat_LG = dict(
    (atom, jnp.array(ghat)) for atom, ghat in zip(atoms_list, ghat_list)
  )
  phase_G = dict(
    (atom, jnp.array(phase)) for atom, phase in zip(atoms_list, phase_list)
  )
  nct_G = sum(nct_terms) if nct_terms else 0.0
  vbar_G = sum(vbar_terms) if vbar_terms else 0.0
  e_zero0 = sum(e_zero_terms) if e_zero_terms else 0.0

  nct_G = jnp.array(nct_G)
  vbar_G = jnp.array(vbar_G)
  e_zero0 = jnp.array(e_zero0)

  nct_g = jnp.fft.ifftn(nct_G, axes=range(-3, 0)).real

  # NOTE: test the total charge of the core electrons
  atom = paw.atoms_list[-1]
  _ = _beta_sbt_single_atom(
    np.asarray(paw.r_g[atom]),
    np.asarray(paw.dr_g[atom]),
    np.asarray(paw.nc_g[atom])[None, :],
    np.array([0]),
    g_vec_np,
    None,
  )[0, 0] * 4 * np.pi * num_grids / crystal.vol
  nc = np.sum(
    np.asarray(paw.nc_g[atom]) * 4 * np.pi
    * np.asarray(paw.r_g[atom])**2 * np.asarray(paw.dr_g[atom])
  )
  print(f"Core charge from real space integration: {nc:.6f}")
  return ghat_LG, phase_G, nct_G, vbar_G, e_zero0, nct_g


def build_paw_precompute(paw, crystal, g_vec):
  """Precompute PAW compensation charge terms on the PW grid.

  Returns:
    Tuple of (ghat_LG, phase_G, nct_G, vbar_G, e_zero0, nct_g).
  """
  from .spherical_harmonics import Yarr

  def precompute_ghat_LG(g_vec_grid, g_lg_radial, r_radial, dr_radial, lmax_val):
    g_vec_np = np.array(g_vec_grid)
    g_norm = np.linalg.norm(g_vec_np, axis=-1)
    g_hat = np.zeros_like(g_vec_np)
    mask = g_norm > 0
    g_hat[mask] = g_vec_np[mask] / g_norm[mask][..., None]
    Lmax_val = (lmax_val + 1) ** 2
    Y_LG = Yarr(list(range(Lmax_val)), g_hat)

    l_list = np.arange(lmax_val + 1, dtype=int)
    radial_int_lG = _beta_sbt_single_atom(
      np.array(r_radial),
      np.array(dr_radial),
      np.array(g_lg_radial),
      l_list,
      g_vec_np,
      None,
    )[0]

    ghat_LG = np.zeros((Lmax_val, *g_norm.shape), dtype=np.complex128)
    for L in range(Lmax_val):
      ell = int(np.floor(np.sqrt(L)))
      ghat_LG[L] = 4 * np.pi * (-1j) ** ell * radial_int_lG[ell] * Y_LG[L]
    # Match FFT normalization used by density_grid_reciprocal
    num_grids = np.prod(g_vec_grid.shape[:-1])
    ghat_LG *= num_grids / crystal.vol
    ghat_LG = jnp.array(ghat_LG)
    return ghat_LG

  num_grids = np.prod(g_vec.shape[:-1])
  atoms_list = paw.atoms_list
  positions = [
    crystal.positions[paw.atom_index_map[atom]] for atom in atoms_list
  ]
  g_lg_list = [paw.g_lg[atom] for atom in atoms_list]
  r_g_list = [paw.r_g[atom] for atom in atoms_list]
  dr_g_list = [paw.dr_g[atom] for atom in atoms_list]
  lmax_list = [paw.lmax[atom] for atom in atoms_list]
  nct_g_list = [paw.nct_g[atom] for atom in atoms_list]
  vbar_g_list = [paw.vbar_g[atom] for atom in atoms_list]

  @map_over_atoms
  def _precompute_atom(position, g_lg, r_g, dr_g, lmax, nct_g, vbar_g):
    ghat_LG = precompute_ghat_LG(g_vec, g_lg, r_g, dr_g, lmax)
    phase_G = jnp.exp(
      -1j * jnp.einsum(
        "xyzc,c->xyz",
        g_vec,
        position
      )
    )
    nct_G_atom = _beta_sbt_single_atom(
      r_g,
      dr_g,
      nct_g[None, :],
      np.array([0]),
      g_vec,
      None,
    )[0, 0] * 4 * np.pi * num_grids / crystal.vol
    vbar_G_atom = _beta_sbt_single_atom(
      r_g,
      dr_g,
      vbar_g[None, :] / jnp.sqrt(4 * jnp.pi),
      np.array([0]),
      g_vec,
      None,
    )[0, 0] * 4 * np.pi * num_grids / crystal.vol
    e_zero0_atom = jnp.sum(
      nct_g * jnp.sqrt(4 * jnp.pi) * vbar_g
      * r_g**2 * dr_g
    )
    return (
      ghat_LG,
      phase_G,
      phase_G * nct_G_atom,
      phase_G * vbar_G_atom,
      e_zero0_atom,
    )

  outputs = _precompute_atom(
    positions, g_lg_list, r_g_list, dr_g_list, lmax_list, nct_g_list,
    vbar_g_list
  )
  if outputs:
    ghat_list, phase_list, nct_terms, vbar_terms, e_zero_terms = zip(
      *outputs
    )
  else:
    ghat_list = []
    phase_list = []
    nct_terms = []
    vbar_terms = []
    e_zero_terms = []

  ghat_LG = dict(zip(atoms_list, ghat_list))
  phase_G = dict(zip(atoms_list, phase_list))
  nct_G = sum(nct_terms) if nct_terms else 0.0
  vbar_G = sum(vbar_terms) if vbar_terms else 0.0
  e_zero0 = sum(e_zero_terms) if e_zero_terms else 0.0

  nct_g = jnp.fft.ifftn(nct_G, axes=range(-3, 0)).real

  # NOTE: test the total charge of the core electrons
  atom = paw.atoms_list[-1]
  _ = _beta_sbt_single_atom(
    paw.r_g[atom],
    paw.dr_g[atom],
    paw.nc_g[atom][None, :],
    np.array([0]),
    g_vec,
    None,
  )[0, 0] * 4 * np.pi * num_grids / crystal.vol
  nc = jnp.sum(
    paw.nc_g[atom] * 4 * jnp.pi * paw.r_g[atom]**2 * paw.dr_g[atom]
  )
  print(f"Core charge from real space integration: {nc:.6f}")
  return ghat_LG, phase_G, nct_G, vbar_G, e_zero0, nct_g


def load_gpaw_coeff_metadata(config) -> Tuple[Optional[np.lib.npyio.NpzFile], Optional[np.ndarray]]:
  """Load GPAW exported coefficients metadata from config if provided.

  Returns:
    Tuple of (gpaw_coeff_data, gpaw_kpts_frac).
    `gpaw_coeff_data` is None when no path is provided.
  """
  gpaw_coeff_data = None
  gpaw_kpts_frac = None
  gpaw_coeff_path = getattr(config, "gpaw_coeff_path", None)
  if gpaw_coeff_path:
    gpaw_coeff_data = np.load(gpaw_coeff_path, allow_pickle=True)
    if "grid_sizes" in gpaw_coeff_data:
      grid_sizes = [int(x) for x in gpaw_coeff_data["grid_sizes"]]
      if len(set(grid_sizes)) != 1:
        raise ValueError(
          f"GPAW grid_sizes {grid_sizes} are not cubic; set config.grid_sizes "
          "accordingly before importing coefficients."
        )
      config.grid_sizes = int(grid_sizes[0])
    kpts_frac = gpaw_coeff_data.get("kpts_frac", None)
    if kpts_frac is not None:
      gpaw_kpts_frac = np.asarray(kpts_frac)
  return gpaw_coeff_data, gpaw_kpts_frac


def compare_gpaw_coefficients(
  gpaw_coeff_data,
  crystal,
  paw,
  params_pw,
  params_occ,
  g_vec,
  calc_atomic_density_matrix,
  total_energy,
):
  """Run GPAW-vs-Jrystal projector and energy split comparisons."""
  coeff = gpaw_coeff_data["coeff"]
  occ = gpaw_coeff_data["occupation"]
  coeff_cmp = jnp.array(np.asarray(coeff))
  occ_cmp = jnp.array(occ)
  if coeff_cmp.ndim == 4:
    coeff_cmp = coeff_cmp[None, ...]
  if occ_cmp.ndim == 2:
    occ_cmp = occ_cmp[None, ...]
  ngrid = np.prod(coeff_cmp.shape[-3:])
  coeff_cmp = coeff_cmp * (jnp.sqrt(crystal.vol) / ngrid)
  d_p_cmp_list, f_matrix_cmp = calc_atomic_density_matrix(
    coeff_cmp, occ_cmp, return_f_matrix=True
  )

  # Compare P_ni via exported f_GI from GPAW
  for s in range(coeff_cmp.shape[0]):
    for k in range(coeff_cmp.shape[1]):
      key_f = f"proj_f_GI_k{k}"
      key_q = f"proj_Q_G_k{k}"
      key_idx = f"proj_indices_k{k}"
      if key_f not in gpaw_coeff_data.files:
        continue
      f_GI = np.asarray(gpaw_coeff_data[key_f])
      Q_G = np.asarray(gpaw_coeff_data[key_q])
      indices = np.asarray(gpaw_coeff_data[key_idx])
      flat = coeff_cmp[s, k].reshape(coeff_cmp.shape[2], -1)
      psit_nG = flat[:, Q_G]
      P_alt = psit_nG @ f_GI.conj()
      for a, I1, I2 in indices:
        key = f"P_ani_{a}_s{s}_k{k}"
        if key not in gpaw_coeff_data.files:
          continue
        gpaw_P = np.asarray(gpaw_coeff_data[key])
        diff = float(np.max(np.abs(P_alt[:, I1:I2] - gpaw_P)))
        print(f"P_ni (from f_GI) compare atom {a} s{s} k{k}: max|Δ| = {diff:.6e}")

  atoms_list = paw.atoms_list
  idx_list = [paw.atom_index_map[atom] for atom in atoms_list]
  l_m_list = [paw.index_map[atom] for atom in atoms_list]

  @map_over_atoms
  def _compare_p_ni(atom, idx, l_m):
    l_idx, m_idx = l_m
    P_cmp = f_matrix_cmp[..., l_idx, m_idx]
    for s in range(P_cmp.shape[0]):
      for k in range(P_cmp.shape[1]):
        key = f"P_ani_{idx}_s{s}_k{k}"
        if key not in gpaw_coeff_data.files:
          continue
        gpaw_P = np.asarray(gpaw_coeff_data[key])
        diff = float(jnp.max(jnp.abs(P_cmp[s, k].conj() - gpaw_P)))
        print(
          f"P_ni compare atom {atom} s{s} k{k}: max|Δ| = {diff:.6e}"
        )

  @map_over_atoms
  def _compare_d_asp(atom, idx, D_p_atom):
    key = f"D_asp_{idx}"
    if key not in gpaw_coeff_data.files:
      return
    gpaw_packed = np.asarray(gpaw_coeff_data[key])[0]
    diff = float(jnp.max(jnp.abs(gpaw_packed - pack(D_p_atom))))
    print(f"D_asp compare atom {atom}: max|Δ| = {diff:.6e}")

  _compare_p_ni(atoms_list, idx_list, l_m_list)
  _compare_d_asp(atoms_list, idx_list, d_p_cmp_list)

  total, kinetic, hartree, exc = total_energy(
    params_pw,
    params_occ,
    g_vec,
    coeff_occ_override=(coeff_cmp, occ_cmp),
  )
  total = jax.block_until_ready(total)
  kinetic = jax.block_until_ready(kinetic)
  hartree = jax.block_until_ready(hartree)
  exc = jax.block_until_ready(exc)
  e_zero = total - kinetic - hartree - exc

  required_keys = (
    "gpaw_e_kinetic",
    "gpaw_e_coulomb",
    "gpaw_e_zero",
    "gpaw_e_xc",
  )
  if all(key in gpaw_coeff_data.files for key in required_keys):
    gpaw_kinetic = float(gpaw_coeff_data["gpaw_e_kinetic"])
    gpaw_coulomb = float(gpaw_coeff_data["gpaw_e_coulomb"])
    gpaw_zero = float(gpaw_coeff_data["gpaw_e_zero"])
    gpaw_xc = float(gpaw_coeff_data["gpaw_e_xc"])
    jr_kinetic = float(jnp.real(kinetic))
    jr_coulomb = float(jnp.real(hartree))
    jr_zero = float(jnp.real(e_zero))
    jr_xc = float(jnp.real(exc))
    logging.info("GPAW vs Jrystal four energy components (Ha):")
    logging.info(f"  Kinetic: {jr_kinetic:.6f} vs {gpaw_kinetic:.6f}")
    logging.info(f"  Coulomb: {jr_coulomb:.6f} vs {gpaw_coulomb:.6f}")
    logging.info(f"  E_zero: {jr_zero:.6f} vs {gpaw_zero:.6f}")
    logging.info(f"  XC: {jr_xc:.6f} vs {gpaw_xc:.6f}")
    logging.info("GPAW vs Jrystal four-component deltas (Ha):")
    logging.info(f"  ΔKinetic: {(jr_kinetic-gpaw_kinetic):.6e}")
    logging.info(f"  ΔCoulomb: {(jr_coulomb-gpaw_coulomb):.6e}")
    logging.info(f"  ΔE_zero: {(jr_zero-gpaw_zero):.6e}")
    logging.info(f"  ΔXC: {(jr_xc-gpaw_xc):.6e}")
  else:
    logging.info(
      "Skip four-component energy comparison: missing keys %s",
      required_keys,
    )
  return


def setup_gpaw(atom_type: str, xc_name: str = "PBE"):
  """Load and parse GPAW PAW setup file.
  
  Grid Properties:
  - Units: Bohr (atomic units)
  - Grid equation: r = a * i / (n - i)
  - Integration: ∫n_stored * r² * dr * √(4π) = N_electrons
  
  Returns:
    Tuple with all PAW quantities including g_lg for compensation charges
  """
  
  # Load GPAW setup file
  setup_path = find_gpaw_setup(None, atom_type, xc=xc_name)
  pp_data = parse_paw_setup(setup_path)
  
  # Extract basic properties
  Z = int(pp_data['atom']['Z'])  # Total atomic number
  valence = pp_data['atom'].get('valence')
  if valence is None:
    # Fallback: if valence not provided, use Z - core when available
    core = pp_data['atom'].get('core')
    if core is not None:
      valence = float(Z) - float(core)
    else:
      valence = float(Z)
  
  # Construct radial grid
  grid_info = pp_data['radial_grid']
  a = grid_info['a']
  n = grid_info['n']
  i = np.arange(n, dtype=np.float64)
  r_g = a * i / (n - i)  # Keep original grid for g_lg calculation
  dr_g = a * n / (n - i) ** 2
  
  # Angular momentum information from valence states
  l_j = np.array([state['l'] for state in pp_data['valence_states']], dtype=int)
  lcut = int(np.max(l_j))
  rc_values = [
    state.get('rc') for state in pp_data.get('valence_states', [])
    if state.get('rc') is not None
  ]
  if rc_values:
    rcutmax = float(np.max(rc_values))
  else:
    shape_rc = pp_data.get('shape_function', {}).get('rc')
    rcutmax = float(shape_rc) if shape_rc is not None else float(r_g[-1])

  rcut2 = 2.0 * rcutmax
  gcut2 = int(np.searchsorted(r_g, rcut2, side='left'))
  if gcut2 > len(r_g):
    gcut2 = len(r_g)
  
  # Apply cutoff to grid
  r_g = r_g[:gcut2]
  dr_g = dr_g[:gcut2]

  phi_jg = np.array(
    [wave['values'][:gcut2] for wave in pp_data['ae_partial_waves']],
    dtype=np.float64
  )
  phit_jg = np.array(
    [wave['values'][:gcut2] for wave in pp_data['pseudo_partial_waves']],
    dtype=np.float64
  )
  pt_jg = np.array(
    [proj['values'][:gcut2] for proj in pp_data['projector_functions']],
    dtype=np.float64
  )
  # Integration: \sqrt{4π} * ∫ n_c(r) * r² dr = N_core, n_c is the radial component
  nc_g = np.array(pp_data['ae_core_density'][:gcut2], dtype=np.float64) / np.sqrt(
    4 * np.pi
  )
  nct_g = np.array(
    pp_data['pseudo_core_density'][:gcut2], dtype=np.float64
  ) / np.sqrt(4 * np.pi)
  
  # Local potential, skip r=0 point and apply cutoff
  vbar_g = np.array(
    pp_data.get('zero_potential', np.zeros(n, dtype=np.float64))[:gcut2],
    dtype=np.float64
  )
  
  lmax = lcut  # Maximum l for augmentation should equal lcut for GPAW compatibility
  
  shape_params = pp_data.get('shape_function')
  rc = shape_params.get('rc', None)
  sf_type = shape_params.get('type', 'gauss')
  
  # Initialize g_lg for all l values up to lmax
  g_lg = np.zeros((lmax + 1, gcut2), dtype=np.float64)
  if sf_type == 'gauss':
      # Gaussian shape functions following GPAW's convention
      # g_lg[0] = 4 / rc^3 / sqrt(pi) * exp(-(r/rc)^2)
      g_lg[0] = 4 / rc**3 / np.sqrt(np.pi) * np.exp(-(r_g / rc)**2)
      
      # Higher l components: g_lg[l] = 2/(2l+1)/rc^2 * r * g_lg[l-1]
      for ell in range(1, lmax + 1):
          g_lg[ell] = 2.0 / (2 * ell + 1) / rc**2 * r_g * g_lg[ell - 1]
      
      # Normalize each l-component according to GPAW convention
      # GPAW normalizes so that rgd.integrate(g_lg[l], l) = 4π
      # Since integrate multiplies by 4π, the raw integral should be 1.0
      for ell in range(lmax + 1):
          # Calculate integral with 4π factor (like rgd.integrate does)
          # Skip r=0 point in integration like GPAW does
          integral_with_4pi = float(
            np.sum(g_lg[ell, 1:] * r_g[1:]**(ell + 2) * dr_g[1:])
            * 4 * np.pi
          )
          if integral_with_4pi > 1e-10:
              # Divide by integral and multiply by 4π to get correct normalization
              g_lg[ell] = g_lg[ell] / integral_with_4pi * (4 * np.pi)
  else:
      # For other shape function types, use simple fallback
      print(f"Warning: Shape function type '{sf_type}' not fully implemented, using simplified version")
      g_lg[0] = 4 / rc**3 / np.sqrt(np.pi) * np.exp(-(r_g / rc)**2)
      for ell in range(1, lmax + 1):
          g_lg[ell] = 2.0 / (2 * ell + 1) / rc**2 * r_g * g_lg[ell - 1]

  return {
    'r_g': r_g,
    'dr_g': dr_g,
    'phi_jg': phi_jg,
    'phit_jg': phit_jg,
    'nc_g': nc_g,
    'nct_g': nct_g,
    'vbar_g': vbar_g,
    'l_j': l_j,
    'pt_jg': pt_jg,
    'Z': Z,
    'valence': valence,
    'lmax': lmax,
    'lcut': lcut,
    'gcut2': gcut2,
    'g_lg': g_lg,
    'K_p': pp_data['kinetic_energy_differences'],
    'K_c': pp_data['core_energy']['kinetic'] - pp_data['ae_energy']['kinetic'],
    'e_xc': pp_data['ae_energy']["xc"]
  }


def setup_qe(upf_path: Optional[str] = None):
  """Load and parse QE UPF pseudopotential file.
  
  WARNING NOTE: this function is deprecated and jrystal currently only supports PAW
  calculations using the pp data from GPAW

  This function reads a Quantum ESPRESSO UPF file and extracts PAW data.
  Values are returned in QE's native storage convention without conversion.
  QE UPF Storage Conventions (as documented in paw_pp_file_documentation.md)
  
  Returns:
    Tuple of arrays containing PAW data in QE native convention
  """

  if upf_path is None:
    raise ValueError(
      "setup_qe is deprecated and no default UPF path is configured. "
      "Provide an explicit `upf_path` if you still need this code path."
    )

  # load the pseudopotential
  pp_dict = parse_upf(upf_path)
  Z = 6  # Atomic number for Carbon
  lmax = int(pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['l_max_aug'])  # Max l for augmentation
  l_j = jnp.array([int(proj['angular_momentum']) for proj in pp_dict['PP_NONLOCAL']['PP_BETA']])  # l for each projector
  lcut = max(l_j)  # Maximum l among projectors
  gcut_j = jnp.array([int(proj['cutoff_radius_index']) for proj in pp_dict['PP_NONLOCAL']['PP_BETA']])  # Grid indices
  gcut = jnp.max(gcut_j)  # Use maximum grid index for uniform cutoff
  
  # Extract radial grid (units: Bohr)
  r_g = jnp.array(pp_dict['PP_MESH']['PP_R'])[:gcut]  # Radial points
  dr_g = jnp.array(pp_dict['PP_MESH']['PP_RAB'])[:gcut]  # r * dr for integration
  
  # Extract radial functions (in QE storage convention)
  pt_jg = jnp.array([proj['values'] for proj in pp_dict['PP_NONLOCAL']['PP_BETA']])[:, :gcut]  # β(r) * r * √(4π)
  phi_jg = jnp.array([phi['values'] for phi in pp_dict['PP_FULL_WFC']['PP_AEWFC']])[:, :gcut]  # φ(r) * r * √(4π)
  phit_jg = jnp.array([phi['values'] for phi in pp_dict['PP_FULL_WFC']['PP_PSWFC']])[:, :gcut]  # φ̃(r) * r * √(4π)
  # Core densities (stored as n(r) without factors in QE)
  # Integration: ∫ n_c(r) * 4π * r² dr = N_core
  nc_g = jnp.array(pp_dict['PP_PAW']['PP_AE_NLCC'])[:gcut]  # AE core density n_c(r)
  nct_g = jnp.array(pp_dict['PP_NLCC'])[:gcut]  # Pseudo core density ñ_c(r)
  vbar_g = jnp.array(pp_dict['PP_LOCAL'])[:gcut]  # Local pseudopotential V_loc(r)

  # Augmentation charge setup
  nj = len(l_j)  # Number of projector radial functions
  nq = nj * (nj + 1) // 2  # Number of unique pairs (upper triangular)
  
  # Augmentation functions Q_ij^l(r) - stored as Q(r) * r² in QE
  # Convert to physical Q(r) by dividing by r² and 4π
  n_lqg = jnp.zeros((2 * lcut + 1, nq, gcut))
  
  # Multipole moments Δ_lq from PP_MULTIPOLES
  # NOTE: this is the same as our calculation, we do not use it
  Delta_lq = jnp.array(pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['PP_MULTIPOLES']).reshape(lmax + 1, nj, nj)
  Delta_lq = jnp.transpose(Delta_lq, (1, 2, 0))[jnp.triu_indices(nj)].T
  
  # Extract augmentation functions Q_ij^l(r) from PP_QIJ
  for qijl in pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['PP_QIJ']:
    # QE stores Q(r) * r², convert to Q(r) / 4π for internal use
    n_lqg = n_lqg.at[
      int(qijl['angular_momentum']),
      int(qijl['first_index']) * nj + int(qijl['second_index'])
    ].set(jnp.array(qijl['values'][:gcut]) / r_g[:gcut]**2 / 4 / jnp.pi)

  assert r_g.shape[0] == gcut
  assert dr_g.shape[0] == gcut
  assert phi_jg.shape[1] == gcut
  assert phit_jg.shape[1] == gcut
  assert nc_g.shape[0] == gcut
  assert nct_g.shape[0] == gcut
  
  return r_g, dr_g, phi_jg, phit_jg, nc_g, nct_g, vbar_g, l_j, pt_jg, Z, lmax, lcut, gcut
