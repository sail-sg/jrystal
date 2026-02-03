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

import time
from dataclasses import dataclass
from math import ceil
from typing import List, Union

import jax
import numpy as np
import optax
from absl import logging
from einops import einsum
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

from .._src import energy, entropy, occupation, pw, xc
from .._src.crystal import Crystal
from .._src.grid import proper_grid_size
from ..config import JrystalConfigDict
from ..pseudopotential import normcons
from ..pseudopotential.utils import map_over_atoms
from .convergence import create_convergence_checker
from ..pseudopotential.paw_setup import build_paw_precompute, build_paw_setup
from ..pseudopotential.paw_calc import compute_proj_pw_overlap
from .opt_utils import (
  create_crystal,
  create_freq_mask,
  create_grids,
  create_optimizer,
  set_env_params,
)
from .pre_calc import pre_calc_beta_sbt


@dataclass
class GroundStateEnergyOutput:
  """Output of the ground state energy calculation.

  Args:
    config (JrystalConfigDict): The configuration for the calculation.
    crystal (Crystal): The crystal object.
    params_pw (dict): Parameters for the plane wave basis.
    params_occ (dict): Parameters for the occupation.
    total_energy (Union[float, jax.Array]): The total energy of the crystal.
    total_energy_history (List[float]): The optimization history of the total
    energy.
  """
  config: JrystalConfigDict
  crystal: Crystal
  params_pw: dict
  params_occ: dict
  total_energy: Union[float, jax.Array]
  total_energy_history: List[float]


def pack(D_p: jnp.ndarray) -> jnp.ndarray:
  """Pack a Hermitian matrix for better efficiency.

  The diagonal elements are halved to calculate the inner product.
  """
  n = D_p.shape[-1]
  tmp = D_p.copy()
  tmp = tmp.at[..., jnp.arange(n), jnp.arange(n)].set(
    tmp[..., jnp.arange(n), jnp.arange(n)] / 2
  )
  return tmp[0, 0][jnp.triu_indices(n)].real * 2


def calc(config: JrystalConfigDict) -> GroundStateEnergyOutput:
  """Calculate the ground state energy of a crystal with norm-conserving
  pseudopotential.

  Args:
      config (JrystalConfigDict): The configuration for the calculation.

  Returns:
      GroundStateEnergyOutput: The ground state energy output of the crystal.
  """
  # Initialize and Prepare variables.
  set_env_params(config)
  key = jax.random.PRNGKey(config.seed)
  temp = config.smearing
  gpaw_coeff_data = None
  gpaw_kpts_frac = None
  gpaw_coeff_path = getattr(config, "gpaw_coeff_path", None)
  # if gpaw_coeff_path:
  #   gpaw_coeff_data = np.load(gpaw_coeff_path, allow_pickle=True)
  #   if "grid_sizes" in gpaw_coeff_data:
  #     grid_sizes = [int(x) for x in gpaw_coeff_data["grid_sizes"]]
  #     if len(set(grid_sizes)) != 1:
  #       raise ValueError(
  #         f"GPAW grid_sizes {grid_sizes} are not cubic; set config.grid_sizes "
  #         "accordingly before importing coefficients."
  #       )
  #     config.grid_sizes = int(grid_sizes[0])
  #   kpts_frac = gpaw_coeff_data.get("kpts_frac", None)
  #   if kpts_frac is not None:
  #     gpaw_kpts_frac = np.asarray(kpts_frac)

  crystal = create_crystal(config)
  xc_name = "LDA" if "lda" in config.xc.lower() else "PBE"
  paw = build_paw_setup(crystal, xc_name)
  pseudopot = paw.pseudopot
  logging.info(f"Crystal: {crystal.symbols}")
  EPS = config.eps

  # Initialize the mesh and sharding for the parallelization.
  num_devices = len(jax.devices())
  util_devices = num_devices if config.parallel_over_k_mesh else 1
  logging.info(f"Parallel over k-mesh: {config.parallel_over_k_mesh}.")
  logging.info(f"Number of devices (used): {num_devices}({util_devices}).")

  mesh = Mesh(
    np.array(jax.devices()[:util_devices]).reshape([1, -1]), ('s', 'k')
  )
  sharding = NamedSharding(mesh, P('s', 'k'))  # shard by the kpt dimension.

  g_vec, r_vec, k_vec = create_grids(config)
  if gpaw_kpts_frac is not None:
    bvec = 2 * jnp.pi * jnp.linalg.inv(crystal.cell_vectors).T
    k_vec = jnp.array(gpaw_kpts_frac) @ bvec
  num_kpts = k_vec.shape[0]
  logging.info(f"Number of G-vectors: {proper_grid_size(config.grid_sizes)}")
  logging.info(f"Number of k-vectors: {proper_grid_size(config.k_grid_sizes)}")
  num_bands = ceil(paw.valence_charges / 2) + config.empty_bands
  logging.info(f"num_bands: {num_bands}")
  logging.info(f"XC functional: {config.xc}")
  freq_mask = create_freq_mask(config)
  (
    ghat_LG,
    phase_G,
    nct_G,
    vbar_G,
    e_zero0,
    nct_g_,
  ) = build_paw_precompute(paw, crystal, g_vec)

  convergence_checker = create_convergence_checker(config)
  converged = False
  # initialize pseudopotential
  logging.info("Initializing pseudopotential (local)...")
  start = time.time()

  k_vec = jax.device_put(k_vec, NamedSharding(mesh, P('k')))
  logging.info(
    f"Local pseudopotential done. Time: {time.time() - start:.2f} seconds"
  )
  logging.info("Initializing pseudopotential (Spherical Bessel Transform)...")
  start = time.time()
  beta_gk = pre_calc_beta_sbt(
    pseudopot,
    np.array(g_vec),
    np.array(k_vec)
  )
  beta_gk = jax.device_put(beta_gk, NamedSharding(mesh, P('k')))
  end = time.time()
  logging.info(
    f"Spherical Bessel Transform done. Times: {end - start:.2f} seconds"
  )
  logging.info("Initializing pseudopotential (nonlocal)...")
  start = time.time()
  proj_pw_overlap = normcons.potential_nonlocal_psi_reciprocal(
    crystal.positions,
    g_vec,
    k_vec,
    pseudopot.r_grid,
    pseudopot.nonlocal_beta_grid,
    pseudopot.nonlocal_angular_momentum,
    [jnp.eye(q.shape[0]) for q in pseudopot.nonlocal_d_matrix],
    beta_gk
  )
  # NOTE: we have checked that the overlap matrix we obtained is correct
  # tmp = beta_gk[0].reshape(5, -1).T
  # overlap2 = compute_proj_pw_overlap(g_vec.reshape(-1, 3), crystal.positions[0])
  # idx = index_map[atoms_list[0]]
  # overlap1 = proj_pw_overlap[0, idx[0], idx[1], ...].reshape(13, config.grid_sizes**3).T
  # print(jnp.abs(overlap1 - overlap2).max())

  end = time.time()
  logging.info(f"Nonlocal potential done. Times: {end - start:.2f} seconds")
  logging.info("Deploying pseudopotential (nonlocal)...")
  start = time.time()
  # potential_nl = jax.device_put(potential_nl, NamedSharding(mesh, P('k')))
  end = time.time()
  logging.info(
    f"Deploying pseudopotential (nonlocal) done. "
    f"Times: {end - start:.2f} seconds"
  )

  from ..pseudopotential.ultrasoft import (
    check_uspp_overlap,
    get_ultrasoft_coeff_fun,
  )
  get_ultrasoft_coeff = get_ultrasoft_coeff_fun(
    crystal.positions,
    k_vec,
    g_vec,
    freq_mask,
    crystal.vol,
    pseudopot.r_grid,
    pseudopot.nonlocal_beta_grid,
    pseudopot.nonlocal_angular_momentum,
    pseudopot.nonlocal_d_matrix,
    beta_gk
  )
  del beta_gk

  # Define functions for energy calculation.
  def get_occupation(params):
    return occupation.occupation(
      params,
      num_kpts,
      num_electrons=np.sum(pseudopot.valence_charges),
      spin=crystal.spin,
      method=config.occupation,
      spin_restricted=config.spin_restricted
    )
  
  from gpaw.sphere.lebedev import weight_n, Y_nL
  weight_n = jnp.array(weight_n)
  Y_nL = jnp.array(Y_nL)

  def calc_atomic_density_matrix(coeff, occ, return_f_matrix: bool = False):

    _f_matrix = einsum(
      coeff.conj(),
      proj_pw_overlap,
      "s k band x y z, k  beta phi x y z -> s k band beta phi"
    )
    """
    NOTE: 
    `proj_pw_overlap` evaluates $<G|p_i>$ without normalization factor
    $\sqrt{\Omega}$ of the plane wave basis. The orbital is defined as:

    $$  \phi_n = \sum_G c_{nG} \frac{1}{\sqrt{\Omega}} e^{i(G+k)r} $$

    Therefore, when we evaluate the overlap between the orbital and the projector,
    we should include the normalization factor as below.
    """
    _f_matrix /= jnp.sqrt(crystal.vol)

    atoms_list = paw.atoms_list
    idx_list = [paw.index_map[atom] for atom in atoms_list]

    @map_over_atoms
    def _calc_d_p(idx):
      l_idx, m_idx = idx
      return einsum(
        _f_matrix[..., l_idx, m_idx].conj(),
        occ,
        _f_matrix[..., l_idx, m_idx],
        "s k band proj1, s k band, s k band proj2 -> s k proj1 proj2"
      )

    D_p_list = _calc_d_p(idx_list)

    if return_f_matrix:
      return D_p_list, _f_matrix
    return D_p_list
  
  def calc_paw_xc_correction(atom: str, D_p_packed):
      def _calculate_xc_energy(D_sLq, n_qg, nc0_sg):

        n_sLg = jnp.dot(D_sLq, n_qg)  # shape: [n_spin, Lmax, n_g]
        n_sLg = n_sLg.at[0].add(nc0_sg * jnp.sqrt(4 * jnp.pi))
        Y_nL_local = Y_nL[:, :Lmax_]  # Only use L up to Lmax
        # vectorized version
        n = jnp.dot(Y_nL_local, n_sLg)
        # TODO: here we encounter negative density, we use a quick fix, should reconsider
        n = jnp.where(n > 0, n, 0)
        # e_g = -3/4 * (3 / np.pi)**(1/3) * n**(4/3)
        def _exc_density(n_sg):
          if n_sg.ndim == 1:
            n_sg = n_sg[None, :]
          return xc.xc_density(n_sg, g_vec, xc_type=config.xc)
        exc_density = jax.vmap(_exc_density)(n)
        n_total = n if n.ndim == 2 else jnp.sum(n, axis=1)
        # E_xc_ = einsum(weight_n, e_g, dr_g[atom] * r_g[atom]**2, "i, ij, j") * 4 * jnp.pi
        E_xc_ = jnp.einsum(
          "i, ij, j",
          weight_n,
          n_total * exc_density,
          paw.dr_g[atom] * paw.r_g[atom]**2
        ) * 4 * jnp.pi
        return E_xc_

      n_qg_ = paw.n_qg[atom]
      nt_qg_ = paw.nt_qg[atom]
      nc_g_ = paw.nc_g[atom]
      nct_g_ = paw.nct_g[atom]
      T_Lqp_ = paw.T_Lqp[atom]
      e_xc0_ = paw.e_xc0[atom]
      Lmax_ = (2 * paw.lmax[atom] + 1)**2
      D_sLq = jnp.inner(D_p_packed, T_Lqp_)
      e_ae = _calculate_xc_energy(D_sLq, n_qg_, nc_g_)
      e_ps = _calculate_xc_energy(D_sLq, nt_qg_, nct_g_)
      return e_ae - e_ps - e_xc0_

  atoms_list = paw.atoms_list
  delta_pL_list = [paw.Delta_pL[atom] for atom in atoms_list]
  delta0_list = [paw.Delta0[atom] for atom in atoms_list]
  phase_list = [phase_G[atom] for atom in atoms_list]
  ghat_list = [ghat_LG[atom] for atom in atoms_list]
  k_p_list = [paw.K_p[atom] for atom in atoms_list]
  k_c_list = [paw.K_c[atom] for atom in atoms_list]
  mb_p_list = [paw.MB_p[atom] for atom in atoms_list]
  mb_list = [paw.MB[atom] for atom in atoms_list]
  m_list = [paw.M[atom] for atom in atoms_list]
  m_p_list = [paw.M_p[atom] for atom in atoms_list]
  m_pp_list = [paw.M_pp[atom] for atom in atoms_list]

  def total_energy(params_pw, params_occ, g_vec, pseudopot=pseudopot, return_components: bool = False):
    coeff = pw.coeff(params_pw, freq_mask, sharding=sharding)
    coeff = get_ultrasoft_coeff(coeff)
    occ = get_occupation(params_occ)
    if gpaw_coeff_data is not None:
      coeff = coeff_cmp
      occ = occ_cmp
    kinetic = energy.kinetic(g_vec, k_vec, coeff, occ)
    kinetic_pseudo = kinetic

    density = pw.density_grid(coeff, crystal.vol, occ)
    density = density.at[0].add(nct_g_)
    density = density.at[0].set(jnp.where(density[0] > 0, density[0], 0))
    exc = energy.xc_energy(
      density, g_vec, crystal.vol, config.xc, kohn_sham=False
    )
    exc_pseudo = exc
    d_p_list = calc_atomic_density_matrix(coeff, occ)

    @map_over_atoms
    def _rho_comp_term(D_p_atom, Delta_pL, Delta0, phase, ghat):
      D_p_packed = pack(D_p_atom)
      Q_L = jnp.dot(D_p_packed, Delta_pL)
      Q_L = Q_L.at[0].add(Delta0)
      return phase * jnp.tensordot(Q_L, ghat, axes=[0, 0])

    rho_terms = _rho_comp_term(
      d_p_list, delta_pL_list, delta0_list, phase_list, ghat_list
    )
    rho_comp_G = sum(rho_terms) if rho_terms else 0.0
    density_reciprocal = pw.density_grid_reciprocal(coeff, crystal.vol, occ)
    e_zero = normcons.energy_local(density_reciprocal, vbar_G, crystal.vol) + e_zero0
    e_zero_pseudo = e_zero
    density_reciprocal = density_reciprocal.at[0].add(rho_comp_G + nct_G)
    hartree = energy.hartree(density_reciprocal, g_vec, crystal.vol)
    hartree_pseudo = hartree

    @map_over_atoms
    def _atomic_terms(
      atom, D_p_atom, K_p, K_c, MB_p, MB, M, M_p, M_pp
    ):
      D_p_packed = pack(D_p_atom)
      kin_add = jnp.sum(K_p * D_p_atom[0, 0]).real + K_c
      # nct contribution to e_zero is canceled out with MB
      e_zero_add = jnp.sum(MB_p * D_p_packed) + MB
      hartree_add = M + jnp.dot(
        D_p_packed, (M_p + jnp.dot(M_pp, D_p_packed))
      )
      exc_add = calc_paw_xc_correction(atom, D_p_packed)
      return kin_add, e_zero_add, hartree_add, exc_add

    atom_terms = _atomic_terms(
      atoms_list, d_p_list, k_p_list, k_c_list, mb_p_list, mb_list,
      m_list, m_p_list, m_pp_list
    )
    if atom_terms:
      kin_terms, e_zero_terms, hartree_terms, exc_terms = zip(*atom_terms)
      kinetic += sum(kin_terms)
      e_zero += sum(e_zero_terms)
      hartree += sum(hartree_terms)
      exc += sum(exc_terms)

    total = kinetic + hartree + e_zero + exc
    if return_components:
      comps = {
        "kinetic_pseudo": kinetic_pseudo,
        "kinetic_atomic": kinetic - kinetic_pseudo,
        "coulomb_pseudo": hartree_pseudo,
        "coulomb_atomic": hartree - hartree_pseudo,
        "zero_pseudo": e_zero_pseudo,
        "zero_atomic": e_zero - e_zero_pseudo,
        "xc_pseudo": exc_pseudo,
        "xc_atomic": exc - exc_pseudo,
      }
      return total, kinetic, hartree, exc, e_zero, comps
    return total, kinetic, hartree, exc

  def get_entropy(params_occ):
    occ = get_occupation(params_occ)
    return entropy.fermi_dirac(occ, eps=EPS)

  def free_energy(
    params_pw, params_occ, temp, g_vec
  ):
    total, kinetic, hartree, exc = total_energy(
      params_pw, params_occ, g_vec
    )
    etro = get_entropy(params_occ)
    free = total - temp * etro
    return free, (total, etro, kinetic, hartree, exc)

  # Initialize parameters and optimizer.
  optimizer = create_optimizer(config)
  params_pw = pw.param_init(
    key,
    num_bands,
    num_kpts,
    freq_mask,
    spin_restricted=config.spin_restricted,
    sharding=sharding
  )
  check_uspp_overlap(
    params_pw,
    freq_mask=freq_mask,
    sharding=sharding,
    get_ultrasoft_coeff=get_ultrasoft_coeff,
    proj_pw_overlap=proj_pw_overlap,
    crystal_vol=crystal.vol,
    pseudopot=pseudopot
  )
  params_occ = occupation.param_init(
    key, num_bands, paw.valence_charges, num_kpts, crystal.spin, config.occupation
  )
  params_occ = jax.device_put(params_occ, sharding)
  params = {"pw": params_pw, "occ": params_occ}
  opt_state = optimizer.init(params)

  if gpaw_coeff_data is not None:
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
    print(f"GPAW coeff keys: {gpaw_coeff_data.files}")
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
    total, kinetic, hartree, exc, e_zero, comps = total_energy(
      params_pw, params_occ, g_vec, return_components=True
    )
    total = jax.block_until_ready(total)
    if "gpaw_e_total_free" in gpaw_coeff_data.files:
      if "gpaw_e_kinetic_pseudo" in gpaw_coeff_data.files:
        gpaw_k_p = float(gpaw_coeff_data["gpaw_e_kinetic_pseudo"])
        gpaw_k_a = float(gpaw_coeff_data["gpaw_e_kinetic_atomic"])
        gpaw_c_p = float(gpaw_coeff_data["gpaw_e_coulomb_pseudo"])
        gpaw_c_a = float(gpaw_coeff_data["gpaw_e_coulomb_atomic"])
        gpaw_z_p = float(gpaw_coeff_data["gpaw_e_zero_pseudo"])
        gpaw_z_a = float(gpaw_coeff_data["gpaw_e_zero_atomic"])
        gpaw_x_p = float(gpaw_coeff_data["gpaw_e_xc_pseudo"])
        gpaw_x_a = float(gpaw_coeff_data["gpaw_e_xc_atomic"])
        logging.info("GPAW vs Jrystal split (pseudo / atomic) (Ha):")
        logging.info(f"  Kinetic: {comps['kinetic_pseudo']:.6f}/{comps['kinetic_atomic']:.6f} vs {gpaw_k_p:.6f}/{gpaw_k_a:.6f}")
        logging.info(f"  Coulomb: {comps['coulomb_pseudo']:.6f}/{comps['coulomb_atomic']:.6f} vs {gpaw_c_p:.6f}/{gpaw_c_a:.6f}")
        logging.info(f"  E_zero: {comps['zero_pseudo']:.6f}/{comps['zero_atomic']:.6f} vs {gpaw_z_p:.6f}/{gpaw_z_a:.6f}")
        logging.info(f"  XC: {comps['xc_pseudo']:.6f}/{comps['xc_atomic']:.6f} vs {gpaw_x_p:.6f}/{gpaw_x_a:.6f}")
        logging.info("GPAW vs Jrystal split deltas (Ha):")
        logging.info(f"  ΔKinetic (pseudo/atomic): {(comps['kinetic_pseudo']-gpaw_k_p):.6e} / {(comps['kinetic_atomic']-gpaw_k_a):.6e}")
        logging.info(f"  ΔCoulomb (pseudo/atomic): {(comps['coulomb_pseudo']-gpaw_c_p):.6e} / {(comps['coulomb_atomic']-gpaw_c_a):.6e}")
        logging.info(f"  ΔE_zero (pseudo/atomic): {(comps['zero_pseudo']-gpaw_z_p):.6e} / {(comps['zero_atomic']-gpaw_z_a):.6e}")
        logging.info(f"  ΔXC (pseudo/atomic): {(comps['xc_pseudo']-gpaw_x_p):.6e} / {(comps['xc_atomic']-gpaw_x_a):.6e}")
      breakpoint()
    return

  # Define update function.
  with mesh:

    @jax.jit
    def update(params, opt_state, temp, g_vec):
      loss = lambda x: free_energy(
        x["pw"], x["occ"], temp, g_vec
      )
      (loss_val, es), grad = jax.value_and_grad(loss, has_aux=True)(params)
      updates, opt_state = optimizer.update(grad, opt_state)
      params = optax.apply_updates(params, updates)
      return params, opt_state, loss_val, es

    # Define scheduler for temperature annealing.
    if config.smearing > 0.:
      temperature_scheduler = optax.exponential_decay(
        init_value=100.,
        transition_steps=config.epoch // 2,
        decay_rate=config.smearing / 100,
        end_value=config.smearing
      )
    else:

      def temperature_scheduler(i):
        return 0.

    logging.info(f"smearing: {config.smearing}")

    # The main loop for optimization.
    if config.verbose:
      iters = tqdm(range(config.epoch))
    else:
      iters = tqdm(range(config.epoch), disable=True)

    train_time = 0
    for i in iters:
      temp = temperature_scheduler(i)
      start = time.time()
      params, opt_state, loss_val, es = update(
        params, opt_state, temp, g_vec
      )
      etot, entro, kinetic, hartree, exc = es
      etot = jax.block_until_ready(etot)
      train_time += time.time() - start
      converged = convergence_checker.check(etot)
      if converged:
        logging.info("Converged.")
        break

      iters.set_description(
        f"Loss: {loss_val:.4f}|Energy: {etot:.4f}|"
        f"Kinetic: {kinetic:.4f}|Hartree: {hartree:.4f}|XC: {exc:.4f}|E_zero: {etot - kinetic - hartree - exc:.4f}|"
      )

  if not converged:
    logging.warning("Did not converge.")

  #####################################
  #        END OF OPTIMIZATION        #
  #####################################
  coeff = pw.coeff(params["pw"], freq_mask)
  coeff = get_ultrasoft_coeff(coeff)
  occ = get_occupation(params["occ"])
  density = pw.density_grid(coeff, crystal.vol, occ)
  density_reciprocal = pw.density_grid_reciprocal(coeff, crystal.vol, occ)
  # kinetic = energy.kinetic(g_vec, k_vec, coeff, occ)
  # hartree = energy.hartree(density_reciprocal, g_vec, crystal.vol)
  e_zero = normcons.energy_local(density_reciprocal, vbar_G, crystal.vol)
  d_p_list = calc_atomic_density_matrix(coeff, occ)
  atoms_list = paw.atoms_list
  mb_list = [paw.MB[atom] for atom in atoms_list]
  mb_p_list = [paw.MB_p[atom] for atom in atoms_list]

  @map_over_atoms
  def _e_zero_term(D_p_atom, MB, MB_p):
    return MB + jnp.sum(MB_p * pack(D_p_atom))

  e_zero_terms = _e_zero_term(d_p_list, mb_list, mb_p_list)
  if e_zero_terms:
    e_zero += sum(e_zero_terms)

  exc = energy.xc_energy(density, g_vec, crystal.vol, config.xc, kohn_sham=False)

  # logging.info(f"Hartree Energy: {hartree:.4f} Ha")
  # # logging.info(f"External (local) Energy: {external_local:.4f} Ha")
  # # logging.info(f"External (nonlocal) Energy: {external_nonlocal:.4f} Ha")
  # logging.info(f"XC Energy: {exc:.4f} Ha")
  # logging.info(f"Kinetic Energy: {kinetic:.4f} Ha")
  # logging.info(f"Total Energy: {etot:.4f} Ha")

  return density
