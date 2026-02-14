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
from functools import partial
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

from .._src import energy, entropy, occupation, pw
from .._src.crystal import Crystal
from .._src.grid import proper_grid_size
from ..config import JrystalConfigDict
from ..pseudopotential import normcons
from ..pseudopotential.utils import map_over_atoms, pack
from .convergence import create_convergence_checker
from ..pseudopotential.paw_setup import (
  build_paw_precompute,
  build_paw_setup,
  compare_gpaw_coefficients,
  load_gpaw_coeff_metadata,
)
from ..pseudopotential.paw_calc import (
  build_paw_xc_correction,
  compute_proj_pw_overlap,
)
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


def calc(config: JrystalConfigDict) -> None:
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
  paw_debug = config.paw_debug

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
  if paw_debug:
    gpaw_coeff_data, gpaw_kpts_frac = load_gpaw_coeff_metadata(config)
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
  # NOTE: we have checked that the overlap matrix we obtained is correct
  # tmp = beta_gk[0].reshape(5, -1).T
  # overlap2 = compute_proj_pw_overlap(g_vec.reshape(-1, 3), crystal.positions[0])
  # idx = index_map[atoms_list[0]]
  # overlap1 = proj_pw_overlap[0, idx[0], idx[1], ...].reshape(13, config.grid_sizes**3).T
  # print(jnp.abs(overlap1 - overlap2).max())
  # but the alignment with gpaw is still of numerical accuracy, causing remaining misalignment
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
      # NOTE: may have problem if XC functional distinguishes spin
      return einsum(
        _f_matrix[..., l_idx, m_idx].conj(),
        occ,
        _f_matrix[..., l_idx, m_idx],
        "s k band proj1, s k band, s k band proj2 -> proj1 proj2"
      )

    D_p_list = _calc_d_p(idx_list)

    if return_f_matrix:
      return D_p_list, _f_matrix
    return D_p_list
  
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
  calc_paw_xc_correction = build_paw_xc_correction(
    paw, g_vec, config.xc
  )

  def total_energy(
    params_pw,
    params_occ,
    g_vec,
    pseudopot=pseudopot,
    coeff_occ_override=None,
  ):
    coeff = pw.coeff(params_pw, freq_mask, sharding=sharding)
    coeff = get_ultrasoft_coeff(coeff)
    occ = get_occupation(params_occ)
    if coeff_occ_override is not None:
      coeff, occ = coeff_occ_override

    kinetic = energy.kinetic(g_vec, k_vec, coeff, occ)
    density = pw.density_grid(coeff, crystal.vol, occ)
    density = density.at[0].add(nct_g_)
    exc = energy.xc_energy(
      density, g_vec, crystal.vol, config.xc, kohn_sham=False
    )
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
    density_reciprocal = density_reciprocal.at[0].add(rho_comp_G + nct_G)
    hartree = energy.hartree(density_reciprocal, g_vec, crystal.vol)

    @map_over_atoms
    def _atomic_terms(
      atom, D_p_atom, K_p, K_c, MB_p, MB, M, M_p, M_pp
    ):
      D_p_packed = pack(D_p_atom)
      kin_add = jnp.sum(K_p * D_p_atom).real + K_c
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

  if paw_debug:
    compare_gpaw_coefficients(
      gpaw_coeff_data=gpaw_coeff_data,
      crystal=crystal,
      paw=paw,
      params_pw=params_pw,
      params_occ=params_occ,
      g_vec=g_vec,
      calc_atomic_density_matrix=calc_atomic_density_matrix,
      total_energy=total_energy,
    )
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
  total, kinetic, hartree, exc = total_energy(
    params["pw"],
    params["occ"],
    g_vec,
    pseudopot=pseudopot,
  )

  logging.info(f"Hartree Energy: {hartree:.4f} Ha")
  logging.info(f"XC Energy: {exc:.4f} Ha")
  logging.info(f"Kinetic Energy: {kinetic:.4f} Ha")
  logging.info(f"Total Energy: {total:.4f} Ha")

  return
