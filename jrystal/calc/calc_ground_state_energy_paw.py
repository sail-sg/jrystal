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

from .._src import energy, entropy, occupation, pw
from .._src.crystal import Crystal
from .._src.grid import proper_grid_size
from ..config import JrystalConfigDict
from ..pseudopotential import normcons
from .convergence import create_convergence_checker
from .calc_paw import setup_gpaw, calc_paw, compute_proj_pw_overlap
from .opt_utils import (
  create_crystal,
  create_freq_mask,
  create_grids,
  create_optimizer,
  create_pseudopotential,
  get_ewald_coulomb_repulsion,
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

  crystal = create_crystal(config)
  pseudopot = create_pseudopotential(config)
  valence_charges = np.sum(pseudopot.valence_charges)
  logging.info(f"Crystal: {crystal.symbols}")
  EPS = config.eps

  def expand(data: list, n_proj, l_j) -> jnp.ndarray:
    """Expand the matrix from the size of radial component to the size of the projectors."""
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

  # preprocessing for PAW pp file
  atoms_list = ["C1", "C2"]
  K_p = {}
  K_c = {}
  M = {}
  M_p = {}
  M_pp = {}
  Kc = {}
  MB = {}
  MB_p = {}
  n_qg = {}
  nt_qg = {}
  nc_g = {}
  nct_g = {}
  lmax = {}
  e_xc0 = {}
  r_g = {}
  dr_g = {}
  T_Lqp = {}
  pseudopot.r_grid = []
  pseudopot.nonlocal_beta_grid = []
  pseudopot.nonlocal_angular_momentum = []
  pseudopot.nonlocal_d_matrix = []
  # index to convert packed 2D indices to 1D index 
  index1 = jnp.array([
    [0, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4],
    [2, 1, 2, 3, 2, 1, 2, 3, 0, 1, 2, 3, 4]
  ])
  index2 = jnp.array([
    [5, 6, 6, 6, 7, 8, 8, 8, 9, 9, 9, 9, 9],
    [2, 1, 2, 3, 2, 1, 2, 3, 0, 1, 2, 3, 4]
  ])

  for a in atoms_list:
    # NOTE: current only support the element with one character, e.g. C is okay while He is not
    # TODO: modify the pseudopot object to dict structure
    setup_data = setup_gpaw(a[0])
    pseudopot.r_grid.append(setup_data['r_g'])
    pseudopot.nonlocal_beta_grid.append(setup_data['pt_jg'])
    pseudopot.nonlocal_angular_momentum.append(setup_data['l_j'])
    results = calc_paw(setup_data)
    # pseudopot.nonlocal_d_matrix.append(np.eye(setup_data['pt_jg'].shape[0]))
    n_proj = setup_data['pt_jg'].shape[0]
    tmp_mat = np.zeros((n_proj, n_proj))
    tmp_mat[np.triu_indices(n_proj)] = results['Delta_lq'][0]
    tmp_mat = tmp_mat + tmp_mat.T - np.diag(np.diag(tmp_mat))
    # NOTE: the jnp.sqrt(4 * jnp.pi) factor here is questionable
    pseudopot.nonlocal_d_matrix.append(tmp_mat / jnp.sqrt(4 * jnp.pi))
    K_p[a] = expand(setup_data['K_p'], 13, setup_data['l_j'])
    K_c[a] = setup_data['K_c']
    M[a] = results["M"]
    M_p[a] = results["M_p"]
    M_pp[a] = results["M_pp"]
    MB[a] = results["MB"]
    MB_p[a] = results["MB_p"]
    n_qg[a] = results["n_qg"]
    nt_qg[a] = results["nt_qg"]
    T_Lqp[a] = results["T_Lqp"]

    nc_g[a] = setup_data["nc_g"]
    nct_g[a] = setup_data["nct_g"]
    lmax[a] = int(setup_data["lmax"])
    e_xc0[a] = setup_data["e_xc"]
    gcut = 258
    a_ = 0.400000
    n = 300
    i = np.arange(gcut)
    r_g[a] = a_ * i / (n - i)
    dr_g[a] = a_ * n / (n - i)**2

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
  num_kpts = k_vec.shape[0]
  logging.info(f"Number of G-vectors: {proper_grid_size(config.grid_sizes)}")
  logging.info(f"Number of k-vectors: {proper_grid_size(config.k_grid_sizes)}")
  num_bands = ceil(valence_charges / 2) + config.empty_bands
  logging.info(f"num_bands: {num_bands}")
  logging.info(f"XC functional: {config.xc}")
  freq_mask = create_freq_mask(config)
  ew = get_ewald_coulomb_repulsion(config)
  valence_charges = np.sum(pseudopot.valence_charges)

  convergence_checker = create_convergence_checker(config)
  converged = False
  # initialize pseudopotential
  logging.info("Initializing pseudopotential (local)...")
  start = time.time()
  # potential_loc = normcons.potential_local_reciprocal(
  #   crystal.positions,
  #   g_vec,
  #   pseudopot.r_grid,
  #   pseudopot.local_potential_grid,
  #   pseudopot.local_potential_charge,
  #   crystal.vol
  # )

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
  # tmp = beta_gk[0].reshape(5, 64).T
  # overlap2 = compute_proj_pw_overlap(g_vec.reshape(-1, 3), crystal.positions[0])
  # overlap1 = proj_pw_overlap[0, index1[0], index1[1], ...].reshape(13, config.grid_sizes**3).T
  # print(jnp.abs(overlap1 - overlap2).max())

  # NOTE: check the othorgonality of the wavefunctions here
  # def expand(data: list, n_proj, l_j) -> jnp.ndarray:
  #   """Expand the matrix from the size of radial component to the size of the projectors."""
  #   data = jnp.array(data).reshape((len(l_j), len(l_j)))
  #   expanded_data = jnp.zeros((n_proj, n_proj))
  #   i1 = 0
  #   for j1, l1 in enumerate(l_j):
  #     for m1 in range(2 * l1 + 1):
  #       i2 = 0
  #       for j2, l2 in enumerate(l_j):
  #         for m2 in range(2 * l2 + 1):
  #           if l1 == l2 and m1 == m2:
  #             expanded_data = expanded_data.at[i1, i2].set(data[j1, j2])
  #           i2 += 1
  #       i1 += 1
  #   return expanded_data
  # dO = expand(pseudopot.nonlocal_d_matrix[0], 13, setup_data['l_j'])
  # overlap1 = einsum(
  #   proj_pw_overlap[0, index1[0], index1[1], ...].reshape(13, config.grid_sizes**3),
  #   proj_pw_overlap[0, index1[0], index1[1], ...].conj().reshape(13, config.grid_sizes**3),
  #   dO, "proj1 g1, proj2 g2, proj1 proj2 -> g1 g2"
  # )
  # dO = expand(pseudopot.nonlocal_d_matrix[1], 13, setup_data['l_j'])
  # overlap2 = einsum(
  #   proj_pw_overlap[0, index2[0], index2[1], ...].reshape(13, config.grid_sizes**3),
  #   proj_pw_overlap[0, index2[0], index2[1], ...].conj().reshape(13, config.grid_sizes**3),
  #   dO, "proj1 g1, proj2 g2, proj1 proj2 -> g1 g2"
  # )
  # S = jnp.eye(config.grid_sizes**3) + overlap1 + overlap2
  # print(jnp.abs(S - S.T.conj()).max())
  # print(jnp.linalg.eigvals(S).min())

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

  from ..pseudopotential.ulatrsoft import get_ultrasoft_coeff_fun
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
  
  def total_energy(params_pw, params_occ, g_vec, pseudopot=pseudopot):
    coeff = pw.coeff(params_pw, freq_mask, sharding=sharding)
    # this is the original overlap without PAW correction
    # overlap1 = einsum(coeff, coeff.conj(), "s k band1 x y z, s k band2 x y z -> s k band1 band2")
    coeff = get_ultrasoft_coeff(coeff)
    occ = get_occupation(params_occ)
    density = pw.density_grid(coeff, crystal.vol, occ)
    density_reciprocal = pw.density_grid_reciprocal(coeff, crystal.vol, occ)
    kinetic = energy.kinetic(g_vec, k_vec, coeff, occ)
    hartree = energy.hartree(density_reciprocal, g_vec, crystal.vol)
    # external_local = normcons.energy_local(
    #   density_reciprocal, potential_loc, vol=crystal.vol
    # )
    # external_nonlocal = normcons.energy_nonlocal(
    #   coeff, potential_nl, vol=crystal.vol, occupation=occ
    # )

    xc = energy.xc_energy(
      density, g_vec, crystal.vol, config.xc, kohn_sham=False
    )

    _f_matrix = einsum(
      coeff,
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

    # NOTE: check the othorgonality of the wavefunctions here
    # def expand(data: list, n_proj, l_j) -> jnp.ndarray:
    #   """Expand the matrix from the size of radial component to the size of the projectors."""
    #   data = jnp.array(data).reshape((len(l_j), len(l_j)))
    #   expanded_data = jnp.zeros((n_proj, n_proj))
    #   i1 = 0
    #   for j1, l1 in enumerate(l_j):
    #     for m1 in range(2 * l1 + 1):
    #       i2 = 0
    #       for j2, l2 in enumerate(l_j):
    #         for m2 in range(2 * l2 + 1):
    #           if l1 == l2 and m1 == m2:
    #             expanded_data = expanded_data.at[i1, i2].set(data[j1, j2])
    #           i2 += 1
    #       i1 += 1
    #   return expanded_data
    # dO = expand(pseudopot.nonlocal_d_matrix[0], 13, setup_data['l_j'])
    # overlap1 = einsum(coeff, coeff.conj(), "s k band1 x y z, s k band2 x y z -> s k band1 band2")
    # # correction1 = einsum(_f_matrix[..., index1[0], index1[1]], dO, _f_matrix[..., index1[0], index1[1]], "s k band1 proj1, proj1 proj2, s k band2 proj2 -> s k band1 band2")
    # correction1 = einsum(_f_matrix[..., index1[0], index1[1]], _f_matrix[..., index1[0], index1[1]], "s k band1 proj1, s k band2 proj1 -> s k band1 band2") 
    # dO = expand(pseudopot.nonlocal_d_matrix[1], 13, setup_data['l_j'])
    # # correction2 = einsum(_f_matrix[..., index2[0], index2[1]], dO, _f_matrix[..., index2[0], index2[1]], "s k band1 proj1, proj1 proj2, s k band2 proj2 -> s k band1 band2")
    # correction2 = einsum(_f_matrix[..., index2[0], index2[1]], _f_matrix[..., index2[0], index2[1]], "s k band1 proj1, s k band2 proj1 -> s k band1 band2")
    # overlap = overlap1 + (correction1 + correction2)
    # overlap = overlap1 + (correction1 + correction2) / crystal.vol

    # for a, D_sii in wfs.D_asii.items():
    #   Q_L = jnp.einsum('ij, ijL -> L',
    #     D_sii[:wfs.ndensities].real, wfs.delta_aiiL[a])
    #   Q_L[0] += wfs.delta0_a[a]
    D_p = {}
    D_p["C1"] = einsum(
      _f_matrix[..., index1[0], index1[1]].conj(),
      occ,
      _f_matrix[..., index1[0], index1[1]],
      "s k band proj1, s k band, s k band proj2 -> s k proj1 proj2"
    ).real
    D_p["C2"] = einsum(
      _f_matrix[..., index2[0], index2[1]].conj(),
      occ,
      _f_matrix[..., index2[0], index2[1]],
      "s k band proj1, s k band, s k band proj2 -> s k proj1 proj2"
    ).real

    def pack(D_p: jnp.ndarray) -> jnp.ndarray:
      """Pack a Hermitian matrix for better efficiency

      The diagonal elements are halfed to calculate the inner product
      """

      n = D_p.shape[-1]
      D_p = D_p.at[..., jnp.diag_indices(n)].set(D_p[..., jnp.diag_indices(n)] / 2)
      return D_p[0,0][jnp.triu_indices(n)].real * 2
    
    def unpack(D_p: jnp.ndarray, n_proj: int) -> jnp.ndarray:
      tmp_mat = np.zeros((n_proj, n_proj))
      tmp_mat[np.triu_indices(n_proj)] = D_p
      tmp_mat = tmp_mat + tmp_mat.T - np.diag(np.diag(tmp_mat))
      return tmp_mat

    def calc_paw_xc_correction(atom: str):
      def _calculate_xc_energy(D_sLq, n_qg, nc0_sg):

        n_sLg = jnp.dot(D_sLq, n_qg)  # shape: [n_spin, Lmax, n_g]
        n_sLg = n_sLg.at[0].add(nc0_sg)
        Y_nL_local = Y_nL[:, :Lmax_]  # Only use L up to Lmax
        # for n in range(50):  # 50 Lebedev points
        #   w = weight_n[n]
        #   Y_L = Y_nL_local[n]  # shape: [Lmax]
        #   n_sg = jnp.dot(Y_L, n_sLg)  # shape: [n_spin, n_g]
        #   n_sg = jnp.where(n_sg > 0, n_sg, 0)
        #   e_g = -3/4 * (3 / np.pi)**(1/3) * n_sg**(4/3)
        #   E_xc += w * jnp.sum(e_g * dr_g[atom] * r_g[atom]**2, axis=-1) * 4 * jnp.pi

        # vectorized version
        n = jnp.dot(Y_nL_local, n_sLg)
        # TODO: here we encounter negative density, we use a quick fix, should reconsider
        n = jnp.where(n > 0, n, 0)
        e_g = -3/4 * (3 / np.pi)**(1/3) * n**(4/3)
        # E_xc_ = einsum(weight_n, e_g, dr_g[atom] * r_g[atom]**2, "i, ij, j") * 4 * jnp.pi
        E_xc_ = jnp.einsum("i, ij, j", weight_n, e_g, dr_g[atom] * r_g[atom]**2) * 4 * jnp.pi
        return E_xc_

      n_qg_ = n_qg[atom]
      nt_qg_ = nt_qg[atom]
      nc_g_ = nc_g[atom]
      nct_g_ = nct_g[atom]
      T_Lqp_ = T_Lqp[atom]
      e_xc0_ = e_xc0[atom]
      Lmax_ = (2 * lmax[atom] + 1)**2
      D_sLq = jnp.inner(D_p_packed, T_Lqp_)
      e_ae = _calculate_xc_energy(D_sLq, n_qg_, nc_g_)
      e_ps = _calculate_xc_energy(D_sLq, nt_qg_, nct_g_)
      return e_ae - e_ps - e_xc0_

    e_zero = 0
    for atom in atoms_list:
      D_p_packed = pack(D_p[atom])
      kinetic += jnp.sum(K_p[atom] * D_p[atom][0,0]).real + K_c[atom]
      e_zero += MB[atom] + jnp.sum(MB_p[atom] * D_p_packed)
      hartree += M[atom] + jnp.dot(
        D_p_packed, (M_p[atom] + jnp.dot(M_pp[atom], D_p_packed))
      )
      xc += calc_paw_xc_correction(atom)

    return kinetic + hartree + e_zero + xc, kinetic, hartree, xc

  def get_entropy(params_occ):
    occ = get_occupation(params_occ)
    return entropy.fermi_dirac(occ, eps=EPS)

  def free_energy(
    params_pw, params_occ, temp, g_vec
  ):
    total, kinetic, hartree, xc = total_energy(
      params_pw, params_occ, g_vec
    )
    etro = get_entropy(params_occ)
    free = total - temp * etro
    return free, (total, etro, kinetic, hartree, xc)

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
  params_occ = occupation.param_init(
    key, num_bands, valence_charges, num_kpts, crystal.spin, config.occupation
  )
  params_occ = jax.device_put(params_occ, sharding)
  params = {"pw": params_pw, "occ": params_occ}
  opt_state = optimizer.init(params)

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
      etot, entro, kinetic, hartree, xc = es
      etot = jax.block_until_ready(etot)
      train_time += time.time() - start
      converged = convergence_checker.check(etot)
      if converged:
        logging.info("Converged.")
        break

      # iters.set_description(
      #   f"Loss: {loss_val:.4f}|Energy: {etot:.4f}|"
      #   f"Entropy: {entro:.4f}|T: {temp:.2E}"
      # )
      iters.set_description(
        f"Loss: {loss_val:.4f}|Energy: {etot:.4f}|"
        f"Kinetic: {kinetic:.4f}|Hartree: {hartree:.4f}|XC: {xc:.4f}"
      )

  if not converged:
    logging.warning("Did not converge.")

  #####################################
  #        END OF OPTIMIZATION        #
  #####################################
  coeff = pw.coeff(params["pw"], freq_mask)
  occ = get_occupation(params["occ"])
  density = pw.density_grid(coeff, crystal.vol, occ)
  density_reciprocal = pw.density_grid_reciprocal(coeff, crystal.vol, occ)
  kinetic = energy.kinetic(g_vec, k_vec, coeff, occ)
  hartree = energy.hartree(density_reciprocal, g_vec, crystal.vol)
  # external_local = normcons.energy_local(
  #   density_reciprocal, potential_loc, vol=crystal.vol
  # )
  # external_nonlocal = normcons.energy_nonlocal(
  #   coeff, potential_nl, vol=crystal.vol, occupation=occ
  # )

  xc = energy.xc_energy(density, g_vec, crystal.vol, config.xc, kohn_sham=False)

  logging.info(f"Hartree Energy: {hartree:.4f} Ha")
  logging.info(f"External (local) Energy: {external_local:.4f} Ha")
  logging.info(f"External (nonlocal) Energy: {external_nonlocal:.4f} Ha")
  logging.info(f"XC Energy: {xc:.4f} Ha")
  logging.info(f"Kinetic Energy: {kinetic:.4f} Ha")
  logging.info(f"Nuclear repulsion Energy: {ew:.4f} Ha")
  logging.info(f"Total Energy: {etot+ew:.4f} Ha")

  return density
