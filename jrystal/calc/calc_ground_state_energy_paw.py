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
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

from .._src import energy, entropy, occupation, pw
from .._src.crystal import Crystal
from .._src.grid import proper_grid_size
from ..config import JrystalConfigDict
from ..pseudopotential import normcons
from .convergence import create_convergence_checker
import calc_paw
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
  potential_loc = normcons.potential_local_reciprocal(
    crystal.positions,
    g_vec,
    pseudopot.r_grid,
    pseudopot.local_potential_grid,
    pseudopot.local_potential_charge,
    crystal.vol
  )

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
  potential_nl = normcons.potential_nonlocal_psi_reciprocal(
    crystal.positions,
    g_vec,
    k_vec,
    pseudopot.r_grid,
    pseudopot.nonlocal_beta_grid,
    pseudopot.nonlocal_angular_momentum,
    pseudopot.nonlocal_d_matrix,
    beta_gk
  )  # shape "kpt beta phi x y z"
  del beta_gk
  end = time.time()
  logging.info(f"Nonlocal potential done. Times: {end - start:.2f} seconds")
  logging.info("Deploying pseudopotential (nonlocal)...")
  start = time.time()
  potential_nl = jax.device_put(potential_nl, NamedSharding(mesh, P('k')))
  end = time.time()
  logging.info(
    f"Deploying pseudopotential (nonlocal) done. "
    f"Times: {end - start:.2f} seconds"
  )

  # preprocessing for 
  atoms_list = ["C", "C"]
  K_p = {}
  M = {}
  M_p = {}
  M_pp = {}
  Kc = {}
  MB = {}
  MB_p = {}
  for a in atoms_list:
    setup_data = calc_paw.setup_gpaw(a)
    results = calc_paw.calc(*setup_data)
    K_p[a] = results["K_p"]
    M[a] = results["M"]
    M_p[a] = results["M_p"]
    M_pp[a] = results["M_pp"]
    Kc[a] = results["Kc"]
    MB[a] = results["MB"]
    MB_p[a] = results["MB_p"]

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
  
  import jax.numpy as jnp
  def total_energy(params_pw, params_occ, g_vec):
    coeff = pw.coeff(params_pw, freq_mask, sharding=sharding)
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

    proj_pw_overlap = pseudopotential.nloc.potential_nonlocal_psi_reciprocal(
    potential_nl = normcons.potential_nonlocal_psi_reciprocal(
    crystal.positions,
    g_vec,
    k_vec,
    pseudopot.r_grid,
    pseudopot.nonlocal_beta_grid,
    pseudopot.nonlocal_angular_momentum,
    pseudopot.nonlocal_d_matrix,
    beta_gk
  )
    _f_matrix = einsum(
      coeff,
      proj_pw_overlap,
      "s k band x y z, k beta phi x y z -> s k band beta phi"
    )

    for D_sii, P_ni in zip(D_asii.values(), P_ani.values()):
      D_sii[self.spin] += jnp.einsum('ni, n, nj -> ij',
        P_ni.conj(), occ_n, P_ni).real
      
    for a, D_sii in wfs.D_asii.items():
      Q_L = jnp.einsum('ij, ijL -> L',
        D_sii[:wfs.ndensities].real, wfs.delta_aiiL[a])
      Q_L[0] += wfs.delta0_a[a]

    for atom in atoms_list:
      kinetic += jnp.dot(K_p[atom], D_p[atom]) + Kc[atom]
      e_zero += MB[atom] + jnp.dot(MB_p[atom], D_p[atom])
      e_coulomb += M[atom] + jnp.dot(
        D_p[atom], (M_p[atom] + jnp.dot(M_pp[atom], D_p[atom]))
      )
      # TODO: PAW xc energy
      # e_xc += calculate_paw_correction(self.setups[a], D_sp,
      #                                                   dH_asp[a], a=a)

    return kinetic + hartree + external_local + external_nonlocal + xc

  def get_entropy(params_occ):
    occ = get_occupation(params_occ)
    return entropy.fermi_dirac(occ, eps=EPS)

  def free_energy(
    params_pw, params_occ, temp, g_vec, potential_loc, potential_nl
  ):
    total = total_energy(
      params_pw, params_occ, g_vec, potential_loc, potential_nl
    )
    etro = get_entropy(params_occ)
    free = total - temp * etro
    return free, (total, etro)

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
    def update(params, opt_state, temp, g_vec, potential_nl):
      loss = lambda x: free_energy(
        x["pw"], x["occ"], temp, g_vec, potential_loc, potential_nl
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
        params, opt_state, temp, g_vec, potential_nl
      )
      etot, entro = es
      etot = jax.block_until_ready(etot)
      train_time += time.time() - start
      converged = convergence_checker.check(etot)
      if converged:
        logging.info("Converged.")
        break

      iters.set_description(
        f"Loss: {loss_val:.4f}|Energy: {etot+ew:.4f}|"
        f"Entropy: {entro:.4f}|T: {temp:.2E}"
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
  external_local = normcons.energy_local(
    density_reciprocal, potential_loc, vol=crystal.vol
  )
  external_nonlocal = normcons.energy_nonlocal(
    coeff, potential_nl, vol=crystal.vol, occupation=occ
  )

  xc = energy.xc_energy(density, g_vec, crystal.vol, config.xc, kohn_sham=False)

  logging.info(f"Hartree Energy: {hartree:.4f} Ha")
  logging.info(f"External (local) Energy: {external_local:.4f} Ha")
  logging.info(f"External (nonlocal) Energy: {external_nonlocal:.4f} Ha")
  logging.info(f"XC Energy: {xc:.4f} Ha")
  logging.info(f"Kinetic Energy: {kinetic:.4f} Ha")
  logging.info(f"Nuclear repulsion Energy: {ew:.4f} Ha")
  logging.info(f"Total Energy: {etot+ew:.4f} Ha")

  return density
