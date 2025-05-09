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
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

from .._src import braket, energy, entropy, mdmm, occupation, potential, pw
from .._src.crystal import Crystal
from .._src.grid import proper_grid_size
from .._src.xc import xc_density
from ..config import JrystalConfigDict
from ..pseudopotential import local, nloc
from .convergence import create_convergence_checker
from .opt_utils import (
  create_crystal,
  create_freq_mask,
  create_grids,
  create_optimizer,
  create_pseudopotential,
  get_ewald_coulomb_repulsion,
  set_env_params
)


@dataclass
class GroundStateEnergyOutput:
  """Output of the ground state energy calculation.

  Args:
    config (JrystalConfigDict): The configuration for the calculation.
    crystal (Crystal): The crystal object.
    params_pw (dict): Parameters for the plane wave basis.
    params_occ (dict): Parameters for the occupation.
    total_energy (Union[float, jax.Array]): The total energy of the crystal.
    total_energy_history (List[float]): The optimization history of the total energy.
  """
  config: JrystalConfigDict
  crystal: Crystal
  params_pw: dict
  params_occ: dict
  total_energy: Union[float, jax.Array]
  total_energy_history: List[float]


def calc(config: JrystalConfigDict) -> GroundStateEnergyOutput:
  """Calculate the ground state energy of a crystal with norm-conserving pseudopotential.

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
  logging.info(f"Crystal: {crystal.symbol}")
  EPS = config.eps

  mesh = Mesh(np.array(jax.devices()).reshape([1, -1]), ('s', 'k'))
  sharding = NamedSharding(mesh, P('s', 'k'))  # shard by the kpt dimension.
  num_devices = len(jax.devices())
  logging.info(f"Number of devices: {num_devices}")

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
  potential_loc = local._potential_local_reciprocal(
    crystal.positions,
    g_vec,
    pseudopot.r_grid,
    pseudopot.local_potential_grid,
    pseudopot.local_potential_charge,
    crystal.vol
  )

  k_vec = jax.device_put(k_vec, NamedSharding(mesh, P('k')))
  logging.info("Initializing pseudopotential (Spherical Bessel Transform)...")
  start = time.time()
  beta_gk = nloc.beta_sbt_grid_multi_atoms(
    pseudopot.r_grid,
    pseudopot.nonlocal_beta_grid,
    pseudopot.nonlocal_angular_momentum,
    g_vec,
    k_vec,
  )
  beta_gk = jax.device_put(beta_gk, NamedSharding(mesh, P('k')))
  end = time.time()
  logging.info(
    f"Spherical Bessel Transform done. Times: {end - start:.2f} seconds"
  )
  logging.info("Initializing pseudopotential (nonlocal)...")
  start = time.time()
  potential_nl = nloc._potential_nonlocal_square_root(
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
    f"Deploying pseudopotential (nonlocal) done. Times: {end - start:.2f} seconds"
  )

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

  def total_energy(
    params_pw,
    params_occ,
    g_vec,
    potential_loc,
    potential_nl,
    density_old=None,
    temp=None,
  ):
    is_mdmm = density_old is not None
    coeff = pw.coeff(params_pw, freq_mask, sharding=sharding)

    if is_mdmm:  # don't contrat orbitals
      occ = None
    else:
      occ = get_occupation(params_occ)

    density = pw.density_grid(coeff, crystal.vol, occ)
    density_reciprocal = pw.density_grid_reciprocal(coeff, crystal.vol, occ)

    kinetic = energy.kinetic(g_vec, k_vec, coeff, occ)
    # hartree = energy.hartree(density_reciprocal, g_vec, crystal.vol)
    external_local = local._energy_local(
      density_reciprocal, v_local_reciprocal=potential_loc, vol=crystal.vol
    )
    external_nonlocal = nloc._energy_nonlocal(
      coeff, potential_nl, vol=crystal.vol, occupation=occ
    )

    if is_mdmm:  # compute Hxc using old density
      # NOTE: v_hartree does not have the 0.5 prefactor.
      v_hartree = potential.hartree_reciprocal(
        density_old, g_vec, kohn_sham=True
      )
      hamil_hartree = braket.reciprocal_braket(
        v_hartree, density_reciprocal, crystal.vol
      )

      exc = xc_density(density_old, g_vec, kohn_sham=False, xc_type=config.xc)
      hamil_exc = braket.real_braket(exc, density, crystal.vol)
      vxc = xc_density(density_old, g_vec, kohn_sham=True, xc_type=config.xc)
      hamil_vxc = braket.real_braket(vxc, density, crystal.vol)

      # computes occupation via Fermi-Dirac distribution
      hamil_diag = kinetic + hamil_hartree + external_local + external_nonlocal + hamil_vxc
      mu = params_occ  # negative
      occ = jax.nn.sigmoid(
        (jax.lax.stop_gradient(hamil_diag.real) + mu) / temp
      ) / num_kpts
      occ = jax.lax.stop_gradient(occ)

      # computes the total energy using current occupation
      e_diag = kinetic + 0.5 * hamil_hartree + external_local + external_nonlocal + hamil_exc
      e_tot = (e_diag * occ).sum()

      return jnp.squeeze(e_tot), occ

    else:

      hartree = energy.hartree(density_reciprocal, g_vec, crystal.vol)
      xc = energy.xc_energy(
        density, g_vec, crystal.vol, config.xc, kohn_sham=False
      )

      e_tot = kinetic + hartree + external_local + external_nonlocal + xc
      return jnp.squeeze(e_tot), None

  def get_entropy(params_occ):
    occ = get_occupation(params_occ)
    return entropy.fermi_dirac(occ, eps=EPS)

  def free_energy(
    params_pw,
    params_occ,
    temp,
    g_vec,
    potential_loc,
    potential_nl,
    density_old
  ):
    total, occ = total_energy(
      params_pw,
      params_occ,
      g_vec,
      potential_loc,
      potential_nl,
      density_old,
      temp
    )
    if config.occupation == "mdmm":  # add the multiplier part to the loss
      num_electrons = np.sum(pseudopot.valence_charges)
      infeas = occ.sum() - num_electrons
      mu = params_occ[0]
      damping = 10.  # TODO: make this a config parameter
      etro = entropy.fermi_dirac(occ, eps=EPS)
      free = total - temp * etro
      free += 100 * (-mu * infeas + damping * infeas**2 / 2)
    else:
      etro = get_entropy(params_occ)
      free = total - temp * etro
    return free, (total, etro, occ)

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
  if config.occupation != "mdmm":
    params_occ = jax.device_put(params_occ, sharding)
  # else:  # mark Lagrangian multiplier for gradient ascent
  #   optimizer = optax.chain(optimizer, mdmm.mdmm_descent_ascent())
  params = {"pw": params_pw, "occ": params_occ}
  opt_state = optimizer.init(params)

  # Define update function.
  with mesh:

    @jax.jit
    def update(params, opt_state, temp, g_vec, potential_nl, density_old):
      loss = lambda x: free_energy(
        x["pw"], x["occ"], temp, g_vec, potential_loc, potential_nl, density_old
      )
      (loss_val, es), grad = jax.value_and_grad(loss, has_aux=True)(params)
      updates, opt_state = optimizer.update(grad, opt_state)
      params = optax.apply_updates(params, updates)
      if config.occupation == "capped-simplex-proj":
        occ_ = occupation.capped_simplex(
          params["occ"],
          num_electrons=np.sum(pseudopot.valence_charges),
          spin=crystal.spin,
          spin_restricted=False,
        )
        params["occ"]["param_up"] = occ_[0]
        params["occ"]["param_down"] = occ_[1]
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

    @jax.jit
    def density_fn_jit(params, occ):
      coeff = pw.coeff(params["pw"], freq_mask)
      return pw.density_grid(coeff, crystal.vol, occ)

    if config.occupation == "mdmm":  # init occupation
      occ = occupation.uniform(
        num_kpts,
        np.sum(pseudopot.valence_charges),
        config.spin,
        num_bands,
        config.spin_restricted
      )
    else:
      occ = None

    for i in iters:
      temp = temperature_scheduler(i)
      start = time.time()

      if config.occupation == "mdmm":
        density_old = density_fn_jit(params, occ)
      else:
        density_old = None

      params, opt_state, loss_val, es = update(
        params, opt_state, temp, g_vec, potential_nl, density_old
      )

      etot, entro, occ = es

      etot = jax.block_until_ready(etot)
      train_time += time.time() - start
      converged = convergence_checker.check(etot)
      if converged:
        logging.info("Converged.")
        break

      desc_str = (
        f"Loss: {loss_val:.4f}|Energy: {etot+ew:.4f}|"
        f"Entropy: {entro:.4f}|T: {temp:.2E}"
      )
      if config.occupation == "mdmm":
        desc_str += f"|occ_sum: {occ.sum():.4f}|mu: {params['occ'][0]:.4f}"
      iters.set_description(desc_str)

  if not converged:
    logging.warning("Did not converge.")

  #####################################
  #        END OF OPTIMIZATION        #
  #####################################
  coeff = pw.coeff(params["pw"], freq_mask)
  if not config.occupation == "mdmm":
    occ = get_occupation(params["occ"])
  density = pw.density_grid(coeff, crystal.vol, occ)
  density_reciprocal = pw.density_grid_reciprocal(coeff, crystal.vol, occ)
  kinetic = energy.kinetic(g_vec, k_vec, coeff, occ)
  hartree = energy.hartree(density_reciprocal, g_vec, crystal.vol)
  external_local = local._energy_local(
    density_reciprocal, v_local_reciprocal=potential_loc, vol=crystal.vol
  )
  external_nonlocal = nloc._energy_nonlocal(
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

  np.save(
    f"occ-{config.occupation}-k{k_vec.shape[0]}-s{config.smearing}-b{config.empty_bands}",
    occ
  )

  return density
