import jax
import optax
from math import ceil

import jrystal as jr
from jrystal.config import ConfigDict
from .opt_utils import (
  create_crystal,
  create_optimizer,
  # create_occupation,
  create_grids,
  set_env_params,
  get_ewald_coulomb_repulsion,
  create_fft_mask
)

from dataclasses import dataclass
from absl import logging
from tqdm import tqdm
import time
from typing import List


@dataclass
class GroundStateEnergyOutput:
  config: ConfigDict
  params_wave: jax.Array
  params_occ: jax.Array
  total_energy: float | jax.Array
  total_energy_history: List[float]


def calc(config: ConfigDict) -> GroundStateEnergyOutput:
  # Initialize and Prepare variables.
  set_env_params(config)
  key = jax.random.PRNGKey(config.seed)
  temp = config.smearing

  crystal = create_crystal(config)
  logging.info(f"Crystal: {crystal.symbol}")
  EPS = config.eps

  g_vec, r_vec, k_vec = create_grids(config)
  num_kpts = k_vec.shape[0]
  num_bands = ceil(crystal.num_electron/2) + config.empty_bands
  fft_mask = create_fft_mask(config)
  ew = get_ewald_coulomb_repulsion(config)

  # Define functions for energy calculation.
  # assume fermi-dirac occupation.
  def occupation(params):
    return jr.occupation.idempotent(
      params, crystal.num_electron, num_kpts, crystal.spin
    )

  def total_energy(params_pw, params_occ):
    coeff = jr.pw.coeff(params_pw, fft_mask)
    occ = occupation(params_occ)
    return jr.energy.total_energy(
      coeff, crystal.positions, crystal.charges, g_vec, k_vec, crystal.vol, occ
    )

  def entropy(params_occ):
    occ = occupation(params_occ)
    return jr.entropy.fermi_dirac(occ, eps=EPS)

  def free_energy(params_pw, params_occ, temp):
    total = total_energy(params_pw, params_occ)
    etro = entropy(params_occ)
    free = total + temp * etro
    return free, (total, etro)

  # Initialize parameters and optimizer.
  optimizer = create_optimizer(config)
  params_pw = jr.pw.param_init(key, num_bands, num_kpts, fft_mask)
  params_occ = jr.occupation.idempotent_param_init(key, num_bands, num_kpts)
  params = {"pw": params_pw, "occ": params_occ}
  opt_state = optimizer.init(params)

  # Define update function.
  @jax.jit
  def update(params, opt_state, temp):
    loss = lambda x: free_energy(x["pw"], x["occ"], temp)
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
      params, opt_state, temp
    )
    etot, entro = es
    etot = jax.block_until_ready(etot)
    train_time += time.time() - start

    iters.set_description(
      f"Loss: {loss_val:.4f}|Energy: {etot+ew:.4f}|"
      f"Entropy: {entro:.4f}|T: {temp:.2E}"
    )

  #####################################
  #        END OF OPTIMIZATION        #
  #####################################
  coeff = jr.pw.coeff(params["pw"], fft_mask)
  occ = occupation(params["occ"])
  density = jr.pw.density_grid(coeff, crystal.vol, occ)
  density_reciprocal = jr.pw.density_grid_reciprocal(coeff, crystal.vol, occ)
  kinetic = jr.energy.kinetic(g_vec, k_vec, coeff, occ)
  hartree = jr.energy.hartree(density_reciprocal, g_vec, crystal.vol)
  external = jr.energy.external(
    density_reciprocal,
    crystal.positions,
    crystal.charges,
    g_vec,
    crystal.vol
  )
  lda = jr.energy.xc_lda(density, crystal.vol)

  logging.info(f"Hartree Energy: {hartree:.4f}")
  logging.info(f"External Energy: {external:.4f}")
  logging.info(f"LDA Energy: {lda:.4f}")
  logging.info(f"Kinetic Energy: {kinetic:.4f}")
  logging.info(f"Nuclear repulsion Energy: {ew:.4f}")
  logging.info(f"Total Energy: {etot+ew:.4f}")

  return GroundStateEnergyOutput(
    config, params["pw"], params["occ"], etot+ew, []
  )
