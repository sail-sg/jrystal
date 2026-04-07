from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

import jrystal as jr
from jrystal.calc.types import (
  BandStructureResult,
  EnergyDecomposition,
  GroundStateResult,
  KSampling,
)


def make_config():
  config = jr.config.get_config(None)
  config.basis.grid_sizes = 4
  config.ksampling.k_grid_sizes = [1, 1, 1]
  config.occupation.empty_bands = 2
  config.io.save_density = True
  config.io.save_wavefunction = True
  config.io.save_ground_state_spectrum = True
  config.io.save_checkpoint = True
  config.band.plot.enabled = False
  config.io.log_level = "quiet"
  config.execution.verbose = False
  return config


def make_dummy_crystal():
  return SimpleNamespace(
    num_electron=np.asarray(8),
    num_atom=2,
    symbols=["Si", "Si"],
  )


def make_dummy_ctx(config):
  crystal = make_dummy_crystal()
  grid_size = int(config.basis.grid_sizes)
  num_bands = int(np.ceil(8 / 2.0) + config.occupation.empty_bands)
  return SimpleNamespace(
    crystal=crystal,
    basis=SimpleNamespace(
      grid_sizes=(grid_size, grid_size, grid_size),
      num_g=3,
    ),
    ksampling=SimpleNamespace(
      kpts=jnp.zeros((1, 3), dtype=jnp.float32),
      weights=jnp.ones((1,), dtype=jnp.float32),
    ),
    ewald_energy=0.0,
    execution=SimpleNamespace(num_devices=1, parallel_over_k=False),
    num_bands=num_bands,
  )


def make_ground_state_result(config, *, converged=True):
  crystal = make_dummy_crystal()
  num_bands = int(np.ceil(8 / 2.0) + config.occupation.empty_bands)
  coeff_shape = (1, 1, 3, num_bands)
  density_shape = (1, 4, 4, 4)
  coefficients = {
    "w_re": jnp.zeros(coeff_shape, dtype=jnp.float32),
    "w_im": jnp.zeros(coeff_shape, dtype=jnp.float32),
  }
  occupations = jnp.asarray(
    [[[2.0, 2.0, 2.0, 2.0] + [0.0] * (num_bands - 4)]],
    dtype=jnp.float32,
  )
  eigenvalues = jnp.linspace(-1.0, 1.0, num_bands, dtype=jnp.float32).reshape(
    1,
    1,
    num_bands,
  )
  return GroundStateResult(
    config=config,
    crystal=crystal,
    params_pw=coefficients,
    params_occ={},
    total_energy=-7.5,
    energy_terms=EnergyDecomposition(
      kinetic=1.0,
      hartree=2.0,
      xc=-3.0,
      external=-8.0,
      ewald=0.5,
    ),
    converged=converged,
    density=jnp.zeros(density_shape, dtype=jnp.float32),
    coefficients=coefficients,
    eigenvalues=eigenvalues,
    occupations=occupations,
    actual_solver="scf",
    requested_solver_mode="auto",
    num_iterations=3,
    wall_time=1.25,
    fermi_energy=-0.1,
    convergence_history=[
      {
        "step": 1,
        "total_energy": -7.0,
        "delta_energy": None,
        "delta_density": 0.5,
        "wall_time": 0.4,
      },
      {
        "step": 2,
        "total_energy": -7.4,
        "delta_energy": 0.4,
        "delta_density": 0.1,
        "wall_time": 0.4,
      },
      {
        "step": 3,
        "total_energy": -7.5,
        "delta_energy": 0.1,
        "delta_density": 0.01,
        "wall_time": 0.45,
      },
    ],
    total_energy_history=[-7.0, -7.4, -7.5],
  )


def make_band_result(config):
  crystal = make_dummy_crystal()
  kpath = KSampling(
    mode="path",
    kpts=jnp.asarray(
      [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]],
      dtype=jnp.float32,
    ),
    weights=jnp.ones((3,), dtype=jnp.float32),
    labels=["G", "X", "M"],
    segments=[(0, 1), (1, 2)],
  )
  num_bands = int(np.ceil(8 / 2.0) + config.occupation.empty_bands)
  eigenvalues = jnp.linspace(-0.5, 0.5, 3 * num_bands, dtype=jnp.float32)
  return BandStructureResult(
    config=config,
    crystal=crystal,
    kpath=kpath,
    eigenvalues=eigenvalues.reshape(1, 3, num_bands),
    ground_state_energy=-7.5,
    reference_energy=-0.1,
  )
