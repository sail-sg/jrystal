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
"""Runtime-context construction for calc workflows."""

from typing import Literal, Optional

import jax.numpy as jnp
import numpy as np
from absl import logging
from chex import dataclass
from jaxtyping import Array, Bool, Float

from .._src.band import get_k_path
from .._src.crystal import Crystal
from ..config import JrystalConfigDict
from ..pseudopotential import normcons
from .opt_utils import (
  create_crystal,
  create_freq_mask,
  create_grids,
  create_pseudopotential,
  get_ewald_coulomb_repulsion,
)
from .pre_calc import pre_calc_beta_sbt
from .types import KSampling


@dataclass
class RuntimeContext:
  """Collected runtime state for a calculation workflow."""

  crystal: Crystal
  g_vec: Float[Array, "x y z 3"]
  r_vec: Float[Array, "x y z 3"]
  ksampling: KSampling
  freq_mask: Bool[Array, "x y z"]
  ewald_energy: float
  pseudopotential: Optional[object] = None
  potential_local: Optional[object] = None
  potential_nonlocal: Optional[object] = None


def build_kpath_sampling(
  config: JrystalConfigDict,
  crystal: Crystal,
) -> KSampling:
  """Build KSampling for a band-structure path."""
  if config.band.k_path_file is not None:
    kpts = np.load(config.band.k_path_file)
  else:
    if config.band.k_path_special_points is None:
      raise ValueError(
        "Band calculations require `band.k_path_special_points` or "
        "`band.k_path_file`."
      )
    kpts = get_k_path(
      crystal.cell_vectors,
      path=config.band.k_path_special_points,
      num=config.band.num_kpoints,
      fractional=False,
    )
  kpts = jnp.asarray(kpts)
  return KSampling(
    mode="path",
    kpts=kpts,
    weights=jnp.ones(kpts.shape[0]),
  )


def _build_normcons_potentials(
  config: JrystalConfigDict,
  crystal: Crystal,
  g_vec,
  ksampling: KSampling,
):
  pseudopot = create_pseudopotential(config, crystal=crystal)
  logging.info("Initializing pseudopotential (local)...")
  potential_loc = normcons.potential_local_reciprocal(
    crystal.positions,
    g_vec,
    pseudopot.r_grid,
    pseudopot.local_potential_grid,
    pseudopot.local_potential_charge,
    crystal.vol,
  )

  logging.info("Initializing pseudopotential (Spherical Bessel Transform)...")
  beta_gk = pre_calc_beta_sbt(
    pseudopot,
    np.array(g_vec),
    np.array(ksampling.kpts),
  )

  if ksampling.mode == "mesh":
    logging.info("Initializing pseudopotential (nonlocal)...")
    potential_nl = normcons.potential_nonlocal_psi_reciprocal(
      crystal.positions,
      g_vec,
      ksampling.kpts,
      pseudopot.r_grid,
      pseudopot.nonlocal_beta_grid,
      pseudopot.nonlocal_angular_momentum,
      pseudopot.nonlocal_d_matrix,
      beta_gk,
    )
  else:
    # For band-path workflows, keep the SBT cache and assemble one k-point
    # nonlocal operator at a time inside the band solver.
    potential_nl = beta_gk

  return pseudopot, potential_loc, potential_nl


def build_runtime_context(
  config: JrystalConfigDict,
  *,
  mode: Literal["mesh", "path"] = "mesh",
) -> RuntimeContext:
  """One-shot initialization of all workflow runtime data."""
  crystal = create_crystal(config)

  if mode == "mesh":
    g_vec, r_vec, ksampling = create_grids(config, crystal=crystal)
  elif mode == "path":
    ksampling = build_kpath_sampling(config, crystal)
    g_vec, r_vec, ksampling = create_grids(
      config,
      crystal=crystal,
      ksampling=ksampling,
    )
  else:
    raise ValueError(f"Unsupported runtime mode: {mode}")

  freq_mask = create_freq_mask(config, crystal=crystal)
  ew = get_ewald_coulomb_repulsion(
    config,
    crystal=crystal,
    g_vector_grid=g_vec,
  )

  pseudopot = None
  potential_loc = None
  potential_nl = None
  if config.method.use_pseudopotential:
    if config.method.pseudopotential_type not in ["normcons", "normconserving", "nc"]:
      raise NotImplementedError(
        "RuntimeContext currently supports only norm-conserving "
        "pseudopotentials in step 1."
      )
    pseudopot, potential_loc, potential_nl = _build_normcons_potentials(
      config,
      crystal,
      g_vec,
      ksampling,
    )

  return RuntimeContext(
    crystal=crystal,
    g_vec=g_vec,
    r_vec=r_vec,
    ksampling=ksampling,
    freq_mask=freq_mask,
    ewald_energy=ew,
    pseudopotential=pseudopot,
    potential_local=potential_loc,
    potential_nonlocal=potential_nl,
  )


__all__ = ["RuntimeContext", "build_kpath_sampling", "build_runtime_context"]
