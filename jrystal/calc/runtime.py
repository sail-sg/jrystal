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

import jax
import jax.numpy as jnp
import numpy as np
from chex import dataclass
from jaxtyping import Array, Float

from .._src.band import get_k_path
from .._src.crystal import Crystal
from .._src.grid import proper_grid_size
from ..config import JrystalConfigDict
from .opt_utils import (
  create_crystal,
  create_freq_mask,
  create_grids,
  get_ewald_coulomb_repulsion,
)
from .types import ExecutionPlan, KSampling, PlaneWaveBasis


@dataclass
class RuntimeContext:
  """Collected runtime state for a calculation workflow."""

  crystal: Crystal
  g_vec: Float[Array, "x y z 3"]
  r_vec: Float[Array, "x y z 3"]
  ksampling: KSampling
  basis: PlaneWaveBasis
  ewald_energy: float
  execution: ExecutionPlan
  pseudopotential: Optional[object] = None
  pseudo_cache: Optional[object] = None
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


def build_runtime_context(
  config: JrystalConfigDict,
  *,
  mode: Literal["mesh", "path"] = "mesh",
  backend=None,
) -> RuntimeContext:
  """One-shot initialization of all workflow runtime data.

  Args:
    config: Jrystal configuration.
    mode: ``"mesh"`` for ground-state k-mesh, ``"path"`` for band k-path.
    backend: An :class:`ElectronicBackend` instance.  When provided,
      ``backend.build_potentials(ctx)`` is called to attach
      backend-specific potentials to the context.

  Returns:
    Fully initialised :class:`RuntimeContext`.
  """
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

  grid_sizes = tuple(int(x) for x in proper_grid_size(config.basis.grid_sizes))

  basis = PlaneWaveBasis(
    freq_mask=freq_mask,
    grid_sizes=grid_sizes,
    num_g=int(np.sum(np.asarray(freq_mask))),
  )

  execution = ExecutionPlan(
    num_devices=len(jax.devices()),
    parallel_over_k=config.execution.parallel_over_k_mesh
    if mode == "mesh" else config.execution.parallel_over_k_path,
  )

  ew = get_ewald_coulomb_repulsion(
    config,
    crystal=crystal,
    g_vector_grid=g_vec,
  )

  ctx = RuntimeContext(
    crystal=crystal,
    g_vec=g_vec,
    r_vec=r_vec,
    ksampling=ksampling,
    basis=basis,
    ewald_energy=ew,
    execution=execution,
  )

  if backend is not None:
    ctx = backend.build_potentials(ctx)

  return ctx


__all__ = ["RuntimeContext", "build_kpath_sampling", "build_runtime_context"]
