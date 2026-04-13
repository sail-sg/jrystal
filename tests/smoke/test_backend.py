from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from jrystal.calc import backend as backend_module
from jrystal.calc.backend import (
  AllElectronBackend,
  NormConservingBackend,
  UltrasoftBackend,
  get_backend,
)
from jrystal.calc.runtime import RuntimeContext
from jrystal.calc.types import ExecutionPlan, KSampling, PlaneWaveBasis
from jrystal.pseudopotential.kernel import UltrasoftMeshCache
from tests.smoke.helpers import make_config


def _make_runtime_context(*, num_kpts: int = 3) -> RuntimeContext:
  return RuntimeContext(
    crystal=SimpleNamespace(vol=1.0),
    g_vec=jnp.zeros((2, 2, 2, 3), dtype=jnp.float32),
    r_vec=jnp.zeros((2, 2, 2, 3), dtype=jnp.float32),
    ksampling=KSampling(
      mode="mesh",
      kpts=jnp.zeros((num_kpts, 3), dtype=jnp.float32),
      weights=jnp.ones((num_kpts,), dtype=jnp.float32) / float(num_kpts),
    ),
    basis=PlaneWaveBasis(
      freq_mask=jnp.ones((2, 2, 2), dtype=bool),
      grid_sizes=(2, 2, 2),
      num_g=8,
    ),
    ewald_energy=0.0,
    execution=ExecutionPlan(num_devices=1, parallel_over_k=False),
  )


def _make_uspp_mesh_cache(*, num_kpts: int = 3) -> UltrasoftMeshCache:
  return UltrasoftMeshCache(
    family="us",
    species_setups=(),
    atom_species_map=SimpleNamespace(),
    vloc_g=jnp.zeros((2, 2, 2), dtype=jnp.float32),
    beta_radial_gk=(),
    projector_gk=None,
    projector_mask=None,
    q_matrices=jnp.zeros((2, 1, 1), dtype=jnp.float32),
    channel_qii=jnp.zeros((2, 8, 8), dtype=jnp.float32),
    channel_mask=jnp.ones((2, 8), dtype=jnp.float32),
    channel_beta=jnp.zeros((2, 8), dtype=jnp.int32),
    channel_dii=jnp.zeros((2, 8, 8), dtype=jnp.float32),
    channel_coupling=jnp.zeros((2, 8, 8, 1, 1), dtype=jnp.float32),
    augmentation_radial_fields_g=jnp.zeros(
      (2, 1, 1, 1, 2, 2, 2), dtype=jnp.float32
    ),
    augmentation_harmonics_g=jnp.zeros((2, 1, 1, 2, 2, 2), dtype=jnp.float32),
    nlcc_g=None,
    augmentation_l_max=0,
    has_angular_augmentation=False,
    channel_projectors_gk=None,
    channel_projectors_compact_gk=jnp.zeros(
      (2, num_kpts, 8, 5),
      dtype=jnp.complex64,
    ),
  )


def test_get_backend_returns_all_electron_by_default():
  config = make_config()
  config.method.use_pseudopotential = False

  backend = get_backend(config)

  assert isinstance(backend, AllElectronBackend)


def test_get_backend_returns_normconserving_when_enabled():
  config = make_config()
  config.method.use_pseudopotential = True
  config.method.pseudopotential_type = "nc"

  backend = get_backend(config)

  assert isinstance(backend, NormConservingBackend)


def test_get_backend_returns_ultrasoft_when_enabled():
  config = make_config()
  config.method.use_pseudopotential = True
  config.method.pseudopotential_type = "ultrasoft"

  backend = get_backend(config)

  assert isinstance(backend, UltrasoftBackend)


def test_ultrasoft_backend_accepts_multik_mesh_cache(monkeypatch):
  config = make_config()
  config.method.use_pseudopotential = True
  config.method.pseudopotential_type = "us"
  config.ksampling.k_grid_sizes = [2, 2, 2]
  backend = UltrasoftBackend(config)
  ctx = _make_runtime_context(num_kpts=3)
  cache = _make_uspp_mesh_cache(num_kpts=3)

  monkeypatch.setattr(
    "jrystal.calc.opt_utils.create_pseudopotential",
    lambda _config, crystal=None: SimpleNamespace(
      species_setups=(),
      atom_species_map=SimpleNamespace(),
      valence_charges=jnp.asarray([4.0, 4.0], dtype=jnp.float32),),
  )
  monkeypatch.setattr(
    backend_module, "build_pseudo_cache", lambda *args, **kwargs: cache
  )

  updated = backend.build_potentials(ctx)

  assert updated.pseudo_cache is cache
  assert updated.potential_local.shape == (2, 2, 2)


def test_ultrasoft_backend_rejects_multik_mesh_cache_with_wrong_projector_dim(
  monkeypatch,
):
  config = make_config()
  config.method.use_pseudopotential = True
  config.method.pseudopotential_type = "us"
  config.ksampling.k_grid_sizes = [2, 2, 2]
  backend = UltrasoftBackend(config)
  ctx = _make_runtime_context(num_kpts=3)
  cache = _make_uspp_mesh_cache(num_kpts=2)

  monkeypatch.setattr(
    "jrystal.calc.opt_utils.create_pseudopotential",
    lambda _config, crystal=None: SimpleNamespace(
      species_setups=(),
      atom_species_map=SimpleNamespace(),
      valence_charges=jnp.asarray([4.0, 4.0], dtype=jnp.float32),),
  )
  monkeypatch.setattr(
    backend_module, "build_pseudo_cache", lambda *args, **kwargs: cache
  )

  with pytest.raises(ValueError, match="k-point dimension"):
    backend.build_potentials(ctx)
