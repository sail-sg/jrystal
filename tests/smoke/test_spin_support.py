from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from jrystal._src import hamiltonian as hamiltonian_module
from jrystal.calc import solver_nscf as solver_nscf_module
from jrystal.calc.backend import UltrasoftBackend
from jrystal.calc.solver_scf import _compute_occupation, _fixed_occupation
from jrystal.calc.types import KPointOperatorBundle, KSampling
from jrystal.pseudopotential.kernel import UltrasoftPathCache
from tests.smoke.helpers import make_config


def test_scf_fixed_occupation_respects_unrestricted_spin_counts():
  evals = jnp.asarray(
    [
      [[-4.0, -3.0, -2.0, -1.0]],
      [[-4.0, -3.0, -2.0, -1.0]],
    ],
    dtype=jnp.float32,
  )

  occ = _fixed_occupation(
    evals,
    num_electrons=6,
    spin=2,
    spin_restricted=False,
  )

  np.testing.assert_allclose(
    np.asarray(occ[0, 0]),
    np.asarray([1.0, 1.0, 1.0, 1.0]),
  )
  np.testing.assert_allclose(
    np.asarray(occ[1, 0]),
    np.asarray([1.0, 1.0, 0.0, 0.0]),
  )


def test_scf_smearing_occupation_uses_per_spin_targets():
  evals = jnp.asarray(
    [
      [[-4.0, -3.0, -2.0, -1.0]],
      [[-3.0, -2.0, -1.0, 0.0]],
    ],
    dtype=jnp.float32,
  )

  occ, chemical_potential = _compute_occupation(
    evals,
    num_electrons=6,
    k_weights=jnp.ones((1,), dtype=jnp.float32),
    smearing=0.1,
    spin=2,
    spin_restricted=False,
  )

  np.testing.assert_allclose(
    np.asarray(jnp.sum(occ, axis=(1, 2))),
    np.asarray([4.0, 2.0]),
    atol=1e-5,
  )
  assert tuple(np.asarray(chemical_potential).shape) == (2,)


def test_ae_hamiltonian_matrix_keeps_spin_resolved_density(monkeypatch):
  received = {}

  def _effective(density_grid, *args, **kwargs):
    received["density"] = jnp.asarray(density_grid)
    return jnp.zeros_like(density_grid)

  def _expectation(wave_or_coeff, *_args, diagonal=False, **_kwargs):
    return jnp.zeros(
      (
        wave_or_coeff.shape[0],
        wave_or_coeff.shape[1],
        wave_or_coeff.shape[2],
        wave_or_coeff.shape[2],
      ),
      dtype=jnp.float32,
    )

  monkeypatch.setattr(hamiltonian_module.potential, "effective", _effective)
  monkeypatch.setattr(
    hamiltonian_module.pw, "wave_grid", lambda coefficient, vol: coefficient
  )
  monkeypatch.setattr(hamiltonian_module.braket, "expectation", _expectation)
  monkeypatch.setattr(
    hamiltonian_module, "kinetic_operator", lambda g_vec, kpts: None
  )

  coefficient = jnp.zeros((2, 1, 1, 1, 1, 1), dtype=jnp.complex64)
  density = jnp.asarray(
    [
      [[[1.0]]],
      [[[0.25]]],
    ],
    dtype=jnp.float32,
  )

  hamiltonian_module.hamiltonian_matrix(
    coefficient,
    positions=jnp.zeros((0, 3), dtype=jnp.float32),
    charges=jnp.zeros((0,), dtype=jnp.int32),
    effictive_density_grid=density,
    g_vector_grid=jnp.zeros((1, 1, 1, 3), dtype=jnp.float32),
    kpts=jnp.zeros((1, 3), dtype=jnp.float32),
    vol=jnp.asarray(1.0, dtype=jnp.float32),
    xc="lda_x",
    kohn_sham=True,
  )

  np.testing.assert_allclose(
    np.asarray(received["density"]), np.asarray(density)
  )


def test_uspp_xc_density_splits_nlcc_between_spin_channels():
  config = make_config()
  backend = UltrasoftBackend(config)
  total_density = jnp.zeros((2, 2, 2, 2), dtype=jnp.float32)
  ctx = SimpleNamespace(
    pseudo_cache=SimpleNamespace(nlcc_g=jnp.ones((2, 2, 2), dtype=jnp.float32))
  )

  xc_density = backend._xc_density(total_density, ctx)

  np.testing.assert_allclose(
    np.asarray(xc_density[0]),
    np.full((2, 2, 2), 0.5, dtype=np.float32),
  )
  np.testing.assert_allclose(
    np.asarray(xc_density[1]),
    np.full((2, 2, 2), 0.5, dtype=np.float32),
  )


def test_uspp_prepare_iteration_returns_spin_resolved_operator_state(
  monkeypatch
):
  config = make_config()
  backend = UltrasoftBackend(config)
  density = jnp.zeros((2, 2, 2, 2), dtype=jnp.float32)
  ctx = SimpleNamespace(
    g_vec=jnp.zeros((2, 2, 2, 3), dtype=jnp.float32),
    potential_local=jnp.zeros((2, 2, 2), dtype=jnp.float32),
    pseudo_cache=SimpleNamespace(
      nlcc_g=None,
      channel_dii=jnp.zeros((1, 1, 1), dtype=jnp.float32),
      channel_coupling=jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.float32),
      channel_beta=jnp.zeros((1, 1), dtype=jnp.int32),
      augmentation_radial_fields_g=jnp.zeros(
        (1, 1, 1, 1, 2, 2, 2), dtype=jnp.float32
      ),
      augmentation_harmonics_g=jnp.zeros((1, 1, 1, 2, 2, 2), dtype=jnp.float32),
      channel_mask=jnp.ones((1, 1), dtype=jnp.float32),
    ),
    crystal=SimpleNamespace(vol=1.0),
  )

  monkeypatch.setattr(
    backend,
    "_effective_local_potential",
    lambda *_args, **_kwargs: jnp.stack(
      [
        jnp.ones((2, 2, 2), dtype=jnp.float32),
        2.0 * jnp.ones((2, 2, 2), dtype=jnp.float32),],
      axis=0,),
  )

  state = backend.prepare_iteration(density, ctx)

  assert tuple(state.local_potential_r.shape) == (2, 2, 2, 2)
  assert tuple(state.channel_dii.shape) == (2, 1, 1, 1)


def test_band_num_bands_uses_unrestricted_occ_max(monkeypatch):
  config = make_config()
  config.system.spin_restricted = False
  config.band.empty_bands = 2
  captured = {}

  def _fake_run_nscf_ae(
    _config, _ctx, _density, num_bands, *, phase_timer=None
  ):
    del phase_timer
    captured["num_bands"] = num_bands
    return jnp.zeros((2, 1, num_bands), dtype=jnp.float32)

  def _make_band_result(**kwargs):
    return SimpleNamespace(**kwargs)

  monkeypatch.setattr(solver_nscf_module, "_run_nscf_ae", _fake_run_nscf_ae)
  monkeypatch.setattr(
    solver_nscf_module,
    "BandStructureResult",
    _make_band_result,
  )

  backend = SimpleNamespace(num_electrons=lambda _ctx: 8)
  ctx = SimpleNamespace(
    crystal=SimpleNamespace(),
    ksampling=KSampling(
      mode="path",
      kpts=jnp.zeros((1, 3), dtype=jnp.float32),
      weights=jnp.ones((1,), dtype=jnp.float32),
    ),
  )
  ground_state_result = SimpleNamespace(
    density=jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
    total_energy=-1.0,
    fermi_energy=None,
  )

  result = solver_nscf_module.run_nscf(
    config,
    ctx,
    backend,
    ground_state_result,
  )

  assert captured["num_bands"] == 10
  assert tuple(result.eigenvalues.shape) == (2, 1, 10)


def test_uspp_band_batches_two_spin_channels_when_unrestricted(monkeypatch):
  config = make_config()
  config.system.spin_restricted = False
  config.solver.scf.eigensolver.max_iter = 1
  config.execution.profile = False

  recorded = []

  def _fake_lobpcg(*, v0, k, **_kwargs):
    recorded.append(tuple(v0.shape))
    evals = jnp.zeros((v0.shape[0], k), dtype=jnp.float32)
    return evals, v0

  monkeypatch.setattr(solver_nscf_module, "batched_lobpcg", _fake_lobpcg)

  ctx = SimpleNamespace(
    g_vec=jnp.zeros((1, 1, 1, 3), dtype=jnp.float32),
    basis=SimpleNamespace(freq_mask=jnp.ones((1, 1, 1), dtype=bool)),
    ksampling=KSampling(
      mode="path",
      kpts=jnp.zeros((2, 3), dtype=jnp.float32),
      weights=jnp.ones((2,), dtype=jnp.float32) / 2.0,
    ),
    crystal=SimpleNamespace(vol=1.0),
    pseudo_cache=UltrasoftPathCache(
      family="us",
      species_setups=(),
      atom_species_map=SimpleNamespace(),
      vloc_g=jnp.zeros((1, 1, 1), dtype=jnp.float32),
      beta_radial_gk=(),
      q_matrices=jnp.zeros((1, 1, 1), dtype=jnp.float32),
      channel_qii=jnp.zeros((1, 1, 1), dtype=jnp.float32),
      channel_mask=jnp.ones((1, 1), dtype=jnp.float32),
      channel_beta=jnp.zeros((1, 1), dtype=jnp.int32),
      channel_dii=jnp.zeros((1, 1, 1), dtype=jnp.float32),
      channel_coupling=jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.float32),
      augmentation_radial_fields_g=jnp.zeros(
        (1, 1, 1, 1, 1, 1, 1), dtype=jnp.float32
      ),
      augmentation_harmonics_g=jnp.zeros((1, 1, 1, 1, 1, 1), dtype=jnp.float32),
      nlcc_g=None,
      augmentation_l_max=0,
      has_angular_augmentation=False,
    ),
  )

  backend = SimpleNamespace(
    prepare_nscf=lambda density, _ctx: "state",
    build_kpoint_operator=lambda kpt_index, nscf_state, _ctx:
    KPointOperatorBundle(
      kpt=_ctx.ksampling.kpts[kpt_index:kpt_index + 1],
      local_potential_r=jnp.zeros((2, 1, 1, 1), dtype=jnp.float32),
      projector_channels_g=jnp.zeros((1, 1, 1, 1, 1, 1), dtype=jnp.complex64),
      channel_qii=jnp.zeros((1, 1, 1), dtype=jnp.float32),
      channel_dii_eff=jnp.zeros((2, 1, 1, 1), dtype=jnp.float32),
      channel_mask=jnp.ones((1, 1), dtype=jnp.float32),
      solver_kind="generalized",),
  )

  eigenvalues = solver_nscf_module._run_nscf_us(
    config,
    ctx,
    density=jnp.zeros((2, 1, 1, 1), dtype=jnp.float32),
    num_bands=1,
    backend=backend,
  )

  assert recorded == [(2, 1, 1), (2, 1, 1)]
  assert tuple(eigenvalues.shape) == (2, 2, 1)
