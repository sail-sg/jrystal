"""Smoke tests for electronic backends."""
import jax
import jax.numpy as jnp
from absl.testing import absltest

from jrystal.calc.backend import AllElectronBackend, NormConservingBackend, get_backend
from jrystal.calc.runtime import build_runtime_context
from jrystal.config import JrystalConfigDict, _normalize_config, get_config
from jrystal._src import pw, occupation

jax.config.update("jax_enable_x64", True)


def _make_ae_config():
  return get_config()


def _make_nc_config():
  raw = {
    "system": {"crystal": "diamond"},
    "method": {"use_pseudopotential": True, "pseudopotential_type": "nc"},
    "basis": {"grid_sizes": 24, "cutoff_energy": 50},
    "ksampling": {"k_grid_sizes": [1, 1, 1]},
  }
  return JrystalConfigDict(_normalize_config(raw))


class BackendFactoryTest(absltest.TestCase):

  def test_ae_backend_from_config(self):
    config = _make_ae_config()
    backend = get_backend(config)
    self.assertIsInstance(backend, AllElectronBackend)

  def test_nc_backend_from_config(self):
    config = _make_nc_config()
    backend = get_backend(config)
    self.assertIsInstance(backend, NormConservingBackend)


class AEBackendTest(absltest.TestCase):

  def test_total_energy_runs(self):
    config = _make_ae_config()
    config.basis.grid_sizes = 16
    config.basis.cutoff_energy = 20
    config.ksampling.k_grid_sizes = [1, 1, 1]

    backend = AllElectronBackend(config)
    ctx = build_runtime_context(config, backend=backend)

    num_e = backend.num_electrons(ctx)
    num_bands = num_e // 2 + 2
    num_kpts = ctx.ksampling.kpts.shape[0]

    key = jax.random.PRNGKey(0)
    params_pw = pw.param_init(key, num_bands, num_kpts, ctx.basis.freq_mask)
    coeff = pw.coeff(params_pw, ctx.basis.freq_mask)
    params_occ = occupation.params_init(num_bands, num_kpts)
    occ_fn = occupation.get_occupation_fn(num_e, spin=0, spin_restricted=True)
    occ = occ_fn(params_occ)

    e = backend.total_energy(coeff, occ, ctx)
    self.assertTrue(jnp.isfinite(e), f"Energy is not finite: {e}")

  def test_energy_decomposition_runs(self):
    config = _make_ae_config()
    config.basis.grid_sizes = 16
    config.basis.cutoff_energy = 20
    config.ksampling.k_grid_sizes = [1, 1, 1]

    backend = AllElectronBackend(config)
    ctx = build_runtime_context(config, backend=backend)

    num_e = backend.num_electrons(ctx)
    num_bands = num_e // 2 + 2
    num_kpts = ctx.ksampling.kpts.shape[0]

    key = jax.random.PRNGKey(0)
    params_pw = pw.param_init(key, num_bands, num_kpts, ctx.basis.freq_mask)
    coeff = pw.coeff(params_pw, ctx.basis.freq_mask)
    params_occ = occupation.params_init(num_bands, num_kpts)
    occ_fn = occupation.get_occupation_fn(num_e, spin=0, spin_restricted=True)
    occ = occ_fn(params_occ)

    decomp = backend.energy_decomposition(coeff, occ, ctx)
    self.assertIn("kinetic", decomp)
    self.assertIn("hartree", decomp)
    self.assertIn("xc", decomp)
    for v in decomp.values():
      self.assertTrue(jnp.isfinite(v), f"Decomp value not finite: {v}")


if __name__ == "__main__":
  absltest.main()
