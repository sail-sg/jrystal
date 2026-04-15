from __future__ import annotations

import jax
import jax.numpy as jnp

import jrystal as jr
from jrystal.calc import solver_direct_opt
from jrystal.calc.runtime import build_runtime_context
from jrystal.calc.solver_direct_opt import run_direct_opt


class _ToyBackend:

  requires_canonical_transform = False

  def num_electrons(self, ctx):
    return int(ctx.crystal.num_electron)

  def total_energy(self, coeff, occ, ctx):
    del ctx
    return (0.01 * jnp.sum(jnp.abs(coeff)**2) + 0.001 * jnp.sum(occ**2))

  def overlap_inv_sqrt_apply(self, coeff, ctx):
    del ctx
    return coeff

  def energy_decomposition(self, coeff, occ, ctx):
    total = self.total_energy(coeff, occ, ctx)
    return {
      "kinetic": total,
      "hartree": jnp.asarray(0.0, dtype=total.dtype),
      "xc": jnp.asarray(0.0, dtype=total.dtype),
      "external": jnp.asarray(0.0, dtype=total.dtype),
    }


def test_direct_opt_builds_a_single_update_jit(monkeypatch):
  config = jr.config.get_config(None)
  config.io.log_level = "quiet"
  config.execution.verbose = False
  config.io.save_checkpoint = False
  config.solver.mode = "direct_opt"
  config.solver.direct_opt.max_steps = 3
  config.solver.direct_opt.optimizer.learning_rate = 1e-3
  config.solver.direct_opt.occupation_optimizer.warmup_steps = 2
  config.basis.grid_sizes = 4
  config.basis.cutoff_energy = 5
  config.ksampling.k_grid_sizes = [1, 1, 1]
  config.occupation.empty_bands = 1

  ctx = build_runtime_context(config, mode="mesh")
  backend = _ToyBackend()

  jit_calls = []
  original_jit = jax.jit

  def counting_jit(fun=None, **kwargs):

    def _wrap(f):
      jit_calls.append((getattr(f, "__name__", "<lambda>"), kwargs))
      return original_jit(f, **kwargs)

    if fun is None:
      return _wrap
    return _wrap(fun)

  monkeypatch.setattr(solver_direct_opt.jax, "jit", counting_jit)

  result = run_direct_opt(config, ctx, backend)

  assert result.actual_solver == "direct_opt"
  assert sum(1 for name, _ in jit_calls if name == "update") == 1
