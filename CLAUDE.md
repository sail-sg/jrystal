# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important Rules
Before executing any of the following operations, always stop and ask the user for confirmation:
- git commit
- git push
- git merge
- git rebase
- Any operation that modifies git history

Commit messages must be clear, concise summaries of **what was done** — focus on actions and outcomes. Never reference step numbers, plan indices, file lists, or internal task tracking (e.g. "Step 3", "WP2", "from cc1.md"). The git history should read as a standalone narrative of changes, not as an execution log of a plan.

## What This Project Is

Jrystal is a JAX-based differentiable plane-wave density functional theory (DFT) framework for periodic crystalline systems. It computes ground-state energies, band structures, and (eventually) forces/stress for solid-state materials. All numerical kernels run on GPU via JAX.

## Build & Run

```bash
# Install in dev mode (use project venv)
.venv/bin/pip install -e .

# Run a calculation
.venv/bin/python -m jrystal.cli energy config.yaml
.venv/bin/python -m jrystal.cli band config.yaml

# Python API
import jrystal as jr
config = jr.config.get_config("config.yaml")
result = jr.calc.energy(config)
bands = jr.calc.band(config, result)
```

## Tests

```bash
# All smoke tests (CPU, ~30s)
.venv/bin/python -m pytest tests/smoke/ -v

# Single test file
.venv/bin/python -m pytest tests/smoke/test_backend.py -v

# Kernel unit tests (co-located with source, pytest)
.venv/bin/python -m pytest jrystal/_src/smearing_test.py -v
.venv/bin/python -m pytest jrystal/_src/crystal_test.py -v

# Run baseline benchmark (takes ~2min on CPU)
.venv/bin/python tests/reference/run_baseline.py
```

Tests use `pytest` as the unified runner. CI runs `tests/smoke/` on push/PR.

## Lint & Format

```bash
# Ruff (configured in pyproject.toml)
ruff check jrystal/
ruff format --check jrystal/

# Project style: google-based, 2-space indent, 80 col
```

Key ruff ignores: `F722` (jaxtyping annotations), `E501` (long lines), `N802` (function naming), `PLR0913`/`PLR0914` (arg counts).

## Architecture

Two layers, strict separation:

### `_src/` — Kernel layer (pure JAX math, do not add workflow logic here)
Stateless functions operating on arrays. Key modules: `pw.py` (plane-wave coefficients/density), `energy.py` (kinetic/hartree/xc/external), `hamiltonian.py` (H matrix/trace), `grid.py` (G/R/k vectors), `occupation.py`, `fft/`, `linalg/batch_lobpcg.py`, `ewald.py`, `xc.py` (wraps jxc).

### `calc/` — Workflow layer (orchestration, config, IO)
Backend-agnostic solvers that delegate physics to `ElectronicBackend`:

```
calc/__init__.py          # Public API: energy(config), band(config)
  → backend.py            # AllElectronBackend / NormConservingBackend
  → solver_direct_opt.py  # run_direct_opt(config, ctx, backend)
  → solver_scf.py         # run_scf(config, ctx, backend)
  → solver_nscf.py        # run_nscf(config, ctx, backend, gs_result)
  → runtime.py            # RuntimeContext + build_runtime_context()
  → types.py              # KSampling, PlaneWaveBasis, ExecutionPlan, result dataclasses
  → density_mixing.py     # DIIS, simple_mixing, kerker
```

**Data flow**: `config` → `get_backend(config)` → `build_runtime_context(config, backend=backend)` → `run_*solver*(config, ctx, backend)` → `GroundStateResult`

**Backend protocol**: `build_potentials(ctx)`, `total_energy(coeff, occ, ctx)`, `hamiltonian_apply(coeff, density, ctx)`, `energy_decomposition(coeff, occ, ctx)`, `num_electrons(ctx)`. Adding a new backend (e.g. ultrasoft) means implementing these methods.

### Config
Nested YAML schema v1 with groups: `system`, `method`, `basis`, `ksampling`, `solver`, `occupation`, `ewald`, `band`, `execution`, `io`. Legacy flat configs auto-migrate via `_normalize_config()`. Validation in `validate_config()`.

### Proxy modules
`jrystal/energy.py`, `jrystal/pw.py`, etc. are thin re-exports from `_src/` for the public `jr.energy.*` namespace. `calc/` bypasses them and imports `_src` directly.

## Key Conventions

- Internal units: Hartree (energy), Bohr (length). QE uses Rydberg — multiply by 2 when comparing.
- `chex.dataclass` for objects crossing `jax.jit` boundaries (KSampling, RuntimeContext). Standard `dataclass` for result containers.
- `_src` functions use jaxtyping shape annotations: `Float[Array, "spin kpt band x y z"]`.
- Config access: `config.method.xc`, `config.basis.cutoff_energy`, `config.solver.mode`, `config.solver.scf.max_iter`, etc.
- Pseudopotential files live in `pseudopotential/normconserving/`. The `*.pz-vbc.UPF` files are QE-compatible LDA norm-conserving PPs.

## Things to Watch Out For

- `jxc-python` (XC functional library) is sourced locally via `[tool.uv.sources]` pointing to `../jxc`. It may not be available in all environments.
- No GPU on CI — smoke tests must be CPU-compatible with small grids (grid_sizes=16, cutoff=20, k_grid=[1,1,1], epoch=2-3).
- The `_src/occupation.py` API was refactored: use `get_occupation_fn()` + `params_init()`, not the old `occupation()` / `param_init()`.
- `chatdocs/` and `.claude/` are local-only directories — never commit them.
