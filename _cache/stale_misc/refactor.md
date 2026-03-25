# Jrystal Refactor Plan by GPT


## 1. Refactor Plan

:white_check_mark: Support K point symmetry reduction.
:white_check_mark: energy/potential/... support kpts weights.
:black_square_button: Add test and examples.
:black_square_button: Add documentation.
:white_check_mark: reimplement jax.fft.ifftn what can shard on GPUs.
:black_square_button: Refactor src APIs.

:black_square_button: replace jax-xc with jxc.  (?? need to improve jxc first)
:black_square_button: Add SCF related features.
:black_square_button: Add HF related features.
:white_check_mark: refactor SPMD
:black_square_button: PAW


## 2. Main Changes

- K points symmetry and weights supports. Weights normalized to 1.
- Occupation number sums to total number of electrons * k points. The way of calling occupation function is changed.


<!--
## 1. Goal and Scope

### North Star
Build `jrystal` into a production-grade, JAX-native, GPU-first plane-wave DFT platform that can eventually cover the practical scope of Quantum ESPRESSO workflows.

### Practical framing
"Do all things QE can do" is a multi-stage program, not a single refactor. The right approach is:
1. Stabilize and harden the existing core.
2. Build a modular architecture that supports feature growth.
3. Deliver QE-parity features in tiers (pw.x core first, then advanced physics and post-processing).

## 2. Current Codebase Assessment

## Strengths
- Clear focus on plane-wave periodic DFT with JAX auto-diff and GPU execution.
- Core modules exist for crystal/grid/pw/energy/hamiltonian and pseudopotential handling.
- Multiple execution workflows already exist: all-electron, norm-conserving, ultrasoft, band structure.
- Initial multi-device work exists (`jrystal/spmd/`).

## Critical quality issues (must fix first)
- Build is currently broken: `jrystal/_src/xc.py` is truncated and raises `IndentationError` at import (`jrystal/_src/xc.py:34`).
- XC potential path appears inconsistent/invalid (`jrystal/_src/potential.py:190-196`), with incompatible call signatures and conditional variable usage.
- Public API/version mismatch: package version differs between code and metadata (`jrystal/__init__.py:15` vs `pyproject.toml:8`).
- Calculator return-contract mismatch: functions annotated to return dataclasses but return raw density arrays (`jrystal/calc/calc_ground_state_energy_normcons.py:67`, `:313`, similar in other calculators).
- Zero-byte source files in active package tree:
  - `jrystal/calc/geo_opt_all_electron.py`
  - `jrystal/calc/preconditioned_sgd.py`
  - `jrystal/perturbation/hypergradient.py`
  - `jrystal/pseudopotential/ultrasoft_test.py`
- `python -m compileall jrystal` fails due syntax/indentation issues.

## High-priority maintainability issues
- Heavy duplication across calculators (`calc_ground_state_energy_*` and `calc_band_structure_*`) with diverging behavior.
- In-package backup and experiment code (`jrystal/calc/_backup`, `jrystal/pseudopotential/backup`) pollutes production package surface.
- Config schema drift: fields used in code but missing in typed config/defaults (e.g., `pseudopotential_type` used in `jrystal/calc/opt_utils.py:121` but absent in `jrystal/config.py`).
- Sample config quality problems (e.g., invalid value `parallel_over_k_path: TrFalseue` in `config_hf.yaml:139`).
- Repeated full object construction in utility helpers (`create_crystal` called repeatedly across utilities).
- Library code sets multiprocessing start method globally (`jrystal/calc/pre_calc.py:23`, `:32`), which is unsafe for host applications.

## Testing and CI gaps
- CI has lint checks only (`.github/workflows/lint.yml`), no required unit/integration performance gates.
- Several tests look like scripts/manual experiments and rely on outdated API (e.g., legacy function names in `jrystal/_src/energy_test.py`).
- Dependency-gated runtime checks fail locally without explicit environment bootstrap (`ase`, `pytest`, etc.).

## Docs/product gaps
- Docs are partially stale/incomplete (`docs/examples/scf.rst` is unfinished).
- Feature roadmap is out of sync with current tree (it says ultrasoft/SPMD are upcoming while code already includes partial implementations).
- CLI surface is minimal (`energy`, `band` only in `main.py:18`) and not aligned with QE-like workflow breadth.

## 3. QE-Parity Gap Analysis

## Broad feature tiers vs current state

### Tier A: Core pw.x parity (must-have)
- Solid SCF engine (mixing, robust convergence controls, restart/checkpoint).
- Norm-conserving and ultrasoft pseudopotentials, stable and validated.
- Spin modes: restricted/unrestricted, collinear magnetism.
- Forces and stress with geometry/cell relaxation.
- k-point, smearing, occupations, DOS/bands outputs.

Current status: partial. Direct minimization and bands exist, but architecture and reliability are not yet production-ready.

### Tier B: Advanced electronic structure parity
- Hybrid DFT, DFT+U, SOC/non-collinear magnetism.
- Better diagonalization and iterative eigensolvers.
- Improved symmetry reduction and performance scaling.

Current status: mostly missing/early.

### Tier C: Extended QE ecosystem parity
- DFPT phonons, dielectric/response, EPW-style pipelines, NEB/MD/post-processing ecosystem.

Current status: not present.

## 4. Target Architecture (Refactor End-State)

Adopt a layered architecture with strict boundaries:

1. **Domain layer**
- `Crystal`, `Lattice`, `KMesh`, `Pseudopotential`, `XCModel`, typed immutable data models.

2. **Numerical kernel layer**
- Pure JAX kernels for FFT/PW ops, Hamiltonian application, density/potential evaluation.
- No IO/logging/argparse side effects.

3. **Solver layer**
- SCF/direct-minimization/eigensolver components sharing common interfaces.
- Convergence logic, mixing/preconditioners, line search, checkpoints.

4. **Workflow layer**
- Ground-state, band structure, DOS, relaxation, phonon workflows as composable pipelines.

5. **Interface layer**
- CLI + Python API + configuration parser + provenance/result serialization.

## Key engineering constraints
- Keep JAX compilation boundaries explicit and stable-shape.
- Enforce deterministic execution modes for tests and reproducibility.
- Treat GPU memory and XLA compile time as first-class metrics.

## 5. Refactor Roadmap

## Phase 0: Stabilization and Correctness (2-4 weeks)
- Fix hard breakages (`xc.py`, `potential.py`, compile/import path).
- Define single source of truth for versioning.
- Remove/relocate backup code from installable package.
- Replace zero-byte placeholder modules with explicit stubs or real implementations.
- Add config validation (schema + defaults + strict unknown-field handling).
- Make all calculators return structured result objects consistently.

### Exit criteria
- `import jrystal` works in clean env.
- `python -m compileall jrystal` passes.
- Minimal smoke tests pass on CPU and single GPU.

## Phase 1: API and Module Unification (3-6 weeks)
- Introduce a unified `CalculationEngine` contract shared by AE/NC/USPP.
- Deduplicate calculator pipelines into reusable components.
- Separate pure compute kernels from orchestration and side effects.
- Normalize naming conventions (`effective` typo variants, logging text, argument semantics).

### Exit criteria
- Single ground-state/band workflow abstraction with backend-specific plugins.
- >50% reduction in duplicated calculator logic.

## Phase 2: SCF/Optimization Core (4-8 weeks)
- Add robust SCF loop with mixing options (Pulay/Broyden/simple).
- Keep direct minimization as an alternative backend.
- Add restart/checkpoint and convergence diagnostics.
- Add strict numerical regression baselines vs QE for representative systems.

### Exit criteria
- Reliable convergence across benchmark set (Si, Al, C, Na, Mg, selected USPP cases).
- Automated QE comparison reports (energy, forces, band edges).

## Phase 3: Physics Feature Expansion (8-16 weeks)
- Complete spin-polarized workflows and stabilize XC family support.
- Harden ultrasoft and nonlocal projectors with validated tests.
- Add forces and stress tensors with geometry relaxation.
- Improve symmetry and k-point reduction infrastructure.

### Exit criteria
- Production-ready ground-state + relaxation + bands for NC/USPP.
- Consistent force/stress validation against QE references.

## Phase 4: QE-Parity Growth Tracks (ongoing)
- Hybrid/DFT+U, SOC/non-collinear.
- DFPT/phonon and response-property workflows.
- Performance scaling: multi-GPU domain/k-point/band parallel strategies.
- Post-processing ecosystem (DOS/PDOS, projected bands, workflow artifacts).

### Exit criteria
- Defined parity matrix with pass/fail status per feature family.
- Release cadence with benchmark dashboards and compatibility matrix.

## 6. Code Quality Program (Continuous)

## Required quality gates
- Pre-commit: format + lint + type checks.
- CI: unit tests, deterministic integration tests, regression vs saved references.
- GPU CI lane for key kernels/workflows.
- Performance CI: compile time + runtime + memory budget checks.

## Test strategy
- Unit tests for every pure kernel.
- Integration tests for each workflow (AE/NC/USPP).
- Golden comparisons against QE for canonical crystals and pseudopotentials.
- Property-based tests for shape/dtype invariants.

## 7. Suggested Repository Reorganization

```text
jrystal/
  core/            # domain models and validation
  kernels/         # pure JAX kernels (pw, xc, hamiltonian, pp)
  solvers/         # scf/minimization/eigensolver/mixing
  workflows/       # energy, band, relax, dos, phonon (later)
  io/              # config, upf parsing, serialization
  cli/             # user entrypoints
  benchmarks/      # performance + QE parity scripts
  tests/
    unit/
    integration/
    regression_qe/
```

## 8. Immediate Action List (first PR series)

1. Repair import/build blockers (`xc.py`, potential-XC path, compile errors).
2. Introduce strict config model and migrate existing YAML files.
3. Replace calculator return values with typed result dataclasses.
4. Extract shared ground-state loop from duplicated calculator files.
5. Move `_backup` and exploratory scripts out of package import path.
6. Stand up baseline CI: lint + unit + one CPU integration + one GPU smoke test.
7. Publish parity benchmark table (Jrystal vs QE) for 3-5 reference systems.

## 9. Risks and Mitigations

- **Risk:** JAX/XLA compile overhead dominates iterative workflows.
  - **Mitigation:** stable shapes, cached compiled functions, controlled recompilation boundaries.
- **Risk:** Feature growth without architecture cleanup causes permanent divergence.
  - **Mitigation:** enforce layered architecture before adding major physics features.
- **Risk:** Numerical drift vs QE under different pseudopotentials/settings.
  - **Mitigation:** reference dataset with tolerance contracts per observable.

## 10. Definition of Done for "QE-like"

`jrystal` can be considered QE-like for practical pw workflows when it delivers:
- Reproducible, validated ground-state energies/forces/stresses.
- Reliable SCF and direct minimization options.
- NC/USPP production support.
- Spin-polarized and key XC modes in routine use.
- Automated regression/performance tracking and documented parity status.

This plan intentionally prioritizes correctness and architecture first; without that, adding more QE features will increase instability and technical debt. -->
