# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## MUST FOLLOW

### Math mode
Use markdown grammar when you are creating a markdown file, while use direct math expression display when you output to the terminal, that is better for user to read

## Project Overview

**Jrystal** - A JAX-based Differentiable Density Functional Theory (DFT) Framework for solid-state materials calculations. The project implements quantum mechanical calculations for crystalline systems using plane wave basis sets with GPU acceleration and automatic differentiation.

## High-Level Architecture

The codebase follows a modular architecture with clear separation between:

1. **Calculation Engines** (`jrystal/calc/`): Separate implementations for all-electron vs pseudopotential calculations. Each calculation type (energy vs band) has its own module.

2. **Core DFT Components** (`jrystal/`): Crystal structure, Hamiltonian, kinetic/potential energy, plane waves, Ewald summation - each concept is isolated in its own module.

3. **Pseudopotential Handling** (`jrystal/pseudopotential/`): Complete abstraction for norm-conserving and PAW (Projector Augmented Wave) pseudopotentials.

4. **Configuration System**: Type-safe configuration using `ml_collections` with `JrystalConfigDict` containing all calculation parameters.

The key architectural insight: The code routes calculations based on `use_pseudopotential` flag in config, choosing between fundamentally different computational paths (all-electron vs pseudopotential).

## Common Development Commands

```bash
# Run energy calculation
python main.py -m energy -c config.yaml

# Run band structure calculation
python main.py -m band -c config.yaml

# Format and lint code
make py-format-fix    # Auto-format with isort + yapf
make flake8          # Lint check
make mypy            # Type checking

# Documentation
make doc-build       # Build Sphinx docs
make doc-dev         # Serve docs with autobuild (localhost:8000)

# Add Apache license headers
make addlicense-fix
```

## Testing Single Components

For testing specific calculations without full pipeline:
1. Modify `config.yaml` with test parameters
2. Run with specific mode: `python main.py -m [energy|band] -c config.yaml`
3. Check output in `log/` directory for detailed logs
4. Use `-l path/to/pickle` to load previous calculation results

## Key Configuration Parameters

Edit `config.yaml` to control calculations:
- `use_pseudopotential`: Switch between all-electron (False) and pseudopotential (True)
- `pseudopotential_type`: "nc" (norm-conserving) or "paw" (PAW method)
- `cutoff_energy`: Plane wave cutoff in Hartree
- `grid_size`: Real-space grid dimensions
- `kpt_grid`: K-point sampling
- `optimizer`: Adam with configurable learning rate and scheduler
- `max_epoch`: Maximum optimization iterations
- `convergence_criterion`: Energy convergence threshold

## Current Development Focus

The `paw-minimal` branch is actively developing PAW pseudopotential support. Recent modifications in:
- `jrystal/calc/calc_paw.py`: Core PAW calculations
- `jrystal/calc/calc_ground_state_energy_paw.py`: PAW energy minimization
- Integration with existing energy/band calculation pipelines

## Important Implementation Notes

1. **Direct Optimization**: Uses direct energy minimization without SCF cycles, leveraging JAX's automatic differentiation.

2. **Parallelization**: Built-in support for GPU acceleration and custom sharding via `jrystal/_src/spmd/`.

3. **Pseudopotential Files**: Stored in `/home/aiops/zhaojx/jrystal/pseudopotential/normconserving/` or project's `pseudopotential/` directory.

4. **Coordinate Systems**: Uses fractional coordinates for atoms, reciprocal space for k-points.

5. **Energy Units**: Internal calculations in Hartree atomic units.

## Debugging Tips

- Enable JAX debugging: Set `JAX_debug_nans: true` in config
- Check convergence: Monitor `convergence_criterion_history` in output
- Visualize bands: Use `tests/*/plot.py` scripts for band structure plots
- Compare with Quantum ESPRESSO benchmarks in documentation
