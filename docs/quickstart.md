# Quickstart

This project now uses the nested config schema from `config.yaml`.

## Run From CLI

Ground-state calculation using the configured solver mode:

```bash
jrystal energy config.yaml
```

Force SCF regardless of config:

```bash
jrystal scf config.yaml
```

Force direct optimisation regardless of config:

```bash
jrystal direct-opt config.yaml
```

Band structure calculation:

```bash
jrystal band config.yaml
```

The config path is a positional argument and defaults to `config.yaml`, so these
are also valid:

```bash
jrystal energy
jrystal band
```

Any config field can be overridden from the command line:

```bash
jrystal energy config.yaml --basis.cutoff_energy=200 --solver.mode=scf --solver.scf.max_iter=50
```

## Config Layout

The main groups in the config are:

- `system`
- `method`
- `basis`
- `ksampling`
- `solver`
- `occupation`
- `ewald`
- `band`
- `execution`
- `io`

For example:

```yaml
system:
  crystal: "diamond"
  crystal_file_path: null
  spin: 0
  spin_restricted: true

method:
  xc: "lda_x"
  use_pseudopotential: false
  pseudopotential_type: "nc"

basis:
  cutoff_energy: 100
  grid_sizes: 48

ksampling:
  k_grid_sizes: [4, 4, 4]
  symmetry_reduction: true

solver:
  mode: "auto"
  auto:
    primary: "scf"
    fallback: "direct_opt"
  scf:
    max_iter: 100
  direct_opt:
    max_steps: 10000
```

## Python API

Ground-state:

```python
import jrystal as jr

config = jr.config.get_config("config.yaml")
result = jr.calc.energy(config)

print(f"Total energy: {result.total_energy:.6f} Ha")
print(f"Converged: {result.converged}")
print(result.energy_terms)
```

Band structure:

```python
import jrystal as jr

config = jr.config.get_config("config.yaml")
ground_state = jr.calc.energy(config)
band_result = jr.calc.band(config, ground_state_result=ground_state)

print(band_result.eigenvalues.shape)
print(f"Ground-state energy: {band_result.ground_state_energy:.6f} Ha")
```

`jr.calc.energy()` automatically selects the configured backend and solver mode.
`jr.calc.band()` reuses a provided ground-state result when available.
