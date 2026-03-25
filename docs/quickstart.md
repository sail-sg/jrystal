# Quickstart

This project now uses the nested config schema from `config.yaml`.

## Run From CLI

Ground-state calculation:

```bash
jrystal -m energy -c config.yaml
```

Band structure calculation:

```bash
jrystal -m band -c config.yaml
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

`jr.calc.energy()` automatically selects the configured backend and solver.
`jr.calc.band()` reuses a provided ground-state result when available.
