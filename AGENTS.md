# Repository Guidelines

## Project Structure & Module Organization
- `jrystal/`: main Python package.
  - `jrystal/calc/`: calculation pipelines for energy/band and all-electron vs pseudopotential paths.
  - `jrystal/pseudopotential/`: PAW and norm-conserving implementations.
  - `jrystal/_src/`: lower-level kernels, tests, and experimental components.
- `docs/`: Sphinx documentation and tutorials.
- `tests/`: script-based regression examples (each folder contains `config.yaml`, `main.sh`, and `plot.py`).
- `geometry/` and `pseudopotential/`: input structures and pseudopotential data.
- `config.yaml`: default run configuration; `log/`: calculation outputs.

## Build, Test, and Development Commands
- `pip install -e .` installs Jrystal in editable mode.
- `jrystal -m energy -c config.yaml` runs a ground-state energy calculation (CLI entry point).
- `python main.py -m band -c config.yaml` runs band-structure directly via `main.py`.
- `make py-format-fix` / `make flake8` / `make mypy` format and lint the codebase.
- `make doc-build` or `make doc-dev` builds or serves Sphinx docs (`localhost:8000`).

## Coding Style & Naming Conventions
- Python uses 2-space indentation and 80-column lines (see `setup.cfg`).
- Format with `isort` + `yapf`; lint with `flake8`.
- Use `snake_case` for functions/variables and `CamelCase` for classes.
- Keep numerical kernels in `jrystal/_src/` and orchestration logic in `jrystal/calc/`.

## Testing Guidelines
- No centralized test runner is configured; tests are script-driven.
- Example: `cd tests/normconserving/al && ./main.sh` (runs `jrystal`, then `plot.py`).
- Add new tests under `tests/<type>/<system>/` with a local `config.yaml` and `main.sh`.

## Commit & Pull Request Guidelines
- Commit messages are short, lowercase, and direct (e.g., “fix ...”, “debug”); keep them concise and action-oriented.
- PRs should state the calculation mode, key config changes, and reproduction steps (command + config path).
- If results change (bands/energies), include a plot or a brief comparison note.
