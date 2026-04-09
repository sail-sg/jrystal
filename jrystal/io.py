"""Run output, checkpoint, and plotting helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import ase
import jax.numpy as jnp
import numpy as np
import yaml
from jax import ShapeDtypeStruct

from ._src.linalg import batched_lobpcg
from ._src.utils import expand_coefficient, squeeze_coefficient
from .calc.density_mixing import kerker_preconditioner
from .config import JrystalConfigDict, get_config
from .smearing import find_chemical_potential
from .terminal_ui import stage_warning

HARTREE_TO_EV = 27.211386245988


def _require_orbax():
  try:
    import orbax.checkpoint as ocp
  except ImportError as exc:  # pragma: no cover - dependency failure path
    raise ImportError(
      "orbax-checkpoint is required for jrystal checkpoint/restart support."
    ) from exc
  return ocp


def _config_to_dict(
  config: JrystalConfigDict | Mapping[str, Any]
) -> dict[str, Any]:
  if hasattr(config, "to_dict"):
    return config.to_dict()
  return dict(config)


def _jsonable(value: Any) -> Any:
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, (list, tuple)):
    return [_jsonable(item) for item in value]
  if isinstance(value, dict):
    return {str(key): _jsonable(item) for key, item in value.items()}
  return value


def _formula_from_crystal(crystal) -> str:
  if getattr(crystal, "symbols", None):
    return ase.Atoms(symbols=crystal.symbols).get_chemical_formula()
  return "system"


def resolve_output_root(config: JrystalConfigDict) -> Path:
  output_dir = config.io.output_dir
  if output_dir is None and config.io.save_dir is not None:
    output_dir = config.io.save_dir
  if output_dir is None:
    output_dir = "out"
  return Path(output_dir).expanduser().resolve()


def setup_output_dir(
  config: JrystalConfigDict,
  crystal,
  *,
  task: str,
) -> Path:
  """Create a per-run output directory."""
  output_root = resolve_output_root(config)
  output_root.mkdir(parents=True, exist_ok=True)

  if config.io.run_label:
    run_dir = output_root / config.io.run_label
    if run_dir.exists():
      raise FileExistsError(
        f"Output directory already exists: {run_dir}. "
        "Choose a new `io.run_label` or remove the old directory."
      )
  else:
    formula = _formula_from_crystal(crystal)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{formula}_{task}_{timestamp}"
    suffix = 1
    while run_dir.exists():
      suffix += 1
      run_dir = output_root / f"{formula}_{task}_{timestamp}_{suffix}"

  run_dir.mkdir(parents=True, exist_ok=False)
  return run_dir


def save_config_snapshot(config: JrystalConfigDict, run_dir: Path) -> None:
  path = run_dir / "config.yaml"
  with open(path, "w", encoding="utf-8") as file:
    yaml.safe_dump(
      _config_to_dict(config),
      file,
      sort_keys=False,
      allow_unicode=False,
    )


def save_run_metadata(run_dir: Path, metadata: Mapping[str, Any]) -> None:
  path = run_dir / "run.json"
  with open(path, "w", encoding="utf-8") as file:
    json.dump(_jsonable(dict(metadata)), file, indent=2, sort_keys=True)
    file.write("\n")


def _energy_terms_dict(result) -> dict[str, float]:
  return {
    key: float(value) for key, value in asdict(result.energy_terms).items()
  }


def _infer_fermi_energy(
  result,
  *,
  k_weights,
) -> float | None:
  if result.eigenvalues is None or result.occupations is None:
    return None

  eigenvalues = jnp.asarray(result.eigenvalues)
  occupations = jnp.asarray(result.occupations)
  smearing = float(result.config.occupation.smearing)
  num_electrons = int(np.asarray(result.crystal.num_electron))
  occ_array = np.asarray(occupations)
  eig_array = np.asarray(eigenvalues)
  occ_max = 2.0 if bool(result.config.system.spin_restricted) else 1.0
  occ_tol = max(occ_max * 1e-3, 1e-8)

  partially_occupied = (
    (occ_array > occ_tol) & (occ_array < (occ_max - occ_tol))
  )
  occupied_mask = occ_array > (0.5 * occ_max)
  empty_mask = occ_array <= (0.5 * occ_max)
  occupied = eig_array[occupied_mask]
  empty = eig_array[empty_mask]

  if (occupied.size > 0 and empty.size > 0 and not np.any(partially_occupied)):
    return float(np.max(occupied))

  if smearing > 0:
    return float(
      find_chemical_potential(
        eigenvalues,
        num_electrons,
        smearing=smearing,
        k_weights=jnp.asarray(k_weights),
      )
    )

  occ_mask = occ_array > occ_tol
  occupied = eig_array[occ_mask]
  if occupied.size == 0:
    return None
  return float(np.max(occupied))


def _convergence_payload(result) -> dict[str, Any]:
  if result.convergence_history:
    columns = list(result.convergence_history[0].keys())
    units = []
    for column in columns:
      if column in {"step"}:
        units.append("")
      elif column in {"delta_density"}:
        units.append("")
      elif column in {"wall_time"} or column.endswith("_s"):
        units.append("s")
      else:
        units.append("Ha")
    data = [
      [_jsonable(record.get(column))
       for column in columns]
      for record in result.convergence_history
    ]
  else:
    columns = ["step", "total_energy"]
    units = ["", "Ha"]
    data = [
      [idx + 1, float(value)]
      for idx, value in enumerate(result.total_energy_history)
    ]

  return {
    "solver": result.actual_solver,
    "columns": columns,
    "units": units,
    "data": data,
  }


def _ground_state_energy_payload(result, *, k_weights) -> dict[str, Any]:
  fermi_energy = result.fermi_energy
  if fermi_energy is None:
    fermi_energy = _infer_fermi_energy(result, k_weights=k_weights)

  return {
    "total_energy_ha": float(result.total_energy),
    "total_energy_ev": float(result.total_energy * HARTREE_TO_EV),
    "decomposition": _energy_terms_dict(result),
    "chemical_potential_ha": (
      None if fermi_energy is None else float(fermi_energy)
    ),
    "fermi_energy_ha": None if fermi_energy is None else float(fermi_energy),
    "fermi_energy_available": fermi_energy is not None,
    "num_electrons": int(np.asarray(result.crystal.num_electron)),
    "requested_solver_mode": result.requested_solver_mode,
    "actual_solver": result.actual_solver,
    "converged": bool(result.converged),
    "num_iterations": int(result.num_iterations),
    "wall_time_sec": float(result.wall_time),
  }


def _coefficients_from_result(result) -> dict[str, np.ndarray] | None:
  source = result.coefficients if result.coefficients is not None else result.params_pw
  if not source:
    return None
  if "w_re" not in source or "w_im" not in source:
    return None
  return {
    "w_re": np.asarray(source["w_re"]),
    "w_im": np.asarray(source["w_im"]),
  }


def _compute_ground_state_spectrum(config, ctx, backend, result):
  if result.eigenvalues is not None:
    return result

  coeffs = _coefficients_from_result(result)
  if coeffs is None:
    return result

  freq_mask = ctx.basis.freq_mask
  coeff_guess = jnp.asarray(coeffs["w_re"]) + 1.0j * jnp.asarray(coeffs["w_im"])
  density = jnp.asarray(result.density)
  iteration_state = backend.prepare_iteration(density, ctx)
  precond = kerker_preconditioner(ctx.g_vec, freq_mask)
  lobpcg_max_iter = config.solver.scf.eigensolver.max_iter
  s, k, g, b = coeff_guess.shape

  def _hvp(coeff_compact):
    coeff_full = expand_coefficient(coeff_compact.conj(), freq_mask)
    hpsi_full = backend.hamiltonian_apply(coeff_full, iteration_state, ctx)
    return squeeze_coefficient(hpsi_full, freq_mask)

  def _svp(c):
    coeff_batch = c.reshape(s, k, g, -1)
    coeff_full = expand_coefficient(coeff_batch.conj(), freq_mask)
    spsi_full = backend.overlap_apply(coeff_full, ctx)
    return squeeze_coefficient(spsi_full.conj(), freq_mask).reshape(s * k, g, -1)

  def _matmul(c):
    coeff_batch = c.reshape(s, k, g, -1)
    return _hvp(coeff_batch).reshape(s * k, g, -1)

  eigval, eigvec = batched_lobpcg(
    matmul=_matmul,
    b_matmul=_svp,
    k=b,
    v0=coeff_guess.reshape(s * k, g, b),
    which="smallest",
    preconditioner=precond,
    maxit=lobpcg_max_iter,
    tol=1e-8,
  )
  coeff_compact = eigvec.reshape(s, k, g, b).conj()
  return replace(
    result,
    coefficients={
      "w_re": coeff_compact.real,
      "w_im": coeff_compact.imag,
    },
    eigenvalues=eigval.reshape(s, k, b),
  )


def save_ground_state(
  config: JrystalConfigDict,
  result,
  run_dir: Path,
  *,
  ctx=None,
  backend=None,
):
  """Persist ground-state outputs under ``ground_state/``."""
  ground_state_dir = run_dir / "ground_state"
  ground_state_dir.mkdir(parents=True, exist_ok=True)

  if config.io.save_ground_state_spectrum and result.eigenvalues is None and (
    ctx is not None and backend is not None
  ):
    result = _compute_ground_state_spectrum(config, ctx, backend, result)

  fermi_energy = result.fermi_energy
  if fermi_energy is None:
    fermi_energy = _infer_fermi_energy(result, k_weights=ctx.ksampling.weights)
  if fermi_energy is not None and result.fermi_energy is None:
    result = replace(result, fermi_energy=fermi_energy)

  with open(ground_state_dir / "energy.json", "w", encoding="utf-8") as file:
    json.dump(
      _ground_state_energy_payload(result, k_weights=ctx.ksampling.weights),
      file,
      indent=2,
      sort_keys=True,
    )
    file.write("\n")

  with open(
    ground_state_dir / "convergence.json",
    "w",
    encoding="utf-8",
  ) as file:
    json.dump(_convergence_payload(result), file, indent=2, sort_keys=True)
    file.write("\n")

  if config.io.save_density:
    np.save(ground_state_dir / "density.npy", np.asarray(result.density))

  if config.io.save_ground_state_spectrum and result.eigenvalues is not None:
    np.save(
      ground_state_dir / "eigenvalues.npy",
      np.asarray(result.eigenvalues),
    )
    if result.occupations is not None:
      np.save(
        ground_state_dir / "occupations.npy",
        np.asarray(result.occupations),
      )

  if config.io.save_wavefunction:
    coeffs = _coefficients_from_result(result)
    if coeffs is not None:
      np.savez(ground_state_dir / "coefficients.npz", **coeffs)

  return result


def _kpath_payload(result) -> dict[str, Any]:
  return {
    "mode": result.kpath.mode,
    "kpts": np.asarray(result.kpath.kpts).tolist(),
    "weights": np.asarray(result.kpath.weights).tolist(),
    "labels": result.kpath.labels,
    "segments": result.kpath.segments,
  }


def save_band_structure(
  config: JrystalConfigDict,
  result,
  run_dir: Path,
) -> None:
  band_dir = run_dir / "band"
  band_dir.mkdir(parents=True, exist_ok=True)
  np.save(band_dir / "eigenvalues.npy", np.asarray(result.eigenvalues))
  with open(band_dir / "kpath.json", "w", encoding="utf-8") as file:
    json.dump(_kpath_payload(result), file, indent=2, sort_keys=True)
    file.write("\n")

  if config.io.save_dir is not None:
    legacy_name = "".join(result.crystal.symbols or []) + "_band_structure.npy"
    np.save(
      resolve_output_root(config) / legacy_name, np.asarray(result.eigenvalues)
    )

  if not config.band.plot.enabled:
    return

  try:
    from .plot import band_structure as plot_band_structure
  except ImportError:
    stage_warning("Band", "matplotlib not installed — skipping plot")
    return

  try:
    plot_band_structure(
      result,
      reference_energy=result.reference_energy,
      unit=config.band.plot.unit,
      y_min=config.band.plot.y_min,
      y_max=config.band.plot.y_max,
      save_path=band_dir / "band_structure.pdf",
    )
  except Exception as exc:  # pragma: no cover - plotting failure path
    stage_warning("Band", f"Failed to save band plot: {exc}")


def make_checkpoint_manager(run_dir: Path):
  ocp = _require_orbax()
  ckpt_dir = run_dir / "checkpoint"
  ckpt_dir.mkdir(parents=True, exist_ok=True)
  return ocp.CheckpointManager(
    ckpt_dir,
    options=ocp.CheckpointManagerOptions(max_to_keep=2),
  )


def save_checkpoint(
  manager, physical_state: Mapping[str, Any], step: int
) -> None:
  ocp = _require_orbax()
  from . import __version__

  manager.save(step, args=ocp.args.StandardSave(dict(physical_state)))
  meta_file = Path(manager.directory) / "meta.json"
  if not meta_file.exists():
    with open(meta_file, "w", encoding="utf-8") as file:
      json.dump(
        {
          "jrystal_version": __version__,
          "format_version": 1,
        },
        file,
        indent=2,
        sort_keys=True,
      )
      file.write("\n")


def make_abstract_state(config: JrystalConfigDict, ctx) -> dict[str, Any]:
  num_spin = 1 if config.system.spin_restricted else 2
  grid = tuple(int(size) for size in ctx.basis.grid_sizes)
  num_kpts = int(ctx.ksampling.kpts.shape[0])
  num_bands = int(
    np.ceil(
      float(np.asarray(ctx.crystal.num_electron)) /
      (2.0 if config.system.spin_restricted else 1.0)
    )
  ) + int(config.occupation.empty_bands)
  num_g = int(ctx.basis.num_g)
  return {
    "density":
      ShapeDtypeStruct((num_spin, *grid), jnp.float32),
    "coefficients":
      {
        "w_re":
          ShapeDtypeStruct((num_spin, num_kpts, num_g, num_bands), jnp.float32),
        "w_im":
          ShapeDtypeStruct((num_spin, num_kpts, num_g, num_bands), jnp.float32),
      },
    "occupations":
      ShapeDtypeStruct((num_spin, num_kpts, num_bands), jnp.float32),
    "eigenvalues":
      ShapeDtypeStruct((num_spin, num_kpts, num_bands), jnp.float32),
    "has_eigenvalues":
      False,
    "step":
      0,
    "total_energy":
      0.0,
  }


def _load_previous_config(prev_config_path: Path) -> JrystalConfigDict:
  if not prev_config_path.exists():
    raise FileNotFoundError(
      f"Missing frozen config for restart: {prev_config_path}"
    )
  return get_config(str(prev_config_path))


def _compare_restart_field(
  path: str, prev_value: Any, current_value: Any
) -> None:
  if prev_value != current_value:
    raise ValueError(
      f"Restart incompatibility at `{path}`: previous={prev_value!r}, current={current_value!r}."
    )


def _validate_restart_compatibility(
  prev_config_path: Path,
  current_config: JrystalConfigDict,
) -> None:
  prev_config = _load_previous_config(prev_config_path)

  _compare_restart_field(
    "basis.cutoff_energy",
    prev_config.basis.cutoff_energy,
    current_config.basis.cutoff_energy,
  )
  _compare_restart_field(
    "basis.grid_sizes",
    tuple(np.asarray(prev_config.basis.grid_sizes).tolist())
    if isinstance(prev_config.basis.grid_sizes,
                  (list, tuple)) else prev_config.basis.grid_sizes,
    tuple(np.asarray(current_config.basis.grid_sizes).tolist())
    if isinstance(current_config.basis.grid_sizes,
                  (list, tuple)) else current_config.basis.grid_sizes,
  )
  _compare_restart_field(
    "ksampling.k_grid_sizes",
    tuple(prev_config.ksampling.k_grid_sizes),
    tuple(current_config.ksampling.k_grid_sizes),
  )
  _compare_restart_field(
    "system.spin",
    prev_config.system.spin,
    current_config.system.spin,
  )
  _compare_restart_field(
    "system.spin_restricted",
    prev_config.system.spin_restricted,
    current_config.system.spin_restricted,
  )
  _compare_restart_field(
    "occupation.empty_bands",
    prev_config.occupation.empty_bands,
    current_config.occupation.empty_bands,
  )
  _compare_restart_field(
    "method.pseudopotential_type",
    prev_config.method.pseudopotential_type,
    current_config.method.pseudopotential_type,
  )

  from .calc.opt_utils import create_crystal

  prev_crystal = create_crystal(prev_config)
  current_crystal = create_crystal(current_config)
  _compare_restart_field(
    "system.num_atoms",
    int(prev_crystal.num_atom),
    int(current_crystal.num_atom),
  )
  _compare_restart_field(
    "system.species",
    tuple(prev_crystal.symbols or []),
    tuple(current_crystal.symbols or []),
  )


def load_checkpoint(
  restart_path: str,
  config: JrystalConfigDict,
  ctx,
) -> tuple[dict[str, Any], int]:
  ocp = _require_orbax()
  run_dir = Path(restart_path)
  ckpt_dir = run_dir / "checkpoint"
  if not (ckpt_dir / "meta.json").exists():
    raise FileNotFoundError(
      f"No checkpoint found at {ckpt_dir}. Cannot restart from {restart_path}."
    )

  _validate_restart_compatibility(run_dir / "config.yaml", config)
  manager = ocp.CheckpointManager(ckpt_dir)
  step = manager.latest_step()
  if step is None:
    raise FileNotFoundError(f"No checkpoint steps found in {ckpt_dir}.")
  abstract_state = make_abstract_state(config, ctx)
  physical_state = manager.restore(
    step,
    args=ocp.args.StandardRestore(abstract_state),
  )
  return physical_state, int(step)
