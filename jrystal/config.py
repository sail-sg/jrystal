# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
import warnings
from typing import Any, Mapping, Optional

import yaml
from ml_collections import ConfigDict


class JrystalConfigDict(ConfigDict):
  """Typed wrapper for Jrystal configuration."""


default_config = {
  "schema_version": 1,
  "system":
    {
      "crystal": "diamond",
      "crystal_file_path": None,
      "spin": 0,
      "spin_restricted": True,
    },
  "method":
    {
      "xc": "lda_x+lda_c_pw",
      "family": "ae",
      "pseudopotential_file_dir": None,
    },
  "basis":
    {
      "freq_mask_method": "spherical",
      "cutoff_energy": 100,
      "grid_sizes": 48,
    },
  "ksampling": {
    "k_grid_sizes": [4, 4, 4],
    "symmetry_reduction": True,
  },
  "solver":
    {
      "mode": "auto",
      "auto":
        {
          "primary": "scf",
          "fallback": "direct_opt",
          "fallback_on_nonconverged": True,
          "fallback_on_error": True,
        },
      "scf":
        {
          "max_iter": 100,
          "eigensolver": {
            "method": "lobpcg",
            "max_iter": 6,
          },
          "mixing": {
            "method": "diis",
            "beta": 0.8,
            "history_size": 8,
          },
          "convergence": {
            "density_tol": 1e-3,
            "energy_tol": 1e-6,
          },
        },
      "direct_opt":
        {
          "max_steps": 10000,
          "canonical_transform": False,
          "optimizer":
            {
              "name": "adam",
              "learning_rate": 0.01,
              "b1": 0.9,
              "b2": 0.99,
            },
          "scheduler": None,
          "occupation_optimizer":
            {
              "warmup_steps": 500,
              "learning_rate": 0.001,
              "scheduler": None,
            },
          "convergence": {
            "window_size": 20,
            "energy_std_tol": 1e-6,
          },
        },
    },
  "occupation": {
    "method": "fermi-dirac",
    "smearing": 0.0,
    "empty_bands": 20,
  },
  "ewald": {
    "eta": 0.1,
    "cutoff": 2e4,
  },
  "band":
    {
      "empty_bands": 20,
      "k_path_special_points": None,
      "num_kpoints": 64,
      "k_path_file": None,
      "epoch": 5000,
      "fine_tuning": True,
      "fine_tuning_epoch": 300,
      "eigensolver_max_iter": 100,
      "plot": {
        "enabled": True,
        "unit": "eV",
        "y_min": None,
        "y_max": None,
      },
    },
  "execution":
    {
      "seed": 123,
      "parallel_over_k_mesh": False,
      "parallel_over_k_path": True,
      "profile": False,
      "compile_cache": True,
      "compile_cache_dir": "~/.cache/jrystal/jax_compile_cache",
      "xla_preallocate": True,
      "jax_enable_x64": True,
      "jax_debug_nans": False,
      "verbose": True,
      "eps": 1e-8,
    },
  "experimental": {
    "disable_jit": False,
  },
  "io":
    {
      "output_dir": "out/",
      "run_label": None,
      "save_density": True,
      "save_wavefunction": False,
      "save_ground_state_spectrum": False,
      "save_checkpoint": True,
      "checkpoint_interval": 50,
      "restart": "from_scratch",
      "log_level": "normal",
    },
}

_GROUP_FIELDS = {
  "system": {
    "crystal",
    "crystal_file_path",
    "spin",
    "spin_restricted",
  },
  "method":
    {
      "xc",
      "family",
      "use_pseudopotential",
      "pseudopotential_type",
      "pseudopotential_file_dir",
    },
  "basis":
    {
      "freq_mask_method",
      "cutoff_energy",
      "cutoff_energy_ha",
      "grid_sizes",
    },
  "ksampling": {
    "k_grid_sizes",
    "symmetry_reduction",
  },
  "solver":
    {
      "mode",
      "auto",
      "scf",
      "direct_opt",
      "type",
      "optimizer",
      "optimizer_args",
      "scheduler",
      "occupation_optimizer",
      "epoch",
      "scf_max_iter",
      "scf_max_iteration",
      "lobpcg_max_iter",
      "mixing_beta",
      "diis_max_hist",
      "convergence_window_size",
      "convergence_condition",
    },
  "occupation":
    {
      "method",
      "smearing",
      "empty_bands",
      "warmup_steps",
      "warmup_step",
      "learning_rate",
      "scheduler",
    },
  "ewald": {
    "eta",
    "cutoff",
  },
  "band":
    {
      "empty_bands",
      "k_path_special_points",
      "num_kpoints",
      "k_path_file",
      "epoch",
      "fine_tuning",
      "fine_tuning_epoch",
      "eigensolver_max_iter",
      "plot",
    },
  "execution":
    {
      "seed",
      "parallel_over_k_mesh",
      "parallel_over_k_path",
      "parallel_over_k",
      "profile",
      "compile_cache",
      "compile_cache_dir",
      "xla_preallocate",
      "jax_enable_x64",
      "jax_debug_nans",
      "verbose",
      "eps",
    },
  "experimental": {"disable_jit",},
  "io":
    {
      "output_dir",
      "save_dir",
      "run_label",
      "save_density",
      "save_wavefunction",
      "save_ground_state_spectrum",
      "save_checkpoint",
      "checkpoint_interval",
      "restart",
      "save_band_plot",
      "log_level",
    },
}

_SOLVER_AUTO_FIELDS = {
  "primary",
  "fallback",
  "fallback_on_nonconverged",
  "fallback_on_error",
}

_SOLVER_SCF_FIELDS = {
  "max_iter",
  "scf_max_iter",
  "scf_max_iteration",
  "lobpcg_max_iter",
  "mixing_beta",
  "diis_max_hist",
  "convergence_condition",
  "eigensolver",
  "mixing",
  "convergence",
}

_SOLVER_SCF_EIGENSOLVER_FIELDS = {
  "method",
  "max_iter",
  "lobpcg_max_iter",
}

_SOLVER_SCF_MIXING_FIELDS = {
  "method",
  "beta",
  "mixing_beta",
  "history_size",
  "diis_max_hist",
}

_SOLVER_SCF_CONVERGENCE_FIELDS = {
  "density_tol",
  "energy_tol",
  "convergence_condition",
}

_SOLVER_DIRECT_OPT_FIELDS = {
  "max_steps",
  "canonical_transform",
  "epoch",
  "optimizer",
  "optimizer_args",
  "scheduler",
  "occupation_optimizer",
  "convergence",
  "convergence_window_size",
  "convergence_condition",
}

_SOLVER_DIRECT_OPT_OCCUPATION_OPT_FIELDS = {
  "warmup_steps",
  "warmup_step",
  "learning_rate",
  "scheduler",
}

_SOLVER_DIRECT_OPT_CONVERGENCE_FIELDS = {
  "window_size",
  "energy_std_tol",
  "convergence_window_size",
  "convergence_condition",
}

_OCCUPATION_SCHEDULER_FIELDS = {
  "name",
  "factor",
  "patience",
  "rtol",
  "atol",
  "cooldown",
  "accumulation_size",
  "min_scale",
}

_DEFAULT_OCCUPATION_SCHEDULER = {
  "name": "reduce_on_plateau",
  "factor": 0.5,
  "patience": 50,
  "rtol": 1e-4,
  "atol": 0.0,
  "cooldown": 20,
  "accumulation_size": 1,
  "min_scale": 1e-2,
}

_BAND_PLOT_FIELDS = {
  "enabled",
  "unit",
  "y_min",
  "y_max",
}

_LEGACY_FIELD_MAP = {
  "crystal": ("system", "crystal"),
  "crystal_file_path": ("system", "crystal_file_path"),
  "crystal_file_path_path": ("system", "crystal_file_path"),
  "spin": ("system", "spin"),
  "spin_restricted": ("system", "spin_restricted"),
  "xc": ("method", "xc"),
  "family": ("method", "family"),
  "pseudopotential_file_dir": ("method", "pseudopotential_file_dir"),
  "freq_mask_method": ("basis", "freq_mask_method"),
  "cutoff_energy": ("basis", "cutoff_energy"),
  "cutoff_energy_ha": ("basis", "cutoff_energy"),
  "grid_sizes": ("basis", "grid_sizes"),
  "k_grid_sizes": ("ksampling", "k_grid_sizes"),
  "symmetry_reduction": ("ksampling", "symmetry_reduction"),
  "type": ("solver", "mode"),
  "epoch": ("solver", "direct_opt", "max_steps"),
  "scf_max_iter": ("solver", "scf", "max_iter"),
  "scf_max_iteration": ("solver", "scf", "max_iter"),
  "lobpcg_max_iter": ("solver", "scf", "eigensolver", "max_iter"),
  "mixing_beta": ("solver", "scf", "mixing", "beta"),
  "diis_max_hist": ("solver", "scf", "mixing", "history_size"),
  "optimizer": ("solver", "direct_opt", "optimizer", "name"),
  "optimizer_args": ("solver", "direct_opt", "optimizer"),
  "scheduler": ("solver", "direct_opt", "scheduler"),
  "convergence_window_size":
    (
      "solver",
      "direct_opt",
      "convergence",
      "window_size",
    ),
  "convergence_condition":
    (
      "solver",
      "direct_opt",
      "convergence",
      "energy_std_tol",
    ),
  "occupation": ("occupation", "method"),
  "smearing": ("occupation", "smearing"),
  "empty_bands": ("occupation", "empty_bands"),
  "occupation_warmup_steps":
    ("solver", "direct_opt", "occupation_optimizer", "warmup_steps"),
  "occupation_warmup_step":
    ("solver", "direct_opt", "occupation_optimizer", "warmup_steps"),
  "occupation_learning_rate":
    ("solver", "direct_opt", "occupation_optimizer", "learning_rate"),
  "band_structure_empty_bands": ("band", "empty_bands"),
  "k_path_special_points": ("band", "k_path_special_points"),
  "num_kpoints": ("band", "num_kpoints"),
  "k_path_file": ("band", "k_path_file"),
  "band_structure_epoch": ("band", "epoch"),
  "k_path_fine_tuning": ("band", "fine_tuning"),
  "k_path_fine_tuning_epoch": ("band", "fine_tuning_epoch"),
  "band_eigensolver_max_iter": ("band", "eigensolver_max_iter"),
  "seed": ("execution", "seed"),
  "parallel_over_k": ("execution", "parallel_over_k"),
  "parallel_over_k_mesh": ("execution", "parallel_over_k_mesh"),
  "parallel_over_k_path": ("execution", "parallel_over_k_path"),
  "profile": ("execution", "profile"),
  "compile_cache": ("execution", "compile_cache"),
  "compile_cache_dir": ("execution", "compile_cache_dir"),
  "xla_preallocate": ("execution", "xla_preallocate"),
  "jax_enable_x64": ("execution", "jax_enable_x64"),
  "jax_debug_nans": ("execution", "jax_debug_nans"),
  "verbose": ("execution", "verbose"),
  "eps": ("execution", "eps"),
  "disable_jit": ("experimental", "disable_jit"),
  "jax_disable_jit": ("experimental", "disable_jit"),
  "output_dir": ("io", "output_dir"),
  "save_dir": ("io", "output_dir"),
  "run_label": ("io", "run_label"),
  "save_density": ("io", "save_density"),
  "save_wavefunction": ("io", "save_wavefunction"),
  "save_ground_state_spectrum": ("io", "save_ground_state_spectrum"),
  "save_checkpoint": ("io", "save_checkpoint"),
  "checkpoint_interval": ("io", "checkpoint_interval"),
  "restart": ("io", "restart"),
  "save_band_plot": ("band", "plot", "enabled"),
  "log_level": ("io", "log_level"),
}


def _set_nested_value(
  config: dict[str, Any], path: tuple[str, ...], value: Any
):
  target = config
  for part in path[:-1]:
    target = target[part]
  target[path[-1]] = copy.deepcopy(value)


def _deep_merge(target: dict[str, Any], updates: Mapping[str, Any]) -> None:
  for key, value in updates.items():
    if isinstance(value, Mapping) and isinstance(target.get(key), dict):
      _deep_merge(target[key], value)
    else:
      target[key] = copy.deepcopy(value)


def _warn_unknown_fields(config: Mapping[str, Any]) -> None:
  for key, value in config.items():
    if key == "schema_version":
      continue

    if key in _GROUP_FIELDS and isinstance(value, Mapping):
      if key == "solver":
        _warn_unknown_solver_fields(value)
        continue
      if key == "band":
        _warn_unknown_band_fields(value)
        continue
      if key == "occupation":
        _warn_unknown_occupation_fields(value)
        continue
      unknown_fields = sorted(set(value) - _GROUP_FIELDS[key])
      for unknown_field in unknown_fields:
        warnings.warn(
          f"Unknown config field: {key}.{unknown_field}",
          stacklevel=3,
        )
      continue

    if key not in _LEGACY_FIELD_MAP and key != "ewald_args":
      warnings.warn(f"Unknown config field: {key}", stacklevel=3)


def _warn_unknown_solver_fields(solver_config: Mapping[str, Any]) -> None:
  unknown_fields = sorted(set(solver_config) - _GROUP_FIELDS["solver"])
  for unknown_field in unknown_fields:
    warnings.warn(
      f"Unknown config field: solver.{unknown_field}",
      stacklevel=4,
    )

  auto_config = solver_config.get("auto")
  if isinstance(auto_config, Mapping):
    for unknown_field in sorted(set(auto_config) - _SOLVER_AUTO_FIELDS):
      warnings.warn(
        f"Unknown config field: solver.auto.{unknown_field}",
        stacklevel=4,
      )

  scf_config = solver_config.get("scf")
  if isinstance(scf_config, Mapping):
    for unknown_field in sorted(set(scf_config) - _SOLVER_SCF_FIELDS):
      warnings.warn(
        f"Unknown config field: solver.scf.{unknown_field}",
        stacklevel=4,
      )
    eigensolver = scf_config.get("eigensolver")
    if isinstance(eigensolver, Mapping):
      for unknown_field in sorted(
        set(eigensolver) - _SOLVER_SCF_EIGENSOLVER_FIELDS,
      ):
        warnings.warn(
          f"Unknown config field: solver.scf.eigensolver.{unknown_field}",
          stacklevel=4,
        )
    mixing = scf_config.get("mixing")
    if isinstance(mixing, Mapping):
      for unknown_field in sorted(set(mixing) - _SOLVER_SCF_MIXING_FIELDS):
        warnings.warn(
          f"Unknown config field: solver.scf.mixing.{unknown_field}",
          stacklevel=4,
        )
    convergence = scf_config.get("convergence")
    if isinstance(convergence, Mapping):
      for unknown_field in sorted(
        set(convergence) - _SOLVER_SCF_CONVERGENCE_FIELDS,
      ):
        warnings.warn(
          f"Unknown config field: solver.scf.convergence.{unknown_field}",
          stacklevel=4,
        )

  direct_opt_config = solver_config.get("direct_opt")
  if isinstance(direct_opt_config, Mapping):
    for unknown_field in sorted(
      set(direct_opt_config) - _SOLVER_DIRECT_OPT_FIELDS,
    ):
      warnings.warn(
        f"Unknown config field: solver.direct_opt.{unknown_field}",
        stacklevel=4,
      )
    convergence = direct_opt_config.get("convergence")
    if isinstance(convergence, Mapping):
      for unknown_field in sorted(
        set(convergence) - _SOLVER_DIRECT_OPT_CONVERGENCE_FIELDS,
      ):
        warnings.warn(
          f"Unknown config field: solver.direct_opt.convergence.{unknown_field}",
          stacklevel=4,
        )
    occupation_optimizer = direct_opt_config.get("occupation_optimizer")
    if isinstance(occupation_optimizer, Mapping):
      for unknown_field in sorted(
        set(occupation_optimizer) - _SOLVER_DIRECT_OPT_OCCUPATION_OPT_FIELDS
      ):
        warnings.warn(
          "Unknown config field: "
          f"solver.direct_opt.occupation_optimizer.{unknown_field}",
          stacklevel=4,
        )


def _warn_unknown_band_fields(band_config: Mapping[str, Any]) -> None:
  unknown_fields = sorted(set(band_config) - _GROUP_FIELDS["band"])
  for unknown_field in unknown_fields:
    warnings.warn(
      f"Unknown config field: band.{unknown_field}",
      stacklevel=4,
    )

  plot_config = band_config.get("plot")
  if isinstance(plot_config, Mapping):
    for unknown_field in sorted(set(plot_config) - _BAND_PLOT_FIELDS):
      warnings.warn(
        f"Unknown config field: band.plot.{unknown_field}",
        stacklevel=4,
      )


def _warn_unknown_occupation_fields(
  occupation_config: Mapping[str, Any]
) -> None:
  unknown_fields = sorted(set(occupation_config) - _GROUP_FIELDS["occupation"])
  for unknown_field in unknown_fields:
    warnings.warn(
      f"Unknown config field: occupation.{unknown_field}",
      stacklevel=4,
    )

  scheduler_config = occupation_config.get("scheduler")
  if isinstance(scheduler_config, Mapping):
    for unknown_field in sorted(
      set(scheduler_config) - _OCCUPATION_SCHEDULER_FIELDS,
    ):
      warnings.warn(
        f"Unknown config field: occupation.scheduler.{unknown_field}",
        stacklevel=4,
      )


def _apply_legacy_field(
  config: dict[str, Any],
  key: str,
  value: Any,
) -> None:
  if key == "use_pseudopotential":
    if not isinstance(value, bool):
      raise TypeError("Config field `use_pseudopotential` must be a bool.")
    config["method"]["family"] = "nc" if value else "ae"
    return

  if key == "pseudopotential_type":
    config["method"]["family"] = _normalize_method_family(value)
    return

  if key == "ewald_args":
    if not isinstance(value, Mapping):
      raise TypeError("Config field `ewald_args` must be a mapping.")
    if "ewald_eta" in value:
      config["ewald"]["eta"] = copy.deepcopy(value["ewald_eta"])
    if "ewald_cutoff" in value:
      config["ewald"]["cutoff"] = copy.deepcopy(value["ewald_cutoff"])
    unknown_fields = sorted(set(value) - {"ewald_eta", "ewald_cutoff"})
    for unknown_field in unknown_fields:
      warnings.warn(
        f"Unknown config field: ewald_args.{unknown_field}",
        stacklevel=3,
      )
    return

  if key == "parallel_over_k":
    config["execution"]["parallel_over_k_mesh"] = copy.deepcopy(value)
    config["execution"]["parallel_over_k_path"] = copy.deepcopy(value)
    return

  if key == "optimizer_args":
    if not isinstance(value, Mapping):
      raise TypeError("Config field `optimizer_args` must be a mapping.")
    _deep_merge(config["solver"]["direct_opt"]["optimizer"], value)
    return

  if key == "convergence_condition":
    config["solver"]["direct_opt"]["convergence"]["energy_std_tol"] = (
      copy.deepcopy(value)
    )
    config["solver"]["scf"]["convergence"]["energy_tol"] = copy.deepcopy(value,)
    return

  _set_nested_value(config, _LEGACY_FIELD_MAP[key], value)


def _normalize_solver_mode(value: Any) -> Any:
  if value == "direct":
    return "direct_opt"
  return value


def _normalize_method_family(value: Any) -> str:
  if not isinstance(value, str):
    raise TypeError("Config field `method.family` must be a string.")
  normalized = value.lower().replace("-", "_")
  if normalized in {"ae", "all_electron", "allelectron"}:
    return "ae"
  if normalized in {"nc", "normcons", "normconserving"}:
    return "nc"
  if normalized in {"us", "ultrasoft"}:
    return "us"
  raise ValueError(
    "Config field `method.family` must be one of 'ae', 'nc', or 'us'."
  )


def _normalize_method_group(
  target: dict[str, Any],
  group_value: Mapping[str, Any],
) -> None:
  method_value = copy.deepcopy(dict(group_value))
  family = method_value.pop("family", None)
  use_pseudopotential = method_value.pop("use_pseudopotential", None)
  pseudopotential_type = method_value.pop("pseudopotential_type", None)

  explicit_family = (
    _normalize_method_family(family) if family is not None else None
  )
  legacy_family = None
  if use_pseudopotential is not None:
    if not isinstance(use_pseudopotential, bool):
      raise TypeError(
        "Config field `method.use_pseudopotential` must be a bool."
      )
    if not use_pseudopotential:
      legacy_family = "ae"
    else:
      legacy_family = (
        _normalize_method_family(pseudopotential_type)
        if pseudopotential_type is not None else "nc"
      )
  elif pseudopotential_type is not None:
    legacy_family = _normalize_method_family(pseudopotential_type)

  if explicit_family is not None and legacy_family is not None and (
    explicit_family != legacy_family
  ):
    raise ValueError(
      "Config fields `method.family` and legacy "
      "`method.use_pseudopotential`/`method.pseudopotential_type` "
      "must agree."
    )

  if explicit_family is not None:
    target["family"] = explicit_family
  elif legacy_family is not None:
    target["family"] = legacy_family

  _deep_merge(target, method_value)


def _normalize_solver_scf_group(
  target: dict[str, Any],
  group_value: Mapping[str, Any],
) -> None:
  scf_value = copy.deepcopy(dict(group_value))
  scf_max_iteration = scf_value.pop("scf_max_iteration", None)
  if scf_max_iteration is not None and "max_iter" not in scf_value:
    scf_value["max_iter"] = scf_max_iteration
  scf_max_iter = scf_value.pop("scf_max_iter", None)
  if scf_max_iter is not None and "max_iter" not in scf_value:
    scf_value["max_iter"] = scf_max_iter

  eigensolver = scf_value.pop("eigensolver", None)
  if eigensolver is not None:
    if not isinstance(eigensolver, Mapping):
      raise TypeError(
        "Config group `solver.scf.eigensolver` must be a mapping."
      )
    eigensolver_value = copy.deepcopy(dict(eigensolver))
    lobpcg_max_iter = eigensolver_value.pop("lobpcg_max_iter", None)
    if lobpcg_max_iter is not None and "max_iter" not in eigensolver_value:
      eigensolver_value["max_iter"] = lobpcg_max_iter
    _deep_merge(target["eigensolver"], eigensolver_value)

  mixing = scf_value.pop("mixing", None)
  if mixing is not None:
    if not isinstance(mixing, Mapping):
      raise TypeError("Config group `solver.scf.mixing` must be a mapping.")
    mixing_value = copy.deepcopy(dict(mixing))
    mixing_beta = mixing_value.pop("mixing_beta", None)
    if mixing_beta is not None and "beta" not in mixing_value:
      mixing_value["beta"] = mixing_beta
    diis_max_hist = mixing_value.pop("diis_max_hist", None)
    if diis_max_hist is not None and "history_size" not in mixing_value:
      mixing_value["history_size"] = diis_max_hist
    _deep_merge(target["mixing"], mixing_value)

  convergence = scf_value.pop("convergence", None)
  if convergence is not None:
    if not isinstance(convergence, Mapping):
      raise TypeError(
        "Config group `solver.scf.convergence` must be a mapping."
      )
    convergence_value = copy.deepcopy(dict(convergence))
    convergence_condition = convergence_value.pop("convergence_condition", None)
    if convergence_condition is not None and "energy_tol" not in convergence_value:
      convergence_value["energy_tol"] = convergence_condition
    _deep_merge(target["convergence"], convergence_value)

  if "lobpcg_max_iter" in scf_value:
    target["eigensolver"]["max_iter"] = copy.deepcopy(
      scf_value.pop("lobpcg_max_iter"),
    )
  if "mixing_beta" in scf_value:
    target["mixing"]["beta"] = copy.deepcopy(scf_value.pop("mixing_beta"))
  if "diis_max_hist" in scf_value:
    target["mixing"]["history_size"] = copy.deepcopy(
      scf_value.pop("diis_max_hist"),
    )
  if "convergence_condition" in scf_value:
    target["convergence"]["energy_tol"] = copy.deepcopy(
      scf_value.pop("convergence_condition"),
    )

  _deep_merge(target, scf_value)


def _normalize_solver_direct_opt_group(
  target: dict[str, Any],
  group_value: Mapping[str, Any],
) -> None:
  direct_opt_value = copy.deepcopy(dict(group_value))
  epoch = direct_opt_value.pop("epoch", None)
  if epoch is not None and "max_steps" not in direct_opt_value:
    direct_opt_value["max_steps"] = epoch

  optimizer = direct_opt_value.pop("optimizer", None)
  if optimizer is not None:
    if isinstance(optimizer, Mapping):
      _deep_merge(target["optimizer"], copy.deepcopy(dict(optimizer)))
    else:
      target["optimizer"]["name"] = copy.deepcopy(optimizer)

  optimizer_args = direct_opt_value.pop("optimizer_args", None)
  if optimizer_args is not None:
    if not isinstance(optimizer_args, Mapping):
      raise TypeError(
        "Config field `solver.direct_opt.optimizer_args` must be a mapping."
      )
    _deep_merge(target["optimizer"], copy.deepcopy(dict(optimizer_args)))

  occupation_optimizer = direct_opt_value.pop("occupation_optimizer", None)
  if occupation_optimizer is not None:
    if not isinstance(occupation_optimizer, Mapping):
      raise TypeError(
        "Config group `solver.direct_opt.occupation_optimizer` "
        "must be a mapping."
      )
    _normalize_occupation_optimizer_group(
      target["occupation_optimizer"],
      occupation_optimizer,
    )

  convergence = direct_opt_value.pop("convergence", None)
  if convergence is not None:
    if not isinstance(convergence, Mapping):
      raise TypeError(
        "Config group `solver.direct_opt.convergence` must be a mapping."
      )
    convergence_value = copy.deepcopy(dict(convergence))
    convergence_window_size = convergence_value.pop(
      "convergence_window_size",
      None,
    )
    if convergence_window_size is not None and "window_size" not in convergence_value:
      convergence_value["window_size"] = convergence_window_size
    convergence_condition = convergence_value.pop(
      "convergence_condition",
      None,
    )
    if convergence_condition is not None and "energy_std_tol" not in convergence_value:
      convergence_value["energy_std_tol"] = convergence_condition
    _deep_merge(target["convergence"], convergence_value)

  if "convergence_window_size" in direct_opt_value:
    target["convergence"]["window_size"] = copy.deepcopy(
      direct_opt_value.pop("convergence_window_size"),
    )
  if "convergence_condition" in direct_opt_value:
    target["convergence"]["energy_std_tol"] = copy.deepcopy(
      direct_opt_value.pop("convergence_condition"),
    )

  _deep_merge(target, direct_opt_value)


def _normalize_basis_group(
  target: dict[str, Any],
  group_value: Mapping[str, Any],
) -> None:
  basis_value = copy.deepcopy(dict(group_value))
  cutoff_energy_ha = basis_value.pop("cutoff_energy_ha", None)
  if cutoff_energy_ha is not None and "cutoff_energy" not in basis_value:
    basis_value["cutoff_energy"] = cutoff_energy_ha
  _deep_merge(target, basis_value)


def _normalize_solver_group(
  target: dict[str, Any],
  group_value: Mapping[str, Any],
) -> None:
  solver_value = copy.deepcopy(dict(group_value))
  solver_type = solver_value.pop("type", None)
  if solver_type is not None and "mode" not in solver_value:
    solver_value["mode"] = solver_type

  if "mode" in solver_value:
    target["mode"] = _normalize_solver_mode(solver_value.pop("mode"))

  auto = solver_value.pop("auto", None)
  if auto is not None:
    if not isinstance(auto, Mapping):
      raise TypeError("Config group `solver.auto` must be a mapping.")
    auto_value = copy.deepcopy(dict(auto))
    for field in ("primary", "fallback"):
      if field in auto_value:
        auto_value[field] = _normalize_solver_mode(auto_value[field])
    _deep_merge(target["auto"], auto_value)

  scf = solver_value.pop("scf", None)
  if scf is not None:
    if not isinstance(scf, Mapping):
      raise TypeError("Config group `solver.scf` must be a mapping.")
    _normalize_solver_scf_group(target["scf"], scf)

  direct_opt = solver_value.pop("direct_opt", None)
  if direct_opt is not None:
    if not isinstance(direct_opt, Mapping):
      raise TypeError("Config group `solver.direct_opt` must be a mapping.")
    _normalize_solver_direct_opt_group(target["direct_opt"], direct_opt)

  for key in (
    "optimizer",
    "optimizer_args",
    "scheduler",
    "epoch",
    "convergence_window_size",
    "convergence_condition",
  ):
    if key in solver_value:
      _apply_legacy_field(
        config={"solver": target},
        key=key,
        value=solver_value.pop(key),
      )

  for key in (
    "scf_max_iter",
    "scf_max_iteration",
    "lobpcg_max_iter",
    "mixing_beta",
    "diis_max_hist",
  ):
    if key in solver_value:
      _apply_legacy_field(
        config={"solver": target},
        key=key,
        value=solver_value.pop(key),
      )

  _deep_merge(target, solver_value)


def _normalize_band_group(
  target: dict[str, Any],
  group_value: Mapping[str, Any],
) -> None:
  band_value = copy.deepcopy(dict(group_value))
  if band_value.get("empty_bands") is None:
    band_value.pop("empty_bands", None)
  plot = band_value.pop("plot", None)
  if plot is not None:
    if not isinstance(plot, Mapping):
      raise TypeError("Config group `band.plot` must be a mapping.")
    _deep_merge(target["plot"], copy.deepcopy(dict(plot)))

  _deep_merge(target, band_value)


def _normalize_occupation_optimizer_group(
  target: dict[str, Any],
  group_value: Mapping[str, Any],
) -> None:
  occupation_optimizer_value = copy.deepcopy(dict(group_value))
  if "warmup_step" in occupation_optimizer_value:
    occupation_optimizer_value.setdefault(
      "warmup_steps",
      occupation_optimizer_value.pop("warmup_step"),
    )

  scheduler = occupation_optimizer_value.pop("scheduler", None)
  if scheduler is not None:
    if not isinstance(scheduler, Mapping):
      raise TypeError(
        "Config group `solver.direct_opt.occupation_optimizer.scheduler` "
        "must be a mapping."
      )
    scheduler_value = copy.deepcopy(_DEFAULT_OCCUPATION_SCHEDULER)
    _deep_merge(scheduler_value, copy.deepcopy(dict(scheduler)))
    occupation_optimizer_value["scheduler"] = scheduler_value

  _deep_merge(target, occupation_optimizer_value)


def _normalize_occupation_group(
  normalized: dict[str, Any],
  group_value: Mapping[str, Any],
) -> None:
  occupation_value = copy.deepcopy(dict(group_value))
  legacy_optimizer_fields = {}
  for key in ("warmup_steps", "warmup_step", "learning_rate", "scheduler"):
    if key in occupation_value:
      legacy_optimizer_fields[key] = occupation_value.pop(key)

  _deep_merge(normalized["occupation"], occupation_value)
  if legacy_optimizer_fields:
    _normalize_occupation_optimizer_group(
      normalized["solver"]["direct_opt"]["occupation_optimizer"],
      legacy_optimizer_fields,
    )


def _normalize_io_group(
  target: dict[str, Any],
  group_value: Mapping[str, Any],
) -> None:
  io_value = copy.deepcopy(dict(group_value))
  legacy_save_dir = io_value.pop("save_dir", None)
  explicit_output_dir = "output_dir" in io_value

  _deep_merge(target, io_value)

  if legacy_save_dir is not None and explicit_output_dir and (
    legacy_save_dir != target["output_dir"]
  ):
    warnings.warn(
      "Config fields `io.output_dir` and legacy `io.save_dir` differ; "
      "using `io.output_dir` for new outputs.",
      stacklevel=3,
    )
  if legacy_save_dir is not None and (
    not explicit_output_dir or
    target["output_dir"] == default_config["io"]["output_dir"]
  ):
    target["output_dir"] = copy.deepcopy(legacy_save_dir)


def _sync_log_level_and_verbose(
  normalized: dict[str, Any],
  *,
  log_level_explicit: bool,
  verbose_explicit: bool,
) -> None:
  if log_level_explicit:
    normalized["execution"]["verbose"] = normalized["io"]["log_level"
                                                         ] != "quiet"
    return
  if verbose_explicit:
    normalized["io"]["log_level"] = (
      "normal" if normalized["execution"]["verbose"] else "quiet"
    )


def _normalize_config(config: Optional[Mapping[str, Any]]) -> dict[str, Any]:
  normalized = copy.deepcopy(default_config)
  if config is None:
    config = {}
  log_level_explicit = False
  verbose_explicit = False
  band_plot_enabled_explicit = False
  legacy_band_plot_enabled = None

  for key, value in config.items():
    if key == "schema_version":
      normalized["schema_version"] = copy.deepcopy(value)
      continue

    if key in _GROUP_FIELDS and isinstance(value, Mapping):
      group_value = dict(value)
      if key == "system" and "crystal_file_path_path" in group_value:
        group_value["crystal_file_path"] = group_value.pop(
          "crystal_file_path_path"
        )
      if key == "execution" and "parallel_over_k" in group_value:
        parallel_over_k = group_value.pop("parallel_over_k")
        group_value.setdefault("parallel_over_k_mesh", parallel_over_k)
        group_value.setdefault("parallel_over_k_path", parallel_over_k)
      if key == "execution" and "verbose" in group_value:
        verbose_explicit = True
      if key == "io" and "log_level" in group_value:
        log_level_explicit = True
      if key == "io" and "save_band_plot" in group_value:
        legacy_band_plot_enabled = copy.deepcopy(
          group_value.pop("save_band_plot"),
        )
      if key == "band":
        plot_value = group_value.get("plot")
        if isinstance(plot_value, Mapping) and "enabled" in plot_value:
          band_plot_enabled_explicit = True
      if key == "solver":
        _normalize_solver_group(normalized["solver"], group_value)
      elif key == "method":
        _normalize_method_group(normalized["method"], group_value)
      elif key == "basis":
        _normalize_basis_group(normalized["basis"], group_value)
      elif key == "band":
        _normalize_band_group(normalized["band"], group_value)
      elif key == "occupation":
        _normalize_occupation_group(normalized, group_value)
      elif key == "io":
        _normalize_io_group(normalized["io"], group_value)
      else:
        _deep_merge(normalized[key], group_value)
      continue

    if key == "ewald_args" or key in _LEGACY_FIELD_MAP:
      if key == "verbose":
        verbose_explicit = True
      if key == "log_level":
        log_level_explicit = True
      _apply_legacy_field(normalized, key, value)
      continue

    if key in _GROUP_FIELDS:
      raise TypeError(f"Config group `{key}` must be a mapping.")

  if (not band_plot_enabled_explicit and legacy_band_plot_enabled is not None):
    normalized["band"]["plot"]["enabled"] = legacy_band_plot_enabled

  _sync_log_level_and_verbose(
    normalized,
    log_level_explicit=log_level_explicit,
    verbose_explicit=verbose_explicit,
  )

  return normalized


def _migrate_flat_config(flat: dict[str, Any]) -> dict[str, Any]:
  """Convert a legacy flat config to nested schema v1."""
  if "schema_version" in flat:
    return copy.deepcopy(flat)

  migrated = copy.deepcopy(default_config)
  for key, value in flat.items():
    if key == "ewald_args" or key in _LEGACY_FIELD_MAP:
      _apply_legacy_field(migrated, key, value)

  return migrated


def _is_number(value: Any) -> bool:
  return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_optional_string(value: Any, path: str) -> None:
  if value is not None and not isinstance(value, str):
    raise TypeError(f"Config field `{path}` must be a string or None.")


def _validate_bool(value: Any, path: str) -> None:
  if not isinstance(value, bool):
    raise TypeError(f"Config field `{path}` must be a bool.")


def _validate_int(value: Any, path: str) -> None:
  if not isinstance(value, int) or isinstance(value, bool):
    raise TypeError(f"Config field `{path}` must be an int.")


def _validate_number(value: Any, path: str) -> None:
  if not _is_number(value):
    raise TypeError(f"Config field `{path}` must be a number.")


def _validate_optional_number(value: Any, path: str) -> None:
  if value is not None:
    _validate_number(value, path)


def _validate_grid_sizes(value: Any, path: str) -> None:
  if isinstance(value, int) and not isinstance(value, bool):
    return
  if isinstance(value, (list, tuple)) and all(
    isinstance(item, int) and not isinstance(item, bool) for item in value
  ):
    return
  raise TypeError(f"Config field `{path}` must be an int or list/tuple of int.")


def validate_config(config: Mapping[str, Any]) -> None:  # noqa: PLR0915
  """Validate a nested schema v1 config."""
  _validate_int(config["schema_version"], "schema_version")
  if config["schema_version"] != 1:
    raise ValueError("Only config schema_version=1 is supported.")

  _validate_optional_string(config["system"]["crystal"], "system.crystal")
  _validate_optional_string(
    config["system"]["crystal_file_path"],
    "system.crystal_file_path",
  )
  _validate_int(config["system"]["spin"], "system.spin")
  _validate_bool(
    config["system"]["spin_restricted"],
    "system.spin_restricted",
  )

  if not isinstance(config["method"]["xc"], str):
    raise TypeError("Config field `method.xc` must be a string.")
  if not isinstance(config["method"]["family"], str):
    raise TypeError("Config field `method.family` must be a string.")
  if config["method"]["family"] not in {"ae", "nc", "us"}:
    raise ValueError(
      "Config field `method.family` must be one of 'ae', 'nc', or 'us'."
    )
  _validate_optional_string(
    config["method"]["pseudopotential_file_dir"],
    "method.pseudopotential_file_dir",
  )

  if not isinstance(config["basis"]["freq_mask_method"], str):
    raise TypeError("Config field `basis.freq_mask_method` must be a string.")
  _validate_number(config["basis"]["cutoff_energy"], "basis.cutoff_energy")
  _validate_grid_sizes(config["basis"]["grid_sizes"], "basis.grid_sizes")
  _validate_grid_sizes(
    config["ksampling"]["k_grid_sizes"],
    "ksampling.k_grid_sizes",
  )
  _validate_bool(
    config["ksampling"]["symmetry_reduction"],
    "ksampling.symmetry_reduction",
  )

  if not isinstance(config["solver"]["mode"], str):
    raise TypeError("Config field `solver.mode` must be a string.")
  if config["solver"]["mode"] not in {"auto", "scf", "direct_opt"}:
    raise ValueError(
      "Config field `solver.mode` must be one of "
      "'auto', 'scf', or 'direct_opt'."
    )

  if not isinstance(config["solver"]["auto"]["primary"], str):
    raise TypeError("Config field `solver.auto.primary` must be a string.")
  if not isinstance(config["solver"]["auto"]["fallback"], str):
    raise TypeError("Config field `solver.auto.fallback` must be a string.")
  if config["solver"]["auto"]["primary"] not in {"scf", "direct_opt"}:
    raise ValueError(
      "Config field `solver.auto.primary` must be 'scf' or 'direct_opt'."
    )
  if config["solver"]["auto"]["fallback"] not in {"scf", "direct_opt"}:
    raise ValueError(
      "Config field `solver.auto.fallback` must be 'scf' or 'direct_opt'."
    )
  if config["solver"]["auto"]["primary"] == config["solver"]["auto"]["fallback"
                                                                    ]:
    raise ValueError(
      "Config fields `solver.auto.primary` and `solver.auto.fallback` "
      "must be different."
    )
  _validate_bool(
    config["solver"]["auto"]["fallback_on_nonconverged"],
    "solver.auto.fallback_on_nonconverged",
  )
  _validate_bool(
    config["solver"]["auto"]["fallback_on_error"],
    "solver.auto.fallback_on_error",
  )

  _validate_int(config["solver"]["scf"]["max_iter"], "solver.scf.max_iter")
  if not isinstance(config["solver"]["scf"]["eigensolver"]["method"], str):
    raise TypeError(
      "Config field `solver.scf.eigensolver.method` must be a string."
    )
  if config["solver"]["scf"]["eigensolver"]["method"] != "lobpcg":
    raise ValueError(
      "Config field `solver.scf.eigensolver.method` must be 'lobpcg'."
    )
  _validate_int(
    config["solver"]["scf"]["eigensolver"]["max_iter"],
    "solver.scf.eigensolver.max_iter",
  )
  if not isinstance(config["solver"]["scf"]["mixing"]["method"], str):
    raise TypeError("Config field `solver.scf.mixing.method` must be a string.")
  if config["solver"]["scf"]["mixing"]["method"] != "diis":
    raise ValueError("Config field `solver.scf.mixing.method` must be 'diis'.")
  _validate_number(
    config["solver"]["scf"]["mixing"]["beta"],
    "solver.scf.mixing.beta",
  )
  _validate_int(
    config["solver"]["scf"]["mixing"]["history_size"],
    "solver.scf.mixing.history_size",
  )
  _validate_number(
    config["solver"]["scf"]["convergence"]["density_tol"],
    "solver.scf.convergence.density_tol",
  )
  _validate_number(
    config["solver"]["scf"]["convergence"]["energy_tol"],
    "solver.scf.convergence.energy_tol",
  )

  _validate_int(
    config["solver"]["direct_opt"]["max_steps"],
    "solver.direct_opt.max_steps",
  )
  if not isinstance(
    config["solver"]["direct_opt"]["canonical_transform"], bool
  ):
    raise TypeError(
      "Config field `solver.direct_opt.canonical_transform` must be a bool."
    )
  if not isinstance(config["solver"]["direct_opt"]["optimizer"], Mapping):
    raise TypeError(
      "Config field `solver.direct_opt.optimizer` must be a mapping."
    )
  if not isinstance(config["solver"]["direct_opt"]["optimizer"]["name"], str):
    raise TypeError(
      "Config field `solver.direct_opt.optimizer.name` must be a string."
    )
  _validate_number(
    config["solver"]["direct_opt"]["optimizer"]["learning_rate"],
    "solver.direct_opt.optimizer.learning_rate",
  )
  _validate_optional_string(
    config["solver"]["direct_opt"]["scheduler"],
    "solver.direct_opt.scheduler",
  )
  _validate_int(
    config["solver"]["direct_opt"]["convergence"]["window_size"],
    "solver.direct_opt.convergence.window_size",
  )
  _validate_number(
    config["solver"]["direct_opt"]["convergence"]["energy_std_tol"],
    "solver.direct_opt.convergence.energy_std_tol",
  )
  occupation_optimizer = config["solver"]["direct_opt"]["occupation_optimizer"]
  _validate_int(
    occupation_optimizer["warmup_steps"],
    "solver.direct_opt.occupation_optimizer.warmup_steps",
  )
  if occupation_optimizer["warmup_steps"] < 0:
    raise ValueError(
      "Config field `solver.direct_opt.occupation_optimizer.warmup_steps` "
      "must be non-negative."
    )
  _validate_number(
    occupation_optimizer["learning_rate"],
    "solver.direct_opt.occupation_optimizer.learning_rate",
  )
  if occupation_optimizer["learning_rate"] <= 0:
    raise ValueError(
      "Config field `solver.direct_opt.occupation_optimizer.learning_rate` "
      "must be positive."
    )
  scheduler_config = occupation_optimizer["scheduler"]
  if scheduler_config is not None:
    if not isinstance(scheduler_config, Mapping):
      raise TypeError(
        "Config field `solver.direct_opt.occupation_optimizer.scheduler` "
        "must be a mapping or None."
      )
    if scheduler_config.get("name") != "reduce_on_plateau":
      raise ValueError(
        "Config field `solver.direct_opt.occupation_optimizer.scheduler.name` "
        "must be 'reduce_on_plateau'."
      )
    _validate_number(
      scheduler_config["factor"],
      "solver.direct_opt.occupation_optimizer.scheduler.factor",
    )
    _validate_int(
      scheduler_config["patience"],
      "solver.direct_opt.occupation_optimizer.scheduler.patience",
    )
    _validate_number(
      scheduler_config["rtol"],
      "solver.direct_opt.occupation_optimizer.scheduler.rtol",
    )
    _validate_number(
      scheduler_config["atol"],
      "solver.direct_opt.occupation_optimizer.scheduler.atol",
    )
    _validate_int(
      scheduler_config["cooldown"],
      "solver.direct_opt.occupation_optimizer.scheduler.cooldown",
    )
    _validate_int(
      scheduler_config["accumulation_size"],
      "solver.direct_opt.occupation_optimizer.scheduler.accumulation_size",
    )
    _validate_number(
      scheduler_config["min_scale"],
      "solver.direct_opt.occupation_optimizer.scheduler.min_scale",
    )
    if not (0.0 < scheduler_config["factor"] < 1.0):
      raise ValueError(
        "Config field `solver.direct_opt.occupation_optimizer.scheduler.factor` "
        "must satisfy 0 < factor < 1."
      )
    if scheduler_config["patience"] < 1:
      raise ValueError(
        "Config field `solver.direct_opt.occupation_optimizer.scheduler.patience` "
        "must be positive."
      )
    if scheduler_config["cooldown"] < 0:
      raise ValueError(
        "Config field `solver.direct_opt.occupation_optimizer.scheduler.cooldown` "
        "must be non-negative."
      )
    if scheduler_config["accumulation_size"] < 1:
      raise ValueError(
        "Config field "
        "`solver.direct_opt.occupation_optimizer.scheduler.accumulation_size` "
        "must be positive."
      )
    if scheduler_config["min_scale"] < 0:
      raise ValueError(
        "Config field `solver.direct_opt.occupation_optimizer.scheduler.min_scale` "
        "must be non-negative."
      )

  if not isinstance(config["occupation"]["method"], str):
    raise TypeError("Config field `occupation.method` must be a string.")
  _validate_number(config["occupation"]["smearing"], "occupation.smearing")
  _validate_int(config["occupation"]["empty_bands"], "occupation.empty_bands")

  _validate_number(config["ewald"]["eta"], "ewald.eta")
  _validate_number(config["ewald"]["cutoff"], "ewald.cutoff")

  _validate_int(config["band"]["empty_bands"], "band.empty_bands")
  _validate_optional_string(
    config["band"]["k_path_special_points"],
    "band.k_path_special_points",
  )
  _validate_int(config["band"]["num_kpoints"], "band.num_kpoints")
  _validate_optional_string(config["band"]["k_path_file"], "band.k_path_file")
  _validate_int(config["band"]["epoch"], "band.epoch")
  _validate_bool(config["band"]["fine_tuning"], "band.fine_tuning")
  _validate_int(
    config["band"]["fine_tuning_epoch"],
    "band.fine_tuning_epoch",
  )
  _validate_int(
    config["band"]["eigensolver_max_iter"],
    "band.eigensolver_max_iter",
  )
  _validate_bool(config["band"]["plot"]["enabled"], "band.plot.enabled")
  if config["band"]["plot"]["unit"] not in {"eV", "Ha", "Ry"}:
    raise ValueError(
      "Config field `band.plot.unit` must be 'eV', 'Ha', or 'Ry'."
    )
  _validate_optional_number(config["band"]["plot"]["y_min"], "band.plot.y_min")
  _validate_optional_number(config["band"]["plot"]["y_max"], "band.plot.y_max")
  if (
    config["band"]["plot"]["y_min"] is not None and
    config["band"]["plot"]["y_max"] is not None and
    config["band"]["plot"]["y_min"] >= config["band"]["plot"]["y_max"]
  ):
    raise ValueError(
      "Config fields `band.plot.y_min` and `band.plot.y_max` "
      "must satisfy y_min < y_max."
    )

  _validate_int(config["execution"]["seed"], "execution.seed")
  _validate_bool(
    config["execution"]["parallel_over_k_mesh"],
    "execution.parallel_over_k_mesh",
  )
  _validate_bool(
    config["execution"]["parallel_over_k_path"],
    "execution.parallel_over_k_path",
  )
  _validate_bool(config["execution"]["profile"], "execution.profile")
  _validate_bool(
    config["execution"]["compile_cache"], "execution.compile_cache"
  )
  _validate_optional_string(
    config["execution"]["compile_cache_dir"],
    "execution.compile_cache_dir",
  )
  _validate_bool(
    config["execution"]["xla_preallocate"],
    "execution.xla_preallocate",
  )
  _validate_bool(
    config["execution"]["jax_enable_x64"],
    "execution.jax_enable_x64",
  )
  _validate_bool(
    config["execution"]["jax_debug_nans"],
    "execution.jax_debug_nans",
  )
  _validate_bool(config["execution"]["verbose"], "execution.verbose")
  _validate_number(config["execution"]["eps"], "execution.eps")
  _validate_bool(
    config["experimental"]["disable_jit"],
    "experimental.disable_jit",
  )

  if not isinstance(config["io"]["output_dir"], str):
    raise TypeError("Config field `io.output_dir` must be a string.")
  _validate_optional_string(config["io"]["run_label"], "io.run_label")
  _validate_bool(config["io"]["save_density"], "io.save_density")
  _validate_bool(
    config["io"]["save_wavefunction"],
    "io.save_wavefunction",
  )
  _validate_bool(
    config["io"]["save_ground_state_spectrum"],
    "io.save_ground_state_spectrum",
  )
  _validate_bool(config["io"]["save_checkpoint"], "io.save_checkpoint")
  _validate_int(
    config["io"]["checkpoint_interval"],
    "io.checkpoint_interval",
  )
  if config["io"]["checkpoint_interval"] <= 0:
    raise ValueError("Config field `io.checkpoint_interval` must be positive.")
  if not isinstance(config["io"]["restart"], str):
    raise TypeError("Config field `io.restart` must be a string.")
  if config["io"]["log_level"] not in {"quiet", "normal", "verbose"}:
    raise ValueError(
      "Config field `io.log_level` must be 'quiet', 'normal', or 'verbose'."
    )

  if config["system"]["spin_restricted"] and config["system"]["spin"] != 0:
    raise ValueError(
      "Config fields `system.spin_restricted=true` and `system.spin != 0` "
      "are incompatible."
    )

  if (
    config["method"]["family"] == "ae" and
    config["method"]["pseudopotential_file_dir"] is not None
  ):
    raise ValueError(
      "Config field `method.pseudopotential_file_dir` cannot be set when "
      "`method.family` is 'ae'."
    )

  if config["method"]["family"] in {"nc", "us"}:
    pseudopotential_dir = config["method"]["pseudopotential_file_dir"]
    if pseudopotential_dir is not None:
      from pathlib import Path

      if not Path(pseudopotential_dir).expanduser().exists():
        raise ValueError(
          "Config field `method.pseudopotential_file_dir` must point to an "
          "existing directory."
        )

  if (
    config["band"]["k_path_special_points"] is not None and
    config["band"]["k_path_file"] is not None
  ):
    raise ValueError(
      "Config fields `band.k_path_special_points` and `band.k_path_file` "
      "cannot both be set."
    )


def get_config(config_file: Optional[str] = None) -> JrystalConfigDict:
  if config_file is not None:
    with open(config_file, "r", encoding="utf-8") as file:
      raw_config = yaml.safe_load(file) or {}
  else:
    raw_config = {}

  if not isinstance(raw_config, Mapping):
    raise TypeError("Config file must define a mapping at the top level.")

  _warn_unknown_fields(raw_config)
  normalized_config = _normalize_config(raw_config)
  validate_config(normalized_config)
  return JrystalConfigDict(normalized_config)
