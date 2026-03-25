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
  "system": {
    "crystal": "diamond",
    "crystal_file_path": None,
    "spin": 0,
    "spin_restricted": True,
  },
  "method": {
    "xc": "lda_x",
    "use_pseudopotential": False,
    "pseudopotential_type": "nc",
    "pseudopotential_file_dir": None,
  },
  "basis": {
    "freq_mask_method": "spherical",
    "cutoff_energy": 100,
    "grid_sizes": 48,
  },
  "ksampling": {
    "k_grid_sizes": [4, 4, 4],
    "symmetry_reduction": True,
  },
  "solver": {
    "type": "direct_opt",
    "optimizer": "adam",
    "optimizer_args": {
      "learning_rate": 0.01,
      "b1": 0.9,
      "b2": 0.99,
    },
    "scheduler": None,
    "epoch": 10000,
    "convergence_window_size": 20,
    "convergence_condition": 1e-6,
  },
  "occupation": {
    "method": "uniform",
    "smearing": 0.0,
    "empty_bands": 20,
  },
  "ewald": {
    "eta": 0.1,
    "cutoff": 2e4,
  },
  "band": {
    "empty_bands": None,
    "k_path_special_points": None,
    "num_kpoints": 64,
    "k_path_file": None,
    "epoch": 5000,
    "fine_tuning": True,
    "fine_tuning_epoch": 300,
  },
  "execution": {
    "seed": 123,
    "parallel_over_k_mesh": False,
    "parallel_over_k_path": True,
    "xla_preallocate": True,
    "jax_enable_x64": True,
    "jax_debug_nans": False,
    "verbose": True,
    "eps": 1e-8,
  },
  "io": {
    "save_dir": None,
  },
}

_GROUP_FIELDS = {
  "system": {
    "crystal",
    "crystal_file_path",
    "spin",
    "spin_restricted",
  },
  "method": {
    "xc",
    "use_pseudopotential",
    "pseudopotential_type",
    "pseudopotential_file_dir",
  },
  "basis": {
    "freq_mask_method",
    "cutoff_energy",
    "grid_sizes",
  },
  "ksampling": {
    "k_grid_sizes",
    "symmetry_reduction",
  },
  "solver": {
    "type",
    "optimizer",
    "optimizer_args",
    "scheduler",
    "epoch",
    "convergence_window_size",
    "convergence_condition",
  },
  "occupation": {
    "method",
    "smearing",
    "empty_bands",
  },
  "ewald": {
    "eta",
    "cutoff",
  },
  "band": {
    "empty_bands",
    "k_path_special_points",
    "num_kpoints",
    "k_path_file",
    "epoch",
    "fine_tuning",
    "fine_tuning_epoch",
  },
  "execution": {
    "seed",
    "parallel_over_k_mesh",
    "parallel_over_k_path",
    "parallel_over_k",
    "xla_preallocate",
    "jax_enable_x64",
    "jax_debug_nans",
    "verbose",
    "eps",
  },
  "io": {
    "save_dir",
  },
}

_LEGACY_FIELD_MAP = {
  "crystal": ("system", "crystal"),
  "crystal_file_path": ("system", "crystal_file_path"),
  "crystal_file_path_path": ("system", "crystal_file_path"),
  "spin": ("system", "spin"),
  "spin_restricted": ("system", "spin_restricted"),
  "xc": ("method", "xc"),
  "use_pseudopotential": ("method", "use_pseudopotential"),
  "pseudopotential_type": ("method", "pseudopotential_type"),
  "pseudopotential_file_dir": ("method", "pseudopotential_file_dir"),
  "freq_mask_method": ("basis", "freq_mask_method"),
  "cutoff_energy": ("basis", "cutoff_energy"),
  "grid_sizes": ("basis", "grid_sizes"),
  "k_grid_sizes": ("ksampling", "k_grid_sizes"),
  "symmetry_reduction": ("ksampling", "symmetry_reduction"),
  "epoch": ("solver", "epoch"),
  "optimizer": ("solver", "optimizer"),
  "optimizer_args": ("solver", "optimizer_args"),
  "scheduler": ("solver", "scheduler"),
  "convergence_window_size": ("solver", "convergence_window_size"),
  "convergence_condition": ("solver", "convergence_condition"),
  "occupation": ("occupation", "method"),
  "smearing": ("occupation", "smearing"),
  "empty_bands": ("occupation", "empty_bands"),
  "band_structure_empty_bands": ("band", "empty_bands"),
  "k_path_special_points": ("band", "k_path_special_points"),
  "num_kpoints": ("band", "num_kpoints"),
  "k_path_file": ("band", "k_path_file"),
  "band_structure_epoch": ("band", "epoch"),
  "k_path_fine_tuning": ("band", "fine_tuning"),
  "k_path_fine_tuning_epoch": ("band", "fine_tuning_epoch"),
  "seed": ("execution", "seed"),
  "parallel_over_k": ("execution", "parallel_over_k"),
  "parallel_over_k_mesh": ("execution", "parallel_over_k_mesh"),
  "parallel_over_k_path": ("execution", "parallel_over_k_path"),
  "xla_preallocate": ("execution", "xla_preallocate"),
  "jax_enable_x64": ("execution", "jax_enable_x64"),
  "jax_debug_nans": ("execution", "jax_debug_nans"),
  "verbose": ("execution", "verbose"),
  "eps": ("execution", "eps"),
  "save_dir": ("io", "save_dir"),
}


def _set_nested_value(
  config: dict[str, Any], path: tuple[str, str], value: Any
):
  config[path[0]][path[1]] = copy.deepcopy(value)


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
      unknown_fields = sorted(set(value) - _GROUP_FIELDS[key])
      for unknown_field in unknown_fields:
        warnings.warn(
          f"Unknown config field: {key}.{unknown_field}",
          stacklevel=3,
        )
      continue

    if key not in _LEGACY_FIELD_MAP and key != "ewald_args":
      warnings.warn(f"Unknown config field: {key}", stacklevel=3)


def _apply_legacy_field(
  config: dict[str, Any],
  key: str,
  value: Any,
) -> None:
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

  _set_nested_value(config, _LEGACY_FIELD_MAP[key], value)


def _normalize_config(config: Optional[Mapping[str, Any]]) -> dict[str, Any]:
  normalized = copy.deepcopy(default_config)
  if config is None:
    config = {}

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
      _deep_merge(normalized[key], group_value)
      continue

    if key == "ewald_args" or key in _LEGACY_FIELD_MAP:
      _apply_legacy_field(normalized, key, value)
      continue

    if key in _GROUP_FIELDS:
      raise TypeError(f"Config group `{key}` must be a mapping.")

  if normalized["band"]["empty_bands"] is None:
    normalized["band"]["empty_bands"] = normalized["occupation"]["empty_bands"]

  return normalized


def _migrate_flat_config(flat: dict[str, Any]) -> dict[str, Any]:
  """Convert a legacy flat config to nested schema v1."""
  if "schema_version" in flat:
    return copy.deepcopy(flat)

  migrated = copy.deepcopy(default_config)
  for key, value in flat.items():
    if key == "ewald_args" or key in _LEGACY_FIELD_MAP:
      _apply_legacy_field(migrated, key, value)

  if migrated["band"]["empty_bands"] is None:
    migrated["band"]["empty_bands"] = migrated["occupation"]["empty_bands"]

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
  _validate_bool(
    config["method"]["use_pseudopotential"],
    "method.use_pseudopotential",
  )
  if not isinstance(config["method"]["pseudopotential_type"], str):
    raise TypeError(
      "Config field `method.pseudopotential_type` must be a string."
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

  if not isinstance(config["solver"]["type"], str):
    raise TypeError("Config field `solver.type` must be a string.")
  if not isinstance(config["solver"]["optimizer"], str):
    raise TypeError("Config field `solver.optimizer` must be a string.")
  if not isinstance(config["solver"]["optimizer_args"], Mapping):
    raise TypeError("Config field `solver.optimizer_args` must be a mapping.")
  _validate_optional_string(config["solver"]["scheduler"], "solver.scheduler")
  _validate_int(config["solver"]["epoch"], "solver.epoch")
  _validate_int(
    config["solver"]["convergence_window_size"],
    "solver.convergence_window_size",
  )
  _validate_number(
    config["solver"]["convergence_condition"],
    "solver.convergence_condition",
  )

  if not isinstance(config["occupation"]["method"], str):
    raise TypeError("Config field `occupation.method` must be a string.")
  _validate_number(config["occupation"]["smearing"], "occupation.smearing")
  _validate_int(config["occupation"]["empty_bands"], "occupation.empty_bands")

  _validate_number(config["ewald"]["eta"], "ewald.eta")
  _validate_number(config["ewald"]["cutoff"], "ewald.cutoff")

  if config["band"]["empty_bands"] is not None:
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

  _validate_int(config["execution"]["seed"], "execution.seed")
  _validate_bool(
    config["execution"]["parallel_over_k_mesh"],
    "execution.parallel_over_k_mesh",
  )
  _validate_bool(
    config["execution"]["parallel_over_k_path"],
    "execution.parallel_over_k_path",
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

  _validate_optional_string(config["io"]["save_dir"], "io.save_dir")


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
