from __future__ import annotations

from pathlib import Path

import jrystal as jr
from jrystal.calc import opt_utils


def test_compile_cache_defaults_are_present():
  config = jr.config.get_config(None)

  assert config.execution.compile_cache is True
  assert config.execution.compile_cache_dir == "~/.cache/jrystal/jax_compile_cache"


def test_set_env_params_configures_compile_cache(monkeypatch, tmp_path):
  config = jr.config.get_config(None)
  config.execution.compile_cache = True
  config.execution.compile_cache_dir = str(tmp_path / "jax_compile_cache")
  config.execution.verbose = True

  config_updates = []
  lines = []

  def _fake_update(key, value):
    config_updates.append((key, value))

  monkeypatch.setattr(opt_utils.jax.config, "update", _fake_update)
  monkeypatch.setattr(
    opt_utils, "stage_line", lambda *args, **kwargs: lines.append(args)
  )

  opt_utils.set_env_params(config)

  cache_dir = str(Path(config.execution.compile_cache_dir).expanduser())
  assert Path(cache_dir).is_dir()
  assert ("jax_compilation_cache_dir", cache_dir) in config_updates
  assert ("jax_persistent_cache_min_entry_size_bytes", -1) in config_updates
  assert ("jax_persistent_cache_min_compile_time_secs", 1.0) in config_updates
  assert any(
    "JAX compile cache:" in call[1] for call in lines if len(call) >= 2
  )


def test_set_env_params_skips_compile_cache_when_disabled(
  monkeypatch, tmp_path
):
  config = jr.config.get_config(None)
  config.execution.compile_cache = False
  config.execution.compile_cache_dir = str(tmp_path / "jax_compile_cache")
  config.execution.verbose = False

  config_updates = []

  def _fake_update(key, value):
    config_updates.append((key, value))

  monkeypatch.setattr(opt_utils.jax.config, "update", _fake_update)

  opt_utils.set_env_params(config)

  assert (
    "jax_compilation_cache_dir", str(tmp_path / "jax_compile_cache")
  ) not in config_updates
  assert not (tmp_path / "jax_compile_cache").exists()
