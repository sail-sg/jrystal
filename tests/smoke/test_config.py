from __future__ import annotations

import textwrap

import pytest

import jrystal as jr


def test_io_defaults_are_available():
  config = jr.config.get_config(None)
  assert config.io.output_dir == "out/"
  assert config.io.restart == "from_scratch"
  assert config.io.log_level == "normal"
  assert config.execution.verbose is True


def test_legacy_io_alias_and_verbose_sync(tmp_path):
  config_path = tmp_path / "config.yaml"
  config_path.write_text(
    textwrap.dedent(
      """
        io:
          save_dir: legacy-out
        execution:
          verbose: false
        """
    ).strip() + "\n",
    encoding="utf-8",
  )
  config = jr.config.get_config(str(config_path))
  assert config.io.output_dir == "legacy-out"
  assert config.io.log_level == "quiet"
  assert config.execution.verbose is False


def test_log_level_drives_verbose(tmp_path):
  config_path = tmp_path / "config.yaml"
  config_path.write_text(
    textwrap.dedent("""
        io:
          log_level: verbose
        """).strip() + "\n",
    encoding="utf-8",
  )
  config = jr.config.get_config(str(config_path))
  assert config.io.log_level == "verbose"
  assert config.execution.verbose is True


def test_invalid_log_level_is_rejected():
  config = jr.config.get_config(None).to_dict()
  config["io"]["log_level"] = "loud"
  with pytest.raises(ValueError, match="io.log_level"):
    jr.config.validate_config(config)
