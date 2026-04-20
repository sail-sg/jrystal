from __future__ import annotations

import jrystal as jr
from jrystal import cli


def test_band_command_runs_band_only(monkeypatch):
  config = jr.config.get_config(None)
  calls = []

  monkeypatch.setattr(cli, "render_logo", lambda **_: None)
  monkeypatch.setattr(cli.jr.config, "get_config", lambda _: config)
  monkeypatch.setattr(cli.jr.config, "validate_config", lambda _: None)
  monkeypatch.setattr(
    cli.jr.calc,
    "energy",
    lambda *_args, **_kwargs: (_ for _ in ()).throw(
      AssertionError("jr.calc.energy should not be called for `jrystal band`")
    ),
  )
  monkeypatch.setattr(
    cli.jr.calc,
    "band",
    lambda cfg, ground_state_result=None: calls.append(
      (cfg.solver.mode, ground_state_result)
    ),
  )

  cli.main(["band", "config.yaml"])

  assert calls == [("auto", None)]


def test_cli_override_updates_io_fields():
  config = jr.config.get_config(None)
  cli._apply_overrides(
    config,
    {
      "io.output_dir": "results",
      "io.run_label": "manual",
      "io.log_level": "quiet",
    },
  )
  assert config.io.output_dir == "results"
  assert config.io.run_label == "manual"
  assert config.io.log_level == "quiet"
