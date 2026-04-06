from __future__ import annotations

from jrystal import terminal_ui


def test_stage_output_is_mirrored_without_ansi(tmp_path):
  log_path = tmp_path / "jrystal.log"
  terminal_ui.open_log(log_path)
  try:
    terminal_ui.stage_line("SCF", "hello")
    terminal_ui.stage_warning("SCF", "not converged", color="red")
  finally:
    terminal_ui.close_log()
  text = log_path.read_text(encoding="utf-8")
  assert "hello" in text
  assert "not converged" in text
  assert "\x1b[" not in text


def test_ascii_logo_variant_is_available():
  logo = terminal_ui.get_logo_text(variant="ascii")
  assert "##" in logo
