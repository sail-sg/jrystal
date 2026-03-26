"""Terminal styling and lightweight live-status helpers."""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import TextIO

_RESET = "\033[0m"
_CLEAR_LINE = "\r\033[2K"
_COLOR_CODES = {
  "blue": "34",
  "cyan": "36",
  "green": "32",
  "magenta": "35",
  "red": "31",
  "white": "97",
  "yellow": "33",
}

_STAGE_STYLES = {
  "AUTO": ("yellow", "=>"),
  "Band": ("magenta", "::"),
  "DirectOpt": ("cyan", "<>"),
  "Init": ("blue", ".."),
  "SCF": ("cyan", "<>"),
}
_OUTPUT_LOCK = threading.RLock()
_UTF8_LOGO_LINES = (
  "       ██    ███     █   █    ███    █████    ██     █",
  "        █    █  █     █ █    █         █     █  █    █",
  "        █    ███       █      ██       █     ████    █",
  "     █  █    █ █       █        █      █     █  █    █",
  "      ██     █  █      █     ███       █     █  █    ███",
)
_ASCII_LOGO_LINES = (
  "       ##    ###     #   #    ###    #####    ##     #",
  "        #    #  #     # #    #         #     #  #    #",
  "        #    ###       #      ##       #     ####    #",
  "     #  #    # #       #        #      #     #  #    #",
  "      ##     #  #      #     ###       #     #  #    ###",
)


def _env_flag(name: str) -> bool:
  value = os.environ.get(name)
  return value is not None and value.lower() not in {"", "0", "false", "no"}


def supports_color(stream: TextIO | None = None) -> bool:
  """Return True when ANSI colour output is appropriate."""
  stream = sys.stdout if stream is None else stream
  if _env_flag("FORCE_COLOR"):
    return True
  if os.environ.get("NO_COLOR"):
    return False
  term = os.environ.get("TERM", "")
  return hasattr(stream, "isatty") and stream.isatty() and term.lower() != "dumb"


def supports_live_output(stream: TextIO | None = None) -> bool:
  """Return True when transient animation output is safe."""
  stream = sys.stderr if stream is None else stream
  return hasattr(stream, "isatty") and stream.isatty()


def _stream_encoding(stream: TextIO | None = None) -> str:
  stream = sys.stdout if stream is None else stream
  return (getattr(stream, "encoding", None) or "").lower()


def _use_utf8_logo(stream: TextIO | None = None) -> bool:
  if _env_flag("JRYSTAL_ASCII_LOGO"):
    return False
  if _env_flag("JRYSTAL_UTF8_LOGO"):
    return True
  return "utf" in _stream_encoding(stream)


def style(
  text: str,
  *,
  color: str | None = None,
  bold: bool = False,
  stream: TextIO | None = None,
) -> str:
  """Apply ANSI styles when the target stream supports colour."""
  if not supports_color(stream):
    return text

  codes = []
  if bold:
    codes.append("1")
  if color is not None:
    codes.append(_COLOR_CODES[color])
  if not codes:
    return text
  return f"\033[{';'.join(codes)}m{text}{_RESET}"


def stage_prefix(
  stage_name: str,
  *,
  stream: TextIO | None = None,
  color_override: str | None = None,
) -> str:
  """Return a styled stage prefix such as ``<> [SCF]``."""
  stream = sys.stderr if stream is None else stream
  color, icon = _STAGE_STYLES.get(stage_name, ("cyan", "--"))
  return style(
    f"{icon} [{stage_name}]",
    color=color_override or color,
    bold=True,
    stream=stream,
  )


def _warning_prefix(
  *,
  stream: TextIO | None = None,
  color: str = "yellow",
) -> str:
  stream = sys.stderr if stream is None else stream
  return style("!!", color=color, bold=True, stream=stream)


def console_line(text: str, *, stream: TextIO | None = None) -> None:
  """Write a stable terminal line without logger prefixes."""
  stream = sys.stderr if stream is None else stream
  with _OUTPUT_LOCK:
    if supports_live_output(stream):
      stream.write(_CLEAR_LINE)
    stream.write(text + "\n")
    stream.flush()


def stage_line(
  stage_name: str,
  message: str,
  *,
  stream: TextIO | None = None,
) -> None:
  """Write a stage-labelled stable terminal line."""
  stream = sys.stderr if stream is None else stream
  console_line(f"{stage_prefix(stage_name, stream=stream)} {message}", stream=stream)


def stage_warning(
  stage_name: str,
  message: str,
  *,
  stream: TextIO | None = None,
  color: str = "yellow",
) -> None:
  """Write a warning line for a stage."""
  stream = sys.stderr if stream is None else stream
  console_line(
    (
      f"{_warning_prefix(stream=stream, color=color)} "
      f"{stage_prefix(stage_name, stream=stream, color_override=color)} "
      f"{style(message, color=color, bold=(color == 'red'), stream=stream)}"
    ),
    stream=stream,
  )


def get_logo_text(
  *,
  stream: TextIO | None = None,
  variant: str = "utf8",
) -> str:
  """Return a logo variant.

  Supported variants are ``utf8``, ``ascii``, and ``auto``.
  """
  if variant == "utf8":
    lines = _UTF8_LOGO_LINES
  elif variant == "ascii":
    lines = _ASCII_LOGO_LINES
  elif variant == "auto":
    lines = _UTF8_LOGO_LINES if _use_utf8_logo(stream) else _ASCII_LOGO_LINES
  else:
    raise ValueError(
      f"Unknown logo variant '{variant}'. Use 'utf8', 'ascii', or 'auto'."
    )
  return "\n".join(lines) + "\n"


def colorize_logo(
  logo: str,
  *,
  stream: TextIO | None = None,
  scan_col: int | None = None,
  band_width: int = 10,
) -> str:
  """Colourise the logo, optionally with a moving highlight band."""
  if not supports_color(stream):
    return logo

  colors = ["cyan", "blue", "magenta", "yellow", "green"]
  lines = logo.splitlines()
  coloured_lines = []
  for i, line in enumerate(lines):
    if not line:
      coloured_lines.append("")
      continue

    base_color = colors[i % len(colors)]
    if scan_col is None:
      coloured_lines.append(
        style(line, color=base_color, bold=True, stream=stream)
      )
      continue

    start = max(0, min(len(line), scan_col))
    end = max(start, min(len(line), scan_col + band_width))
    coloured_lines.append(
      style(line[:start], color=base_color, bold=True, stream=stream)
      + style(line[start:end], color="white", bold=True, stream=stream)
      + style(line[end:], color=base_color, bold=True, stream=stream)
    )
  return "\n".join(coloured_lines) + ("\n" if logo.endswith("\n") else "")


def render_logo(
  *,
  stream: TextIO | None = None,
  animate: bool | None = None,
  variant: str = "utf8",
) -> None:
  """Render the jrystal logo with an optional short scan animation."""
  stream = sys.stdout if stream is None else stream
  logo = get_logo_text(stream=stream, variant=variant)

  if animate is None:
    animate = (
      supports_live_output(stream)
      and supports_color(stream)
      and not _env_flag("JRYSTAL_NO_ANIM")
      and not _env_flag("CI")
    )

  if not animate:
    stream.write(colorize_logo(logo, stream=stream))
    stream.flush()
    return

  lines = logo.splitlines()
  max_width = max((len(line) for line in lines), default=0)
  scan_positions = list(range(-12, max_width + 12, 6))
  with _OUTPUT_LOCK:
    for frame_index, scan_col in enumerate(scan_positions):
      if frame_index:
        stream.write(f"\033[{len(lines)}A")
      frame = colorize_logo(logo, stream=stream, scan_col=scan_col)
      for line in frame.splitlines():
        stream.write(_CLEAR_LINE + line + "\n")
      stream.flush()
      time.sleep(0.04)


def metric(
  label: str,
  value: str,
  *,
  color: str,
  stream: TextIO | None = None,
) -> str:
  """Return a coloured metric token such as ``E=-10.123``."""
  stream = sys.stderr if stream is None else stream
  label_text = style(label, color=color, bold=True, stream=stream)
  value_text = style(value, color=color, stream=stream)
  return f"{label_text}={value_text}"


class Spinner:
  """Tiny TTY-only spinner for long-running terminal work."""

  def __init__(
    self,
    stage_name: str,
    *,
    stream: TextIO | None = None,
    interval: float = 0.1,
  ):
    self.stage_name = stage_name
    self.stream = sys.stderr if stream is None else stream
    self.interval = interval
    self.enabled = supports_live_output(self.stream)
    self._frames = ["|", "/", "-", "\\"]
    self._message = ""
    self._stop_event = threading.Event()
    self._thread: threading.Thread | None = None

  def start(self, message: str = "") -> "Spinner":
    """Start animating."""
    if not self.enabled or self._thread is not None:
      self._message = message
      return self
    self._message = message
    self._stop_event.clear()
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()
    return self

  def update(self, message: str) -> None:
    """Update the transient message displayed next to the spinner."""
    self._message = message

  def stop(self) -> None:
    """Stop the spinner and clear the transient line."""
    if not self.enabled:
      return
    self._stop_event.set()
    if self._thread is not None:
      self._thread.join(timeout=self.interval * 4)
      self._thread = None
    with _OUTPUT_LOCK:
      self.stream.write(_CLEAR_LINE)
      self.stream.flush()

  def _run(self) -> None:
    frame_index = 0
    while not self._stop_event.is_set():
      prefix = stage_prefix(self.stage_name, stream=self.stream)
      frame = style(
        self._frames[frame_index % len(self._frames)],
        color=_STAGE_STYLES.get(self.stage_name, ("cyan", "--"))[0],
        bold=True,
        stream=self.stream,
      )
      with _OUTPUT_LOCK:
        self.stream.write(_CLEAR_LINE)
        self.stream.write(f"{frame} {prefix} {self._message}")
        self.stream.flush()
      frame_index += 1
      time.sleep(self.interval)
