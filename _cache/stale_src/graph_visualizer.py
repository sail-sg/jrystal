from __future__ import annotations

import os
import warnings
from functools import partial, wraps
from pathlib import Path
from typing import Optional, Tuple

import jax
import jaxlib._jax as xe


def _import_graphviz():
  """Import graphviz lazily so module import does not require graphviz."""
  try:
    import graphviz  # pytype: disable=import-error
    return graphviz
  except Exception:
    return None


def _split_path(filename: str, default_ext: str = "pdf") -> Tuple[Path, str]:
  """Split output path and extension."""
  path = Path(filename)
  ext = path.suffix[1:] if path.suffix else default_ext
  stem = path.with_suffix("")
  return stem, ext


def write_graphviz(
  dot_graph: str,
  filename: str,
  *,
  directory: str = ".",
  cleanup: bool = True,
  fmt: str = "pdf",
  view: bool = False,
):
  """Render a DOT graph into a file and return output path.

  If python-graphviz is unavailable, this function warns and returns ``None``.
  """
  graphviz = _import_graphviz()
  if graphviz is None:
    warnings.warn(
      "graphviz package is not installed; skipping graph rendering.",
      RuntimeWarning,
    )
    return None

  gvz = graphviz.Source(dot_graph)
  output = gvz.render(
    filename=filename,
    directory=directory,
    cleanup=cleanup,
    format=fmt,
    view=view,
  )
  return output


def _extract_hlo(fun, args, kwargs, optimized: bool):
  """Extract HLO text and DOT graph from a lowered/compiled function."""
  lowered = fun.lower(*args, **kwargs)
  if not optimized:
    hlo_comp = lowered.compiler_ir(dialect="hlo")
    hlo_text = hlo_comp.as_hlo_text()
    dot_graph = hlo_comp.as_hlo_dot_graph()
    return hlo_text, dot_graph

  compiled = lowered.compile()
  hlo_text = compiled.as_text()
  dot_graph = None

  try:
    exe = compiled.runtime_executable()
    hlo_modules = exe.hlo_modules()
    if hlo_modules:
      xla_comp = xe.XlaComputation(
        hlo_modules[0].as_serialized_hlo_module_proto()
      )
      dot_graph = xla_comp.as_hlo_dot_graph()
  except Exception as exc:
    warnings.warn(
      f"Failed to extract optimized HLO dot graph: {exc}",
      RuntimeWarning,
    )

  return hlo_text, dot_graph


def _write_hlo_text(hlo_text: str, filename: str) -> str:
  """Write HLO text to disk and return path."""
  path = Path(filename)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(hlo_text)
  return str(path)


def visualize_com_graph(
  filename: str,
  *,
  optimized: bool = True,
  graph: bool = True,
  txt: bool = False,
  view_command: Optional[str] = None,
):
  """Decorator to export HLO graph/text on function invocation.

  Example:
    ``@visualize_com_graph("my_hlo.pdf")``
  """

  def _decorator(fun):

    @wraps(fun)
    def _wrapped_func(*args, **kwargs):
      hlo_text, dot_graph = _extract_hlo(fun, args, kwargs, optimized=optimized)

      if txt:
        stem, _ = _split_path(filename)
        _write_hlo_text(hlo_text, str(stem) + ".txt")

      if graph and dot_graph is not None:
        stem, ext = _split_path(filename, default_ext="pdf")
        output = write_graphviz(dot_graph, str(stem), fmt=ext)
        if output is not None and view_command:
          os.system(f"{view_command} {output}")

      return fun(*args, **kwargs)

    return _wrapped_func

  return _decorator


def view_hlo(
  *args,
  optimized: bool = True,
  graph: bool = True,
  txt: bool = False,
  view_command: Optional[str] = None,
):
  """Decorator helper for ad-hoc HLO visualization.

  Usage:
    ``@view_hlo``
    ``@view_hlo(optimized=False, txt=True)``
  """
  if len(args) == 1 and callable(args[0]):
    fun = args[0]
    filename = "optimized_" * optimized + f"hlo_of_{fun.__name__}@{id(fun)}.pdf"
    return visualize_com_graph(
      filename,
      optimized=optimized,
      graph=graph,
      txt=txt,
      view_command=view_command,
    )(fun)
  return partial(
    _view_hlo,
    optimized=optimized,
    graph=graph,
    txt=txt,
    view_command=view_command,
  )


def _view_hlo(fun, *, optimized, graph, txt, view_command):
  filename = "optimized_" * optimized + f"hlo_of_{fun.__name__}@{id(fun)}.pdf"
  return visualize_com_graph(
    filename,
    optimized=optimized,
    graph=graph,
    txt=txt,
    view_command=view_command,
  )(fun)
