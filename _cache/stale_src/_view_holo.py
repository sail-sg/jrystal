from functools import partial

import graphviz
import jax
from jaxlib.xla_extension import XlaComputation, hlo_module_from_text


def write_graphviz(gvz, filename):
  gvz.render(
    filename=filename,
    directory=".",
    cleanup=True,
    format="pdf",
    view=False,
  )
  return filename + ".pdf"


def view_hlo(
  *args,
  optimized=True,
  graph=True,
  txt=False,
  view_command=None,
):
  if len(args) == 1 and callable(args[0]):
    return _view_hlo(
      args[0],
      optimized=optimized,
      graph=graph,
      txt=txt,
      view_command=view_command,
    )
  return partial(
    _view_hlo,
    optimized=optimized,
    graph=graph,
    txt=txt,
    view_command=view_command,
  )


def _view_hlo(fun, *, optimized, graph, txt, view_command):
  """Decorator to view the HLO graph of a function.

  Usage:

  .. code-block:: python

    @view_hlo
    @jax.jit
    def f(x):
      return jnp.sin(x)

  Args:
    fun: a function decorated with `jax.jit`.
  Returns:
    A wrapped function that will display the HLO graph when called.
  """

  def _wrapped_func(*args, **kwargs):
    filename = "optimized_" * optimized + f"hlo_of_{fun.__name__}@{id(fun)}"
    if not optimized:
      xla_comp = jax.xla_computation(fun)(*args, **kwargs)
      if graph:
        dot = xla_comp.as_hlo_dot_graph()
        gvz = graphviz.Source(dot)
      if txt:
        hlo_text = xla_comp.as_hlo_text()
    else:
      hlo_text = fun.lower(*args, **kwargs).compile().as_text()
      if graph:
        hlo_module = hlo_module_from_text(hlo_text)
        dot = XlaComputation(hlo_module.as_serialized_hlo_module_proto()
                            ).as_hlo_dot_graph()
        gvz = graphviz.Source(dot)
    write_graphviz(gvz, filename)
    return fun(*args, **kwargs)

  return _wrapped_func
