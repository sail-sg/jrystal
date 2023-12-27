import jax
from jax import core
# import jaxlib
import os
import graphviz
from jaxlib.xla_extension import hlo_module_from_text, XlaComputation
import itertools
from graphviz import Digraph
import webbrowser  # noqa
import subprocess
from functools import partial


def is_jupyter_notebook():
  try:
    from IPython import get_ipython
    if 'IPKernelApp' not in get_ipython().config:
      return False
  except ImportError:
    return False
  except AttributeError:
    return False
  return True


is_jupyter = is_jupyter_notebook()


def view(filename, view_command=None):
  if view_command is not None:
    subprocess.Popen(
      [view_command, filename],
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
    )
  elif is_jupyter:
    from IPython.display import display, Javascript
    display(Javascript('window.open("{filename}");'.format(filename=filename)))
  else:
    webbrowser.open_new_tab(f"file://{os.path.abspath(filename)}")


def write_graphviz(gvz, filename):
  gvz.render(
    filename=filename,
    directory=".",
    cleanup=True,
    format="pdf",
    view=False,
  )
  return filename + ".pdf"


def write_txt(text, filename):
  with open(f"{filename}.txt", "w") as f:
    f.write(text)
  return filename + ".txt"


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
    if graph:
      view(write_graphviz(gvz, filename), view_command=view_command)
    if txt:
      view(write_txt(hlo_text, filename), view_command=view_command)
    return fun(*args, **kwargs)

  return _wrapped_func


def jaxpr_to_dot_graph(jaxpr):
  """code adapted from
  https://gist.github.com/niklasschmitz/559a1f717f3535db0e26d0edccad0b46
  """
  styles = {
    "const": {
      "style": "filled", "color": "goldenrod1"
    },
    "invar": {
      "color": "mediumspringgreen", "style": "filled"
    },
    "outvar":
      {
        "style": "filled,dashed", "fillcolor": "indianred1", "color": "black"
      },
    "op_node": {
      "shape": "box", "color": "lightskyblue", "style": "filled"
    },
    "intermediate": {
      "style": "filled", "color": "cornflowerblue"
    }
  }
  id_names = (f'id{id}' for id in itertools.count())
  graph = Digraph(engine='dot')
  graph.attr(size='6,10!')
  for v in jaxpr.constvars:
    graph.node(
      str(v), core.raise_to_shaped(v.aval).str_short(), styles['const']
    )
  for v in jaxpr.invars:
    graph.node(str(v), v.aval.str_short(), styles['invar'])
  for eqn in jaxpr.eqns:
    for v in eqn.invars:
      if isinstance(v, core.Literal):
        graph.node(
          str(id(v.val)),
          core.raise_to_shaped(core.get_aval(v.val)).str_short(),
          styles['const']
        )
    if eqn.primitive.multiple_results:
      id_name = next(id_names)
      graph.node(id_name, str(eqn.primitive), styles['op_node'])
      for v in eqn.invars:
        graph.edge(
          str(id(v.val) if isinstance(v, core.Literal) else v), id_name
        )
      for v in eqn.outvars:
        graph.node(str(v), v.aval.str_short(), styles['intermediate'])
        graph.edge(id_name, str(v))
    else:
      outv, = eqn.outvars
      graph.node(str(outv), str(eqn.primitive), styles['op_node'])
      for v in eqn.invars:
        graph.edge(
          str(id(v.val) if isinstance(v, core.Literal) else v), str(outv)
        )
  for i, v in enumerate(jaxpr.outvars):
    outv = 'out_' + str(i)
    graph.node(outv, outv, styles['outvar'])
    graph.edge(str(v), outv)
  return graph


def view_jaxpr(*args, graph=True, txt=False, view_command=None):
  if len(args) == 1 and callable(args[0]):
    return _view_jaxpr(args[0], graph=graph, txt=txt, view_command=view_command)
  return partial(_view_jaxpr, graph=graph, txt=txt, view_command=view_command)


def _view_jaxpr(fn, *, graph, txt, view_command):
  """Decorator to view the jaxpr graph of a function.

  Usage:

  .. code-block:: python

    @view_jaxpr
    def f(x):
      return jnp.sin(x)

  Args:
    fun: a python function
  Returns:
    A wrapped function that will display the jaxpr graph when called.
  """

  def _wrapped_func(*args, **kwargs):
    filename = f"jaxpr_of_{fn.__name__}@{id(fn)}"
    closed_jaxpr = jax.make_jaxpr(fn)(*args, **kwargs)
    gvz = jaxpr_to_dot_graph(closed_jaxpr.jaxpr)
    if graph:
      view(write_graphviz(gvz, filename), view_command=view_command)
    if txt:
      view(
        write_txt(str(closed_jaxpr.jaxpr), filename), view_command=view_command
      )
    return fn(*args, **kwargs)

  return _wrapped_func
