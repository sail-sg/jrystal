import jax
from jax import core
# import jaxlib
import graphviz
from jaxlib.xla_extension import hlo_module_from_text, XlaComputation
import itertools
from graphviz import Digraph
import webbrowser


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


def view_hlo(fun, optimized=True):
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
    if not optimized:
      xla_comp = jax.xla_computation(fun)(*args, **kwargs)
      dot = xla_comp.as_hlo_dot_graph()
      gvz = graphviz.Source(dot)
    else:
      hlo_text = fun.lower(*args, **kwargs).compile().as_text()
      hlo_module = hlo_module_from_text(hlo_text)
      dot = XlaComputation(hlo_module.as_serialized_hlo_module_proto()
                          ).as_hlo_dot_graph()
      gvz = graphviz.Source(dot)

    filename = "optimized_" * optimized + f"hlo_of_{fun.__name__}@{id(fun)}"
    is_jupyter = is_jupyter_notebook()
    gvz.render(
      filename=filename,
      directory=".",
      cleanup=True,
      format="pdf",
      view=(not is_jupyter),
    )
    if is_jupyter:
      from IPython.display import display, Javascript
      display(
        Javascript('window.open("{filename}.pdf");'.format(filename=filename))
      )
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


def view_jaxpr(fn):
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
    closed_jaxpr = jax.make_jaxpr(fn)(*args, **kwargs)
    gvz = jaxpr_to_dot_graph(closed_jaxpr.jaxpr)
    filename = f"jaxpr_of_{fn.__name__}@{id(fn)}"
    is_jupyter = is_jupyter_notebook()
    gvz.render(
      filename=filename,
      directory=".",
      cleanup=True,
      format="pdf",
      view=(not is_jupyter),
    )
    if is_jupyter:
      from IPython.display import display, Javascript
      display(
        Javascript('window.open("{filename}.pdf");'.format(filename=filename))
      )
    return fn(*args, **kwargs)

  return _wrapped_func
