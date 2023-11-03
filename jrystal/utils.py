import jax
import jaxlib
import graphviz
from jaxlib.xla_extension import hlo_module_from_text, XlaComputation


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
      graphviz.Source(dot).view()
    else:
      hlo_text = fun.lower(*args, **kwargs).compile().as_text()
      hlo_module = hlo_module_from_text(hlo_text)
      dot = XlaComputation(hlo_module.as_serialized_hlo_module_proto()
                          ).as_hlo_dot_graph()
      graphviz.Source(dot).view()
    return fun(*args, **kwargs)

  return _wrapped_func
