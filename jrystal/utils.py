import jax
import jaxlib
import graphviz


def view_hlo(fun):
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
    print_opt = jaxlib.xla_extension.HloPrintOptions()
    print_opt.print_backend_config = True
    xla_comp = jax.xla_computation(fun)(*args, **kwargs)
    dot = xla_comp.as_hlo_dot_graph()
    graphviz.Source(dot).view()
    return fun(*args, **kwargs)

  return _wrapped_func
