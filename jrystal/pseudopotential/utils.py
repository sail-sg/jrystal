from functools import wraps
from typing import List, Callable, Any


def map_over_atoms(fun: Callable[..., Any]) -> Callable[..., List]:
  """A wrapper function that extends a single-atom operation to apply to all
    atoms in a collection.

    If f is a function that operates on an array x representing a single atom,
    then map_over_atoms(f) transforms f to operate on a list of arrays x_list,
    applying the function f to each array in the list, representing a series of
    atoms, and return a list of the outputs for all the atoms.

  Args:
      fun (Callable[..., Any]): _description_

  Returns:
      Callable: _description_
  """

  @wraps(fun)
  def map_fun(*arrays, **kwargs):

    num_atoms = len(arrays[0])
    output = [
      fun(*(arr[i] for arr in arrays), **kwargs) for i in range(num_atoms)
    ]
    return output

  return map_fun
