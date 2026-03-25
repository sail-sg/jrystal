"""Customized typing modules.

.. note::
   This module might be retired in future version.

Variable Type Conventions
------------------------

The following naming conventions are used for variable types:

* Variables with suffix ``_vector_grid`` have shape ``[*batch x y z 3]``
* Variables with suffix ``_grid`` have shape ``[*batch x y z]``

See Also
--------
For more details, refer to ``/jrystal/docs/symbol.md``

Variable type and associated typings:

- A variable named with postfix ``_vector_grid`` implies it has shape ``[*batch x y z 3]``.

- A variable named with postfix ``_grid`` implies it has shape ``[*batch x y z]``

Ref.: /jrystal/docs/symbol.md

"""
from typing import Tuple, TypeVar, Union

from jaxtyping import Array, Float, Int
from typing_extensions import TypeAlias


class VectorGrid:
  """Defines a vector field, which has shape (*batches, n1, ..., nd, d)
  """

  def __class_getitem__(cls, params: Tuple[TypeVar, int]):
    if len(params) != 2 or not isinstance(params[1], int):
      raise TypeError("VectorGrid takes 2 parameters, dtype and ndim")
    dtype, ndim = params
    grid_size = " ".join([f"n{i+1}" for i in range(ndim)])
    return dtype[Array, f"... {grid_size} {ndim}"]


class ScalarGrid:
  """Defines a scalar field, which has shape (*batches, n1, ..., nd)
  """

  def __class_getitem__(cls, params: Tuple[TypeVar, int]):
    if len(params) != 2 or not isinstance(params[1], int):
      raise TypeError("VectorGrid takes 2 parameters, dtype and ndim")
    dtype, ndim = params
    grid_size = " ".join([f"n{i+1}" for i in range(ndim)])
    return dtype[Array, f"... {grid_size}"]


OccupationArray: TypeAlias = Union[Float[Array, "spin kpts band"],
                                   Int[Array, "spin kpts band"]]
"""Occupation array. Shape: (num_spin, num_k, num_band). """

CellVector: TypeAlias = Float[Array, "3 3"]
"""Cell vectors. Shape: (3, 3). """
