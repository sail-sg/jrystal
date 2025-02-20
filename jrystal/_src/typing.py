"""Customized typing modules.

NOTE: This module might be retired in future version.

Variable type and associated typings:

- A variable named with postfix ``_vector_grid`` implies it has shape ``[*batch n1, n2, n3, 3]``.

- A variable named with postfix ``_grid`` implies it has shape ``[*batch n1, n2, n3]``

Ref.: /jrystal/docs/symbol.md

"""
from typing import Union, Tuple, TypeVar
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
"""occupation"""

CellVector: TypeAlias = Float[Array, "3 3"]
"""cell vectors"""
