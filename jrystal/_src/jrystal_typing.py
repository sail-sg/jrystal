"""Customized typing modules.

NOTE: This module will be retired in future version.

Variable type and associated typings:

- A variable named with postfix ``_vector_grid`` implies it has shape
[*batch n1, n2, n3, 3].

- A variable named with postfix ``_grid`` implies it has shape
[*batch n1, n2, n3]

Ref.: /jrystal/docs/symbol.md

"""
from typing import Union
from jaxtyping import Array, Float, Complex, Bool, Int
from typing_extensions import TypeAlias

RealVecterGrid: TypeAlias = Float[Array, "... n1 n2 n3 3"]
RealGrid: TypeAlias = Float[Array, "... n1 n2 n3"]
RealScalar: TypeAlias = Float[Array, " "]

ComplexVecterGrid: TypeAlias = Complex[Array, "... n1 n2 n3 3"]
ComplexGrid: TypeAlias = Complex[Array, "... n1 n2 n3"]

MaskGrid: TypeAlias = Bool[Array, "... n1 n2 n3"]
OccupationArray: TypeAlias = Union[Float[Array, "2 num_k num_bands"],
                                   Int[Array, "2 num_k num_bands"]]
CellVector: TypeAlias = Float[Array, "d d"]
