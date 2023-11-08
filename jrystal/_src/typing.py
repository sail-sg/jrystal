"""Customized typing modules.

Variable type and associated typings:

- A variable named with postfix ``_vector_grid`` implies it has shape 
[*batch n1, n2, n3, 3].

- A variable named with postfix ``_grid`` implies it has shape 
[*batch n1, n2, n3]

Ref. /jrystal/docs/symbol.md

"""
from jaxtyping import Array, Float, Complex
from typing_extensions import TypeAlias

RealVecterGrid: TypeAlias = Float[Array, "... n1 n2 n3 3"]
RealGrid: TypeAlias = Float[Array, "... n1 n2 n3"]

ComplexVecterGrid: TypeAlias = Complex[Array, "... n1 n2 n3 3"]
ComplexGrid: TypeAlias = Complex[Array, "... n1 n2 n3"]
