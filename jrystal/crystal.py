"""The Crystal class.

This module establishes the interface for the crystal structure. It's important
to note that all values are presented in ATOMIC UNITS inside, specifically
hartree for energy and Bohr for length. The input positions of atoms should be
in angstrom.
"""


from ._src.crystal import Crystal

__all__ = ["Crystal"]
