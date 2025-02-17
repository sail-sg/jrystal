"""Calculation module for band structure and energy."""

# from .calc_band_structure import calc as band
from .calc_ground_state_energy import calc as energy
from .calc_band_structure import calc as band
from .calc_ground_state_energy_ncpp import calc as energy_ncpp
from .calc_band_structure_ncpp import calc as band_ncpp

__all__ = ["band", "energy", "energy_ncpp", "band_ncpp"]
