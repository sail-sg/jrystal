"""
XC Spherical Grid Loader and PAW XC Correction

This module implements a customized PAW XC correction calculation
and compares results with GPAW.

Integration pattern (from GPAW lda.py):
    E_xc = sum_n w_n * integral r^2 dr e_xc(r, Omega_n)

where:
- n indexes 50 Lebedev angular points
- w_n are angular weights
- e_xc(r, Omega_n) is XC energy density at angular point Omega_n
- The radial integral is done separately for each angular point
"""

from math import sqrt, pi
import atexit
import gc

from ase.build import bulk
import numpy as np

from gpaw import GPAW, PW

# Load Lebedev grid directly from GPAW
from gpaw.sphere.lebedev import R_nv, weight_n, Y_nL

import sys
from pathlib import Path
devs_dir = Path(__file__).parent.parent
if str(devs_dir) not in sys.path:
    sys.path.insert(0, str(devs_dir))
from devs import calc_paw


def print_grid_info():
    """Print basic info about the Lebedev grid."""
    print("Lebedev Angular Grid (50 points)")
    print("=" * 40)
    print(f"R_nv shape: {R_nv.shape}  (unit vectors)")
    print(f"weight_n shape: {weight_n.shape}")
    print(f"Y_nL shape: {Y_nL.shape}  (spherical harmonics)")
    print(f"Sum of weights: {weight_n.sum():.10f} (should be 1)")
    print(f"4*pi * sum = {4*np.pi*weight_n.sum():.6f} (should be {4*np.pi:.6f})")

    print("\nFirst 5 angular points:")
    for n in range(5):
        print(f"  n={n}: R=({R_nv[n,0]:+.4f}, {R_nv[n,1]:+.4f}, {R_nv[n,2]:+.4f}), w={weight_n[n]:.6f}")


def get_gpaw_paw_xc_correction(calc):
    """
    Get PAW XC correction from GPAW for comparison.

    Returns:
        delta_e_xc: Total PAW XC correction from GPAW (in Hartree)
    """
    D_asp = calc.density.D_asp
    setups = calc.wfs.setups
    xc = calc.hamiltonian.xc

    delta_e_xc_total = 0.0
    for a, D_sp in D_asp.items():
        setup = setups[a]
        delta_e_xc = xc.calculate_paw_correction(setup, D_sp, a=a)
        delta_e_xc_total += delta_e_xc
        print(f"GPAW Atom {a}: Delta E_xc = {delta_e_xc:.6f} Ha")

    return delta_e_xc_total


if __name__ == '__main__':
    # Setup diamond calculation
    name = 'C-diamond'
    a = 3.5668  # diamond lattice parameter in Angstrom
    atoms = bulk('C', 'diamond', a=a)

    k = 2
    calc = GPAW(mode=PW(200),
                xc='LDA',
                setups='paw',
                kpts=(k, k, k),
                txt=None)

    atoms.calc = calc
    energy = atoms.get_potential_energy()

    # Compare PAW XC corrections
    print("\n" + "=" * 60)
    print("GPAW PAW XC Correction:")
    print("=" * 60)
    gpaw_delta_e_xc = get_gpaw_paw_xc_correction(calc)
    print(f"Total GPAW Delta E_xc: {gpaw_delta_e_xc:.6f} Ha")

    print("\n" + "=" * 60)
    print("Customized PAW XC Correction:")
    print("=" * 60)
    custom_delta_e_xc = calc_paw.calc_paw_xc_correction(
        calc.density.D_asp,
        calc.wfs.setups,
        calc.hamiltonian.xc.kernel,
        calc.wfs.nspins
    )
    print(f"Total Custom Delta E_xc: {custom_delta_e_xc:.6f} Ha")

    print("\n" + "=" * 60)
    print("Comparison:")
    print("=" * 60)
    diff = abs(gpaw_delta_e_xc - custom_delta_e_xc)
    print(f"GPAW:   {gpaw_delta_e_xc:.10f} Ha")
    print(f"Custom: {custom_delta_e_xc:.10f} Ha")
    print(f"Diff:   {diff:.2e} Ha")

    if diff < 1e-6:
        print("\nSUCCESS: Results match within 1e-6 Ha!")
    else:
        print("\nWARNING: Results differ - debugging needed")

    # Explicitly close and delete calculator to avoid cleanup errors at exit
    calc.close()
    del calc
    del atoms
    gc.collect()

    # Suppress remaining cleanup warnings at exit
    import sys
    import os
    sys.stderr = open(os.devnull, 'w')
