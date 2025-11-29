"""Test to understand the complex conjugation convention in GPAW"""
import numpy as np
from ase import Atoms
from gpaw import GPAW, PW

# Create a simple system with non-zero atomic position
a = 3.5668
atoms = Atoms('H2', 
              positions=[[0, 0, 0], [a/4, a/4, a/4]],  # Non-zero position for second atom
              cell=[a, a, a],
              pbc=True)

calc = GPAW(mode=PW(300),
            kpts=(2, 2, 2),  # Use k-points to test phase factors
            txt=None)

atoms.calc = calc
energy = atoms.get_potential_energy()

wfs = calc.wfs
pt = wfs.pt
pd = wfs.pd

# Test at first non-gamma k-point
q = 1  # Non-gamma point
kpt = wfs.kpt_u[q]

print("=== Testing Complex Conjugation Convention ===")
print(f"k-point index: {q}")
print(f"k-vector: {pd.kd.ibzk_qc[q]}")

# Get f_GI with different cc flags
f_GI_nocc = pt.expand(q=q, cc=False)
f_GI_cc = pt.expand(q=q, cc=True)

print(f"\nf_GI shape: {f_GI_nocc.shape}")
print(f"dtype: {wfs.dtype}")

# For complex dtype, check the difference
if wfs.dtype == complex:
    # Check if cc=True gives complex conjugate
    diff_conj = np.abs(f_GI_cc - f_GI_nocc.conj()).max()
    print(f"\nMax diff between f_GI(cc=True) and conj(f_GI(cc=False)): {diff_conj:.2e}")
    
    if diff_conj < 1e-10:
        print("✓ cc=True gives complex conjugate of cc=False")
    
    # Now test P_ani calculation
    psit_nG = kpt.psit_nG
    N_total = pd.gd.N_c.prod()
    alpha = 1.0 / N_total
    
    # Get atomic positions for phase factors
    spos_ac = atoms.get_scaled_positions()
    
    # Calculate P_ani manually with both conventions
    print("\n=== Testing P_ani calculation ===")
    
    # Method 1: Using cc=False (like overlap matrix)
    eikR = np.exp(2j * np.pi * np.dot(pd.kd.ibzk_qc[q], spos_ac.T))
    P_ni_nocc = alpha * psit_nG @ f_GI_nocc @ np.diag(eikR)
    
    # Method 2: Using cc=True (like integrate method)
    P_ni_cc = alpha * psit_nG @ f_GI_cc @ np.diag(eikR)
    
    # Method 3: What we might expect mathematically
    # <ψ_n|p_i> = (1/N) Σ_G c_nG* <G|p_i>
    P_ni_math = alpha * psit_nG.conj() @ f_GI_nocc @ np.diag(eikR)
    
    # Get stored P_ani for comparison
    pt.integrate(psit_nG, kpt.P_ani, q)
    
    # Extract P_ani for both atoms
    P_stored = []
    for a in range(len(atoms)):
        if a in kpt.P_ani:
            P_stored.append(kpt.P_ani[a])
    P_stored = np.hstack(P_stored) if P_stored else None
    
    if P_stored is not None:
        print(f"P_stored shape: {P_stored.shape}")
        
        # Compare different methods
        diff_nocc = np.abs(P_stored - P_ni_nocc).max()
        diff_cc = np.abs(P_stored - P_ni_cc).max()
        diff_math = np.abs(P_stored - P_ni_math).max()
        
        print(f"\nDifferences from stored P_ani:")
        print(f"  Using cc=False: {diff_nocc:.2e}")
        print(f"  Using cc=True:  {diff_cc:.2e}")
        print(f"  Using conj(psit) @ f_GI(cc=False): {diff_math:.2e}")
        
        # Find which one matches
        if diff_cc < 1e-10:
            print("\n✓ GPAW uses: psit @ f_GI(cc=True)")
        elif diff_nocc < 1e-10:
            print("\n✓ GPAW uses: psit @ f_GI(cc=False)")
        elif diff_math < 1e-10:
            print("\n✓ GPAW uses: conj(psit) @ f_GI(cc=False)")