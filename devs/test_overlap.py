"""Diamond test"""
import numpy as np
from ase import Atoms
from gpaw import GPAW, PW

name = 'C-diamond'
a = 3.5668  # diamond lattice parameter
b = a / 2

bulk = Atoms('C',
             cell=[[0, b, b],
                   [b, 0, b],
                   [b, b, 0]],
             pbc=True)

k = 8
calc = GPAW(mode=PW(300),       # cutoff
            setups = 'paw',  # Use default PAW setups
            kpts=(k, k, k),     # k-points
            txt=None        # Suppress GPAW output
      )

bulk.calc = calc

energy = bulk.get_potential_energy()

# Access wave functions
wfs = calc.wfs
pt = wfs.pt  # Projector functions handler

# The expand method computes <G|p_i> for all G and all projector functions i
q = 0  # k-point index
f_GI = pt.expand(q=q)  # This is the plane wave-projector overlap matrix!

print(f"\nPlane wave-projector overlap matrix <G|p_i>:")
print(f"  Shape: {f_GI.shape}")

# Decode the shape (real dtype uses special layout)
if wfs.dtype == complex:
    n_G = f_GI.shape[0]
else:
    n_G = f_GI.shape[0] // 2  # Real dtype interleaves real/imag
n_proj_total = f_GI.shape[1]

# Carbon atom setup details
for key in calc.setups.setups.keys():
    setup = calc.setups.setups[key]
    print(f"\n{key} atom setup:")
    print(f"  Number of projector types: {len(setup.pt_j)}")
    print(f"  Angular momenta l_j: {setup.l_j}")
    n_proj_atom = sum(2*l+1 for l in setup.l_j)
    print(f"  Total projector functions per atom: {n_proj_atom}")
    print(f"  M_pp shape: {setup.M_pp.shape}")
    
    # PAW overlap correction matrix
    print(f"  dO_ii (overlap correction) shape: {setup.dO_ii.shape}")

# Get grid info for normalization
pd = wfs.pd
N_total = pd.gd.N_c.prod()
alpha = 1.0 / N_total

print(f"\n=== Overlap Matrix Construction ===")
S_GG_block = np.eye(n_G, dtype=complex)

# Add PAW corrections - f_GI already contains proper <G|p_i>
# For overlap matrix, we need raw f_GI without additional normalization
if wfs.dtype == complex:
    f_GI_block = f_GI * np.sqrt(alpha)  # Normalize for physical units
else:
    f_GI_block = (f_GI[::2] + 1j * f_GI[1::2]) * np.sqrt(alpha)

# Get overlap correction (use first atom)
setup_key = list(calc.setups.setups.keys())[0]
dO_ii = calc.setups.setups[setup_key].dO_ii

# Add PAW correction: f_GI @ dO_ii @ f_GI†
S_GG_block += f_GI_block @ dO_ii @ f_GI_block.conj().T

print(f"\nOverlap matrix block S_GG':")
print(f"  Shape: {S_GG_block.shape}")
eigenvalues = np.linalg.eigvalsh(S_GG_block)
print(f"  Min: {eigenvalues.min():.6f}")
print(f"  Max: {eigenvalues.max():.6f}")
print(f"  All positive: {np.all(eigenvalues > 0)}")

# Verify projector coefficients P_ni = <ψ_n|p_i>
kpt = wfs.kpt_u[0]  # First k-point (spin up)
psit_nG = kpt.psit_nG
n_bands = psit_nG.shape[0]
print(f"\n=== Projector Coefficient Verification ===")
print(f"  psit_nG shape: {psit_nG.shape} (bands × plane waves)")

# Get phase factor for atom position
kd = pd.kd
if kd is None or kd.gamma:
    eikR = 1.0  # Gamma point
else:
    # For k-point q, phase factor e^(ik·R_a)
    # Get actual atom position
    spos_ac = bulk.get_scaled_positions()  # Get fractional coordinates
    print(f"  Atom position (fractional): {spos_ac[0]}")
    print(f"  k-point (fractional): {kd.ibzk_qc[q]}")
    eikR = np.exp(2j * np.pi * np.dot(kd.ibzk_qc[q], spos_ac.T))[0]

if wfs.dtype == complex:
    # integrate method: uses cc=True in expand for complex dtype
    # which means f_GI is already conjugated, so we need to conjugate it back
    f_GI_cc = pt.expand(q=q, cc=True)  # Get conjugated version like integrate does
    P_ni_manual = alpha * psit_nG @ f_GI_cc * eikR
else:
    # For real dtype, convert f_GI layout
    f_GI_complex = f_GI[::2] + 1j * f_GI[1::2]
    # For real dtype, integrate doesn't use cc
    P_ni_manual = alpha * psit_nG @ f_GI_complex * eikR

print(f"\nProjector coefficients P_ni = <ψ_n|p_i>:")
print(f"  N_total (FFT grid points): {N_total}")
print(f"  Normalization alpha = 1/N_total: {alpha:.6e}")
print(f"  Phase factor eikR: {eikR}")

# Compare with stored P_ani
P_ani = kpt.P_ani
if 0 in P_ani and P_ani[0] is not None:
    P_ni_stored = P_ani[0]
    print(f"\nStored P_ani[0] shape: {P_ni_stored.shape}")
    print(f"  First band stored: {P_ni_stored[0,:5]}...")
    
    if P_ni_stored.shape == P_ni_manual.shape:
        diff = np.abs(P_ni_stored - P_ni_manual).max()
        print(f"  Max difference (stored vs computed): {diff:.2e}")
        
        # Check if it's just a phase difference
        ratio = P_ni_stored[0,0] / P_ni_manual[0,0]
        print(f"  Ratio stored/computed: {np.abs(ratio):.6f}")
        
        if diff < 1e-10:
            print(f"  ✓ PERFECT MATCH! P_ani calculation verified.")

breakpoint()
