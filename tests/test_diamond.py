"""Run a diamond calculation and verify PAW plane-wave orthonormality."""

from ase.build import bulk
import numpy as np

from gpaw import GPAW, PW

RUN_ALIGNMENT = False

"""
coeff: calc.wfs.kpt_u[0].psit_nG
volume: calc.wfs.pd.gd.volume
pw-projector overlap: 
f_GI = calc.wfs.pt.expand(q=k_index, cc=False)
if wfs.dtype == float:
    # Fold GPAW's real layout into complex coefficients
    f_GI = f_GI[::2] + 1j * f_GI[1::2]
f_GI = f_GI[:, :13]
"""

def _compute_overlap_matrix_custom(calc, k_index):
    """
    Build the PAW overlap matrix S_nm in band space from scratch.

    The overlap matrix in band space is:
    S_nm = <ψ_n|S|ψ_m> = δ_nm + Σ_a <ψ_n|p̃_a><p̃_a|S_a|p̃_a><p̃_a|ψ_m>

    Where:
    - ψ_n are the wavefunctions (plane-wave coefficients C_nG)
    - p̃_a are the projector functions
    - S_a are the atomic overlap operators

    Returns:
        S_nm: Overlap matrix in band space (n_bands x n_bands)
    """
    wfs = calc.wfs
    kpt = wfs.kpt_u[k_index]

    # Get plane-wave coefficients C_nG
    C_nG = kpt.psit_nG  # shape: (n_bands, n_pw)
    n_bands, n_pw = C_nG.shape

    print(f'  k={k_index}: n_bands={n_bands}, n_pw={n_pw}')

    # Step 1: Compute plane-wave part of overlap
    # For plane waves: <ψ_n|ψ_m> = Σ_G C*_nG C_mG × normalization
    # The normalization depends on the wavefunction type

    if wfs.dtype == float:
        # Real wavefunctions use special normalization
        # GPAW uses 2*dv for real wavefunctions in matrix_elements
        dv = wfs.pd.gd.volume / wfs.pd.gd.N_c.prod()**2

        # For real wavefunctions, we need to use the actual matrix storage
        if hasattr(kpt.psit, 'matrix'):
            # Use the raw matrix array which has special layout for real wfs
            M = kpt.psit.matrix.array
            S_nm = (2 * dv) * (M @ M.T)

            # Correction for G=0 component (from arrays.py lines 228-235)
            # Note the factor of 0.5 for symmetric case!
            G0_component = M[:, 0]
            correction = np.outer(G0_component, G0_component)
            S_nm -= 0.5 * dv * (correction + correction.T)
        else:
            # Fallback if matrix not available
            S_nm = C_nG @ C_nG.conj().T * (2 * dv)
    else:
        # Complex wavefunctions use standard normalization
        dv = wfs.pd.gd.volume / wfs.pd.gd.N_c.prod()**2
        S_nm = C_nG @ C_nG.conj().T * dv

    print(f'    Plane-wave overlap diagonal: {np.diag(S_nm).real[:4]}...')

    # Step 2: Add PAW corrections using projections
    # The projections P_ni = <p̃_i|ψ_n> should already be computed
    P_ani = kpt.P_ani

    if P_ani is not None:
        # Add PAW correction: Σ_a P*_na dS_a P_ma
        for a, P_ni in P_ani.items():
            dO_ii = calc.setups[a].dO_ii  # This is dS_ii = S_ii - I_ii
            # P_ni is (n_bands, n_projectors)
            PAW_correction = P_ni @ dO_ii @ P_ni.T.conj()
            S_nm += PAW_correction

        print(f'    After PAW corrections diagonal: {np.diag(S_nm).real[:4]}...')

    return S_nm


def _compute_overlap_matrix(calc, k_index):
    """Build the PAW overlap matrix S in G-vector space for a single k-point.

    The overlap matrix in G-space is:
    S_GG' = δ_GG' + Σ_a <φ_G|p̃_a><p̃_a|φ_G'> ΔS_a

    where φ_G = e^{iG·r} are plane waves and p̃_a are projectors at atom a.
    """
    wfs = calc.wfs
    pt = wfs.pt
    f_GI = pt.expand(q=k_index, cc=False)

    if wfs.dtype == float:
        # Fold GPAW's real layout into complex coefficients
        f_GI = f_GI[::2] + 1j * f_GI[1::2]

    n_pw = wfs.ng_k[k_index]
    # Note: f_GI might have different number of G-vectors than n_pw
    # We need to make sure we use the right G-vectors
    if f_GI.shape[0] != n_pw:
        print(f'  Warning: f_GI has {f_GI.shape[0]} G-vectors but n_pw={n_pw}')
        # For now, truncate or pad as needed
        if f_GI.shape[0] > n_pw:
            f_GI = f_GI[:n_pw]
        else:
            # This shouldn't happen, but handle it
            raise ValueError(f"f_GI has fewer G-vectors ({f_GI.shape[0]}) than n_pw ({n_pw})")

    # Build the overlap correction matrix dO
    n_proj = f_GI.shape[1]
    dO = np.zeros((n_proj, n_proj), dtype=np.complex128)

    for atom_index, start, stop in pt.my_indices:
        dO_block = calc.setups[atom_index].dO_ii.astype(np.complex128)
        print(f'    Atom {atom_index}: block ({start},{stop}), shape={dO_block.shape}, max|dO|={np.abs(dO_block).max():.3e}')
        dO[start:stop, start:stop] = dO_block

    # Build S in G-space
    tmp_mat = np.eye(14) * 2
    tmp_mat[0, 0] = 1
    S = np.eye(n_pw, dtype=np.complex128) * wfs.pd.gd.volume * 2 + tmp_mat @ f_GI.conj() @ dO @ f_GI.T @ tmp_mat
    S[0, 0] -= wfs.pd.gd.volume  # Correct G=0 term for real wavefunctions

    C = wfs.kpt_u[k_index].psit_nG
    S_ = np.eye(n_pw, dtype=np.complex128) * wfs.pd.gd.volume * 2
    S_[0, 0] -= wfs.pd.gd.volume
    S_gpaw = calc.wfs.kpt_u[0].psit.matrix_elements(symmetric=True, cc=True)
    S_gpaw.array - C @ S_ @ C.conj().T / 216**2
    # P_ni = (C @ tmp_mat @ f_GI).real
    # NOTE: the f_GI.conj() here is very very important!
    P_ni = (C @ tmp_mat @ f_GI.conj()).real
    # W = np.diag([1,-1,-1,-1,1,-1,-1,-1,1,1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,1,1])
    (C @ S_ @ C.conj().T).real / 216**2 + P_ni @ dO @ P_ni.T / 216**2
    # breakpoint()

    return S, f_GI


def _compute_proj_pw_overlap():
    """
    This is the customized function to compute the projector-plane wave overlap matrix:
    We perform the radial integration in real space and compare the results with f_GI
    """

    import sys
    from pathlib import Path

    # Add devs directory to Python path
    devs_dir = Path(__file__).parent.parent
    if str(devs_dir) not in sys.path:
        sys.path.insert(0, str(devs_dir))

    from jrystal.calc.gpaw_load import parse_paw_setup
  
    # Load GPAW setup file
    pp_data = parse_paw_setup(f'/home/aiops/zhaojx/M_p-align-claude/pseudopotential/N.LDA')

    from scipy.special import spherical_jn

    gcut2 = 258
    grid_info = pp_data['radial_grid']
    a = grid_info['a']
    n = grid_info['n']
    i = np.arange(n)
    r_g = a * i / (n - i)  # Keep original grid for g_lg calculation
    dr_g = a * n / (n - i)**2
    r_g = r_g[:gcut2]
    dr_g = dr_g[:gcut2]
    pt_jg = np.array([proj['values'][:gcut2] for proj in pp_data['projector_functions']])
    overlap = np.zeros((14, 13), dtype=np.complex128)

    proj_list = [0, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4]
    m_list = [0, -1, 0, 1, 0, -1, 0, 1, -2, -1, 0, 1, 2]
    l_list = [0, 1, 1, 1, 0, 1, 1, 1, 2, 2, 2, 2, 2]
    G_grid = calc.wfs.pd.get_reciprocal_vectors(q=0)
    from scipy.special import sph_harm_y
    theta_grid = np.arccos(G_grid[:, 2] / np.linalg.norm(G_grid, axis=1))
    theta_grid[0] = 0.0
    phi_grid = np.arctan2(G_grid[:, 1], G_grid[:, 0])
    phi_grid[0] = 0.0   # handle the G = 0 singular case
    # breakpoint()
    # cell_cv = bulk.get_cell()
    # G_grid @ cell_cv / 0.529177210671212
    for k in range(14):
        for j in range(13):
            bessel_grid = spherical_jn(l_list[j], r_g * np.linalg.norm(G_grid[k]))
            overlap[k, j] = np.sum(bessel_grid * pt_jg[proj_list[j]] * r_g * r_g * dr_g) * 4 * np.pi * (-1j) ** l_list[j] *\
                sph_harm_y(l_list[j], m_list[j], theta_grid[k], phi_grid[k])

    # transform the result from spherical harmonics to real spherical harmonics
    tmp_list = [0, 3, 2, 1, 4, 7, 6, 5, 12, 11, 10, 9, 8] # relate the m to -m indices
    overlap_ = overlap.copy()
    for j in range(13):
        if m_list[j] > 0:
            overlap_[:, j] = (overlap[:, tmp_list[j]] + (-1)**m_list[j] * overlap[:, j]) / np.sqrt(2)
        elif m_list[j] < 0:
            overlap_[:, j] = (overlap[:, j] - (-1)**m_list[j] * overlap[:, tmp_list[j]]) / (-1j * np.sqrt(2))
        
    return overlap_


def align_wavefunction():
    """Align wavefunction calculation with GPAW's IFFT on ALL grid points.

    This compares two approaches:
    1. GPAW's pd.ifft(): Uses FFT to transform C_G -> psi(R) on grid
    2. Direct calculation: psi(R) = (1/N) sum_G C_G * exp(i*G*R)

    Both should give IDENTICAL results on grid points (no interpolation).
    """
    print("=" * 60)
    print("WAVEFUNCTION ALIGNMENT TEST")
    print("=" * 60)

    # Get wavefunction coefficients
    C = calc.wfs.kpt_u[0].psit_nG  # [n_bands, n_G]
    n_bands, n_G = C.shape
    print(f"Number of bands: {n_bands}")
    print(f"Number of G-vectors: {n_G}")

    # Get G-vectors (in 1/Bohr)
    G = calc.wfs.pd.get_reciprocal_vectors(q=0)  # [n_G, 3]

    # Get grid info
    gd = calc.wfs.gd  # Coarse grid descriptor
    N_c = gd.N_c  # Grid dimensions
    N_fft = N_c.prod()
    print(f"FFT grid: {N_c} = {N_fft} points")

    # Get grid point coordinates (in Bohr!)
    r_grid = gd.get_grid_point_coordinates()  # [3, nx, ny, nz] in Bohr
    print(f"Grid coordinates shape: {r_grid.shape}")

    for band_idx in range(G.shape[0]):
        print(f"\n--- Band {band_idx} ---")
        C_G = C[band_idx]  # Coefficients for this band

        # ============================================
        # Method 1: GPAW's IFFT
        # ============================================
        psi_gpaw = calc.wfs.pd.ifft(C_G, q=0)  # Returns [nx, ny, nz]

        # ============================================
        # Method 2: Direct Fourier sum (on all grid points)
        # ============================================
        # For real wavefunctions at Gamma, GPAW stores only half G-vectors
        # The formula is: psi(r) = (1/N) * [C_0 + 2 * sum_{G>0} Re(C_G * exp(i*G*r))]

        # Reshape grid coordinates for broadcasting: [3, nx, ny, nz] -> [nx*ny*nz, 3]
        r_flat = r_grid.reshape(3, -1).T  # [N_fft, 3] in Bohr

        # Compute phase factors: exp(i * G @ r) for all G and all r
        # G: [n_G, 3] in 1/Bohr, r: [N_fft, 3] in Bohr
        # G @ r.T: [n_G, N_fft] - dimensionless (no unit conversion needed!)
        phase = np.exp(1j * G @ r_flat.T)  # [n_G, N_fft]

        # For real wavefunctions: multiply by 2 for G>0, keep 1 for G=0
        tmp_mat = np.eye(n_G) * 2
        tmp_mat[0, 0] = 1  # G=0 has no factor of 2

        # Direct calculation: (1/N) * C @ tmp_mat @ exp(i*G*r)
        psi_direct_flat = (C_G @ tmp_mat @ phase / N_fft).real  # [N_fft]
        psi_direct = psi_direct_flat.reshape(N_c)  # [nx, ny, nz]

        # ============================================
        # Compare
        # ============================================
        diff = np.abs(psi_gpaw - psi_direct)
        max_diff = diff.max()
        mean_diff = diff.mean()
        rel_diff = diff / (np.abs(psi_gpaw) + 1e-16)
        max_rel_diff = rel_diff.max()

        print(f"\n{'='*40}")
        print("COMPARISON RESULTS")
        print(f"{'='*40}")
        print(f"Max absolute difference: {max_diff:.2e}")
        print(f"Mean absolute difference: {mean_diff:.2e}")
        print(f"Max relative difference: {max_rel_diff:.2e}")

    # Check a few specific points
    print(f"\nPoint-by-point comparison:")
    for idx in [(0,0,0), (1,0,0), (0,1,0), (0,0,1), (1,1,1), (2,2,2)]:
        if all(i < N_c[j] for j, i in enumerate(idx)):
            g = psi_gpaw[idx]
            d = psi_direct[idx]
            print(f"  {idx}: GPAW={g:+.8f}, Direct={d:+.8f}, diff={abs(g-d):.2e}")

    if max_diff < 1e-10:
        print("\n✓ WAVEFUNCTION ALIGNED PERFECTLY!")
    else:
        print("\n✗ WAVEFUNCTION MISMATCH - debugging needed")
        breakpoint()

    # breakpoint()
    return psi_gpaw, psi_direct


def align_density(r: np.ndarray):
    """Align pseudo-density calculation with GPAW interpolation"""

    # Get wavefunction coefficients and G-vectors
    C = calc.wfs.kpt_u[0].psit_nG  # [n_bands, n_G]
    G = calc.wfs.pd.get_reciprocal_vectors(q=0)  # [n_G, 3]
    f_n = calc.get_occupation_numbers()  # [n_bands]
    tmp_mat = np.eye(C.shape[1]) * 2
    tmp_mat[0, 0] = 1  # G=0 component has no factor of 2
    N_fft = calc.wfs.gd.N_c.prod()
    phit_r = (C @ tmp_mat @ np.exp(1j * G @ r.T / 0.529177210671212) / N_fft).real  # [n_bands, n_points]
    nt_r = np.sum(f_n.reshape(-1, 1) * np.abs(phit_r)**2, axis=0)

    # Get GPAW's interpolated pseudo-density for comparison
    calc.density.interpolate_pseudo_density()
    nt_sg = calc.density.nt_sg  # Shape: (n_spins, nx, ny, nz)
    nct_G = calc.density.nct_G
    gd = calc.density.finegd  # Fine grid descriptor
    coarse_gd = calc.density.gd
    cell_cv = bulk.get_cell()
    r_scaled = np.linalg.solve(cell_cv.T, r.T).T  # Matrix equation: cell^T · r_scaled = r
    r_scaled = r_scaled % 1.0  # Wrap to [0,1)

    r_G = coarse_gd.get_grid_point_coordinates()
    phit_G = (C @ tmp_mat @ np.exp(1j * G @ r_G.reshape(3, -1)) / N_fft).real.reshape(8, 6, 6, 6)
    nt_G = np.sum(f_n[:, None, None, None] * np.abs(phit_G)**2, axis=0)
    # print(calc.density.nt_sG[0] - nt_G - nct_G)

    # Interpolate GPAW density at same position
    density_gpaw = gd.interpolate_grid_points(r_scaled, nt_sg[0]) -\
        coarse_gd.interpolate_grid_points(r_scaled, nct_G)

    # NOTE: the relative error is of order 3e-4, which seems still improvable
    print(f"GPAW interpolation density at r={r[0]}: {density_gpaw[0]:.6e}")
    print(f"Difference: {abs(nt_r[0] - density_gpaw[0]):.6e}")
    print(f"Relative difference: {abs(nt_r[0] - density_gpaw[0]) / density_gpaw[0] * 100:.4f}%")
    # breakpoint()
    assert np.allclose(nt_r, density_gpaw, rtol=1e-4), f"Density mismatch! Direct: {nt_r[0]:.6e}, GPAW: {density_gpaw[0]:.6e}"


name = 'C-diamond'
a = 3.5668  # diamond lattice parameter in Angstrom

bulk = bulk('C', 'diamond', a=a)

k = 1
Ha = 27.211386245988
cutoff_ev = 40.0 * Ha
calc = GPAW(mode=PW(cutoff_ev),  # cutoff energy in eV (match jrystal 40 Ha)
            xc='LDA',            # LDA (exchange+correlation); LDA_X needs datasets not present
            setups='paw',
            kpts=(k, k, k),      # Monkhorst-Pack grid
            txt=None)

bulk.calc = calc
energy = bulk.get_potential_energy()
print(energy)
h = calc.hamiltonian
print("GPAW energy components (Ha):")
print(f"  e_total_free: {h.e_total_free:.12f}")
print(f"  e_total_extrapolated: {h.e_total_extrapolated:.12f}")
print(f"  e_kinetic: {h.e_kinetic:.12f}")
print(f"  e_coulomb: {h.e_coulomb:.12f}")
print(f"  e_zero: {h.e_zero:.12f}")
print(f"  e_external: {h.e_external:.12f}")
print(f"  e_xc: {h.e_xc:.12f}")
print(f"  e_entropy: {h.e_entropy:.12f}")
print(f"  e_total_free (eV): {h.e_total_free * Ha:.12f}")
breakpoint()

# Split XC into pseudo (smooth grid) and atomic PAW correction parts
dens = calc.density
vtmp_sg = h.finegd.zeros(h.nspins)
e_xc_pseudo = h.xc.calculate(h.finegd, dens.nt_sg, vtmp_sg)
e_xc_pseudo /= h.finegd.comm.size
e_xc_atomic = 0.0
for a, D_sp in dens.D_asp.items():
    e_xc_atomic += h.xc.calculate_paw_correction(calc.setups[a], D_sp, a=a)
print("GPAW XC split (Ha):")
print(f"  e_xc_pseudo: {e_xc_pseudo:.12f}")
print(f"  e_xc_atomic: {e_xc_atomic:.12f}")
print(f"  e_xc_total: {e_xc_pseudo + e_xc_atomic:.12f}")

P_ani = calc.wfs.kpt_u[0].P_ani

if P_ani is not None:
    # Add PAW correction: Σ_a P*_na dS_a P_ma
    for a, P_ni in P_ani.items():
        dO_ii = calc.setups[a].dO_ii  # This is dS_ii = S_ii - I_ii
        # P_ni is (n_bands, n_projectors)
        PAW_correction = P_ni @ dO_ii @ P_ni.T.conj()
        f_GI = calc.wfs.pt.expand(q=0, cc=False)
        # breakpoint()

if RUN_ALIGNMENT:
    # Compare wavefunctions directly first (on ALL grid points)
    # align_wavefunction()

    # Then test density alignment
    align_density(np.array([[1.7834/2, 1.7834/2, 0]]))
    # align_density(np.array([[0, 0, 0]]))


# print('Energy:', energy, 'eV')
# print('Reference energy:', calc.get_reference_energy(), 'eV')
# print('Energy + reference:', energy + calc.get_reference_energy(), 'eV')
# print('Number of atoms:', len(bulk))
# print('Atomic numbers:', bulk.get_atomic_numbers())
# print('Number of bands:', calc.get_number_of_bands())
# print('Number of electrons:', calc.get_number_of_electrons())
# print('Hamiltonian free energy (Ha):', calc.hamiltonian.e_total_free)
# print('Hamiltonian kinetic (Ha):', calc.hamiltonian.e_kinetic)
# print('Hamiltonian coulomb (Ha):', calc.hamiltonian.e_coulomb)
# print('Hamiltonian zero/local (Ha):', calc.hamiltonian.e_zero)
# print('Hamiltonian xc (Ha):', calc.hamiltonian.e_xc)
# print('Hamiltonian entropy (Ha):', calc.hamiltonian.e_entropy)
# print('FFT grid points:', calc.wfs.pd.gd.N_c.prod())
# print('Cell volume:', calc.wfs.pd.gd.volume)

if RUN_ALIGNMENT:
    # verify the overlap matrix calculation
    mat1 = _compute_proj_pw_overlap()
    _, mat2 = _compute_overlap_matrix(calc, 0)
    # breakpoint()

    coefficients = {}
    overlap_matrices = {}

    # Compute G-space overlap matrices for each k-point
    for k_index in range(calc.wfs.kd.nibzkpts):
        overlap_matrices[k_index], _ = _compute_overlap_matrix(calc, k_index)

    for spin in range(calc.wfs.nspins):
        for k_index in range(calc.wfs.kd.nibzkpts):
            print(f'\n--- Spin={spin}, k={k_index} ---')

            # Method 1: Use GPAW's internal method for comparison
            kpt = calc.wfs.kpt_u[k_index]
            psit = kpt.psit
            P = kpt.projections

            S_gpaw = psit.matrix_elements(symmetric=True, cc=True)
            if P is not None:
                P2 = P.new()
                calc.wfs.setups.dS.apply(P, out=P2)
                from gpaw.utilities.blas import mmm
                mmm(1.0, P.array, 'N', P2.array, 'C', 1.0, S_gpaw.array)

            print(f'GPAW internal method:')
            print(f'  S diagonal: {np.diag(S_gpaw.array).real[:4]}...')
            print(f'  max|S - I| = {np.abs(S_gpaw.array - np.eye(len(S_gpaw.array))).max():.3e}')

            # Method 2: Our custom calculation
            print(f'\nCustom calculation:')
            S_custom = _compute_overlap_matrix_custom(calc, k_index)

            nbands = len(S_custom)
            identity = np.eye(nbands, dtype=np.complex128)
            max_dev = np.abs(S_custom - identity).max()
            print(f'  max|S - I| = {max_dev:.3e}')

            # Compare the two methods
            diff = np.abs(S_custom - S_gpaw.array).max()
            print(f'\nDifference between methods: {diff:.3e}')

            if diff > 1e-10:
                print('WARNING: Custom and GPAW methods disagree!')
                print(f'  Custom diagonal: {np.diag(S_custom).real}')
                print(f'  GPAW diagonal: {np.diag(S_gpaw.array).real}')
