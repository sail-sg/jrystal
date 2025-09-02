#!/usr/bin/env python
"""Compare PAW implementation between jrystal and GPAW.

This script loads the same UPF file in both implementations and compares
the calculated PAW correction matrices: K_p (kinetic), M_p and M_pp (Coulomb).
"""

import sys
import numpy as np

# For align_paw.py
sys.path.insert(0, '/home/aiops/zhaojx/jrystal')

# For GPAW
sys.path.insert(0, '/home/aiops/zhaojx/jrystal/gpaw')


def run_align_paw():
    """Run jrystal's align_paw.py and extract matrices."""
    print("=" * 60)
    print("Running jrystal align_paw.py")
    print("=" * 60)
    
    # Import and run align_paw
    exec_globals = {}
    with open('devs/align_paw.py', 'r') as f:
        code = f.read()
    
    # Execute up to the breakpoint to get K_p, M_p, M_pp
    code_lines = code.split('\n')
    # Find the breakpoint after M_p, M_pp calculation
    for i, line in enumerate(code_lines):
        if 'M_p, M_pp = calculate_coulomb_corrections()' in line:
            break_line = i + 1
            break
    
    # Execute up to that point
    code_to_exec = '\n'.join(code_lines[:break_line])
    exec(code_to_exec, exec_globals)
    
    # Extract the matrices
    B_ii_jrystal = exec_globals['B_ii']
    K_p_jrystal = exec_globals['K_p']
    M_p_jrystal = exec_globals['M_p']
    M_pp_jrystal = exec_globals['M_pp']
    
    # Debug: Extract T_Lqp and A_q for comparison
    T_Lqp_jrystal = exec_globals.get('T_Lqp', None)
    A_q_jrystal = exec_globals.get('A_q_debug', None)
    
    return B_ii_jrystal, M_p_jrystal, M_pp_jrystal, T_Lqp_jrystal, A_q_jrystal


def run_gpaw_setup():
    """Run GPAW's setup.py and extract matrices.
    
    NOTE: YOU CAN ONLY MODIFY THE CODE INSIDE THIS FUNCTION
    WHEN EVER YOU MODIFY ANY CODE, MAKE SURE YOU ADD A COMMENT TO EXPLAIN
    THE REASON FOR THIS MODIFICATION AND ASK THE PERMISSION FROM THE HUMAN USER
    """
    print("\n" + "=" * 60)
    print("Running GPAW setup.py")
    print("=" * 60)
    
    from gpaw.setup import Setup
    from gpaw.gaunt import gaunt
    from jrystal.pseudopotential.load import parse_upf
    
    # Load the UPF file using jrystal's parser
    pp_dict = parse_upf('/home/aiops/zhaojx/jrystal/pseudopotential/C.pbe-n-kjpaw_psl.1.0.0.UPF')
    
    # Create a data object with all necessary attributes from pp_dict
    class UPFData:
        """Data container for UPF pseudopotential data."""
        def print_info(self, text, setup):
            """Print setup information (required by Setup.__init__)."""
            pass  # Minimal implementation for testing
        
        def find_core_density_cutoff(self, nc_g):
            """Find cutoff radius for core density."""
            # Find where core density becomes negligible
            return self.rgd.r_g[-1]
        
        def create_compensation_charge_functions(self, lmax):
            """Create compensation charge functions.
            TODO: this g_lg is INCORRECT"""
            # Initialize g_lg array
            g_lg = np.zeros((lmax + 1, self.rgd.N))
            return g_lg
        
        def get_overlap_correction(self, Delta0_ii):
            """Directly copied from GPAW"""
            return np.sqrt(4.0 * np.pi) * Delta0_ii
        
        def get_smooth_core_density_integral(self, Delta0):
            """Directly copied from GPAW"""
            return -Delta0 * np.sqrt(4 * np.pi) - self.Z + self.Nc
        
        def get_linear_kinetic_correction(self, T0_qp):
            """Need to access to e_kin_jj"""
            # e_kin_jj = self.e_kin_jj
            # nj = len(e_kin_jj)
            # K_q = []
            # for j1 in range(nj):
            #     for j2 in range(j1, nj):
            #         K_q.append(e_kin_jj[j1, j2])
            # K_p = np.sqrt(4 * np.pi) * np.dot(K_q, T0_qp)
            K_p = 0
            return K_p
        
        def get_xc_correction(self, rgd2, xc, gcut2, lcut):
            """Get XC correction."""
            # Return a placeholder XC correction object
            # The actual implementation is complex and would need full PAW data
            class XCCorrection:
                def __init__(self):
                    self.rgd2 = rgd2
                    self.e_xc0 = 0.0
                    self.four_pi_sqrt = np.sqrt(4 * np.pi)
                    self.nc_corehole_g = np.zeros(gcut2)
                    self.nct_corehole_g = np.zeros(gcut2)
                    self.Y_nL = None
                    self.rnablaY_nLv = None
                    
                def calculate_paw_correction(self, *args, **kwargs):
                    return 0.0
                    
            return XCCorrection()
    
    data = UPFData()
    
    # Basic information from PP_HEADER
    data.name = 'upf'
    data.symbol = pp_dict['PP_HEADER']['element']
    data.Z = int(float(pp_dict['PP_HEADER']['z_valence']) + 2)  # Total Z (valence + core for C)
    data.Nv = float(pp_dict['PP_HEADER']['z_valence'])  # Valence electrons
    data.Nc = data.Z - data.Nv  # Core electrons
    
    # Radial grid from PP_MESH
    from gpaw.atom.radialgd import AbinitRadialGridDescriptor
    r_g = np.array(pp_dict['PP_MESH']['PP_R'])
    gcut = 741
    i = 100
    assert np.allclose((r_g[i+2] - r_g[i+1])/(r_g[i+1] - r_g[i]), (r_g[2*i+2] - r_g[2*i+1])/(r_g[2*i+1] - r_g[2*i]))
    d = np.log((r_g[i+2] - r_g[i+1])/(r_g[i+1] - r_g[i]))
    a = r_g[0] / (np.exp(d) - 1)
    data.rgd = AbinitRadialGridDescriptor(a, d, gcut)
    data.rgd.r_g = r_g[:gcut]
    data.rgd.dr_g = np.array(pp_dict['PP_MESH']['PP_RAB'])[:gcut]
    
    data.lmax = int(pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['l_max_aug']) # maximum angular momentum of the augmentation charge
    data.l_j = np.array([int(proj['angular_momentum']) for proj in pp_dict['PP_NONLOCAL']['PP_BETA']]) # angular momentum of each projector
    data.lcut = max(data.l_j)
    data.rcut_j = np.array([float(proj['cutoff_radius']) for proj in pp_dict['PP_NONLOCAL']['PP_BETA']]) # projector functions
    data.gcut_j = np.array([int(proj['cutoff_radius_index']) for proj in pp_dict['PP_NONLOCAL']['PP_BETA']]) # projector functions
    data.gcut = np.max(data.gcut_j)
    data.r_g = np.array(pp_dict['PP_MESH']['PP_R'])[:data.gcut] # radial grid
    data.dr_g = np.array(pp_dict['PP_MESH']['PP_RAB'])[:data.gcut] # radial grid integration weight
    data.pt_jg = np.array([proj['values'] for proj in pp_dict['PP_NONLOCAL']['PP_BETA']])[:, :data.gcut] /\
         data.r_g # projector functions
    data.phi_jg = np.array([phi['values'] for phi in pp_dict['PP_FULL_WFC']['PP_AEWFC']])[:, :data.gcut] # all-electron wave functions
    data.phit_jg = np.array([phi['values'] for phi in pp_dict['PP_FULL_WFC']['PP_PSWFC']])[:, :data.gcut] # pseudo wave functions
    data.nc_g = np.array(pp_dict['PP_PAW']['PP_AE_NLCC'])[:data.gcut] # all-electron non-linear core charge
    data.nct_g = np.array(pp_dict['PP_NLCC'])[:data.gcut] # non-linera core charge
    data.vbar_g = np.array(pp_dict['PP_LOCAL'])[:data.gcut] # local pseudopotential
    
    # Occupation numbers for Carbon: 2s^2 2p^2
    data.f_j = []
    s_count = p_count = 0
    for l in data.l_j:
        if l == 0:  # s orbital
            data.f_j.append(2.0 if s_count == 0 else 0.0)
            s_count += 1
        elif l == 1:  # p orbital  
            data.f_j.append(2.0 if p_count == 0 else 0.0)
            p_count += 1
    data.l_orb_J = 0
    data.n_j = np.array([0, 0, 1, 1]) 
    
    # Additional required attributes
    data.generator_version = 2  # Modern generator
    data.tauct_g = None
    data.fingerprint = None
    data.phicorehole_g = None
    data.filename = 'C.pbe-n-kjpaw_psl.1.0.0.UPF'
    data.e_electrostatic = 0.0
    data.e_total = 0.0
    data.e_kinetic_core = 0.0  # Will be calculated in Setup
    data.e_kinetic = 0.0
    data.extra_xc_data = {}
    
    # Add rcutfilter and gcutfilter attributes
    # These define the maximum radius for augmentation functions
    if len(data.rcut_j) > 0:
        data.rcutfilter = max(data.rcut_j)
        data.gcutfilter = max(data.gcut_j)
    else:
        data.rcutfilter = 1.0
        data.gcutfilter = data.rgd.n - 1
     # Create Setup object with minimal initialization
    # We need to provide xc functional
    from gpaw.xc import XC
    xc = XC('PBE')
    
    try:
        # Debug: Check the values before Setup
        # print(f"Debug: max(rcut_j) = {max(data.rcut_j) if data.rcut_j.any() else 'N/A'}")
        # print(f"Debug: rgd.N = {data.rgd.N}")
        # print(f"Debug: rgd.r_g[-1] = {data.rgd.r_g[-1]}")
        # print(f"Debug: pt_jg shapes = {[len(pt) for pt in data.pt_jg] if data.pt_jg.any() else 'N/A'}")
        
        # Initialize Setup with the UPF data
        setup = Setup(data, xc, lmax=2, basis=None)
        
        # Debug: Extract T_Lqp and A_q for comparison
        T_Lqp_gpaw = None
        A_q_gpaw = None
        
        if hasattr(setup, 'local_corr') and hasattr(setup.local_corr, 'T_Lqp'):
            T_Lqp_gpaw = setup.local_corr.T_Lqp
            print(f"\nGPAW T_Lqp[0] shape: {T_Lqp_gpaw[0].shape}")
            print(f"GPAW T_Lqp[0] (first 5x5):\n{T_Lqp_gpaw[0][:5, :5]}")
        
        if hasattr(setup, 'A_q_debug'):
            A_q_gpaw = setup.A_q_debug
            print(f"GPAW A_q shape: {A_q_gpaw.shape}")
            print(f"GPAW A_q (first 5): {A_q_gpaw[:5]}")
        
        # Extract matrices
        B_ii_gpaw = setup.B_ii
        K_p_gpaw = setup.K_p
        M_p_gpaw = setup.M_p
        M_pp_gpaw = setup.M_pp
        
        return B_ii_gpaw, M_p_gpaw, M_pp_gpaw, T_Lqp_gpaw, A_q_gpaw
        
    except Exception as e:
        print(f"Error in GPAW setup: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


def compare_matrices(B_ii_j, M_p_j, M_pp_j, T_Lqp_j, A_q_j, B_ii_g, M_p_g, M_pp_g, T_Lqp_g, A_q_g):
    """Compare matrices from both implementations."""
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    
    # Convert jax arrays to numpy for comparison
    B_ii_j = np.array(B_ii_j)
    M_p_j = np.array(M_p_j)
    M_pp_j = np.array(M_pp_j)
    if T_Lqp_j is not None:
        T_Lqp_j = np.array(T_Lqp_j)
    if A_q_j is not None:
        A_q_j = np.array(A_q_j)
    
    # T_Lqp comparison
    print("\n--- T_Lqp[0] matrix comparison ---")
    if T_Lqp_j is not None and T_Lqp_g is not None:
        T_Lqp_j0 = T_Lqp_j[0]
        T_Lqp_g0 = T_Lqp_g[0]
        if T_Lqp_j0.shape == T_Lqp_g0.shape:
            diff_T = np.abs(T_Lqp_j0 - T_Lqp_g0)
            print(f"Shape match: {T_Lqp_j0.shape}")
            print(f"Max difference: {np.max(diff_T):.6e}")
            print(f"Mean difference: {np.mean(diff_T):.6e}")
        else:
            print(f"Shape mismatch: jrystal {T_Lqp_j0.shape} vs GPAW {T_Lqp_g0.shape}")
    
    # A_q comparison
    print("\n--- A_q vector comparison ---")
    if A_q_j is not None and A_q_g is not None:
        if A_q_j.shape == A_q_g.shape:
            print(f"Shape match: {A_q_j.shape}")
            # Show element-wise ratio to understand scaling
            ratio = A_q_g / (A_q_j + 1e-10)  # Add small value to avoid division by zero
            print(f"A_q ratio (GPAW/jrystal) first 5: {ratio[:5]}")
            print(f"Average ratio: {np.mean(np.abs(ratio)):.2e}")
        else:
            print(f"Shape mismatch: jrystal {A_q_j.shape} vs GPAW {A_q_g.shape}")
    
    # B_ii comparison
    print("\n--- Projector function overlaps B_ii ---")
    # print(f"jrystal B_ii:\n{B_ii_j}")
    # print(f"\nGPAW B_ii:\n{B_ii_g}")
    if B_ii_j.shape == B_ii_g.shape:
        diff_B = np.abs(B_ii_j - B_ii_g)
        print(f"Max difference: {np.max(diff_B):.6e}")
        print(f"Mean difference: {np.mean(diff_B):.6e}")
    else:
        print(f"Shape mismatch: jrystal {B_ii_j.shape} vs GPAW {B_ii_g.shape}")
    
    # K_p comparison
    # print("\n--- Kinetic correction K_p ---")
    # print(f"jrystal K_p:\n{K_p_j}")
    # print(f"\nGPAW K_p:\n{K_p_g}")
    # if K_p_j.shape == K_p_g.shape:
    #     diff_K = np.abs(K_p_j - K_p_g)
    #     print(f"Max difference: {np.max(diff_K):.6e}")
    #     print(f"Mean difference: {np.mean(diff_K):.6e}")
    # else:
    #     print(f"Shape mismatch: jrystal {K_p_j.shape} vs GPAW {K_p_g.shape}")
    
    # M_p comparison
    print("\n--- Coulomb correction M_p ---")
    print(f"jrystal M_p:\n{M_p_j}")
    print(f"\nGPAW M_p:\n{M_p_g}")
    if M_p_j.shape == M_p_g.shape:
        diff_M = np.abs(M_p_j - M_p_g)
        print(f"Max difference: {np.max(diff_M):.6e}")
        print(f"Mean difference: {np.mean(diff_M):.6e}")
    else:
        print(f"Shape mismatch: jrystal {M_p_j.shape} vs GPAW {M_p_g.shape}")
    
    # M_pp comparison
    # print("\n--- Coulomb correction M_pp ---")
    # print(f"jrystal M_pp shape: {M_pp_j.shape}")
    # print(f"GPAW M_pp shape: {M_pp_g.shape}")
    # if M_pp_j.shape == M_pp_g.shape:
    #     diff_Mpp = np.abs(M_pp_j - M_pp_g)
    #     print(f"Max difference: {np.max(diff_Mpp):.6e}")
    #     print(f"Mean difference: {np.mean(diff_Mpp):.6e}")
        
    #     # Show first few elements for debugging
    #     print(f"\njrystal M_pp[:3, :3]:\n{M_pp_j[:3, :3]}")
    #     print(f"\nGPAW M_pp[:3, :3]:\n{M_pp_g[:3, :3]}")
    # else:
    #     print(f"Shape mismatch: jrystal {M_pp_j.shape} vs GPAW {M_pp_g.shape}")


if __name__ == "__main__":
    # Run both implementations
    B_ii_j, M_p_j, M_pp_j, T_Lqp_j, A_q_j = run_align_paw()
    B_ii_g, M_p_g, M_pp_g, T_Lqp_g, A_q_g = run_gpaw_setup()
    
    # Compare results
    compare_matrices(B_ii_j, M_p_j, M_pp_j, T_Lqp_j, A_q_j, B_ii_g, M_p_g, M_pp_g, T_Lqp_g, A_q_g)