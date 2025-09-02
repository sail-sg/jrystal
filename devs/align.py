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


def _extract_jrystal_results(exec_globals):
    """Extract results from jrystal execution context.
    
    Args:
        exec_globals: Dictionary containing executed jrystal variables
        
    Returns:
        Tuple of extracted values
    """
    results = {
        'B_ii': exec_globals.get('B_ii'),
        'M': exec_globals.get('M'),
        'nc_g': exec_globals.get('nc_g'),
        'nct_g': exec_globals.get('nct_g'),
        'n_qg': exec_globals.get('n_qg'),
        'nt_qg': exec_globals.get('nt_qg'),
        'Delta0': exec_globals.get('Delta0'),
        'gcut': exec_globals.get('gcut')
    }
    return results


def run_align_paw():
    """Run jrystal's align_paw.py and extract results.
    
    Returns:
        Tuple containing (B_ii, M, nc_g, nct_g, n_qg, nt_qg, Delta0) from jrystal
    """
    # Import and run align_paw
    exec_globals = {}
    with open('devs/align_paw.py', 'r') as f:
        code = f.read()
    
    # Execute up to the breakpoint to get M_p, M_pp calculation
    code_lines = code.split('\n')
    for i, line in enumerate(code_lines):
        if 'M_p, M_pp = calculate_coulomb_corrections()' in line:
            break_line = i + 1
            break
    
    # Execute up to that point
    code_to_exec = '\n'.join(code_lines[:break_line])
    exec(code_to_exec, exec_globals)
    
    # Extract results
    results = _extract_jrystal_results(exec_globals)
    
    return results


def _extract_gpaw_results(setup):
    """Extract results from GPAW setup object.
    
    Args:
        setup: GPAW Setup object
        
    Returns:
        Dictionary of extracted values
    """
    results = {
        'B_ii': setup.B_ii if hasattr(setup, 'B_ii') else None,
        'M': setup.M if hasattr(setup, 'M') else None,
        'Delta0': setup.Delta0 if hasattr(setup, 'Delta0') else None,
        'gcut2': setup.gcut2 if hasattr(setup, 'gcut2') else None
    }
    
    if hasattr(setup, 'local_corr'):
        results.update({
            'nc_g': setup.local_corr.nc_g if hasattr(setup.local_corr, 'nc_g') else None,
            'nct_g': setup.local_corr.nct_g if hasattr(setup.local_corr, 'nct_g') else None,
            'n_qg': setup.local_corr.n_qg if hasattr(setup.local_corr, 'n_qg') else None,
            'nt_qg': setup.local_corr.nt_qg if hasattr(setup.local_corr, 'nt_qg') else None,
            'rgd2_N': setup.local_corr.rgd2.N if hasattr(setup.local_corr, 'rgd2') else None
        })
    else:
        results.update({
            'nc_g': None, 'nct_g': None, 'n_qg': None, 'nt_qg': None, 'rgd2_N': None
        })
    
    return results


def run_gpaw_setup():
    """Run GPAW's setup.py and extract matrices.
    
    NOTE: YOU CAN ONLY MODIFY THE CODE INSIDE THIS FUNCTION
    WHEN EVER YOU MODIFY ANY CODE, MAKE SURE YOU ADD A COMMENT TO EXPLAIN
    THE REASON FOR THIS MODIFICATION AND ASK THE PERMISSION FROM THE HUMAN USER
    
    Returns:
        Dictionary containing GPAW results
    """
    
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
        
        # Extract results
        results = _extract_gpaw_results(setup)
        
        return results
        
    except Exception as e:
        # Re-raise the exception to be handled by caller
        raise e


def _compare_values(val_j, val_g, name, include_sum=False, show_ratio=False):
    """Compare two values (arrays or scalars) from both implementations.
    
    Args:
        val_j: jrystal value (array or scalar)
        val_g: GPAW value (array or scalar)
        name: Name of the value for display
        include_sum: Whether to include sum comparison (for arrays)
        show_ratio: Whether to show the ratio (for scalars)
    """
    print(f"\n--- {name} comparison ---")
    
    # Check for missing data
    if val_j is None or val_g is None:
        print(f"Missing data: jrystal={val_j is not None}, GPAW={val_g is not None}")
        return
    
    # Convert to numpy arrays for uniform handling
    arr_j = np.atleast_1d(val_j)
    arr_g = np.atleast_1d(val_g)
    
    # Check shapes
    if arr_j.shape != arr_g.shape:
        print(f"Shape mismatch: jrystal {arr_j.shape} vs GPAW {arr_g.shape}")
        return
    
    # For scalars (size 1 arrays)
    if arr_j.size == 1:
        print(f"jrystal {name}: {arr_j[0]:.10f}")
        print(f"GPAW {name}: {arr_g[0]:.10f}")
        print(f"Difference: {abs(arr_j[0] - arr_g[0]):.6e}")
        
        if show_ratio and abs(arr_j[0]) > 1e-10:
            print(f"Ratio (GPAW/jrystal): {arr_g[0]/arr_j[0]:.6f}")
    
    # For arrays
    else:
        diff = np.abs(arr_j - arr_g)
        print(f"Shape match: {arr_j.shape}")
        print(f"Max difference: {np.max(diff):.6e}")
        print(f"Mean difference: {np.mean(diff):.6e}")
        
        if include_sum:
            print(f"Sum difference: {abs(np.sum(arr_j) - np.sum(arr_g)):.6e}")


def compare_results(B_ii_j, M_j, nc_g_j, nct_g_j, n_qg_j, nt_qg_j, Delta0_j, 
                    B_ii_g, M_g, nc_g_g, nct_g_g, n_qg_g, nt_qg_g, Delta0_g):
    """Compare densities and M values from both implementations.
    
    Args:
        B_ii_j, M_j, nc_g_j, nct_g_j, n_qg_j, nt_qg_j, Delta0_j: jrystal results
        B_ii_g, M_g, nc_g_g, nct_g_g, n_qg_g, nt_qg_g, Delta0_g: GPAW results
    """
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    
    # Convert jax arrays to numpy for comparison
    arrays_to_convert = [
        ('B_ii_j', B_ii_j), ('nc_g_j', nc_g_j), ('nct_g_j', nct_g_j),
        ('n_qg_j', n_qg_j), ('nt_qg_j', nt_qg_j)
    ]
    
    converted = {}
    for name, arr in arrays_to_convert:
        if arr is not None:
            converted[name] = np.array(arr)
        else:
            converted[name] = None
    
    # Convert scalars
    M_j = float(M_j) if M_j is not None else None
    Delta0_j = float(Delta0_j) if Delta0_j is not None else None
    
    # Compare all values using unified function
    _compare_values(converted['nc_g_j'], nc_g_g, "Core density nc_g", include_sum=True)
    _compare_values(converted['nct_g_j'], nct_g_g, "Smooth core density nct_g", include_sum=True)
    
    # Compare first element of n_qg
    if converted['n_qg_j'] is not None and n_qg_g is not None:
        _compare_values(converted['n_qg_j'][0], n_qg_g[0], "Augmentation density n_qg[0]")
    
    # Compare scalars
    _compare_values(Delta0_j, Delta0_g, "Delta0")
    _compare_values(M_j, M_g, "Scalar M value", show_ratio=True)
    
    # Compare B_ii
    _compare_values(converted['B_ii_j'], B_ii_g, "Projector function overlaps B_ii")


if __name__ == "__main__":
    results_j = run_align_paw()
    try:
        results_g = run_gpaw_setup()
    except Exception as e:
        print(f"Error in GPAW setup: {e}")
        import traceback
        traceback.print_exc()
        results_g = {'B_ii': None, 'M': None, 'nc_g': None, 'nct_g': None,
                     'n_qg': None, 'nt_qg': None, 'Delta0': None}
    
    # Compare results
    compare_results(
        results_j['B_ii'], results_j['M'], results_j['nc_g'], results_j['nct_g'],
        results_j['n_qg'], results_j['nt_qg'], results_j['Delta0'],
        results_g['B_ii'], results_g['M'], results_g['nc_g'], results_g['nct_g'],
        results_g['n_qg'], results_g['nt_qg'], results_g['Delta0']
    )