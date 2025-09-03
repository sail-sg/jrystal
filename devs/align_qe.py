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
    """Run jrystal's align_paw.py and extract results.
    
    Returns:
        Dictionary containing PAW results from jrystal
    """
    from calc_paw import setup_qe, calc
    # setup_qe returns multiple values, pass them all to calc
    setup_data = setup_qe()
    return calc(*setup_data)


def run_gpaw_setup():
    """Run GPAW's setup.py and extract matrices.
    
    NOTE: YOU CAN ONLY MODIFY THE CODE INSIDE THIS FUNCTION
    WHEN EVER YOU MODIFY ANY CODE, MAKE SURE YOU ADD A COMMENT TO EXPLAIN
    THE REASON FOR THIS MODIFICATION AND ASK THE PERMISSION FROM THE HUMAN USER
    
    Returns:
        Dictionary containing GPAW results
    """
    
    from gpaw.setup import Setup
    from gpaw.setup_data import SetupData
    from jrystal.pseudopotential.load import parse_upf
    from gpaw.atom.radialgd import AbinitRadialGridDescriptor
    
    # Load the UPF file using jrystal's parser
    pp_dict = parse_upf('/home/aiops/zhaojx/jrystal/pseudopotential/C.pbe-n-kjpaw_psl.1.0.0.UPF')
    
    # REFACTORING: Use SetupData directly since it already has all required methods
    # No need for custom subclass - SetupData has everything we need!
    data = SetupData(
        symbol=pp_dict['PP_HEADER']['element'],
        xcsetupname='PBE',
        name='upf',
        readxml=False,  # We'll populate manually from UPF
        generator_version=2
    )
    
    # Populate basic atomic information
    data.Z = int(float(pp_dict['PP_HEADER']['z_valence']) + 2)  # Total Z (valence + core for C)
    data.Nv = float(pp_dict['PP_HEADER']['z_valence'])  # Valence electrons
    data.Nc = data.Z - data.Nv  # Core electrons
    
    # Setup radial grid
    r_g = np.array(pp_dict['PP_MESH']['PP_R'])
    gcut = 741
    i = 100
    assert np.allclose((r_g[i+2] - r_g[i+1])/(r_g[i+1] - r_g[i]), (r_g[2*i+2] - r_g[2*i+1])/(r_g[2*i+1] - r_g[2*i]))
    d = np.log((r_g[i+2] - r_g[i+1])/(r_g[i+1] - r_g[i]))
    a = r_g[0] / (np.exp(d) - 1)
    data.rgd = AbinitRadialGridDescriptor(a, d, gcut)
    data.rgd.r_g = r_g[:gcut]
    data.rgd.dr_g = np.array(pp_dict['PP_MESH']['PP_RAB'])[:gcut]
    
    # Angular momentum and cutoff parameters  
    data.lmax = int(pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['l_max_aug'])
    data.l_j = np.array([int(proj['angular_momentum']) for proj in pp_dict['PP_NONLOCAL']['PP_BETA']])
    data.lcut = max(data.l_j)
    data.rcut_j = np.array([float(proj['cutoff_radius']) for proj in pp_dict['PP_NONLOCAL']['PP_BETA']])
    data.gcut_j = np.array([int(proj['cutoff_radius_index']) for proj in pp_dict['PP_NONLOCAL']['PP_BETA']])
    data.gcut = np.max(data.gcut_j)
    
    # Wave functions and projectors (convert from QE to GPAW convention: divide by r)
    data.pt_jg = np.array([proj['values'] for proj in pp_dict['PP_NONLOCAL']['PP_BETA']])[:, :data.gcut] / data.rgd.r_g
    data.phi_jg = np.array([phi['values'] for phi in pp_dict['PP_FULL_WFC']['PP_AEWFC']])[:, :data.gcut] / data.rgd.r_g
    data.phit_jg = np.array([phi['values'] for phi in pp_dict['PP_FULL_WFC']['PP_PSWFC']])[:, :data.gcut] / data.rgd.r_g
    
    # Core densities (convert from QE to GPAW convention: multiply by sqrt(4π))
    data.nc_g = np.array(pp_dict['PP_PAW']['PP_AE_NLCC'])[:data.gcut] * np.sqrt(4 * np.pi)
    data.nct_g = np.array(pp_dict['PP_NLCC'])[:data.gcut] * np.sqrt(4 * np.pi)
    
    # Local potential
    data.vbar_g = np.array(pp_dict['PP_LOCAL'])[:data.gcut]
    
    # Set shape function to avoid errors
    data.shape_function = {'type': 'gauss', 'rc': 0.5}
    
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
    
    # Only set attributes that differ from SetupData defaults or are required
    data.e_kin_jj = np.zeros((len(data.l_j), len(data.l_j)))  # Required for get_linear_kinetic_correction
    data.ExxC_w = []  # Override default {} to avoid HSE warnings
    
    # Set rcutfilter and gcutfilter (required by Setup)
    data.rcutfilter = max(data.rcut_j) if len(data.rcut_j) > 0 else 1.0
    data.gcutfilter = max(data.gcut_j) if len(data.gcut_j) > 0 else data.rgd.n - 1
     # Create Setup object with minimal initialization
    # We need to provide xc functional
    from gpaw.xc import XC
    xc = XC('PBE')
    
    try:
        # Initialize Setup with the UPF data
        setup = Setup(data, xc, lmax=2, basis=None)
        results = {
        'B_ii': setup.B_ii if hasattr(setup, 'B_ii') else None,
        'M': setup.M if hasattr(setup, 'M') else None,
        'Delta_pL': setup.Delta_pL if hasattr(setup, 'Delta_pL') else None,
        'Delta0': setup.Delta0 if hasattr(setup, 'Delta0') else None,
        'gcut2': setup.gcut2 if hasattr(setup, 'gcut2') else None
        }
        
        if hasattr(setup, 'local_corr'):
            results.update({
                'n_qg': setup.local_corr.n_qg if hasattr(setup.local_corr, 'n_qg') else None,
                'nt_qg': setup.local_corr.nt_qg if hasattr(setup.local_corr, 'nt_qg') else None,
                'rgd2_N': setup.local_corr.rgd2.N if hasattr(setup.local_corr, 'rgd2') else None
            })
        else:
            results.update({
                'n_qg': None, 'nt_qg': None, 'rgd2_N': None
            })
        
        return results
        
    except Exception as e:
        # Re-raise the exception to be handled by caller
        raise e


def _compare_values(val_j, val_g, name):
    """Compare two values (arrays or scalars) from both implementations.
    
    Args:
        val_j: jrystal value (array or scalar)
        val_g: GPAW value (array or scalar)
        name: Name of the value for display
    """
    print(f"\n--- {name} comparison ---")
    
    # Check for missing data
    if val_j is None or val_g is None:
        print(f"Missing data: jrystal={val_j is not None}, GPAW={val_g is not None}")
        return

    # Check shapes
    if val_j.shape != val_g.shape:
        print(f"Shape mismatch: jrystal {val_j.shape} vs GPAW {val_g.shape}")
        return
    
    diff = np.abs(val_j - val_g)
    print(f"Shape match: {val_j.shape}")
    print(f"Max difference: {np.max(diff):.6e}")
    print(f"Mean difference: {np.mean(diff):.6e}")


def compare_results(results_j, results_g):
    """Compare PAW results from both implementations.
    
    Args:
        results_j: Dictionary of jrystal results
        results_g: Dictionary of GPAW results
    """
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    
    # Define comparison settings for each quantity
    # Format: (key, display_name)
    comparisons = [
        ('n_qg', 'Augmentation density n_qg'),
        ('nt_qg', 'Smooth augmentation density nt_qg'),
        ('Delta_pL', 'Delta_pL matrix'),
        ('Delta0', 'Delta0'),
        ('M', 'Scalar M value'),
        ('B_ii', 'Projector function overlaps B_ii'),
    ]
    
    for key, display_name in comparisons:
        # Compare values
        _compare_values(results_j[key], results_g[key], display_name)


if __name__ == "__main__":
    results_j = run_align_paw()
    try:
        results_g = run_gpaw_setup()
    except Exception as e:
        print(f"Error in GPAW setup: {e}")
        import traceback
        traceback.print_exc()
        results_g = {}

    # Compare results using the new simplified interface
    compare_results(results_j, results_g)