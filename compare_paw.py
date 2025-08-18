#!/usr/bin/env python
"""Compare PAW implementation between jrystal and GPAW.

This script loads the same UPF file in both implementations and compares
the calculated PAW correction matrices: K_p (kinetic), M_p and M_pp (Coulomb).
"""

import sys
import numpy as np
import jax.numpy as jnp

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
    with open('align_paw.py', 'r') as f:
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
    K_p_jrystal = exec_globals['K_p']
    M_p_jrystal = exec_globals['M_p']
    M_pp_jrystal = exec_globals['M_pp']
    
    print(f"K_p shape: {K_p_jrystal.shape}")
    print(f"M_p shape: {M_p_jrystal.shape}")
    print(f"M_pp shape: {M_pp_jrystal.shape}")
    
    return K_p_jrystal, M_p_jrystal, M_pp_jrystal


def run_gpaw_setup():
    """Run GPAW's setup.py and extract matrices."""
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
            threshold = 1e-8
            for i in range(len(nc_g) - 1, -1, -1):
                if abs(nc_g[i]) > threshold:
                    return i + 1
            return 0
        
        def get_overlap_correction(self, Delta0_ii):
            """Get overlap correction matrix."""
            # For UPF, return zero correction for now
            return np.zeros_like(Delta0_ii)
        
        def get_smooth_core_density_integral(self, Delta0):
            """Get smooth core density integral."""
            # Calculate integral of smooth core density
            if hasattr(self, 'nct_g'):
                return np.sum(self.nct_g * self.rgd.r_g**2 * self.rgd.dr_g) * 4 * np.pi + Delta0
            return Delta0
        
        def get_linear_kinetic_correction(self, T0_qp):
            """Get linear kinetic correction."""
            # This should be calculated from the difference between AE and PS kinetic energies
            # For now return zeros - the actual calculation is complex and depends on
            # the AE and PS wavefunctions
            ni = sum([2 * l + 1 for l in self.l_j])
            K_p = np.zeros(ni * (ni + 1) // 2)
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
        
        def create_compensation_charge_functions(self, lmax):
            """Create smooth compensation charge functions."""
            # For UPF files, we'll create simple Gaussian-like functions
            # This is a placeholder - actual implementation would use the
            # augmentation charge data from the UPF file
            gcut2 = 797  # This was calculated earlier
            g_lg = np.zeros((lmax + 1, gcut2))
            
            # Create simple compensation charges
            r_g = self.rgd.r_g[:gcut2]
            for l in range(lmax + 1):
                # Simple Gaussian-like function
                sigma = 1.0  # Width parameter
                g_lg[l] = np.exp(-r_g**2 / (2 * sigma**2))
                # Normalize
                norm = np.sum(g_lg[l] * r_g**2 * self.rgd.dr_g[:gcut2]) * 4 * np.pi
                if norm > 0:
                    g_lg[l] /= norm
            
            return g_lg
    
    data = UPFData()
    
    # Store reference to data object for method access
    data.data = data  # Self-reference for methods that need it
    
    # Basic information from PP_HEADER
    data.name = 'upf'
    data.symbol = pp_dict['PP_HEADER']['element']
    data.Z = int(float(pp_dict['PP_HEADER']['z_valence']) + 2)  # Total Z (valence + core for C)
    data.Nv = float(pp_dict['PP_HEADER']['z_valence'])  # Valence electrons
    data.Nc = data.Z - data.Nv  # Core electrons
    
    # Radial grid from PP_MESH
    from gpaw.atom.radialgd import EquidistantRadialGridDescriptor
    r_g = np.array(pp_dict['PP_MESH']['PP_R'])
    dr_g = np.array(pp_dict['PP_MESH']['PP_RAB'])
    
    # Create a proper EquidistantRadialGridDescriptor
    # For UPF files, the grid is typically equidistant
    h = dr_g[0] if len(dr_g) > 0 else 0.02  # Grid spacing
    N = len(r_g)  # Number of grid points
    h0 = r_g[0] if len(r_g) > 0 else 0.0  # Starting point (usually 0)
    
    data.rgd = EquidistantRadialGridDescriptor(h, N, h0)
    # Override with actual grid if different
    data.rgd.r_g = r_g
    data.rgd.dr_g = dr_g
    data.rgd.N = N
    data.rgd.n = N  # Some code expects .n instead of .N
    
    # Add the get_cutoff method that Setup expects
    def get_cutoff(f_g):
        """Find the last non-zero element index."""
        g = len(f_g) - 1
        while g >= 0 and f_g[g] == 0.0:
            g -= 1
        return min(g + 1, N - 1)  # Ensure within bounds
    
    data.rgd.get_cutoff = get_cutoff
    
    # The EquidistantRadialGridDescriptor r2g method needs fixing
    # It should convert radius to grid index, not assume equidistant from 0
    original_r2g = data.rgd.r2g
    def fixed_r2g(r):
        """Convert radius to grid index."""
        # Find the index where r_g[i] >= r
        idx = np.searchsorted(data.rgd.r_g, r)
        return min(idx, data.rgd.N - 1)
    
    # Override the method
    data.rgd.r2g = fixed_r2g
    
    # Also need floor method
    def floor(r):
        return int(np.floor(fixed_r2g(r)))
    data.rgd.floor = floor
    
    # Add spline method wrapper to handle backwards_compatible parameter
    original_spline = data.rgd.spline
    def spline_wrapper(a_g, rcut=None, l=0, points=None, backwards_compatible=True):
        """Wrapper for spline method to handle backwards_compatible parameter."""
        # The base spline method doesn't have backwards_compatible parameter
        return original_spline(a_g, rcut=rcut, l=l, points=points)
    data.rgd.spline = spline_wrapper
    
    # Add new method for creating truncated grids
    def new_grid(gcut):
        """Create a new truncated grid."""
        from gpaw.atom.radialgd import EquidistantRadialGridDescriptor
        # Create new grid with same spacing but truncated
        h = data.rgd.dr_g[0] if len(data.rgd.dr_g) > 0 else 0.02
        h0 = data.rgd.r_g[0] if len(data.rgd.r_g) > 0 else 0.0
        new_rgd = EquidistantRadialGridDescriptor(h, gcut, h0)
        new_rgd.r_g = data.rgd.r_g[:gcut].copy()
        new_rgd.dr_g = data.rgd.dr_g[:gcut].copy()
        new_rgd.N = gcut
        new_rgd.n = gcut
        # Copy over the methods we added
        new_rgd.get_cutoff = get_cutoff
        new_rgd.r2g = fixed_r2g
        new_rgd.floor = floor
        new_rgd.spline = spline_wrapper
        new_rgd.new = new_grid  # Recursive definition
        return new_rgd
    
    data.rgd.new = new_grid
    
    # Projectors from PP_NONLOCAL
    projectors = pp_dict['PP_NONLOCAL']['PP_BETA']
    n_proj = len(projectors)
    
    # Reorder projectors: l=0 before l=1 (following GPAW convention)
    proj_order = []
    for l in [0, 1, 2, 3]:  # Order by angular momentum
        for i, proj in enumerate(projectors):
            # angular_momentum might be string, convert to int
            proj_l = int(proj['angular_momentum'])
            if proj_l == l:
                proj_order.append(i)
    
    if len(proj_order) == 0:
        # If reordering failed, use original order
        proj_order = list(range(n_proj))
    
    # Set l_j, n_j, f_j following the reordered projectors
    data.l_j = [int(projectors[i]['angular_momentum']) for i in proj_order]
    data.n_j = [2 for _ in proj_order]  # Principal quantum number (2 for 2s, 2p)
    data.l_orb_J = data.l_j.copy()
    
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
    
    # Projector functions pt_jg (convert to radial functions)
    data.pt_jg = []
    for i in proj_order:
        proj = projectors[i]
        # Projector values are u(r), convert to u(r)/r for GPAW
        pt_g = proj['values'].copy()
        pt_g[1:] /= data.rgd.r_g[1:len(pt_g)]
        pt_g[0] = pt_g[1]  # Handle r=0
        data.pt_jg.append(np.array(pt_g[:data.rgd.n]))  # Ensure it's numpy array
    
    # Cutoff radii from projector data
    data.rcut_j = []
    data.gcut_j = []  # Grid cutoff indices
    for i in proj_order:
        cutoff_idx = int(projectors[i]['cutoff_radius_index'])
        # Ensure cutoff_idx is within bounds
        cutoff_idx = min(cutoff_idx, data.rgd.n-1)
        data.rcut_j.append(data.rgd.r_g[cutoff_idx])
        data.gcut_j.append(cutoff_idx)
    
    # All-electron and pseudo wavefunctions from PP_FULL_WFC
    data.phi_jg = []  # All-electron wavefunctions
    data.phit_jg = []  # Pseudo wavefunctions
    
    if 'PP_FULL_WFC' in pp_dict:
        aewfc = pp_dict['PP_FULL_WFC']['PP_AEWFC']
        pswfc = pp_dict['PP_FULL_WFC']['PP_PSWFC']
        
        # Reorder wavefunctions to match projector order
        for i in proj_order:
            # AEWFC - convert u(r) to u(r)/r
            phi_g = aewfc[i]['values'].copy()
            phi_g[1:] /= data.rgd.r_g[1:len(phi_g)]
            phi_g[0] = phi_g[1]
            data.phi_jg.append(np.array(phi_g[:data.rgd.n]))  # Ensure it's numpy array
            
            # PSWFC - convert u(r) to u(r)/r
            phit_g = pswfc[i]['values'].copy()
            phit_g[1:] /= data.rgd.r_g[1:len(phit_g)]
            phit_g[0] = phit_g[1]
            data.phit_jg.append(np.array(phit_g[:data.rgd.n]))  # Ensure it's numpy array
    
    # Local potential and vbar
    vbar_full = np.array(pp_dict['PP_LOCAL'])[:data.rgd.n]
    # Find actual cutoff of vbar (last non-zero value)
    vbar_cutoff = data.rgd.n - 1
    while vbar_cutoff > 0 and abs(vbar_full[vbar_cutoff]) < 1e-10:
        vbar_cutoff -= 1
    vbar_cutoff += 1  # Include the zero
    
    # Ensure vbar is zero beyond reasonable cutoff (2 * max projector radius)
    if len(data.rcut_j) > 0:
        # Use floor instead of ceil to ensure we stay below 2*max(rcut_j)
        # Subtract a small margin to avoid rounding issues
        max_reasonable_idx = data.rgd.floor(1.99 * max(data.rcut_j))
        vbar_cutoff = min(vbar_cutoff, max_reasonable_idx)
    
    data.vbar_g = np.zeros(data.rgd.n)
    data.vbar_g[:vbar_cutoff] = vbar_full[:vbar_cutoff]
    
    # Core densities from PP_PAW
    if 'PP_PAW' in pp_dict:
        data.nc_g = np.array(pp_dict['PP_PAW']['PP_AE_NLCC'])[:data.rgd.n]
        data.nct_g = np.array(pp_dict['PP_NLCC'])[:data.rgd.n]
        data.tauc_g = np.zeros_like(data.nc_g)  # Kinetic energy density (not in UPF)
        data.tauct_g = np.zeros_like(data.nct_g)
    else:
        # No core correction
        data.nc_g = np.zeros(data.rgd.n)
        data.nct_g = np.zeros(data.rgd.n)
        data.tauc_g = np.zeros(data.rgd.n)
        data.tauct_g = np.zeros(data.rgd.n)
    
    # Additional required attributes
    data.generator_version = 2  # Modern generator
    data.fingerprint = None
    data.filename = 'C.pbe-n-kjpaw_psl.1.0.0.UPF'
    data.e_kinetic_core = 0.0  # Will be calculated in Setup
    data.e_kinetic = 0.0
    data.e_electrostatic = 0.0  # Electrostatic energy
    data.e_total = float(pp_dict['PP_HEADER'].get('total_psenergy', 0.0))  # Total energy from UPF
    data.extra_xc_data = {}
    data.phicorehole_g = np.zeros(data.rgd.n)
    
    # Add rcutfilter and gcutfilter attributes
    # These define the maximum radius for augmentation functions
    if len(data.rcut_j) > 0:
        data.rcutfilter = max(data.rcut_j)
        data.gcutfilter = max(data.gcut_j)
    else:
        data.rcutfilter = 1.0
        data.gcutfilter = data.rgd.n - 1
    
    # Add augmentation charge data if available
    if 'PP_NONLOCAL' in pp_dict and 'PP_AUGMENTATION' in pp_dict['PP_NONLOCAL']:
        aug = pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']
        data.lmax = int(aug['l_max_aug'])
        
        # Load Q_ij functions (augmentation charges)
        if 'PP_QIJ' in aug:
            # Initialize array for augmentation functions
            nj = len(data.l_j)
            nq = nj * (nj + 1) // 2
            lcut = max(data.l_j) if data.l_j else 0
            data.n_lqg = np.zeros((2 * lcut + 1, nq, data.rgd.n))
            
            for qij in aug['PP_QIJ']:
                l = int(qij['angular_momentum'])
                i = int(qij['first_index']) - 1  # Convert to 0-based
                j = int(qij['second_index']) - 1
                # Map i,j to q index
                if i <= j:
                    q = j + i * nj - i * (i + 1) // 2
                else:
                    q = i + j * nj - j * (j + 1) // 2
                data.n_lqg[l, q, :len(qij['values'])] = qij['values']
        
        # Load multipole moments
        if 'PP_MULTIPOLES' in aug:
            multipoles = np.array(aug['PP_MULTIPOLES'])
            # Reshape if needed
            if multipoles.ndim == 1:
                nj = len(data.l_j) 
                lmax = data.lmax
                # Multipoles should be (lmax+1, nj, nj) then take upper triangle
                expected_size = (lmax + 1) * nj * (nj + 1) // 2
                if len(multipoles) >= expected_size:
                    data.Delta_lq = multipoles[:expected_size].reshape(lmax + 1, -1)
                else:
                    data.Delta_lq = np.zeros((lmax + 1, nj * (nj + 1) // 2))
            else:
                data.Delta_lq = multipoles
    else:
        data.lmax = 0
        data.n_lqg = np.zeros((1, 1, data.rgd.n))
        data.Delta_lq = np.zeros((1, 1))
    
    # Create Setup object with minimal initialization
    # We need to provide xc functional
    from gpaw.xc import XC
    xc = XC('PBE')
    
    try:
        # Debug: Check the values before Setup
        print(f"Debug: max(rcut_j) = {max(data.rcut_j) if data.rcut_j else 'N/A'}")
        print(f"Debug: rgd.N = {data.rgd.N}")
        print(f"Debug: rgd.r_g[-1] = {data.rgd.r_g[-1]}")
        print(f"Debug: pt_jg shapes = {[len(pt) for pt in data.pt_jg] if data.pt_jg else 'N/A'}")
        # Check r2g method
        if hasattr(data.rgd, 'r2g'):
            test_r = max(data.rcut_j) if data.rcut_j else 1.0
            print(f"Debug: r2g({test_r}) = {data.rgd.r2g(test_r)}")
            print(f"Debug: ceil(r2g({test_r})) = {data.rgd.ceil(test_r)}")
        
        # Check vbar_g
        vbar_nonzero = np.where(data.vbar_g != 0.0)[0]
        if len(vbar_nonzero) > 0:
            print(f"Debug: vbar_g last nonzero at index {vbar_nonzero[-1]} (r={data.rgd.r_g[vbar_nonzero[-1]]:.4f})")
            print(f"Debug: 2*max(rcut_j) = {2.0 * max(data.rcut_j) if data.rcut_j else 'N/A'}")
            if data.rcut_j:
                idx_2rcut = data.rgd.ceil(2.0 * max(data.rcut_j))
                print(f"Debug: ceil(2*max(rcut_j)) gives index {idx_2rcut}, r={data.rgd.r_g[min(idx_2rcut, data.rgd.N-1)]:.4f}")
        
        # Initialize Setup with the UPF data
        setup = Setup(data, xc, lmax=2, basis=None)
        
        # Extract matrices
        K_p_gpaw = setup.K_p
        M_p_gpaw = setup.M_p
        M_pp_gpaw = setup.M_pp
        
        print(f"K_p shape: {K_p_gpaw.shape}")
        print(f"M_p shape: {M_p_gpaw.shape}")
        print(f"M_pp shape: {M_pp_gpaw.shape}")
        
        return K_p_gpaw, M_p_gpaw, M_pp_gpaw
        
    except Exception as e:
        print(f"Error in GPAW setup: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def compare_matrices(K_p_j, M_p_j, M_pp_j, K_p_g, M_p_g, M_pp_g):
    """Compare matrices from both implementations."""
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    
    if K_p_g is None:
        print("GPAW setup failed, cannot compare")
        return
    
    # Convert jax arrays to numpy for comparison
    K_p_j = np.array(K_p_j)
    M_p_j = np.array(M_p_j)
    M_pp_j = np.array(M_pp_j)
    
    # K_p comparison
    print("\n--- Kinetic correction K_p ---")
    print(f"jrystal K_p:\n{K_p_j}")
    print(f"\nGPAW K_p:\n{K_p_g}")
    if K_p_j.shape == K_p_g.shape:
        diff_K = np.abs(K_p_j - K_p_g)
        print(f"Max difference: {np.max(diff_K):.6e}")
        print(f"Mean difference: {np.mean(diff_K):.6e}")
    else:
        print(f"Shape mismatch: jrystal {K_p_j.shape} vs GPAW {K_p_g.shape}")
    
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
    print("\n--- Coulomb correction M_pp ---")
    print(f"jrystal M_pp shape: {M_pp_j.shape}")
    print(f"GPAW M_pp shape: {M_pp_g.shape}")
    if M_pp_j.shape == M_pp_g.shape:
        diff_Mpp = np.abs(M_pp_j - M_pp_g)
        print(f"Max difference: {np.max(diff_Mpp):.6e}")
        print(f"Mean difference: {np.mean(diff_Mpp):.6e}")
        
        # Show first few elements for debugging
        print(f"\njrystal M_pp[:3, :3]:\n{M_pp_j[:3, :3]}")
        print(f"\nGPAW M_pp[:3, :3]:\n{M_pp_g[:3, :3]}")
    else:
        print(f"Shape mismatch: jrystal {M_pp_j.shape} vs GPAW {M_pp_g.shape}")


if __name__ == "__main__":
    # Run both implementations
    K_p_j, M_p_j, M_pp_j = run_align_paw()
    K_p_g, M_p_g, M_pp_g = run_gpaw_setup()
    
    # Compare results
    compare_matrices(K_p_j, M_p_j, M_pp_j, K_p_g, M_p_g, M_pp_g)