from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from gpaw.gaunt import gaunt
from gpaw.new import zips
from gpaw.setup_data import SetupData
# from jrystal.pseudopotential.load import parse_upf

from .gpaw_load import parse_paw_setup

def setup_qe():
  """Load and parse QE UPF pseudopotential file.
  
  This function reads a Quantum ESPRESSO UPF file and extracts PAW data.
  Values are returned in QE's native storage convention without conversion.
  QE UPF Storage Conventions (as documented in paw_pp_file_documentation.md)
  
  Returns:
    Tuple of arrays containing PAW data in QE native convention
  """
  
  # load the pseudopotential
  pp_dict = parse_upf('/home/aiops/zhaojx/jrystal/pseudopotential/C.pbe-n-kjpaw_psl.1.0.0.UPF')
  Z = 6  # Atomic number for Carbon
  lmax = int(pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['l_max_aug'])  # Max l for augmentation
  l_j = jnp.array([int(proj['angular_momentum']) for proj in pp_dict['PP_NONLOCAL']['PP_BETA']])  # l for each projector
  lcut = max(l_j)  # Maximum l among projectors
  rcut_j = jnp.array([float(proj['cutoff_radius']) for proj in pp_dict['PP_NONLOCAL']['PP_BETA']])  # Rcut for projectors
  gcut_j = jnp.array([int(proj['cutoff_radius_index']) for proj in pp_dict['PP_NONLOCAL']['PP_BETA']])  # Grid indices
  gcut = jnp.max(gcut_j)  # Use maximum grid index for uniform cutoff
  
  # Extract radial grid (units: Bohr)
  r_g = jnp.array(pp_dict['PP_MESH']['PP_R'])[:gcut]  # Radial points
  dr_g = jnp.array(pp_dict['PP_MESH']['PP_RAB'])[:gcut]  # r * dr for integration
  
  # Extract radial functions (in QE storage convention)
  pt_jg = jnp.array([proj['values'] for proj in pp_dict['PP_NONLOCAL']['PP_BETA']])[:, :gcut]  # β(r) * r * √(4π)
  phi_jg = jnp.array([phi['values'] for phi in pp_dict['PP_FULL_WFC']['PP_AEWFC']])[:, :gcut]  # φ(r) * r * √(4π)
  phit_jg = jnp.array([phi['values'] for phi in pp_dict['PP_FULL_WFC']['PP_PSWFC']])[:, :gcut]  # φ̃(r) * r * √(4π)
  # Core densities (stored as n(r) without factors in QE)
  # Integration: ∫ n_c(r) * 4π * r² dr = N_core
  nc_g = jnp.array(pp_dict['PP_PAW']['PP_AE_NLCC'])[:gcut]  # AE core density n_c(r)
  nct_g = jnp.array(pp_dict['PP_NLCC'])[:gcut]  # Pseudo core density ñ_c(r)
  vbar_g = jnp.array(pp_dict['PP_LOCAL'])[:gcut]  # Local pseudopotential V_loc(r)

  # Augmentation charge setup
  nj = len(l_j)  # Number of projector radial functions
  nq = nj * (nj + 1) // 2  # Number of unique pairs (upper triangular)
  
  # Augmentation functions Q_ij^l(r) - stored as Q(r) * r² in QE
  # Convert to physical Q(r) by dividing by r² and 4π
  n_lqg = jnp.zeros((2 * lcut + 1, nq, gcut))
  
  # Multipole moments Δ_lq from PP_MULTIPOLES
  # NOTE: this is the same as our calculation, we do not use it
  Delta_lq = jnp.array(pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['PP_MULTIPOLES']).reshape(lmax + 1, nj, nj)
  Delta_lq = jnp.transpose(Delta_lq, (1, 2, 0))[jnp.triu_indices(nj)].T
  
  # Extract augmentation functions Q_ij^l(r) from PP_QIJ
  for qijl in pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['PP_QIJ']:
    # QE stores Q(r) * r², convert to Q(r) / 4π for internal use
    n_lqg = n_lqg.at[
      int(qijl['angular_momentum']),
      int(qijl['first_index']) * nj + int(qijl['second_index'])
    ].set(jnp.array(qijl['values'][:gcut]) / r_g[:gcut]**2 / 4 / jnp.pi)

  assert r_g.shape[0] == gcut
  assert dr_g.shape[0] == gcut
  assert phi_jg.shape[1] == gcut
  assert phit_jg.shape[1] == gcut
  assert nc_g.shape[0] == gcut
  assert nct_g.shape[0] == gcut
  
  return r_g, dr_g, phi_jg, phit_jg, nc_g, nct_g, vbar_g, l_j, pt_jg, Z, lmax, lcut, gcut


def setup_gpaw(atom_type: str):
  """Load and parse GPAW PAW setup file.
  
  Grid Properties:
  - Units: Bohr (atomic units)
  - Grid equation: r = a * i / (n - i)
  - Integration: ∫n_stored * r² * dr * √(4π) = N_electrons
  
  Returns:
    Tuple with all PAW quantities including g_lg for compensation charges
  """
  
  # Load GPAW setup file
  pp_data = parse_paw_setup(f'/home/aiops/zhaojx/paw-minimal/pseudopotential/{atom_type}.PBE')
  
  # Also load using GPAW's native loader to get shape_function parameters
  data = SetupData('C', 'PBE', readxml=True)
  
  # Extract basic properties
  Z = int(pp_data['atom']['Z'])  # Total atomic number
  
  # Construct radial grid
  grid_info = pp_data['radial_grid']
  a = grid_info['a']
  n = grid_info['n']
  i = jnp.arange(n)
  r_g_original = a * i / (n - i)  # Keep original grid for g_lg calculation
  dr_g_original = a * n / (n - i) ** 2
  
  # Make a copy for modification
  r_g = r_g_original.copy()
  dr_g = dr_g_original.copy()
  
  # NOTE: this modification is used to avoid the singularity at r=0
  # as well as the integration error
  r_g = r_g.at[0].set(1e-5)
  dr_g = dr_g.at[0].set(0)
  
  # Angular momentum information from valence states
  l_j = jnp.array([state['l'] for state in pp_data['valence_states']])
  lcut = max(l_j)
  gcut2 = 258
  if gcut2 > len(r_g):
    gcut2 = len(r_g)
  
  # Apply cutoff to grid
  r_g = r_g[:gcut2]
  dr_g = dr_g[:gcut2]

  phi_jg = jnp.array([wave['values'][:gcut2] for wave in pp_data['ae_partial_waves']]) * r_g
  phit_jg = jnp.array([wave['values'][:gcut2] for wave in pp_data['pseudo_partial_waves']]) * r_g
  pt_jg = jnp.array([proj['values'][:gcut2] for proj in pp_data['projector_functions']]) * r_g
  nc_g = jnp.array(pp_data['ae_core_density'][:gcut2]) / jnp.sqrt(4 * jnp.pi)
  nct_g = jnp.array(pp_data['pseudo_core_density'][:gcut2]) / jnp.sqrt(4 * jnp.pi)
  
  # Local potential, skip r=0 point and apply cutoff
  vbar_g = jnp.array(pp_data.get('zero_potential', jnp.zeros(n))[:gcut2])
  
  lmax = lcut  # Maximum l for augmentation should equal lcut for GPAW compatibility
  
  # Calculate g_lg shape functions for compensation charges
  # We need to implement this properly for GPAW files
  # Using the same Gaussian shape function approach as GPAW
  
  # Get shape function parameters from SetupData
  shape_params = getattr(data, 'shape_function', {'type': 'gauss', 'rc': 3.794733192202e-01})
  rc = shape_params.get('rc', 3.794733192202e-01)
  sf_type = shape_params.get('type', 'gauss')
  
  # Initialize g_lg for all l values up to lmax
  g_lg = jnp.zeros((lmax + 1, gcut2))
  
  # Use the original grid (with r[0] = 0) for g_lg calculation to match GPAW exactly
  r_g_for_glg = r_g_original[:gcut2]
  dr_g_for_glg = dr_g_original[:gcut2]
  
  if sf_type == 'gauss':
      # Gaussian shape functions following GPAW's convention
      # g_lg[0] = 4 / rc^3 / sqrt(pi) * exp(-(r/rc)^2)
      g_lg = g_lg.at[0].set(4 / rc**3 / jnp.sqrt(jnp.pi) * jnp.exp(-(r_g_for_glg / rc)**2))
      
      # Higher l components: g_lg[l] = 2/(2l+1)/rc^2 * r * g_lg[l-1]
      for l in range(1, lmax + 1):
          g_lg = g_lg.at[l].set(2.0 / (2 * l + 1) / rc**2 * r_g_for_glg * g_lg[l - 1])
      
      # Normalize each l-component according to GPAW convention
      # GPAW normalizes so that rgd.integrate(g_lg[l], l) = 4π
      # Since integrate multiplies by 4π, the raw integral should be 1.0
      for l in range(lmax + 1):
          # Calculate integral with 4π factor (like rgd.integrate does)
          # Skip r=0 point in integration like GPAW does
          integral_with_4pi = jnp.sum(g_lg[l, 1:] * r_g_for_glg[1:]**(l + 2) * dr_g_for_glg[1:]) * 4 * jnp.pi
          if integral_with_4pi > 1e-10:
              # Divide by integral and multiply by 4π to get correct normalization
              g_lg = g_lg.at[l].set(g_lg[l] / integral_with_4pi * (4 * jnp.pi))
  else:
      # For other shape function types, use simple fallback
      print(f"Warning: Shape function type '{sf_type}' not fully implemented, using simplified version")
      g_lg = g_lg.at[0].set(4 / rc**3 / jnp.sqrt(jnp.pi) * jnp.exp(-(r_g / rc)**2))
      for l in range(1, lmax + 1):
          g_lg = g_lg.at[l].set(2.0 / (2 * l + 1) / rc**2 * r_g * g_lg[l - 1])

  return {
    'r_g': r_g,
    'dr_g': dr_g,
    'phi_jg': phi_jg,
    'phit_jg': phit_jg,
    'nc_g': nc_g,
    'nct_g': nct_g,
    'vbar_g': vbar_g,
    'l_j': l_j,
    'pt_jg': pt_jg,
    'Z': Z,
    'lmax': lmax,
    'lcut': lcut,
    'gcut2': gcut2,
    'g_lg': g_lg,
    'K_p': pp_data['kinetic_energy_differences'],
    'K_c': pp_data['core_energy']['kinetic'] - pp_data['ae_energy']['kinetic'],
    # 'K_c': pp_data['core_energy']['kinetic'],
    'e_xc': pp_data['ae_energy']["xc"]
  }


def  calc_paw(
  setup_data: dict):
  """Calculate PAW correction terms using QE UPF data in native convention.
  
  Input Convention (QE UPF as loaded by setup_qe):
  -------------------------------------------------
  Arrays passed to this function maintain QE's native storage convention:
  
  Radial Functions:
  - phi_jg, phit_jg: AE/PS wavefunctions stored as φ(r)*r*√(4π)
  - pt_jg: Projector functions stored as β(r)*r*√(4π)
  - nc_g, nct_g: Core densities stored as n(r) (physical density)
  
  Grid and Integration:
  - r_g: Radial grid points in Bohr
  - dr_g: Integration weights PP_RAB = r*dr (includes r factor)
  - Integration: ∫f(r)dr → Σ f[i]*dr_g[i] for radial integrals
  
  Angular Momentum:
  - l_j: Angular momentum for each projector
  - lmax: Maximum l for augmentation
  - lcut: Maximum l among projectors
  
  This function computes PAW quantities including:
  - Augmentation density n_qg
  - Smooth augmentation density nt_qg  
  - Multipole moments Delta_pL
  - Coulomb correction scalar M
  - Projector overlaps B_ii
  
  All calculations respect QE's storage convention with appropriate
  factor handling for physical correctness.
  
  Args:
    r_g (np.ndarray): Radial grid points, shape (gcut,)
    dr_g (np.ndarray): Radial grid integration weights (dr), shape (gcut,)
    phi_jg (np.ndarray): All-electron partial waves φ(r)*r, shape (nj, gcut)
                        These match the true AE wavefunctions inside core region
    phit_jg (np.ndarray): Pseudo partial waves φ̃(r)*r, shape (nj, gcut)
                         Smooth functions matching φ outside core region
    nc_g (np.ndarray): All-electron core density n_c(r), shape (gcut,)
                       True electron density of core states
    nct_g (np.ndarray): Smooth core density ñ_c(r), shape (gcut,)
                        Pseudized version of nc_g, smooth at origin
    vbar_g (np.ndarray): Local pseudopotential V_loc(r), shape (gcut,)
    l_j (np.ndarray): Angular momentum for each projector, shape (nj,)
    pt_jg (np.ndarray): Projector functions p̃(r)*r, shape (nj, gcut)
                       Dual functions to φ̃, satisfying ⟨p̃_i|φ̃_j⟩ = δ_ij
    Z (int): Atomic number (total nuclear charge)
    lmax (int): Maximum angular momentum for augmentation
    lcut (int): Maximum angular momentum for projectors
    gcut (int): Number of radial grid points (cutoff index)
  
  Returns:
    dict: Dictionary containing PAW correction terms:
      - B_ii: Projector overlap matrix ⟨p̃_i|p̃_j⟩
      - M: Scalar Coulomb correction for core-core interaction
      - n_qg: Augmentation densities from AE waves
      - nt_qg: Augmentation densities from pseudo waves
      - Delta_pL: Multipole moments in (p,L) representation
      - Delta0: Monopole compensation charge deficit
      - gcut: Grid cutoff index
  """
  
  # Calculate derived quantities first
  r_g = setup_data['r_g']
  dr_g = setup_data['dr_g']
  phi_jg = setup_data['phi_jg']
  phit_jg = setup_data['phit_jg']
  nc_g = setup_data['nc_g']
  nct_g = setup_data['nct_g'] 
  vbar_g = setup_data['vbar_g']
  l_j = setup_data['l_j']
  pt_jg = setup_data['pt_jg']
  Z = setup_data['Z']
  lmax = setup_data['lmax']
  lcut = setup_data['lcut']
  gcut = setup_data['gcut2']
  g_lg = setup_data['g_lg']
  
  n_rgd = r_g.shape[0]  # number of grid points
  nj = phi_jg.shape[0]  # number of projectors radial functions
  
  # Generate n_j following GPAW convention:
  # - Use principal quantum number for occupied states (2 for Carbon's 2s, 2p)
  # - Use -1 for unoccupied/virtual states
  # For Carbon: typically [2, 2, -1, -1] for [2s, 2p, virtual_s, virtual_p]
  n_j = []
  # First two are occupied (2s^2 2p^2 for Carbon)
  occupied_count = {'s': 1, 'p': 1}  # Carbon has occupied 2s and 2p
  for i, l in enumerate(l_j):
    if l == 0:  # s orbital
      if occupied_count.get('s', 0) > 0:
        n_j.append(2)  # 2s
        occupied_count['s'] -= 1
      else:
        n_j.append(-1)  # virtual s
    elif l == 1:  # p orbital
      if occupied_count.get('p', 0) > 0:
        n_j.append(2)  # 2p
        occupied_count['p'] -= 1
      else:
        n_j.append(-1)  # virtual p
    else:  # d or higher
      n_j.append(-1)  # virtual
  n_j = jnp.array(n_j)
  
  ni = nj + l_j.sum() * 2  # number of projectors
  nq = nj * (nj + 1) // 2  # number of radial function pairs
  _np = ni * (ni + 1) // 2  # number of projector pairs

  def calculate_T_Lqp():
    """Calculate Gaunt coefficients T_Lqp for angular momentum coupling.
    
    These coefficients couple pairs of projectors (q index) with 
    spherical harmonics (L index) for multipole expansions."""
    Lcut = (2 * lcut + 1)**2
    G_LLL = gaunt(int(lcut))[:, :, :Lcut]
    LGcut = G_LLL.shape[2]
    T_Lqp = jnp.zeros((Lcut, nq, _np))
    i = 0
    j = 0
    jlL_i = []
    for l, n in zips(l_j, n_j):
      for m in range(2 * l + 1):
        jlL_i.append((j, l, l**2 + m))
        i += 1
      j += 1
    p = 0
    i1 = 0
    for j1, l1, L1 in jlL_i:
      for j2, l2, L2 in jlL_i[i1:]:
        if j1 < j2:
          q = j2 + j1 * nj - j1 * (j1 + 1) // 2
        else:
          q = j1 + j2 * nj - j2 * (j2 + 1) // 2
        T_Lqp = T_Lqp.at[:LGcut, q, p].set(G_LLL[L1, L2])
        p += 1
      i1 += 1
    return T_Lqp

  def calculate_projector_overlaps():
    """Compute projector function overlaps B_ii = <pt_i | pt_i>."""
    B_jj = jnp.sum(pt_jg[:, None, :] * pt_jg[None, :, :] * dr_g, axis=2)
    # breakpoint()
    B_ii = jnp.zeros((ni, ni))
    i1 = 0
    for j1, l1 in enumerate(l_j):
      for m1 in range(2 * l1 + 1):
        i2 = 0
        for j2, l2 in enumerate(l_j):
          for m2 in range(2 * l2 + 1):
            if l1 == l2 and m1 == m2:
              B_ii = B_ii.at[i1, i2].set(B_jj[j1, j2])
            i2 += 1
        i1 += 1
    return B_ii

  def calc_compensation_charges():

    index = jnp.triu_indices(nj)
    n_qg = (phi_jg[:, None, :] * phi_jg[None])[index] / r_g**2 / 4 / jnp.pi
    nt_qg = (phit_jg[:, None, :] * phit_jg[None])[index] / r_g**2 / 4 / jnp.pi

    # NOTE: check the calculation of the multipoles moment, similar
    # results can be observed in test_paw.test_augmentation_charge
    Delta_lq = jnp.zeros((lmax + 1, nq))
    for l in range(lmax + 1):
      Delta_lq = Delta_lq.at[l].set(jnp.dot(n_qg - nt_qg, r_g**(l + 2) * dr_g))
    Delta_lq *= 4 * jnp.pi
    # Delta_lq = jnp.array(pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['PP_MULTIPOLES'])
    # index_list = [0, 1, 18, 19, 5, 22, 23, 10, 42, 11, 43, 15, 47]
    # for i in index_list:
    #   print(Delta_lq[i] - Delta_lq_[i])

    Lmax = (lmax + 1)**2
    Delta_pL = jnp.zeros((_np, Lmax))
    for l in range(lmax + 1):
      L = l**2
      for m in range(2 * l + 1):
        Delta_pL = Delta_pL.at[:, L + m].set(jnp.dot(Delta_lq[l], T_Lqp[L + m]))

    Delta0 = jnp.dot(nc_g - nct_g, r_g**2 * dr_g) * jnp.sqrt(4 * jnp.pi) - Z / jnp.sqrt(4 * jnp.pi)
    return (n_qg, nt_qg, Delta_lq, Lmax, Delta_pL, Delta0)

  T_Lqp = calculate_T_Lqp()

  """
  n_qg, nt_qg follows the same convention as that of nc_g, nct_g,
  i.e. no 4\pi \& r^2 factors
  """
  n_qg, nt_qg, Delta_lq, Lmax, Delta_pL, Delta0 = calc_compensation_charges()
  B_ii = calculate_projector_overlaps()

  r_max = jnp.maximum(r_g[None], r_g[:, None])
  r_min = jnp.minimum(r_g[None], r_g[:, None])

  def integrate_radial_function(f_g):
    r"""
    Integrate the radial function over the radial grid.
    NOTE: the integrand DOES NOT contain extra r^2 or 4\pi factor

    .. math::
      \int_0^{r_c} 4\pi f(r) r^2 dr
    
    """
    return jnp.sum(f_g * dr_g * r_g**2, axis=-1) * 4 * jnp.pi

  def poisson_rdl(
    g_L: jnp.ndarray,
    l: int,
  ):
    r"""Solve the Poisson equation over radial grid

    .. math::

    .. warning::

    Args:
      g_L (Real[Array, "spin kpts band x y z"]): radial function.
      l (int): angular momentum quantum number.

    Returns:
      (Real[Array, "spin kpts band x y z"]): radial function.
    """

    return jnp.sum(r_min**l * g_L * dr_g * r_g**2 / r_max**(l + 1), axis=-1) /\
      (2 * l + 1) * 4 * jnp.pi

  A = 0.5 * integrate_radial_function(nc_g * poisson_rdl(nc_g, 0))
  # NOTE: GPAW uses jnp.sqrt(4 * jnp.pi) since integrate_radial_function no longer includes 4*pi
  A -= 4 * jnp.pi * Z * jnp.dot(r_g * dr_g, nc_g)
  g_lg = jnp.array(g_lg)
  mct_g = nct_g + Delta0 * g_lg[0] / jnp.sqrt(4 * jnp.pi)
  A -= 0.5 * integrate_radial_function(mct_g * poisson_rdl(mct_g, 0))
  # NOTE: only for QE file testing
  # M = 0.5 * integrate_radial_function(nc_g * poisson_rdl(nc_g, 0))
  M = A

  # NOTE: currently the following code is not tested for QE pp file, but do not delete
  MB = -jnp.sum(nct_g * vbar_g * r_g**2 * dr_g) * jnp.sqrt(4 * jnp.pi)
  AB_q = -jnp.sum(nt_qg * vbar_g[None, :] * r_g**2 * dr_g, axis=1) * 4 * jnp.pi
  MB_p = jnp.dot(AB_q, T_Lqp[0])

  # calculate the linear kinetic correction
  # dekin_nn = (integrate_radial_function(phit_jg[:, None] * phit_jg * vtr_g) / (4 * jnp.pi) -
  #             integrate_radial_function(phi_jg[:, None] * phi_jg * vr_g) / (4 * jnp.pi) +
  #             dH_nn)

  # def calc_linear_kinetic_correction(T0_qp):
  #   e_kin_jj = e_kin_jj
  #   nj = len(e_kin_jj)
  #   K_q = []
  #   for j1 in range(nj):
  #     for j2 in range(j1, nj):
  #       K_q.append(e_kin_jj[j1, j2])
  #   K_p = jnp.sqrt(4 * jnp.pi) * jnp.dot(K_q, T0_qp)
  #   return K_p

  def calc_kinetic_energy(phi1: jnp.ndarray, phi2: jnp.ndarray, l: int):
    r"""
    The kinetic energy of the two-center integral is given by:

    .. math::
      \big\langle \phi_i \big| -\tfrac{1}{2}\nabla^2 \big| \phi_j \big\rangle
      = \tfrac{1}{2}\,\delta_{\ell_i\ell_j}\,\delta_{m_i m_j}
      \int_0^{r_c}\!\left[ u_i'(r)\,u_j'(r)
      +\frac{\ell(\ell+1)}{r^2}\,u_i(r)\,u_j(r) \right]\; dr 

    This method is based on a derivation with ChatGPT
    https://chatgpt.com/c/68798a15-ae84-8002-815c-d0d1566c9ade
    """

    def df(f: jnp.ndarray):
      # NOTE: we are using forward difference to calculate the derivative here
      f = f.at[1:].add(-f[:-1])
      return f

    def dfdr(f: jnp.ndarray):
      return df(f) / df(r_g)
    
    dphi1dr = dfdr(phi1)
    dphi2dr = dfdr(phi2)
    return (integrate_radial_function(phi1 * phi2 / r_g**4) * l * (l + 1) +
      integrate_radial_function(dphi1dr * dphi2dr / r_g**2)) / 2

  K = jnp.zeros((nj, nj))
  for i in range(nj):
    for j in range(i, nj):
      if l_j[i] == l_j[j]:
        K = K.at[i, j].set(calc_kinetic_energy(jnp.array(phi_jg[i]), jnp.array(phi_jg[j]), l_j[i]))
        K = K.at[j, i].set(K[i, j])

  # K_p = jnp.zeros((ni, ni))
  # for i in range(ni):
  #   for j in range(i, ni):
  #     if proj_l[i] == proj_l[j] and proj_m[i] == proj_m[j]:
  #       K_p = K_p.at[i, j].set(K[proj_r[i], proj_r[j]])
  #       K_p = K_p.at[j, i].set(K_p[i, j])
  # breakpoint()

  poisson_rdl0 = jax.vmap(partial(poisson_rdl, l=0))
  def calculate_coulomb_corrections():
    r"""
    The Coulomb energy corrections are given by:

    .. math::
      A_q = \frac{1}{2} \left( \int_0^{r_c} n_c(r) \nabla^2 G_0(r) dr + \int_0^{r_c} n_q(r) \nabla^2 G_0(r) dr \right)

    """

    # NOTE: these two terms are the same, only for numerical stability
    # 1st term in (46)
    A_q = 0.5 * (integrate_radial_function(nc_g * poisson_rdl0(n_qg)) +
                integrate_radial_function(n_qg * poisson_rdl0(nc_g.reshape(1, -1)))) * jnp.sqrt(4 * jnp.pi)
    # 2nd term + 5th termin (46)
    A_q -= 0.5 * (integrate_radial_function(mct_g * poisson_rdl0(nt_qg)) +
                integrate_radial_function(nt_qg * poisson_rdl0(mct_g.reshape(1, -1)))) * jnp.sqrt(4 * jnp.pi)
    # 3rd term in (46)
    A_q -= 4 * jnp.pi * Z * jnp.dot(n_qg, r_g * dr_g) * jnp.sqrt(4 * jnp.pi)
    # 4th term + 6th term in (46)
    # This is for QE file testing
    # A_q -= 0.5 * (integrate_radial_function(mct_g * poisson_rdl0(n_lqg[0])) +
    #               integrate_radial_function(n_lqg[0] * poisson_rdl0(mct_g.reshape(1, -1))))
    # This is for GPAW file testing
    A_q -= 0.5 * (integrate_radial_function(mct_g * poisson_rdl0(g_lg[0:1])) +
                  integrate_radial_function(g_lg[0] * poisson_rdl0(mct_g.reshape(1, -1)))) * \
        Delta_lq[0] / jnp.sqrt(4 * jnp.pi)
    M_p = jnp.dot(A_q, T_Lqp[0])

    A_lqq = []
    for l in range(2 * lcut + 1):
      poisson_rdl_ = jax.vmap(partial(poisson_rdl, l=l))
      # 1st term in (47)
      A_qq = 0.5 * integrate_radial_function(n_qg[None] * poisson_rdl_(n_qg)[:, None]) * 4 * jnp.pi
      # 2nd term in (47)
      A_qq -= 0.5 * integrate_radial_function(nt_qg[None] * poisson_rdl_(nt_qg)[:, None]) * 4 * jnp.pi  
      # if l <= lmax:
      # #   A_qq -= 0.5 * integrate_radial_function(poisson_rdl_(nt_qg)[None] * n_lqg[l][:, None])
      # #   A_qq -= 0.5 * integrate_radial_function(nt_qg[None] * poisson_rdl_(n_lqg[l])[:, None])
      # #   A_qq -= 0.5 * integrate_radial_function(n_lqg[l][None] * poisson_rdl_(n_lqg[l])[:, None])
      #   A_qq -= 0.5 * integrate_radial_function(poisson_rdl_(nt_qg)[None] * n_lqg[l][:, None])
      #   A_qq -= 0.5 * integrate_radial_function(nt_qg[None] * poisson_rdl_(n_lqg[l])[:, None])
      #   A_qq -= 0.5 * integrate_radial_function(n_lqg[l][None] * poisson_rdl_(n_lqg[l])[:, None])
      A_lqq.append(A_qq)

    M_pp = jnp.zeros((_np, _np))
    L = 0
    for l in range(2 * lcut + 1):
      for m in range(2 * l + 1):  # m?
        M_pp += jnp.dot(jnp.transpose(T_Lqp[L]), jnp.dot(A_lqq[l], T_Lqp[L]))
        L += 1
    return M_p, M_pp

  M_p, M_pp = calculate_coulomb_corrections()

  return {
    'B_ii': B_ii,
    'M': M,
    'M_p': M_p,
    'MB': MB,
    'MB_p': MB_p,
    'n_qg': n_qg * 4 * jnp.pi,
    'nt_qg': nt_qg * 4 * jnp.pi,
    'Delta_lq': Delta_lq,
    'Delta_pL': Delta_pL,
    'Delta0': Delta0,
    'gcut': gcut,
    'g_lg': g_lg,
    'vbar_g': vbar_g,
    'T_Lqp': T_Lqp
  }


def compute_proj_pw_overlap(G_grid: jnp.ndarray):
    """
    This is the customized function to compute the projector-plane wave overlap matrix:
    We perform the radial integration in real space and compare the results with f_GI
    """

    pp_data = parse_paw_setup(f'/home/aiops/zhaojx/M_p-align-claude/pseudopotential/C.PBE')
    from scipy.special import spherical_jn

    gcut2 = 258
    grid_info = pp_data['radial_grid']
    a = grid_info['a']
    n = grid_info['n']
    i = jnp.arange(n)
    r_g = a * i / (n - i)  # Keep original grid for g_lg calculation
    dr_g = a * n / (n - i)**2
    r_g = r_g[:gcut2]
    dr_g = dr_g[:gcut2]
    pt_jg = jnp.array([proj['values'][:gcut2] for proj in pp_data['projector_functions']])
    n_g = G_grid.shape[0]
    overlap = jnp.zeros((n_g, 13), dtype=jnp.complex128)

    proj_list = [0, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4]
    m_list = [0, -1, 0, 1, 0, -1, 0, 1, -2, -1, 0, 1, 2]
    l_list = [0, 1, 1, 1, 0, 1, 1, 1, 2, 2, 2, 2, 2]

    from scipy.special import sph_harm_y
    theta_grid = jnp.arccos(G_grid[:, 2] / jnp.linalg.norm(G_grid, axis=1))
    theta_grid = theta_grid.at[0].set(0.0)
    phi_grid = jnp.arctan2(G_grid[:, 1], G_grid[:, 0])
    phi_grid = phi_grid.at[0].set(0.0)   # handle the G = 0 singular case
    for k in range(n_g):
        for j in range(13):
            bessel_grid = spherical_jn(l_list[j], r_g * jnp.linalg.norm(G_grid[k]))
            overlap = overlap.at[k, j].set(jnp.sum(bessel_grid * pt_jg[proj_list[j]] * r_g * r_g * dr_g) * 4 * jnp.pi * (-1j) ** l_list[j] *\
                sph_harm_y(l_list[j], m_list[j], theta_grid[k], phi_grid[k]))

    # transform the result from spherical harmonics to real spherical harmonics
    tmp_list = [0, 3, 2, 1, 4, 7, 6, 5, 12, 11, 10, 9, 8] # relate the m to -m indices
    overlap_ = overlap.copy()
    for j in range(13):
        if m_list[j] > 0:
            overlap_ = overlap.at[:, j].set((overlap[:, tmp_list[j]] + (-1)**m_list[j] * overlap[:, j]) / jnp.sqrt(2))
        elif m_list[j] < 0:
            overlap_ = overlap.at[:, j].set((overlap[:, j] - (-1)**m_list[j] * overlap[:, tmp_list[j]]) / (-1j * jnp.sqrt(2)))
        
    return overlap_


def calc_paw_xc_correction(
  D_asp: list,
  setups: list,
  xc_kernel: callable,
  nspins: int = 1,
  ):

  from gpaw.sphere.lebedev import R_nv, weight_n, Y_nL
  def _calculate_xc_energy(D_sLq, n_qg, nc0_sg):
    n_sLg = jnp.dot(D_sLq, n_qg)  # shape: [n_spin, Lmax, n_g]
    n_sLg[:, 0, :] += nc0_sg
    E_xc = 0.0
    Y_nL_local = Y_nL[:, :Lmax]  # Only use L up to Lmax
    for n in range(50):  # 50 Lebedev points
      w = weight_n[n]
      Y_L = Y_nL_local[n]  # shape: [Lmax]
      n_sg = jnp.dot(Y_L, n_sLg)  # shape: [n_spin, n_g]
      e_g = rgd.empty()
      dedn_sg = rgd.zeros(nspins)
      xc_kernel.calculate(e_g, n_sg, dedn_sg)
      E_xc += w * rgd.integrate(e_g)

    return E_xc

  delta_e_xc_total = 0.0

  for a, D_sp in D_asp.items():
    setup = setups[a]
    xcc = setup.xc_correction

    if xcc is None:
        continue

    # Get data from xc_correction object
    rgd = setup.rgd
    n_qg = setup.local_corr.n_qg      # AE radial products: phi_j1 * phi_j2
    nt_qg = setup.local_corr.nt_qg    # PS radial products: phit_j1 * phit_j2
    nc_g = setup.local_corr.nc_g      # Core density
    nct_g = setup.local_corr.nct_g    # Smooth core density
    rgd = xcc.rgd   # use the truncated grid!!!
    T_Lqp = setup.local_corr.T_Lqp
    e_xc0 = setup.data.e_xc    # Reference atom XC energy
    Lmax = (2 * setup.lmax + 1)**2
    D_sLq = jnp.inner(D_sp, T_Lqp)

    nc0_sg = nc_g / nspins
    nct0_sg = nct_g / nspins
    e_ae = _calculate_xc_energy(D_sLq, n_qg, nc0_sg)
    e_ps = _calculate_xc_energy(D_sLq, nt_qg, nct0_sg)
    delta_e_xc = e_ae - e_ps - e_xc0
    delta_e_xc_total += delta_e_xc
    print(f"Atom {a}: E_AE={e_ae:.6f}, E_PS={e_ps:.6f}, E_xc0={e_xc0:.6f}, Delta={delta_e_xc:.6f} Ha")

  return delta_e_xc_total