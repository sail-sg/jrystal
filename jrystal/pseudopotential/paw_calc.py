from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .gaunt import gaunt

from .._src import xc
from .load_gpaw import parse_paw_setup


def calc_paw(setup_data: dict):
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
    phi_jg (np.ndarray): All-electron partial waves φ(r), shape (nj, gcut)
                        These match the true AE wavefunctions inside core region
    phit_jg (np.ndarray): Pseudo partial waves φ̃(r), shape (nj, gcut)
                         Smooth functions matching φ outside core region
    nc_g (np.ndarray): All-electron core density n_c(r), shape (gcut,)
                       True electron density of core states
    nct_g (np.ndarray): Smooth core density ñ_c(r), shape (gcut,)
                        Pseudized version of nc_g, smooth at origin
    vbar_g (np.ndarray): Local pseudopotential V_loc(r), shape (gcut,)
    l_j (np.ndarray): Angular momentum for each projector, shape (nj,)
    pt_jg (np.ndarray): Projector functions p̃(r), shape (nj, gcut)
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
  phi_jg = setup_data['phi_jg'] * r_g
  phit_jg = setup_data['phit_jg'] * r_g
  nc_g = setup_data['nc_g']
  nct_g = setup_data['nct_g'] 
  vbar_g = setup_data['vbar_g']
  l_j = setup_data['l_j']
  pt_jg = setup_data['pt_jg'] * r_g
  Z = setup_data['Z']
  lmax = setup_data['lmax']
  lcut = setup_data['lcut']
  gcut = setup_data['gcut2']
  g_lg = setup_data['g_lg']
  
  n_rgd = r_g.shape[0]  # number of grid points
  nj = phi_jg.shape[0]  # number of projectors radial functions
  
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
    for l in l_j:
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
    n_qg = (setup_data['phi_jg'][:, None, :] * setup_data['phi_jg'][None])[index] / 4 / jnp.pi
    nt_qg = (setup_data['phit_jg'][:, None, :] * setup_data['phit_jg'][None])[index] / 4 / jnp.pi

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
  r_max = jnp.where(r_max < 1e-14, 1e-14, r_max)
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

  # K = jnp.zeros((nj, nj))
  # for i in range(nj):
  #   for j in range(i, nj):
  #     if l_j[i] == l_j[j]:
  #       K = K.at[i, j].set(calc_kinetic_energy(jnp.array(phi_jg[i]), jnp.array(phi_jg[j]), l_j[i]))
  #       K = K.at[j, i].set(K[i, j])

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
      # 1st term in (47): 0.5 * n_qg @ poisson(n_qg)
      A_qq = 0.5 * integrate_radial_function(n_qg[None] * poisson_rdl_(n_qg)[:, None]) * 4 * jnp.pi
      # 2nd term in (47): -0.5 * nt_qg @ poisson(nt_qg)
      A_qq -= 0.5 * integrate_radial_function(nt_qg[None] * poisson_rdl_(nt_qg)[:, None]) * 4 * jnp.pi

      # Compensation charge terms (3rd-5th terms)
      # These terms involve the shape function g_lg and multipole moments Delta_lq
      if l <= lmax:
        # Poisson solution of g_lg[l] - a single radial function
        w_g_lg = poisson_rdl(g_lg[l], l)  # shape: (gcut,)

        # Following GPAW's convention: the integrals are computed using rgd.integrate
        # which is sum(f * r^2 * dr) * 4π. But the Poisson solver already includes
        # the 4π/(2l+1) factor, so we need to be careful about double-counting.

        # 3rd term: -0.5 * outer(Delta_lq[l], integrate(poisson(nt_qg) * g_lg[l]))
        term3_q = integrate_radial_function(poisson_rdl_(nt_qg) * g_lg[l])
        A_qq -= 0.5 * jnp.outer(Delta_lq[l], term3_q)

        # 4th term: -0.5 * outer(integrate(nt_qg * poisson(g_lg[l])), Delta_lq[l])
        term4_q = integrate_radial_function(nt_qg * w_g_lg)
        A_qq -= 0.5 * jnp.outer(term4_q, Delta_lq[l])

        # 5th term: -0.5 * integrate(g_lg[l] * poisson(g_lg[l])) * outer(Delta_lq, Delta_lq)
        term5_scalar = integrate_radial_function(g_lg[l] * w_g_lg)
        A_qq -= 0.5 * term5_scalar * jnp.outer(Delta_lq[l], Delta_lq[l]) / 4 / jnp.pi

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
    'M_pp': M_pp,
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


def build_paw_xc_correction(paw, g_vec, xc_type: str):
  """Build a PAW XC correction helper using precomputed Lebedev data."""
  from .lebedev import weight_n, Y_nL

  weight_n = jnp.array(weight_n)
  Y_nL = jnp.array(Y_nL)

  def _calculate_xc_energy(D_sLq, n_qg, nc0_sg, dr_g, r_g, lmax):
    n_sLg = jnp.dot(D_sLq, n_qg)  # shape: [n_spin, Lmax, n_g]
    n_sLg = n_sLg.at[0].add(nc0_sg * jnp.sqrt(4 * jnp.pi))
    Lmax = (2 * lmax + 1)**2
    Y_nL_local = Y_nL[:, :Lmax]  # Only use L up to Lmax
    n = jnp.dot(Y_nL_local, n_sLg)
    # TODO: here we encounter negative density, we use a quick fix, should reconsider
    n = jnp.where(n > 0, n, 0)

    def _exc_density(n_sg):
      if n_sg.ndim == 1:
        n_sg = n_sg[None, :]
      return xc.xc_density(n_sg, g_vec, xc_type=xc_type)

    exc_density = jax.vmap(_exc_density)(n)
    n_total = n if n.ndim == 2 else jnp.sum(n, axis=1)
    E_xc_ = jnp.einsum(
      "i, ij, j",
      weight_n,
      n_total * exc_density,
      dr_g * r_g**2
    ) * 4 * jnp.pi
    return E_xc_

  def calc_paw_xc_correction(atom: str, D_p_packed):
    n_qg = paw.n_qg[atom]
    nt_qg = paw.nt_qg[atom]
    nc_g = paw.nc_g[atom]
    nct_g = paw.nct_g[atom]
    T_Lqp = paw.T_Lqp[atom]
    e_xc0 = paw.e_xc0[atom]
    dr_g = paw.dr_g[atom]
    r_g = paw.r_g[atom]
    lmax = paw.lmax[atom]

    D_sLq = jnp.inner(D_p_packed, T_Lqp)
    e_ae = _calculate_xc_energy(D_sLq, n_qg, nc_g, dr_g, r_g, lmax)
    e_ps = _calculate_xc_energy(D_sLq, nt_qg, nct_g, dr_g, r_g, lmax)
    return e_ae - e_ps - e_xc0

  return calc_paw_xc_correction


def compute_proj_pw_overlap(
  G_grid: jnp.ndarray,
  pos: jnp.ndarray,
):
    """
    This is the customized function to compute the projector-plane wave overlap matrix:
    We perform the radial integration in real space and compare the results with f_GI
    """

    pp_data = parse_paw_setup(f'/home/aiops/zhaojx/paw-minimal/pseudopotential/C.LDA')
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
    theta_grid = jnp.arccos(G_grid[:, 2] / jnp.where(jnp.linalg.norm(G_grid, axis=1) > 0, jnp.linalg.norm(G_grid, axis=1), 1))
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
            overlap_ = overlap_.at[:, j].set((overlap[:, tmp_list[j]] + (-1)**m_list[j] * overlap[:, j]) / jnp.sqrt(2))
        elif m_list[j] < 0:
            overlap_ = overlap_.at[:, j].set((overlap[:, j] - (-1)**m_list[j] * overlap[:, tmp_list[j]]) / (-1j * jnp.sqrt(2)))
        
    # overlap_tmp = jnp.zeros((n_g, 13), dtype=jnp.complex128)
    # y_lm = jnp.zeros((n_g, 13), dtype=jnp.complex128)
    # for k in range(n_g):
    #     for j in range(13):
    #         bessel_grid = spherical_jn(l_list[j], r_g * jnp.linalg.norm(G_grid[k]))
    #         overlap_tmp = overlap_tmp.at[k, j].set(jnp.sum(bessel_grid * pt_jg[proj_list[j]] * r_g * r_g * dr_g))
    #         if m_list[j] > 0:
    #           y_lm = y_lm.at[k, j].set(
    #             sph_harm_y(l_list[j], m_list[j], theta_grid[k], phi_grid[k]).real * (-1)**m_list[j] * jnp.sqrt(2)
    #             * (-1.j)**l_list[j]
    #           )
    #         elif m_list[j] < 0:
    #           y_lm = y_lm.at[k, j].set(
    #             sph_harm_y(l_list[j], -m_list[j], theta_grid[k], phi_grid[k]).imag * (-1)**m_list[j] * jnp.sqrt(2)
    #             * (-1.j)**l_list[j]
    #           )    
    #         else:
    #           y_lm = y_lm.at[k, j].set(sph_harm_y(l_list[j], m_list[j], theta_grid[k], phi_grid[k]) * (-1.j)**l_list[j])
    # overlap_tmp *= y_lm * 4 * jnp.pi

    structure_factor = jnp.exp(-1.j * G_grid @ pos).reshape(-1, 1)
    return overlap_ * structure_factor
