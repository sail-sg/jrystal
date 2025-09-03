from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from gpaw.gaunt import gaunt
from gpaw.new import zips
from jrystal.pseudopotential.load import parse_upf
from jrystal._src.energy import kinetic, hartree, exc_functional

def setup_qe():

  # load the pseudopotential
  pp_dict = parse_upf('/home/aiops/zhaojx/jrystal/pseudopotential/C.pbe-n-kjpaw_psl.1.0.0.UPF')
  Z = 6
  # Z = int(pp_dict["PP_HEADER"]["Z_valence"])
  lmax = int(pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['l_max_aug']) # maximum angular momentum of the augmentation charge
  l_j = np.array([int(proj['angular_momentum']) for proj in pp_dict['PP_NONLOCAL']['PP_BETA']]) # angular momentum of each projector
  lcut = max(l_j)
  rcut_j = jnp.array([float(proj['cutoff_radius']) for proj in pp_dict['PP_NONLOCAL']['PP_BETA']]) # projector functions
  gcut_j = jnp.array([int(proj['cutoff_radius_index']) for proj in pp_dict['PP_NONLOCAL']['PP_BETA']]) # projector functions
  gcut = jnp.max(gcut_j)
  # NOTE: here we use a uniform cutoff to handle all the wave functions and
  # densities
  r_g = jnp.array(pp_dict['PP_MESH']['PP_R'])[:gcut] # radial grid
  dr_g = jnp.array(pp_dict['PP_MESH']['PP_RAB'])[:gcut] # radial grid integration weight
  pt_jg = jnp.array([proj['values'] for proj in pp_dict['PP_NONLOCAL']['PP_BETA']])[:, :gcut] # projector functions
  phi_jg = jnp.array([phi['values'] for phi in pp_dict['PP_FULL_WFC']['PP_AEWFC']])[:, :gcut] # all-electron wave functions
  phit_jg = jnp.array([phi['values'] for phi in pp_dict['PP_FULL_WFC']['PP_PSWFC']])[:, :gcut] # pseudo wave functions
  r"""
  It is the true charge density, i.e. it will be 
  correctly integrated as \sum_i 4 \pi r_i^2 nlcc_i
  Different from the setting in GPAW, see the discussions
  after (41b)
  """
  nc_g = jnp.array(pp_dict['PP_PAW']['PP_AE_NLCC'])[:gcut] # all-electron non-linear core charge
  nct_g = jnp.array(pp_dict['PP_NLCC'])[:gcut] # non-linera core charge
  vbar_g = jnp.array(pp_dict['PP_LOCAL'])[:gcut] # local pseudopotential

  n_lqg = jnp.zeros((2 * lcut + 1, nq, gcut))
  Delta_lq = jnp.array(pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['PP_MULTIPOLES']).reshape(lmax + 1, nj, nj)
  Delta_lq = jnp.transpose(Delta_lq, (1, 2, 0))[jnp.triu_indices(nj)].T
  for qijl in pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['PP_QIJ']:
    n_lqg = n_lqg.at[
      int(qijl['angular_momentum']),
      int(qijl['first_index']) * nj + int(qijl['second_index'])
    ].set(jnp.array(qijl['values'][:gcut]) / r_g[:gcut]**2 / 4 / jnp.pi)

  assert r_g.shape[0] == n_rgd
  assert dr_g.shape[0] == n_rgd
  assert phi_jg.shape[1] == n_rgd
  assert phit_jg.shape[1] == n_rgd
  assert nc_g.shape[0] == n_rgd
  assert nct_g.shape[0] == n_rgd
  return r_g, dr_g, phi_jg, phit_jg, nc_g, nct_g, vbar_g, n_lqg, Delta_lq

def calc_qe(r_g, dr_g, phi_jg, phit_jg, nc_g, nct_g, vbar_g, n_lqg, Delta_lq):

  def calculate_T_Lqp():
    """copied from gpaw"""
    Lcut = (2 * lcut + 1)**2
    G_LLL = gaunt(lcut)[:, :, :Lcut]
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
    # Delta_lq_ = jnp.zeros((lmax + 1, nq))
    # for l in range(lmax + 1):
    #   Delta_lq_ = Delta_lq_.at[l].set(jnp.dot(n_qg - nt_qg, r_g**l * dr_g))
    # Delta_lq_ = Delta_lq_.reshape(-1)
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

  # check the shape of the arrays
  n_rgd = r_g.shape[0] # number of grid points
  nj = phi_jg.shape[0] # number of projectors radial functions
  n_j = np.array([0, 0, 1, 1]) # TODO: main quantum number, not sure how to calculate
  ni = nj + l_j.sum() * 2 # number of projectors
  nq = nj * (nj + 1) // 2 # number of radial functionpairs
  _np = ni * (ni + 1) // 2 # number of projector pairs
  proj_r = []
  proj_l = []
  proj_m = []
  i = 0
  for l in l_j:
    for m in range(-l, l + 1):
      proj_r.append(i)
      proj_l.append(l)
      proj_m.append(m)
    i += 1
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
  A -= jnp.sqrt(4 * jnp.pi) * Z * jnp.dot(r_g * dr_g, nc_g)
  # For QE PP files, we need to construct g_lg[0] - a smooth compensation charge
  # g_lg[0] should be a smooth, normalized function with monopole moment 1/sqrt(4π)
  # Common choice: use a Gaussian-like function or the shape of nct_g

  # Option 1: Use normalized smooth core density shape
  if jnp.sum(nct_g) > 1e-10:
      g0_unnorm = nct_g  # Use smooth core density shape
  else:
      # Option 2: Simple Gaussian if no core density
      sigma = r_g[gcut//4]  # Width ~ 1/4 of cutoff radius  
      g0_unnorm = jnp.exp(-r_g**2 / (2 * sigma**2))

  # Normalize so that ∫ g_lg[0] * r² dr = 1/sqrt(4π)
  g0_integral = jnp.sum(g0_unnorm * r_g**2 * dr_g)
  if g0_integral > 1e-10:
      g_lg = [g0_unnorm / (g0_integral * jnp.sqrt(4 * jnp.pi))]
  else:
      g_lg = [jnp.zeros_like(r_g)]  # Fallback if normalization fails

  mct_g = nct_g + Delta0 * g_lg[0]
  A -= 0.5 * integrate_radial_function(mct_g * poisson_rdl(mct_g, 0))
  # NOTE: THIS IS FOR TESTING, SHOULD BE CHANGED TO THE CORRECT FORMULA
  M = 0.5 * integrate_radial_function(nc_g * poisson_rdl(nc_g, 0))
  return B_ii, M, n_qg, nt_qg, Delta_pL, Delta0, gcut

  # NOTE: currently the following code is not tested for QE pp file, but do not delete
  MB = -integrate_radial_function(nct_g * vbar_g)

  AB_q = -integrate_radial_function(nt_qg * vbar_g)
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
        K = K.at[i, j].set(calc_kinetic_energy(phi_jg[i], phi_jg[j], l_j[i]))
        K = K.at[j, i].set(K[i, j])

  K_p = jnp.zeros((ni, ni))
  for i in range(ni):
    for j in range(i, ni):
      if proj_l[i] == proj_l[j] and proj_m[i] == proj_m[j]:
        K_p = K_p.at[i, j].set(K[proj_r[i], proj_r[j]])
        K_p = K_p.at[j, i].set(K_p[i, j])
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
                integrate_radial_function(n_qg * poisson_rdl0(nc_g.reshape(1, -1))))
    # 2nd term + 5th termin (46)
    A_q -= 0.5 * (integrate_radial_function(mct_g * poisson_rdl0(nt_qg)) +
                integrate_radial_function(nt_qg * poisson_rdl0(mct_g.reshape(1, -1))))
    # 3rd term in (46)
    A_q -= 4 * jnp.pi * Z * jnp.dot(n_qg, r_g * dr_g)
    # 4th term + 6th term in (46)
    A_q -= 0.5 * (integrate_radial_function(mct_g * poisson_rdl0(n_lqg[0])) +
                  integrate_radial_function(n_lqg[0] * poisson_rdl0(mct_g.reshape(1, -1))))
    # A_q -= 0.5 * (integrate_radial_function(mct_g * poisson_rdl0(g_lg[0])) +
    #               integrate_radial_function(g_lg[0] * poisson_rdl0(mct_g.reshape(1, -1)))) * \
    #     Delta_lq[0]
    # Save A_q for debugging
    global A_q_debug
    A_q_debug = A_q
    M_p = jnp.dot(A_q, T_Lqp[0])

    A_lqq = []
    for l in range(2 * lcut + 1):
      poisson_rdl_ = jax.vmap(partial(poisson_rdl, l=l))
      # 1st term in (47)
      A_qq = 0.5 * integrate_radial_function(n_qg[None] * poisson_rdl_(n_qg)[:, None])  
      # 2nd term in (47)
      A_qq -= 0.5 * integrate_radial_function(nt_qg[None] * poisson_rdl_(nt_qg)[:, None])  
      if l <= lmax:
        A_qq -= 0.5 * integrate_radial_function(poisson_rdl_(nt_qg)[None] * n_lqg[l][:, None])
        A_qq -= 0.5 * integrate_radial_function(nt_qg[None] * poisson_rdl_(n_lqg[l])[:, None])
        A_qq -= 0.5 * integrate_radial_function(n_lqg[l][None] * poisson_rdl_(n_lqg[l])[:, None])
      A_lqq.append(A_qq)

    M_pp = jnp.zeros((_np, _np))
    L = 0
    for l in range(2 * lcut + 1):
      for m in range(2 * l + 1):  # m?
        M_pp += jnp.dot(jnp.transpose(T_Lqp[L]), jnp.dot(A_lqq[l], T_Lqp[L]))
        L += 1
    return M_p, M_pp

  M_p, M_pp = calculate_coulomb_corrections()
  # breakpoint()

# # TODO: the xc correction seems to be very messy here
# xc_correction = get_xc_correction(rgd2, xc, gcut2, lcut)


# """Density matrix related quantities"""
# for i in range(n_projectors[atom]):
#   for j in range(n_projectors[atom]):
#     for l in range((lmax + 1)**2):
#       delta_aiiL[atom][i, j, l] = PP_MULTIPOLES[i, j, l]
# delta0_a[atom] = jnp.zeros(1)

# wave_grid_arr = pw.wave_grid(coefficient, vol)

# occupation = jnp.ones(
#   shape=coefficient.shape[:3]
# ) if occupation is None else occupation

# density_grid = wave_to_density(wave_grid_arr, occupation)
# density_grid_rec = wave_to_density_reciprocal(wave_grid_arr, occupation)

# def calculate_pseudo_density(wfs):
#   """similar to the wave to density func"""
#   return None

# def calculate_total_density(wfs):
#   return None

# def calculate_atomic_density_matrices(D_asp):
#   """need to calculate the inner product between the pseudo-wave
#   with the projectors"""
#   return None

# def calculate_multipole_moments():

#   return comp_charge, _Q_aL

# def correction():
#   correction = 0
#   for atom in atoms_list:
#     correction += jnp.einsum("ijkl, ij, kl ->", 
#       M_pp, D_aii[atom], D_aii[atom]
#     )
    
#   return correction

# # delta_aiiL = []
# # delta0_a = []
# # D_aii = []
# # D_aii.append(jnp.zeros((nj, nj)))
# # delta_aiiL.append(jnp.zeros((nj, nj, (lmax + 1)**2)))
# # delta0_a.append(jnp.zeros(1))

# def total_energy():

#   # for D_sii, P_ni in zip(D_asii.values(), P_ani.values()):
#   #   D_sii[self.spin] += jnp.einsum('ni, n, nj -> ij',
#   #     P_ni.conj(), occ_n, P_ni).real
#   for D_sii, P_ni in zip(D_asii.values(), P_ani.values()):
#     D_sii[self.spin] += jnp.einsum('ni, n, nj -> ij',
#       P_ni.conj(), occ_n, P_ni).real
    
#   for a, D_sii in wfs.D_asii.items():
#     Q_L = jnp.einsum('ij, ijL -> L',
#       D_sii[:wfs.ndensities].real, wfs.delta_aiiL[a])
#     Q_L[0] += wfs.delta0_a[a]

#   # calculate the kinetic energy
#   # valence kinetic energy
#   e_kinetic = kinetic(
#     g_vector_grid,
#     kpts,
#     coeff_grid,
#     occupation
#   )
#   # core kinetic energy
#   # TODO: implement core kinetic energy to match the AE total energy
#   ekin_c = 0

#   # calculate the exchange-correlation energy
#   exc_cv = exc_functional(
#     density_grid,
#     g_vector_grid,
#     vol,
#     xc,
#     kohn_sham
#   )
  
#   # core-valence Hartree energy
#   e_coulomb = hartree(density_grid_rec, g_vector_grid, vol, kohn_sham)

#   for atom in atoms_list:
#     e_kinetic += jnp.dot(K_p[atom], D_ij[atom]) + Kc[atom]
#     e_zero += MB[atom] + jnp.dot(MB_p[atom], D_p[atom])
#     e_coulomb += M[atom] + jnp.dot(
#       D_p[atom], (M_p[atom] + jnp.dot(M_pp[atom], D_p[atom]))
#     )
#     e_xc += calculate_paw_correction(self.setups[a], D_sp,
#                                                      dH_asp[a], a=a)

#   return e_kinetic + e_coulomb + e_zero + e_xc
