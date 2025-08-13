import jax.numpy as jnp
import numpy as np

from gpaw.gaunt import gaunt
from gpaw.new import zips
from jrystal.pseudopotential.load import parse_upf
from jrystal._src.energy import kinetic, hartree, exc_functional

def calculate_T_Lqp():
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

pp_dict = parse_upf("/home/aiops/zhaojx/jrystal/pseudopotential/C.pbe-n-kjpaw_psl.1.0.0.UPF")
Z = 6
# Z = int(pp_dict["PP_HEADER"]["Z_valence"])
r_g = jnp.array(pp_dict['PP_MESH']['PP_R'])
dr_g = jnp.array(pp_dict['PP_MESH']['PP_RAB'])
lmax = int(pp_dict["PP_HEADER"]["l_max"])
l_j = np.array([int(proj['angular_momentum']) for proj in pp_dict['PP_NONLOCAL']['PP_BETA']])
lcut = max(l_j)
pt_jg = jnp.array([proj['values'] for proj in pp_dict['PP_NONLOCAL']['PP_BETA']])
phi_jg = jnp.array([phi['values'] for phi in pp_dict['PP_FULL_WFC']['PP_AEWFC']])
phit_jg = jnp.array([phi['values'] for phi in pp_dict['PP_FULL_WFC']['PP_PSWFC']])
nc_g = jnp.array(pp_dict['PP_PAW']['PP_AE_NLCC'])
nct_g = jnp.array(pp_dict['PP_NLCC'])

n_rgd = r_g.shape[0]
nj = phi_jg.shape[0]
n_j = np.array([0, 0, 1, 1])
ni = nj + l_j.sum() * 2
nq = nj * (nj + 1) // 2
_np = ni * (ni + 1) // 2
T_Lqp = calculate_T_Lqp()
assert r_g.shape[0] == n_rgd
assert dr_g.shape[0] == n_rgd
assert phi_jg.shape[1] == n_rgd
assert phit_jg.shape[1] == n_rgd
# assert nc_g.shape[0] == n_rgd
# assert nct_g.shape[0] == n_rgd

# n_projectors = []
# delta_aiiL = []
# delta0_a = []
# D_aii = []
# nj = n_projectors[atom]
# nc_g = jnp.zeros(n_rgd)
# nct_g = jnp.zeros(n_rgd)
# D_aii.append(jnp.zeros((nj, nj)))
# delta_aiiL.append(jnp.zeros((nj, nj, (lmax + 1)**2)))
# delta0_a.append(jnp.zeros(1))

def get_compensation_charges():

  index = jnp.triu_indices(nj)
  n_qg = (phi_jg[:, None, :] * phi_jg[None])[index]
  nt_qg = (phit_jg[:, None, :] * phit_jg[None])[index]

  Delta_lq = jnp.zeros((lmax + 1, nq))
  for l in range(lmax + 1):
    Delta_lq[l] = jnp.dot(n_qg - nt_qg, r_g**(2 + l) * dr_g)

  Lmax = (lmax + 1)**2
  Delta_pL = jnp.zeros((_np, Lmax))
  for l in range(lmax + 1):
    L = l**2
    for m in range(2 * l + 1):
      delta_p = jnp.dot(Delta_lq[l], T_Lqp[L + m])
      Delta_pL[:, L + m] = delta_p

  Delta0 = jnp.dot(nc_g - nct_g, r_g**2 * dr_g) - Z / jnp.sqrt(4 * jnp.pi)
  return (n_qg, nt_qg, Delta_lq, Lmax, Delta_pL, Delta0)

  # g_lg = self.data.create_compensation_charge_functions(lmax)
  # gcut_q = jnp.zeros(nq, dtype=int)
  # N0_q = jnp.zeros(nq)
  # q = 0  # q: common index for j1, j2
  # for j1 in range(nj):
  #   for j2 in range(j1, nj):
  #     gcut = rgd.ceil(min(rcut_j[j1], rcut_j[j2]))
  #     N0_q[q] = sum(n_qg[q, :gcut] * r_g[:gcut]**2 * dr_g[:gcut])
  #     gcut_q[q] = gcut

  #     q += 1

  # self.gcut_q = gcut_q
  # self.N0_q = N0_q

  # # Electron density inside augmentation sphere.  Used for estimating
  # # atomic magnetic moment:
  # N0_p = N0_q @ T_Lqp[0] * sqrt(4 * pi)

  # return (g_lg[:, :gcut2].copy(), n_qg, nt_qg,
  #         Delta_lq, Lmax, Delta_pL, Delta0, N0_p)

breakpoint()
n_qg, nt_qg, Delta_lq, Lmax, Delta_pL, Delta0 = get_compensation_charges()

"""Projectors overlap matrix + overlap correction
I am not sure whether we should implement the overlap matrix for projectors
"""
def calculate_projector_overlaps(pt_jg):
  """Compute projector function overlaps B_ii = <pt_i | pt_i>."""
  nj = len(pt_jg)
  B_jj = jnp.zeros((nj, nj))
  for j1, pt1_g in enumerate(pt_jg):
      for j2, pt2_g in enumerate(pt_jg):
          B_jj[j1, j2] = rgd.integrate(pt1_g * pt2_g) / (4 * jnp.pi)
  B_ii = jnp.zeros((ni, ni))
  i1 = 0
  for j1, l1 in enumerate(l_j):
      for m1 in range(2 * l1 + 1):
          i2 = 0
          for j2, l2 in enumerate(l_j):
              for m2 in range(2 * l2 + 1):
                  if l1 == l2 and m1 == m2:
                      B_ii[i1, i2] = B_jj[j1, j2]
                  i2 += 1
          i1 += 1
  return B_ii
B_ii = calculate_projector_overlaps(pt_jg)

for i in range(n_projectors[atom]):
  for j in range(n_projectors[atom]):
    for l in range((lmax + 1)**2):
      delta_aiiL[atom][i, j, l] = PP_MULTIPOLES[i, j, l]
delta0_a[atom] = jnp.zeros(1)

wave_grid_arr = pw.wave_grid(coefficient, vol)

occupation = jnp.ones(
  shape=coefficient.shape[:3]
) if occupation is None else occupation

density_grid = wave_to_density(wave_grid_arr, occupation)
density_grid_rec = wave_to_density_reciprocal(wave_grid_arr, occupation)

def calculate_pseudo_density(wfs):
  """similar to the wave to density func"""
  return None

def calculate_total_density(wfs):
  return None

def calculate_atomic_density_matrices(D_asp):
  """need to calculate the inner product between the pseudo-wave
  with the projectors"""
  return None

def calculate_multipole_moments():

  return comp_charge, _Q_aL

def poisson_rdl(
  g_L: jnp.ndarray,
  r: jnp.ndarray,
  dr: jnp.ndarray,
  l: int,
):
  r"""Solve the Poisson equation over radial grid

  .. math::

  .. warning::

  Args:
    g_L (Real[Array, "spin kpts band x y z"]): radial function.
    r (Real[Array, "spin kpts band x y z"]): radial grid.
    dr (Real[Array, "spin kpts band x y z"]): radial grid spacing.
    l (int): angular momentum quantum number.

  Returns:
    (Real[Array, "spin kpts band x y z"]): radial function.
  """

  return jnp.cumsum(r**l * g_L * dr) / r**l

def four_center_integral(
  n1: jnp.ndarray,
  n2: jnp.ndarray,
  r: jnp.ndarray,
  dr: jnp.ndarray,
):
  
  Lcut = 25
  result = jnp.zeros((Lcut))
  for l in range(Lcut):
    result[l] = jnp.sum(n1 * poisson_rdl(n2, r, dr, l))

  M_pp = jnp.zeros((91, 91))
  

def correction():
  correction = 0
  for atom in atoms_list:
    correction += jnp.einsum("ijkl, ij, kl ->", 
      M_pp, D_aii[atom], D_aii[atom]
    )
    
  return correction

def total_energy():

  # for D_sii, P_ni in zip(D_asii.values(), P_ani.values()):
  #   D_sii[self.spin] += jnp.einsum('ni, n, nj -> ij',
  #     P_ni.conj(), occ_n, P_ni).real
  for D_sii, P_ni in zip(D_asii.values(), P_ani.values()):
    D_sii[self.spin] += jnp.einsum('ni, n, nj -> ij',
      P_ni.conj(), occ_n, P_ni).real
    
  for a, D_sii in wfs.D_asii.items():
    Q_L = jnp.einsum('ij, ijL -> L',
      D_sii[:wfs.ndensities].real, wfs.delta_aiiL[a])
    Q_L[0] += wfs.delta0_a[a]

  # calculate the kinetic energy
  # valence kinetic energy
  e_kinetic = kinetic(
    g_vector_grid,
    kpts,
    coeff_grid,
    occupation
  )
  # core kinetic energy
  # TODO: implement core kinetic energy to match the AE total energy
  ekin_c = 0

  # calculate the exchange-correlation energy
  exc_cv = exc_functional(
    density_grid,
    g_vector_grid,
    vol,
    xc,
    kohn_sham
  )
  
   
  # core-valence Hartree energy
  e_coulomb = hartree(density_grid_rec, g_vector_grid, vol, kohn_sham)

  for atom in atoms_list:
    e_kinetic += jnp.dot(K_p[atom], D_ij[atom]) + Kc[atom]
    e_zero += MB[atom] + jnp.dot(MB_p[atom], D_p[atom])
    e_coulomb += M[atom] + jnp.dot(
      D_p[atom], (M_p[atom] + jnp.dot(M_pp[atom], D_p[atom]))
    )
    e_xc += calculate_paw_correction(self.setups[a], D_sp,
                                                     dH_asp[a], a=a)

  return e_kinetic + e_coulomb + e_zero + e_xc