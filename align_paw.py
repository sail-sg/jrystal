import jax.numpy as jnp

from jrystal.pseudopotential.load import parse_upf
from jrystal._src.energy import kinetic, hartree, exc_functional

pp_dict = parse_upf("/home/aiops/zhaojx/jrystal/pseudopotential/C.pbe-n-kjpaw_psl.1.0.0.UPF")
lmax = 4   # NOTE: not sure if the n_L is uniform for all atoms
n_projectors = []
delta_aiiL = []
delta0_a = []
D_aii = []
n_rgd = 1000
r_g = pp_dict['PP_MESH']['PP_R']
dr_g = pp_dict['PP_MESH']['PP_RAB']

nj = n_projectors[atom]
psi = jnp.zeros((n_rgd, nj))
psit = jnp.zeros((n_rgd, nj))
nc_g = jnp.zeros(n_rgd)
nct_g = jnp.zeros(n_rgd)
assert r_g.shape[0] == n_rgd
assert psi.shape[0] == n_rgd
assert psit.shape[0] == n_rgd
assert nc_g.shape[0] == n_rgd
assert nct_g.shape[0] == n_rgd
D_aii.append(jnp.zeros((nj, nj)))
delta_aiiL.append(jnp.zeros((nj, nj, (lmax + 1)**2)))
delta0_a.append(jnp.zeros(1))

"""initialization of delta_aiiL & delta0_a, which does not depend on the 
pseudized wave function"""
# nq = nj * (nj + 1) // 2
# Delta_lq = np.zeros((lmax + 1, nq))
# for l in range(lmax + 1):
#   Delta_lq[l] = np.dot(n_qg - nt_qg, r_g**(2 + l) * dr_g)
# Lmax = (lmax + 1)**2
# Delta_pL = np.zeros((_np, Lmax))
# for l in range(lmax + 1):
#     L = l**2
#     for m in range(2 * l + 1):
#         delta_p = np.dot(Delta_lq[l], T_Lqp[L + m])
#         Delta_pL[:, L + m] = delta_p
# Delta0 = np.dot(self.local_corr.nc_g - self.local_corr.nct_g,
#                 r_g**2 * dr_g) - self.Z / sqrt(4 * pi)
n_qg = (psi[..., None] * psi[:, None]).reshape(-1, nj**2)
nt_qg = (psit[..., None] * psit[:, None]).reshape(-1, nj**2)
nq = nj * (nj + 1) // 2
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
Delta0 = jnp.dot(nc_g - nct_g, r_g**2 * dr_g) - Z / sqrt(4 * pi)

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
  phi1: jnp.ndarray,
  phi2: jnp.ndarray,
  phi3: jnp.ndarray,
  phi4: jnp.ndarray,
  r: jnp.ndarray,
  dr: jnp.ndarray,
):
  
  Lcut = 25
  result = jnp.zeros((Lcut))
  for l in range(Lcut):
    result[l] = jnp.sum(phi1 * phi2 * poisson_rdl(phi3 * phi4, r, dr, l))

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