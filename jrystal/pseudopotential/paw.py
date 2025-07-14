import jax.numpy as jnp

from jrystal._src.energy import kinetic, hartree, exc_functional


# parameters for the pseudized wave functions
g_vector_grid = None
vol = None
kpts = None
coeff_grid = None
occupation = None
kohn_sham = True

"""parameters for the augmented wave functions



D_aii (list[Array]): list of ndarray with shape (n_projectors, n_projectors) for each atom

"""
lmax = 4   # NOTE: not sure if the n_L is uniform for all atoms
atoms_list = []
n_projectors = []
delta_aiiL = []
delta0_a = []
D_aii = []
for atom in atoms_list:
  D_aii.append(jnp.zeros((n_projectors[atom], n_projectors[atom])))
  delta_aiiL.append(
    jnp.zeros(
      (n_projectors[atom], n_projectors[atom], (lmax + 1)**2)
    )
  )
  delta0_a.append(jnp.zeros(1))

  # initialize the overlap matrix for the projectors
  # def calculate_projector_overlaps(self, pt_jg):
  #   """Compute projector function overlaps B_ii = <pt_i | pt_i>."""
  #   nj = len(pt_jg)
  #   B_jj = np.zeros((nj, nj))
  #   for j1, pt1_g in enumerate(pt_jg):
  #       for j2, pt2_g in enumerate(pt_jg):
  #           B_jj[j1, j2] = self.rgd.integrate(pt1_g * pt2_g) / (4 * pi)
  #   B_ii = np.zeros((self.ni, self.ni))
  #   i1 = 0
  #   for j1, l1 in enumerate(self.l_j):
  #       for m1 in range(2 * l1 + 1):
  #           i2 = 0
  #           for j2, l2 in enumerate(self.l_j):
  #               for m2 in range(2 * l2 + 1):
  #                   if l1 == l2 and m1 == m2:
  #                       B_ii[i1, i2] = B_jj[j1, j2]
  #                   i2 += 1
  #           i1 += 1
  #   return B_ii

  # initialization of delta_aiiL & delta0_a, which does not depend on the 
  # pseudized wave function
  # GPAW code:
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
  ekin_v = kinetic(
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
  eh_cv = hartree(density_grid_rec, g_vector_grid, vol, kohn_sham)

  e_correction = correction()