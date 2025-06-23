from jrystal._src.energy import kinetic, hartree, exc_functional


g_vector_grid = None
vol = None
kpts = None
coeff_grid = None
occupation = None
kohn_sham = True

wave_grid_arr = pw.wave_grid(coefficient, vol)

occupation = jnp.ones(
  shape=coefficient.shape[:3]
) if occupation is None else occupation

density_grid = wave_to_density(wave_grid_arr, occupation)
density_grid_rec = wave_to_density_reciprocal(wave_grid_arr, occupation)

def calculate_pseudo_density(wfs):
  """similar to the wave to density func"""
  return None

def calculate_atomic_density_matrices(D_asp):
  """need to calculate the inner product between the pseudo-wave
  with the projectors"""
  return None

def calculate_multipole_moments():

  return comp_charge, _Q_aL

def total_energy():

  # calculate the kinetic energy
  # valence kinetic energy
  ekin_v = kinetic(
    g_vector_grid,
    kpts,
    coeff_grid,
    occupation
  ) + kinetic_nloc(
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