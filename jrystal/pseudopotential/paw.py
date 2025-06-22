from jrystal._src.energy import kinetic


g_vector_grid = None
kpts = None
coeff_grid = None
occupation = None

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
  eh_cv = 