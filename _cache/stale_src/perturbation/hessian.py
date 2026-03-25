"""
Hessian matrix of the total energy functional, which respect to the plane wave
coefficients.

.. math::
  H_{i G j G'} = \dfrac{\partial^2 E_{\text{total}}}{\partial c_{i k G} \partial c_{j k G'}

"""
from .._src.energy import total_energy, nuclear_repulsion
from .._src.hessian import complex_hessian


def _get_total_energy_all_electrons(
  position,
  charge,
  cell_vectors,
  g_vector_grid,
  kpts,
  vol,
  occupation,
  kohn_sham,
  xc,
):

  def f(coeff):
    e_tot = total_energy(
      coeff, position, charge, g_vector_grid, kpts, vol, occupation, kohn_sham, xc
    )
    e_nuc = nuclear_repulsion(position, charge, cell_vectors, g_vector_grid, vol)
    return e_tot
  return f


def hessian(coeff, total_energy_func):
  return