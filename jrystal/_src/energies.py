"""Energy functions. """
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, Complex

from jrystal._src.pw import _complex_norm_square
from jrystal._src.grid import _get_ewald_lattice


def energy_hartree(
  ng: Complex[Array, '*nd'], 
  g_vec: Float[Array, '*nd d'], 
  vol: Float[Array, '']
) -> Float[Array, '']:
  """Hartree energy for plane wave orbitals on reciprocal space.
      E = 2\pi \sum_i \sum_k \sum_G \dfrac{|n(G)|^2}{|G|^2}
  Args:
      density_vec_ft (3D array): the FT of density. Shape: [N1, N2, N3]
      gvec (4D array): G-vector. Shape: [N1, N2, N3, 3]
      g_mask (3D array): G point mask, Shape: [N1, N2, N3]
      vol: scalar
      
  Return:
      Hartree energy: float.
  """
  
  g_vec_square = jnp.sum(g_vec**2, axis=-1)  # [N1, N2, N3]
  output = _complex_norm_square(ng)
  g_vec_square = g_vec_square.at[0, 0, 0].set(1e-16)
  output /= g_vec_square
  output = output.at[0, 0, 0].set(0)
  output = jnp.sum(output) / vol / 2 * 4 * jnp.pi
  return output


def energy_external(
  ng: Complex[Array, '*nd'], 
  atom_coord: Float[Array, 'ni nd'], 
  atom_charge: Float[Array, 'ni'], 
  g_vec: Float[Array, '*nd d'], 
  vol: Float[Array, ''],
) -> Float[Array, '']:
  """
    externel energy for plane wave orbitals.
        E = \sum_G \vert \sum_i s_i(G) v_i(G) \vert n(G)
    where
        S_i(G) = exp(jG\tau_i)
        v_i(G) = -4 pi q_i / \Vert G \Vert^2
    Args:
        density_vec_ft (6d array): the FT of density. Shape: [N1, N2, N3]
        atom_coord (2D array): coordinate of atoms in unit cell. Shape: [na, 3]
        atom_charge (1D array): charge of atoms in a unit cell. Shape: [na]
        nocc (3D array): occupation mask. Shape: [2, ni, nk]
        gvec (4D array): G-vector. Shape: [N1, N2, N3, 3]
    Return:
        External energy.Float
  """
  g_vec_square = jnp.sum(g_vec**2, axis=-1) 
  s_i = jnp.exp(1j * jnp.matmul(g_vec, atom_coord.transpose())) 
  v_i = atom_charge[None, None, None, :] / (g_vec_square[:, :, :, None])
  v_i = v_i.at[0, 0, 0, :].set(0)
  v_i *= 4 * jnp.pi 
  v_ext_G = jnp.sum(s_i * v_i, axis=-1)
  output = jnp.vdot(v_ext_G.flatten(), ng.flatten())
  output /= vol
  return -jnp.real(output)


def energy_kin(
  g_vec: Float[Array, '*nd d'],
  k_vec: Float[Array, 'nk d'], 
  coeff: Float[Array, '2 nk ni *nd'], 
  occ: Float[Array, '2 nk ni'],
) -> Float[Array, '']:
  """Kinetic energy
      E = 1/2 \sum_{G} |k + G|^2 c_{i,k,G}^2
  Args:
      gvec (4D array): G-vector. Shape: [N1, N2, N3, 3]
      kpts (2D array): k-points. Shape: [nk, 3]
      _params_w (6D array): pw coefficient. Shape: [2, ni, nk, N1, N2, N3]
      nocc (3D array): occupation mask. Shape: [2, ni, nk]
  Returns:
      scalar
  """
  _g = g_vec[None, None, :, :, :, :]  # shape [1, 1, N1, N2, N3, 3]
  _g = jnp.expand_dims(g_vec, '')
  _k = k_vec[None, :, None, None, None, :]  # shape [1, nk, 1, 1, 1, 3]

  e_kin = jnp.sum((_g + _k)**2, axis=-1)  # [1, nk, N1, N2, N3]
  e_kin = jnp.expand_dims(e_kin, axis=0)  # [1, 1, nk, N1, N2, N3]
  e_kin = jnp.sum(e_kin * _complex_norm_square(coeff), axis=(3, 4, 5))  

  return jnp.sum(e_kin * occ) / 2


def energy_lda(
  nr: Float[Array, '2 *nd'],
  vol: Float[Array, '']
):
  nr = jnp.sum(nr, axis=0)
  ngrid = jnp.prod(jnp.array(nr.shape))
  e_lda = jnp.sum(lda_x_raw(nr) * nr)
  return e_lda * vol / ngrid


def lda_x_raw(r0: Float[Array, "*nd"]):
  t3 = 3**(0.1e1 / 0.3e1)
  t4 = jnp.pi**(0.1e1 / 0.3e1)
  t8 = 2.220446049250313e-16**(0.1e1 / 0.3e1)
  t10 = jnp.where(0.1e1 <= 2.22044604925e-16, t8 * 2.22044604925e-16, 1)
  t11 = r0**(0.1e1 / 0.3e1)
  t15 = jnp.where(r0 / 0.2e1 <= 1e-15, 0, -0.3e1 / 0.8e1 * t3 / t4 * t10 * t11)
  res = 0.2e1 * 1. * t15
  return res


def energy_ewald(
  atom_coords,
  atom_charges,
  a,
  gvec,
  vol,
  ew_eta,
  ew_cut=None,
  lattice=None
):
  """Ewald summation
  
    Args:
      ew_cut (): _description_
      ew_eta (_type_): _description_
      atom_coords (ndarray): 2d array. shape: [na, 3]
      atom_charges (array): 1d array
      a (ndarray): crystal lattice vectors.
      gvec (_type_): _description_
      vol (_type_): _description_
  """
  if lattice is not None:
    translation = lattice
  elif ew_cut:
    translation = _get_ewald_lattice(a, ew_cut)

  tau = atom_coords[None, :, :] - atom_coords[:, None, :]  # [na, na, 3]
  # tau += jnp.expand_dims(jnp.eye(len(atom_charges))*1e12, -1)
  tau_t = tau[:, :, None, :] - translation[None, None, :, :]  # [na, na, nt, 3]
  tau_t_norm = jnp.sqrt(jnp.sum(tau_t**2, axis=-1)+1e-20)  # [na, na, nt]
  tau_t_norm = jnp.where(tau_t_norm <= 1e-9, 1e20, tau_t_norm)
  #  atom-atom part:
  ew_ovlp = jnp.sum(jax.scipy.special.erfc(ew_eta * tau_t_norm) / tau_t_norm,
                    axis=2)

  # the reciprocal space part:
  gvec_norm_sq = jnp.sum(gvec**2, axis=3)  # [N1, N2, N3]
  gvec_norm_sq = gvec_norm_sq.at[0, 0, 0].set(1e16)
  ew_rprcl = jnp.exp(
      -gvec_norm_sq / 4 / ew_eta**2) / gvec_norm_sq  # [N1, N2, N3]
  ew_rprcl = jnp.sum(
      ew_rprcl[:, :, :, None, None, None] *
      jnp.cos(gvec[:, :, :, None, None, :] * tau[None, None, None, :, :, :]),
      axis=-1)  # [N1, N2, N3, na, na]
  ew_rprcl = jnp.sum(ew_rprcl.at[0, 0, 0, :, :].set(0),
                     axis=(0, 1, 2))  # [na, na]
  ew_rprcl = ew_rprcl * 4 * jnp.pi / vol
  ew_aa = jnp.einsum('i,ij->j', atom_charges, ew_ovlp + ew_rprcl)
  ew_aa = jnp.dot(ew_aa, atom_charges) / 2

  # single atom part
  ew_a = -jnp.sum(atom_charges**2) * 2 * ew_eta / jnp.sqrt(jnp.pi) / 2
  ew_a -= jnp.sum(atom_charges)**2 * jnp.pi / ew_eta**2 / vol / 2

  return ew_aa + ew_a
