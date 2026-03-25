import time

import jax
import jax.numpy as jnp

# from jax.experimental.sparse.linalg import lobpcg_standard as jax_lobpcg
from jaxtyping import Array, Float

import jrystal as jr

# from batch_davidson import batch_davidson_matrix_free as davidson
# from batch_lobpcg_triton import batch_lobpcg_triton as lobpcg
from jrystal._src.linalg import batched_lobpcg as lobpcg
from jrystal.scf.diis import diis_init, diis_update

# from batch_lobpcg import batch_lobpcg_matrix_free as lobpcg

jax.config.update("jax_enable_x64", True)

E_CUT = 100
G_VEC_SIZE = [
  48,
] * 3

CONVERGENCE = 1e-8
NUM_BANDS = 40
PATH = "/home/aiops/litb/projects/jrystal/geometry/"

SCF_MAX_ITER = 100
# CRYSTAL_TYPE = "si8"
CRYSTAL_TYPE = "diamond"
K_MESH = [
  1,
] * 3
LOBPCG_MAX_ITER = 6
DIIS_MAX_HIST = 8
MIXING_BETA = 0.8
SMEARING = 0
FORCE_EQUAL_OCC_PER_K = False

crystal = jr.crystal.Crystal.create_from_file(f"{PATH}{CRYSTAL_TYPE}.xyz")

mask = jr.grid.spherical_mask(crystal.cell_vectors, G_VEC_SIZE, E_CUT)
print(f"mask ratio: {jnp.mean(mask):.4f}. num of G vectors {jnp.sum(mask)}")
g_vec = jr.grid.g_vectors(crystal.cell_vectors, G_VEC_SIZE)
# kpts = jnp.zeros([1, 3])
kpts, kpts_weights = jr.grid.k_vectors(
  crystal.cell_vectors, K_MESH, symmetry_reduction=True,
  scaled_positions=crystal.scaled_positions, charges=crystal.charges
)
key = jax.random.PRNGKey(123)
evals = jax.random.normal(key, [1, kpts.shape[0], NUM_BANDS])
evals = jnp.sort(evals, axis=-1, descending=False)

OCC_MAX = 2.0 if evals.shape[0] == 1 else 1.0
_occ_bands = float(crystal.num_electron) / OCC_MAX
NUM_OCC_BANDS = int(round(_occ_bands))
HAS_INTEGER_OCC_BANDS = abs(_occ_bands - NUM_OCC_BANDS) < 1e-10


def fixed_occ_per_k(evals):
  """Assign identical occupied-band count at every k-point (insulator mode)."""
  order = jnp.argsort(evals, axis=-1)
  rank = jnp.argsort(order, axis=-1)
  return jnp.where(rank < NUM_OCC_BANDS, OCC_MAX, 0.0).astype(evals.dtype)


def occ_fun(evals):
  if FORCE_EQUAL_OCC_PER_K and HAS_INTEGER_OCC_BANDS:
    return fixed_occ_per_k(evals)

  mu = jr.smearing.find_chemical_potential(
    evals,
    crystal.num_electron,
    smearing=SMEARING,
    k_weights=kpts_weights,
  )
  return jr.smearing.fermi_dirac(evals, mu, smearing=SMEARING)


occ_init = fixed_occ_per_k(evals)
print(occ_init)

key = jax.random.PRNGKey(111)
param_coeff = jr.pw.param_init(
  key, NUM_BANDS, num_kpts=kpts.shape[0], freq_mask=mask
)
coeff_init = param_coeff['w_re'] + 1.j * param_coeff['w_im']
coeff_init = jnp.linalg.qr(coeff_init)[0]  # [s k g band]


def expand(c):
  return jr.utils.expand_coefficient(c, mask)


def density(
  coeff_compact: Float[Array, "s k g band"],
  occ: Float[Array, "s k b"],
):
  coeff = expand(coeff_compact)
  return jr.pw.density_grid(coeff, crystal.vol, occ, k_weights=kpts_weights)


def kerker(g_vec, mask):
  eff_g = g_vec.at[mask].get()
  g2 = jnp.sum(eff_g**2, axis=-1)
  return g2 / (1 + g2)


precond = kerker(g_vec, mask)


def efun(coeff_compact, dens):
  coeff = expand(coeff_compact)
  etot = jr.hamiltonian.hamiltonian_matrix_trace(
    coeff,
    crystal.positions,
    crystal.charges,
    dens,
    crystal.vol,
    g_vec,
    kpts,
    kpts_weights=kpts_weights,
    xc="gga_x_pbe+gga_c_pbe",
    kohn_sham=True,
    keep_spin_axis=False,
  )
  return etot


def Hvp_fun(
  coeff_compact: Float[Array, "s b g band"],
  dens: Float[Array, "s x y z"],
) -> Float[Array, "s k g band"]:

  return jax.grad(efun)(coeff_compact.conj(), dens) / 2.


def lobpcg_matmul(c: Float[Array, "batch g band"], dens):
  return Hvp_fun(jnp.expand_dims(c, axis=0), dens)


@jax.jit
def update(coeff_compact, effective_dens):
  # jxc(vxc) currently fails under the nested jit+grad trace in this SCF path.
  # Keep this function in eager mode so SCF remains functional.
  s, k, g, b = coeff_compact.shape

  eigval, evec = lobpcg(   # pylint: disable=unbalanced-tuple-unpacking
    matmul=lambda c: lobpcg_matmul(c, effective_dens).reshape(s * k, g, -1),
    k=b,
    v0=coeff_compact.reshape(s * k, g, b),
    which="smallest",
    preconditioner=precond,
    maxit=LOBPCG_MAX_ITER,
    tol=1e-8,
  )  # eigval: [s*k, b], evec: [s*k, g, b]

  coeff_compact = evec.reshape(s, k, g, b)
  eigval = eigval.reshape(s, k, b)

  return coeff_compact, eigval


def check_convergence(new_coeff, old_coeff, new_loss, old_loss):
  coeff_diff = jnp.mean(jnp.abs(new_coeff - old_coeff))
  loss_diff = jnp.mean(jnp.abs(new_loss - old_loss))

  return coeff_diff, loss_diff


def weighted_band_energy(
  eigval: Float[Array, "s k b"],
  occ: Float[Array, "s k b"],
):
  return jnp.sum(eigval * occ * kpts_weights[None, :, None])


def mixing(new_coeff, old_coeff, mixing_factor=0.95):
  return new_coeff * mixing_factor + old_coeff * (1 - mixing_factor)


def scf(iter, convergence_tol):
  coeff_compact = coeff_init
  dens = density(coeff_compact, occ_init)
  etol = weighted_band_energy(evals, occ_init)
  occ = occ_init

  time_jit = 0
  time_total = 0

  diis_state = diis_init(
    max_hist=DIIS_MAX_HIST, density_shape=dens.shape, dtype=dens.dtype
  )

  for i in range(iter):
    print(f"SCF iteration {i+1} started.")
    if i == 0:
      start_time = time.time()
      coeff_compact_new, eigval_new = update(coeff_compact, dens)
      coeff_compact_new = coeff_compact_new.conj()
      end_time = time.time()
      time_jit += end_time - start_time
    else:
      start_time = time.time()
      coeff_compact_new, eigval_new = update(coeff_compact, dens)
      coeff_compact_new = coeff_compact_new.conj()
      end_time = time.time()
      time_total += end_time - start_time

    occ = occ_fun(eigval_new)
    print(jnp.sum(occ * kpts_weights[None, :, None]))
    dens_new = density(coeff_compact_new, occ)
    etol_new = weighted_band_energy(eigval_new, occ)
    dc, de = check_convergence(dens_new, dens, etol_new, etol)

    if dc < 1e-3 and de < convergence_tol:
      print(
        f"SCF converged in {i+1} iterations. "
        f"change of density: {dc: .6e}, change of energy: {de: .6e}"
      )
      break

    dens_error = dens_new - dens
    diis_state, dens_new = diis_update(diis_state, dens_new, dens_error, 1e-8)
    dens = mixing(dens_new, dens, MIXING_BETA)
    etol = etol_new
    coeff_compact = coeff_compact_new

    # print(eigval_new)
    print(f"SCF iteration {i+1} completed. Sum of eigenvalues: {etol: .6e}")
    print(
      f"change of coeffcient (avg.): {dc: .6e}, change of energy: {de: .6e}"
    )

  ewald_grid = jr.grid.translation_vectors(crystal.cell_vectors, 1e4)
  ewald = jr.ewald.ewald_coulomb_repulsion(
    crystal.positions, crystal.charges, g_vec, crystal.vol, 1e-1, ewald_grid
  )

  coeff = expand(coeff_compact_new)
  dens = density(coeff_compact_new, occ)

  # dens_rcpl = jr.pw.density_grid_reciprocal(coeff, crystal.vol, occ)
  dens_rcpl = jnp.fft.fftn(dens, axes=range(-3, 0))
  # print(jnp.sum(dens_rcpl) * crystal.vol / np.prod(g_vec.shape[:3]))
  # print(jnp.sum(dens) * crystal.vol / np.prod(g_vec.shape[:3]))

  # print("sum of eigenvalues:", jnp.sum(eigval_new))
  # print("efun(coeff_compact_new, dens):", efun(coeff_compact_new, dens))
  kin = jr.energy.kinetic(coeff, g_vec, kpts, kpts_weights, occ)
  har = jr.energy.hartree(dens_rcpl, g_vec, crystal.vol)
  ext = jr.energy.external(
    dens_rcpl, crystal.positions, crystal.charges, g_vec, crystal.vol
  )
  lda_x = jr.energy.xc_energy(dens, g_vec, crystal.vol, "lda_x")
  total_energy = kin + har + ext + lda_x + ewald

  print(f"Total energy:   \t {total_energy: .8f} Eh")
  print(f"Ewald energy:   \t {ewald: .6f} Eh")
  print(f"Kinetic energy: \t {kin: .6f} Eh")
  print(f"Hartree energy: \t {har: .6f} Eh")
  print(f"External energy: \t {ext: .6f} Eh")
  print(f"XC energy:      \t {lda_x: .6f} Eh")
  print(f"Time taken for JIT: {time_jit: .6f} seconds")
  print(f"Time taken for total: {time_total: .6f} seconds")

  print(eigval_new)
  print(occ)

  return coeff_compact, etol


if __name__ == "__main__":
  scf(SCF_MAX_ITER, CONVERGENCE)
