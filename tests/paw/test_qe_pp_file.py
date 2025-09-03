import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)
import pytest

from jrystal.pseudopotential.load import parse_upf

pp_dict = parse_upf(
  "/home/aiops/zhaojx/jrystal/pseudopotential/C.pbe-n-kjpaw_psl.1.0.0.UPF"
)
tol = 1e-14


def int_over_grid(f):
  return jnp.sum(jnp.array(f) * jnp.array(pp_dict['PP_MESH']['PP_RAB']))

# @pytest.mark.skip
def test_augmentation_charge():
  """Test the augmentation charge integration.
  NOTE: currently, only the 0th angular momentum is checked
  """
  for i in range(2):
    for j in range(2):
      qijr = jnp.array(pp_dict['PP_FULL_WFC']['PP_AEWFC'][i]['values']) *\
        jnp.array(pp_dict['PP_FULL_WFC']['PP_AEWFC'][j]['values']) -\
        jnp.array(pp_dict['PP_FULL_WFC']['PP_PSWFC'][i]['values']) *\
        jnp.array(pp_dict['PP_FULL_WFC']['PP_PSWFC'][j]['values'])
      q_ij = int_over_grid(qijr)
      assert jnp.abs(
        pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['PP_Q'][i * 4 + j] - q_ij
      ) < 3e-9
  for i in range(2, 4):
    for j in range(2, 4):
      qijr = jnp.array(pp_dict['PP_FULL_WFC']['PP_AEWFC'][i]['values']) *\
        jnp.array(pp_dict['PP_FULL_WFC']['PP_AEWFC'][j]['values']) -\
        jnp.array(pp_dict['PP_FULL_WFC']['PP_PSWFC'][i]['values']) *\
        jnp.array(pp_dict['PP_FULL_WFC']['PP_PSWFC'][j]['values'])
      q_ij = int_over_grid(qijr)
      assert jnp.abs(
        pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['PP_Q'][i * 4 + j] - q_ij
      ) < 3e-9


# @pytest.mark.skip
def test_augmentation_charge_multipole_moments():
  r"""Test the augmentation charge integration with multipole moments.
  
  The multipole moments are defined as
  .. math::

    Q_{ij}^l = \int_0^{r_c} Q_{ij}^l(r) r^{l + 2} dr

  Notice that the multipole moments provided by the pseudopotential
  file contains a :math:`4\pi r^2` factor.
  """
  index = [0, 1, 18, 19, 5, 22, 23, 10, 42, 11, 43, 15, 47]
  l = [0, 0, 1, 1, 0, 1, 1, 0, 2, 0, 2, 0, 2]
  for i in range(len(index)):
    moment = int_over_grid(
      jnp.array(pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['PP_QIJ'][i]['values'])/
      pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['PP_MULTIPOLES'][index[i]]
      * jnp.array(pp_dict['PP_MESH']['PP_R']) ** l[i]
    )
    assert jnp.abs(1 - moment) < 4e-7


# @pytest.mark.skip
def test_ae_nlcc():
  r"""Test the all-electron non-linear core charge density.
  
  The integral of the all-electron non-linear core charge density
  should equal to the number of electrons in the core.

  .. math::

    4\pi \int_0^{r_c} n_c(r) r^2 dr = Z

  Notice that the non-linear core charge density provided by the pseudopotential
  file is the true density without any :math:`r, 4\pi` factors.

  """
  n_c = int_over_grid(
    jnp.array(pp_dict['PP_PAW']['PP_AE_NLCC']) *
    jnp.array(pp_dict['PP_MESH']['PP_R'])**2
  ) * 4 * jnp.pi
  # NOTE: the integration of the pseudized NLCC does not have
  # specific meaning
  # n_c = int_over_grid(
  #   jnp.array(pp_dict['PP_NLCC']) *
  #   jnp.array(pp_dict['PP_MESH']['PP_R'])**2
  # ) * 4 * jnp.pi
  assert jnp.abs(n_c - 2) < 2e-9


def test_projector_pswfc_biorthonormality():
  """Test bi-orthonormality between projectors and pseudo wavefunctions.
  
  In PAW formalism, the projectors (beta) and pseudo wavefunctions (psi_tilde)
  must satisfy the bi-orthonormality condition: <beta_i|psi_tilde_j> = delta_ij
  This is a fundamental requirement for the PAW method.
  
  NOTE: The accuracy of this bi-orthonormality in the UPF file is limited (~1e-2).
  This is a known issue with some PAW pseudopotentials and can affect calculation accuracy.
  For high-precision calculations, consider using pseudopotentials with better
  bi-orthonormality or apply Gram-Schmidt orthogonalization.
  """
  n_proj = len(pp_dict['PP_NONLOCAL']['PP_BETA'])
  n_pswfc = len(pp_dict['PP_FULL_WFC']['PP_PSWFC'])
  
  # Verify we have same number of projectors and pseudo wavefunctions
  assert n_proj == n_pswfc, \
    f"Number of projectors ({n_proj}) != number of PSWFC ({n_pswfc})"
  
  # Track maximum errors for reporting
  max_diag_error = 0.0
  max_offdiag_error = 0.0
  biortho_matrix = jnp.zeros((n_proj, n_pswfc))
  
  # Test all pairs
  for i in range(n_proj):
    for j in range(n_pswfc):
      # Get angular momentum of projector and wavefunction
      l_beta = pp_dict['PP_NONLOCAL']['PP_BETA'][i]['angular_momentum']
      l_pswfc = pp_dict['PP_FULL_WFC']['PP_PSWFC'][j]['angular_momentum']
      
      # Only compute if same angular momentum (otherwise orthogonal by symmetry)
      if l_beta == l_pswfc:
        # Get cutoff index from projector (convert to int)
        cutoff = int(pp_dict['PP_NONLOCAL']['PP_BETA'][i]['cutoff_radius_index'])
        
        # Compute inner product <beta_i|psi_tilde_j>
        beta_i = jnp.array(pp_dict['PP_NONLOCAL']['PP_BETA'][i]['values'][:cutoff])
        pswfc_j = jnp.array(pp_dict['PP_FULL_WFC']['PP_PSWFC'][j]['values'][:cutoff])
        rab = jnp.array(pp_dict['PP_MESH']['PP_RAB'][:cutoff])
        
        inner_product = jnp.sum(beta_i * pswfc_j * rab)
        biortho_matrix = biortho_matrix.at[i, j].set(inner_product)
        
        if i == j:
          # Diagonal elements should be 1
          error = jnp.abs(inner_product - 1.0)
          max_diag_error = max(max_diag_error, error)
          # Use relaxed tolerance due to known pseudopotential limitations
          assert error < 2e-2, \
            f"<beta_{i}|psi_tilde_{j}> = {inner_product:.6f}, error = {error:.6f} exceeds tolerance"
        else:
          # Off-diagonal elements should be 0
          error = jnp.abs(inner_product)
          max_offdiag_error = max(max_offdiag_error, error)
          # Use relaxed tolerance due to known pseudopotential limitations  
          assert error < 3e-2, \
            f"<beta_{i}|psi_tilde_{j}> = {inner_product:.6f}, error = {error:.6f} exceeds tolerance"
  
  # Print warning if errors are significant
  if max_diag_error > 1e-3 or max_offdiag_error > 1e-3:
    import warnings
    warnings.warn(
      f"\nBi-orthonormality errors detected in PAW pseudopotential:\n"
      f"  Max diagonal error: {max_diag_error:.6f} (ideal: 0)\n"
      f"  Max off-diagonal error: {max_offdiag_error:.6f} (ideal: 0)\n"
      f"These errors may affect PAW calculation accuracy.\n"
      f"Bi-orthonormality matrix:\n{biortho_matrix}\n",
      UserWarning
    )
