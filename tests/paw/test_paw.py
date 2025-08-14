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
  """Test the augmentation charge integration."""
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