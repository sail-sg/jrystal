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
      fr = jnp.array(pp_dict['PP_FULL_WFC']['PP_AEWFC'][i]['values']) *\
        jnp.array(pp_dict['PP_FULL_WFC']['PP_AEWFC'][j]['values']) -\
        jnp.array(pp_dict['PP_FULL_WFC']['PP_PSWFC'][i]['values']) *\
        jnp.array(pp_dict['PP_FULL_WFC']['PP_PSWFC'][j]['values'])
      q_ij = int_over_grid(fr)
      assert jnp.abs(
        pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['PP_Q'][i * 4 + j] - q_ij
      ) < 3e-9
  for i in range(2, 4):
    for j in range(2, 4):
      fr = jnp.array(pp_dict['PP_FULL_WFC']['PP_AEWFC'][i]['values']) *\
        jnp.array(pp_dict['PP_FULL_WFC']['PP_AEWFC'][j]['values']) -\
        jnp.array(pp_dict['PP_FULL_WFC']['PP_PSWFC'][i]['values']) *\
        jnp.array(pp_dict['PP_FULL_WFC']['PP_PSWFC'][j]['values'])
      q_ij = int_over_grid(fr)
      assert jnp.abs(
        pp_dict['PP_NONLOCAL']['PP_AUGMENTATION']['PP_Q'][i * 4 + j] - q_ij
      ) < 3e-9


def test_augmentation_charge_multipole_moments():
  """Test the augmentation charge integration with multipole moments."""
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
  """Test the all-electron non-local core charge density."""
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