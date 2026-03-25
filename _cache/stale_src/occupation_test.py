import jax.numpy as jnp

from jrystal._src import occupation


def test_simplex_projector():
  occ_fn = occupation.get_occupation_fn(20)
  params = occupation.params_init(20, 10)
  occ = occ_fn(params)
  print(occ)
  print(jnp.sum(occ)/10)  # should be 20
  assert jnp.sum(occ)/10 == 20


if __name__ == "__main__":
  test_simplex_projector()
