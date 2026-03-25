from jrystal.pseudopotential.clebsch_gordan import batch_clebsch_gordan
from sympy.physics.wigner import clebsch_gordan as sympy_clebsch_gordan
import numpy as np


def test_clebsch_gordan():
  j1 = np.array([0, 1, 2, 3])
  j2 = np.array([0, 1, 2, 3])
  j3 = np.array([0, 1, 2, 3, 4, 5, 6])
  m1 = np.array([-3, -2, -1, 0, 1, 2, 3])
  m2 = np.array([-3, -2, -1, 0, 1, 2, 3])
  m3 = np.array([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])

  cg = batch_clebsch_gordan(j1, j2, j3, m1, m2, m3)

  sympy_cg = np.zeros_like(cg)

  for _j1 in range(len(j1)):
    for _j2 in range(len(j2)):
      for _j3 in range(len(j3)):
        for _m1 in range(len(m1)):
          for _m2 in range(len(m2)):
            for _m3 in range(len(m3)):
              sympy_cg[_j1, _j2, _j3, _m1, _m2, _m3] = sympy_clebsch_gordan(
                j1[_j1], j2[_j2], j3[_j3], m1[_m1], m2[_m2], m3[_m3]
              ).evalf()
  breakpoint()
  assert np.allclose(cg, sympy_cg)


if __name__ == "__main__":
  test_clebsch_gordan()
