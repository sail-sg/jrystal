""" Calculate the Clebsch-Gordan coefficients. """
import numpy as np
from sympy.physics.wigner import clebsch_gordan, wigner_3j, gaunt

# TODO: implement clebsch_gordan with jax.


def batch_clebsch_gordan(
  j1: np.ndarray,
  j2: np.ndarray,
  j3: np.ndarray,
  m1: np.ndarray,
  m2: np.ndarray,
  m3: np.ndarray,
) -> np.ndarray:
  """
  Calculate the Clebsch-Gordan coefficients < j1 m1; j2 m2 | j3 m3 >.
  The input need to be an 1D array. Output is 6D array, where the dimension are
  the same as the dimensions of the input.

  Return:
    np.ndarray: The Clebsch-Gordan coefficients array.

  """

  shape = (len(j1), len(j2), len(j3), len(m1), len(m2), len(m3))
  cg_vals = np.zeros(shape, dtype=np.float64)

  indices = np.indices(shape).reshape(6, -1).T

  for idx in indices:
    i, j, k, l, m, n = idx
    _j1 = j1[i]
    _j2 = j2[j]
    _j3 = j3[k]
    _m1 = m1[l]
    _m2 = m2[m]
    _m3 = m3[n]

    if _m1 + _m2 != _m3:
      continue

    val = clebsch_gordan(_j1, _j2, _j3, _m1, _m2, _m3).doit()
    cg_vals[i, j, k, l, m, n] = val

  return cg_vals


def batch_wigner_3j(
  j1: np.ndarray,
  j2: np.ndarray,
  j3: np.ndarray,
  m1: np.ndarray,
  m2: np.ndarray,
  m3: np.ndarray,
) -> np.ndarray:
  """
  Calculate the Wigner 3j coefficients < j1 m1; j2 m2 | j3 m3 >.
  The input need to be an 1D array. Output is 6D array, where the dimension are
  the same as the dimensions of the input.

  Return:
    np.ndarray: The Wigner 3j coefficients array.

  """

  shape = (len(j1), len(j2), len(j3), len(m1), len(m2), len(m3))
  cg_vals = np.zeros(shape, dtype=np.float64)

  indices = np.indices(shape).reshape(6, -1).T

  for idx in indices:
    i, j, k, l, m, n = idx
    _j1 = j1[i]
    _j2 = j2[j]
    _j3 = j3[k]
    _m1 = m1[l]
    _m2 = m2[m]
    _m3 = m3[n]

    if _m1 + _m2 != _m3:
      continue

    val = wigner_3j(_j1, _j2, _j3, _m1, _m2, _m3).doit()
    cg_vals[i, j, k, l, m, n] = val

  return cg_vals


def batch_gaunt(
  j1: np.ndarray,
  j2: np.ndarray,
  j3: np.ndarray,
  m1: np.ndarray,
  m2: np.ndarray,
  m3: np.ndarray,
) -> np.ndarray:
  """
  Calculate the Gaunt coefficients < j1 m1; j2 m2 | j3 m3 >.
  The input need to be an 1D array. Output is 6D array, where the dimension are
  the same as the dimensions of the input.

  Return:
    np.ndarray: The Gaunt coefficients array.

  """

  shape = (len(j1), len(j2), len(j3), len(m1), len(m2), len(m3))
  cg_vals = np.zeros(shape, dtype=np.float64)

  indices = np.indices(shape).reshape(6, -1).T

  for idx in indices:
    i, j, k, l, m, n = idx
    _j1 = j1[i]
    _j2 = j2[j]
    _j3 = j3[k]
    _m1 = m1[l]
    _m2 = m2[m]
    _m3 = m3[n]

    if _m1 + _m2 != _m3:
      continue

    val = gaunt(_j1, _j2, _j3, _m1, _m2, _m3).doit()
    cg_vals[i, j, k, l, m, n] = val

  return cg_vals
