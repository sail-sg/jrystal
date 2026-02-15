"""Unit tests for grid and k-point utilities using diamond cell."""
from __future__ import annotations

import numpy as np
from ase.io import read
from ase.dft.kpoints import monkhorst_pack

from jrystal import grid as jr_grid


def _build_vector_grid(basis: np.ndarray, grid_sizes, normalize: bool) -> np.ndarray:
  dim = len(grid_sizes)
  components = []
  for i in range(dim):
    freq = np.fft.fftfreq(
      grid_sizes[i], 1 if normalize else 1 / grid_sizes[i]
    )
    shape = tuple(
      grid_sizes[j] if j == i else 1 for j in range(dim)
    ) + (dim,)
    components.append(np.reshape(np.outer(freq, basis[i]), shape))
  return sum(components)


def test_grid_vectors_and_masks() -> None:
  atoms = read("geometry/diamond.xyz")
  cell = np.array(atoms.cell.array)
  grid_sizes = (4, 4, 4)
  k_grid_sizes = (2, 2, 2)
  cutoff_energy = 20.0

  g_jr = np.array(jr_grid.g_vectors(cell, grid_sizes))
  r_jr = np.array(jr_grid.r_vectors(cell, grid_sizes))
  k_jr = np.array(jr_grid.k_vectors(cell, k_grid_sizes))

  b = 2 * np.pi * np.linalg.inv(cell).T
  g_ref = _build_vector_grid(b, grid_sizes, normalize=False)
  r_ref = _build_vector_grid(cell, grid_sizes, normalize=True)
  k_ref = monkhorst_pack(k_grid_sizes) @ b

  assert g_jr.shape == g_ref.shape == (4, 4, 4, 3)
  assert r_jr.shape == r_ref.shape == (4, 4, 4, 3)
  assert k_jr.shape == k_ref.shape == (8, 3)

  assert np.allclose(g_jr, g_ref, rtol=0, atol=1e-12)
  assert np.allclose(r_jr, r_ref, rtol=0, atol=1e-12)
  assert np.allclose(k_jr, k_ref, rtol=0, atol=1e-12)

  # Check FFT ordering on a few indices for G-vectors.
  assert np.allclose(g_jr[0, 0, 0], np.zeros(3), atol=1e-12)
  assert np.allclose(g_jr[1, 0, 0], b[0], atol=1e-12)
  assert np.allclose(g_jr[2, 0, 0], -2 * b[0], atol=1e-12)

  spherical_mask = np.array(
    jr_grid.spherical_mask(cell, grid_sizes, cutoff_energy)
  )
  g_norm = np.linalg.norm(g_ref, axis=-1)
  spherical_ref = g_norm**2 <= cutoff_energy * 2
  assert spherical_mask.shape == spherical_ref.shape
  assert np.array_equal(spherical_mask, spherical_ref)

  cubic_mask = np.array(jr_grid.cubic_mask(grid_sizes))
  masks = []
  for size in grid_sizes:
    g_max = (size - 1) // 2
    lower_bound = -g_max // 2
    upper_bound = g_max // 2
    m = np.ones((size,), dtype=bool)
    m[upper_bound + 1:lower_bound] = False
    masks.append(m)
  cubic_ref = np.einsum("i,j,k->ijk", *masks)
  assert np.array_equal(cubic_mask, cubic_ref)
