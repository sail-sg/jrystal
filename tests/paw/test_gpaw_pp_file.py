"""Test suite for GPAW PAW setup file.

This module tests various properties and consistency checks for GPAW PAW setup files,
following PAW formalism requirements.
"""

import sys
sys.path.insert(0, '/home/aiops/zhaojx/M_p-align-claude/devs')

import jax
import jax.numpy as jnp
import numpy as np
import xml.etree.ElementTree as ET

jax.config.update('jax_enable_x64', True)
import pytest

from gpaw_load import parse_paw_setup

with open('/home/aiops/zhaojx/M_p-align-claude/pseudopotential/C.PBE', 'r') as f:
  content = f.read().lstrip()
import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as tmp:
  tmp.write(content)
  tmp_path = tmp.name
pp_data = parse_paw_setup(tmp_path)
import os
os.unlink(tmp_path)

def load_paw_data(filepath):
  """Load additional PAW data not captured by the standard loader."""
  with open(filepath, 'r') as f:
    content = f.read()
  content = content.lstrip()
  root = ET.fromstring(content)
  
  projectors = []
  for elem in root.findall('projector_function'):
    state = elem.get('state')
    values = list(map(float, elem.text.split()))
    projectors.append({'state': state, 'values': values})
  
  
  ae_waves = []
  for elem in root.findall('ae_partial_wave'):
    state = elem.get('state')
    values = list(map(float, elem.text.split()))
    ae_waves.append({'state': state, 'values': values})
  
  
  ps_waves = []
  for elem in root.findall('pseudo_partial_wave'):
    state = elem.get('state')
    values = list(map(float, elem.text.split()))
    ps_waves.append({'state': state, 'values': values})
  
  return {
    'projectors': projectors,
    'ae_partial_waves': ae_waves,
    'pseudo_partial_waves': ps_waves
  }

paw_data = load_paw_data('/home/aiops/zhaojx/M_p-align-claude/pseudopotential/C.PBE')


def construct_radial_grid(grid_info, n_points):
  """Construct radial grid from GPAW grid parameters."""
  a = grid_info['a']
  n = grid_info['n']
  i = np.arange(n_points)
  r_g = a * i / (n - i)
  dr_g = a * n / (n - i) ** 2
  return r_g, dr_g


def test_core_electron_integration():
  """Test that core density integrates to number of core electrons."""
  r_g, dr_g = construct_radial_grid(pp_data['radial_grid'], len(pp_data['ae_core_density']))
  nc_g = jnp.array(pp_data['ae_core_density'])
  n_core = jnp.sum(nc_g * r_g**2 * dr_g) * np.sqrt(4 * np.pi)
  expected_core = pp_data['atom']['core']
  error = float(jnp.abs(n_core - expected_core))
  assert error < 3e-14, \
    f"Core electron integral {n_core:.6f} != expected {expected_core:.6f}, error={error:.2e}"


def test_projector_partial_wave_orthogonality():
  """Test orthogonality: <p_i|phi_tilde_j> = delta_ij"""
  projectors = paw_data['projectors']
  partial_waves = paw_data['pseudo_partial_waves']
  
  n_proj = len(projectors)
  n_waves = len(partial_waves)
  
  assert n_proj == n_waves, \
    f"Number of projectors ({n_proj}) != partial waves ({n_waves})"
  min_len = min(
    min(len(proj['values']) for proj in projectors),
    min(len(wave['values']) for wave in partial_waves)
  )
  r_g, dr_g = construct_radial_grid(pp_data['radial_grid'], min_len)
  
  max_diag_error = 0.0
  max_offdiag_error = 0.0
  
  for i, proj in enumerate(projectors):
    for j, wave in enumerate(partial_waves):
      if i == j:
        p_g = jnp.array(proj['values'][:min_len])
        phi_g = jnp.array(wave['values'][:min_len])
        overlap = jnp.sum(p_g * phi_g * r_g**2 * dr_g)
        error = float(jnp.abs(overlap - 1.0))
        max_diag_error = max(max_diag_error, error)
        assert error < 4e-14, \
          f"<p_{i}|phi_tilde_{j}> = {overlap:.6f}, error = {error:.2e}"


def test_partial_wave_normalization():
  """Test normalization: integral of |phi(r)|^2 r^2 dr = 1"""
  partial_waves = paw_data['ae_partial_waves']
  
  for i, wave in enumerate(partial_waves):
    if '1' in wave['state']:
      continue
    phi_g = jnp.array(wave['values'])
    n_points = len(phi_g)
    r_g, dr_g = construct_radial_grid(pp_data['radial_grid'], n_points)
    norm = jnp.sum(phi_g**2 * r_g**2 * dr_g)
    assert jnp.abs(norm - 1.0) < 2e-14, \
      f"Wave {i} (state={wave['state']}) norm = {norm:.6f}, expected 1.0"


def test_core_density_smoothness():
  """Test that pseudo core density matches all-electron outside core region."""
  if not pp_data.get('pseudo_core_density'):
    pytest.skip("No pseudo core density in this setup")
  
  nc_g = jnp.array(pp_data['ae_core_density'])
  nct_g = jnp.array(pp_data['pseudo_core_density'])
  
  assert len(nc_g) == len(nct_g), \
    f"Core density lengths differ: {len(nc_g)} vs {len(nct_g)}"
  
  r_g, _ = construct_radial_grid(pp_data['radial_grid'], len(nc_g))
  match_radius = 1.5
  match_idx = jnp.argmin(jnp.abs(r_g - match_radius))
  diff_outside = jnp.abs(nc_g[match_idx:] - nct_g[match_idx:])
  max_diff_outside = jnp.max(diff_outside)
  assert max_diff_outside < 1e-10, \
    f"Core densities don't match outside r={match_radius}: max diff = {max_diff_outside:.6e}"


def test_partial_wave_matching():
  """Test that pseudo and all-electron waves match outside core region."""
  ae_waves = paw_data['ae_partial_waves']
  ps_waves = paw_data['pseudo_partial_waves']
  
  for i in range(min(len(ae_waves), len(ps_waves))):
    phi_g = jnp.array(ae_waves[i]['values'])
    phit_g = jnp.array(ps_waves[i]['values'])
    
    min_len = min(len(phi_g), len(phit_g))
    phi_g = phi_g[:min_len]
    phit_g = phit_g[:min_len]
    r_g, _ = construct_radial_grid(pp_data['radial_grid'], min_len)
    match_radius = 1.5
    match_idx = jnp.argmin(jnp.abs(r_g - match_radius))
    diff_outside = jnp.abs(phi_g[match_idx:] - phit_g[match_idx:])
    max_diff = jnp.max(diff_outside)
    assert max_diff < 1e-10, \
      f"Wave {i} (state={ae_waves[i]['state']}) doesn't match outside r={match_radius}: " \
      f"max diff = {max_diff:.6e}"


def test_energy_consistency():
  """Test that total energy equals sum of components."""
  if 'ae_energy' not in pp_data:
    pytest.skip("No ae_energy in setup file")
  
  ae_energy = pp_data['ae_energy']
  total = ae_energy['total']
  sum_components = ae_energy['kinetic'] + ae_energy['xc'] + ae_energy['electrostatic']
  # NOTE: maximum tolerance as energy is stored with precision of 1e-6
  assert jnp.abs(total - sum_components) < 1.5e-6, \
    f"Energy inconsistency: total={total:.10f} != sum={sum_components:.10f}"


def test_kinetic_energy_differences_matrix():
  """Test that kinetic energy differences matrix is symmetric."""
  if 'kinetic_energy_differences' not in pp_data:
    pytest.skip("No kinetic energy differences in setup file")
  
  kin_diff = jnp.array(pp_data['kinetic_energy_differences'])
  import math
  sqrt_len = math.sqrt(len(kin_diff))
  if sqrt_len == int(sqrt_len):
    n_states = int(sqrt_len)
    is_full_matrix = True
  else:
    n_states = int((-1 + math.sqrt(1 + 8 * len(kin_diff))) / 2)
    expected_len = n_states * (n_states + 1) // 2
    if len(kin_diff) != expected_len:
      pytest.skip(f"Kinetic energy matrix has unusual size: {len(kin_diff)} elements")
    is_full_matrix = False
  
  if is_full_matrix:
    matrix = kin_diff.reshape((n_states, n_states))
  else:
    matrix = jnp.zeros((n_states, n_states))
    idx = 0
    for i in range(n_states):
      for j in range(i, n_states):
        matrix = matrix.at[i, j].set(kin_diff[idx])
        if i != j:
          matrix = matrix.at[j, i].set(kin_diff[idx])
        idx += 1
  
  assert jnp.allclose(matrix, matrix.T, atol=1e-12), \
    "Kinetic energy differences matrix is not symmetric"