"""Exchange-correlation functional interface wrapping jxc."""
import importlib

import jax.numpy as jnp
from jxc.get_params import get_params


def _xc_level(name):
  """Detect functional level from name prefix."""
  name = name.lower()
  if name.startswith(('mgga', 'hyb_mgga')):
    return 'mgga'
  if name.startswith(('gga', 'hyb_gga')):
    return 'gga'
  return 'lda'


def xc_level(xc_type):
  """Return the highest functional level among compound components.

  Args:
    xc_type (str): Functional name or compound ('+'–separated).

  Returns:
    str: One of ``'lda'``, ``'gga'``, or ``'mgga'``.
  """
  order = {'lda': 0, 'gga': 1, 'mgga': 2}
  names = [s.strip() for s in xc_type.split('+')]
  lvl = max(order[_xc_level(n)] for n in names)
  return {v: k for k, v in order.items()}[lvl]


def _raw_exc(name, polarized, rho, sigma=None, tau=None, lapl=None):
  """Compute exc using the raw jxc internal function.

  The public ``jxc.get_xc_functional(..., order='exc')`` wrapper has bugs
  for polarized and MGGA functionals (it passes ``None`` for optional grid
  arguments that the generated Maple code then tries to do arithmetic on).
  This helper calls the underlying generated function directly, providing
  explicit zero arrays where needed.
  """
  params = get_params(name, polarized)
  mod = importlib.import_module(f'jxc.functionals.{name}')
  fn = mod.pol if polarized else mod.unpol

  level = _xc_level(name)
  kwargs = {}
  if level in ('gga', 'mgga') and sigma is not None:
    kwargs['s'] = sigma
  if level == 'mgga':
    kwargs['l'] = (
      lapl if lapl is not None
      else jnp.zeros_like(rho if rho.ndim == 3 else rho[0])
    )
    if tau is not None:
      kwargs['tau'] = tau

  return fn(params, rho, **kwargs)


def compute_sigma(density_grid, g_vector_grid):
  """Compute contracted density gradient for XC functionals.

  Args:
    density_grid (Array): ``(spin, x, y, z)`` real-space density.
    g_vector_grid (Array): ``(x, y, z, 3)`` G-vector grid.

  Returns:
    Array: For spin=1: ``(x, y, z)``.
    For spin=2: ``(3, x, y, z)`` with ``[sigma_uu, sigma_ud, sigma_dd]``.
  """
  density_recip = jnp.fft.fftn(density_grid, axes=range(-3, 0))
  grads = []
  for d in range(3):
    grad_d = jnp.real(jnp.fft.ifftn(
      1j * g_vector_grid[..., d] * density_recip,
      axes=range(-3, 0)
    ))
    grads.append(grad_d)

  num_spin = density_grid.shape[0]
  if num_spin == 1:
    return sum(g[0]**2 for g in grads)
  else:
    return jnp.stack([
      sum(g[0]**2 for g in grads),
      sum(g[0] * g[1] for g in grads),
      sum(g[1]**2 for g in grads),
    ], axis=0)


def xc_energy_density(rho, xc_type, polarized, sigma=None, tau=None, lapl=None):
  """Compute XC energy density (exc) per particle.

  Supports compound functionals (e.g. ``'gga_x_pbe+gga_c_pbe'``).

  Args:
    rho (Array): Electron density.  Unpolarized: ``(x, y, z)``.
      Polarized: ``(2, x, y, z)``.
    xc_type (str): Functional name or compound.
    polarized (bool): Whether spin-polarized.
    sigma (Array): Contracted gradient (required for GGA/MGGA).
    tau (Array): Kinetic energy density (required for MGGA).
    lapl (Array): Density Laplacian (optional for MGGA, defaults to 0).

  Returns:
    Array: ``(x, y, z)`` energy density per particle.
  """
  names = [s.strip() for s in xc_type.split('+')]
  spatial_shape = rho.shape if rho.ndim == 3 else rho.shape[1:]
  exc_total = jnp.zeros(spatial_shape)

  for name in names:
    exc_total = exc_total + _raw_exc(name, polarized, rho, sigma, tau, lapl)

  return exc_total


def _raw_vxc(name, polarized, rho, sigma=None, tau=None, lapl=None):
  """Compute vxc using the raw jxc internal derivative function.

  The public ``jxc.get_xc_functional(..., order='vxc')`` wrapper fails
  inside ``jax.jit + jax.grad`` because its Maple-backend availability
  check converts traced values to numpy arrays, causing a fallback to the
  AD backend which then fails on non-scalar outputs.  This helper calls
  the underlying Maple-generated ``unpol_vxc`` / ``pol_vxc`` directly.
  """
  params = get_params(name, polarized)
  mod = importlib.import_module(f'jxc.functionals.{name}')
  fn = mod.pol_vxc if polarized else mod.unpol_vxc

  level = _xc_level(name)
  kwargs = {}
  if level in ('gga', 'mgga') and sigma is not None:
    kwargs['s'] = sigma
  if level == 'mgga':
    kwargs['l'] = (
      lapl if lapl is not None
      else jnp.zeros_like(rho if rho.ndim == 3 else rho[0])
    )
    if tau is not None:
      kwargs['tau'] = tau

  return fn(params, rho, **kwargs)


def xc_potential(rho, xc_type, polarized, sigma=None, tau=None):
  """Compute XC potential derivatives via ``jxc`` Maple-generated modules.

  Supports compound functionals.

  Args:
    rho (Array): Electron density.
    xc_type (str): Functional name or compound.
    polarized (bool): Whether spin-polarized.
    sigma (Array): Contracted gradient (required for GGA/MGGA).
    tau (Array): Kinetic energy density (required for MGGA).

  Returns:
    dict: Keys depend on level — ``'vrho'`` (always), plus ``'vsigma'``
    for GGA and ``'vsigma'``, ``'vlapl'``, ``'vtau'`` for MGGA.
    For polarized, spin/component axis is **last**
    (jxc convention): ``vrho`` has shape ``(…, 2)``, ``vsigma`` ``(…, 3)``.
  """
  names = [s.strip() for s in xc_type.split('+')]
  result = {}

  for name in names:
    vxc = _raw_vxc(name, polarized, rho, sigma=sigma, tau=tau)
    for key, val in vxc.items():
      if key in result:
        result[key] = result[key] + val
      else:
        result[key] = val

  return result


def _gga_xc_potential(vrho, vsigma, density_grid, g_vector_grid):
  r"""Compute the local GGA XC potential (vxc) on the real-space grid.

  .. math::

    V_\mathrm{xc}^\sigma = v_\rho^\sigma
    - \nabla\!\cdot\!\bigl(f_\sigma\, \nabla\rho\bigr)

  For unpolarized calculations:

  .. math::

    V_\mathrm{xc} = v_\rho - 2\,\nabla\!\cdot\!(v_\sigma\,\nabla\rho)

  Args:
    vrho (Array): ``(x, y, z)`` for unpolarized, ``(x, y, z, 2)`` for
      polarized (jxc convention, spin axis last).
    vsigma (Array): ``(x, y, z)`` for unpolarized, ``(x, y, z, 3)`` for
      polarized.
    density_grid (Array): ``(spin, x, y, z)`` real-space density.
    g_vector_grid (Array): ``(x, y, z, 3)`` G-vector grid.

  Returns:
    Array: ``(spin, x, y, z)`` local XC potential.
  """
  num_spin = density_grid.shape[0]
  density_recip = jnp.fft.fftn(density_grid, axes=range(-3, 0))

  # Compute per-spin density gradients: list of 3 arrays each (spin, x, y, z)
  nabla_rho = []
  for d in range(3):
    grad_d = jnp.real(jnp.fft.ifftn(
      1j * g_vector_grid[..., d] * density_recip,
      axes=range(-3, 0)
    ))
    nabla_rho.append(grad_d)

  def _div(field_components):
    """Divergence in reciprocal space: sum_d IFFT(iG_d FFT(f_d))."""
    div = jnp.zeros(field_components[0].shape)
    for d in range(3):
      f_recip = jnp.fft.fftn(field_components[d], axes=range(-3, 0))
      div = div + jnp.real(jnp.fft.ifftn(
        1j * g_vector_grid[..., d] * f_recip,
        axes=range(-3, 0)
      ))
    return div

  if num_spin == 1:
    # Unpolarized: v_xc = vrho - 2 * div(vsigma * nabla_rho)
    field = [vsigma * nabla_rho[d][0] for d in range(3)]
    v_xc = vrho - 2 * _div(field)
    return v_xc[None, ...]  # (1, x, y, z)
  else:
    # Polarized: vsigma has 3 components (uu, ud, dd) on last axis
    vs_uu = vsigma[..., 0]
    vs_ud = vsigma[..., 1]
    vs_dd = vsigma[..., 2]
    vrho_u = vrho[..., 0]
    vrho_d = vrho[..., 1]

    field_u = [
      2 * vs_uu * nabla_rho[d][0] + vs_ud * nabla_rho[d][1]
      for d in range(3)
    ]
    field_d = [
      vs_ud * nabla_rho[d][0] + 2 * vs_dd * nabla_rho[d][1]
      for d in range(3)
    ]
    v_xc_u = vrho_u - _div(field_u)
    v_xc_d = vrho_d - _div(field_d)
    return jnp.stack([v_xc_u, v_xc_d], axis=0)  # (2, x, y, z)


def _mgga_xc_potential(vrho, vsigma, vtau, vlapl, density_grid, g_vector_grid):
  r"""Compute the local MGGA XC potential and return vtau for the non-local part.

  The local potential extends the GGA form with a Laplacian correction:

  .. math::

    V_\mathrm{xc,local}^\sigma = v_\rho^\sigma
    - \nabla\!\cdot\!\bigl(f_\sigma\,\nabla\rho\bigr)
    + \nabla^2 v_{\nabla^2\!\rho}^\sigma

  The tau-dependent non-local operator
  :math:`-\tfrac{1}{2}\nabla\!\cdot\!(v_\tau\,\nabla\psi_i)` must be applied
  per orbital by the caller; ``vtau`` is returned for this purpose.

  Args:
    vrho (Array): ``(x, y, z)`` for unpolarized, ``(x, y, z, 2)`` for
      polarized (jxc convention, spin axis last).
    vsigma (Array): ``(x, y, z)`` for unpolarized, ``(x, y, z, 3)`` for
      polarized.
    vtau (Array): ``(x, y, z)`` for unpolarized, ``(x, y, z, 2)`` for
      polarized.
    vlapl (Array | None): ``(x, y, z)`` for unpolarized, ``(x, y, z, 2)``
      for polarized.  If ``None``, the Laplacian correction is skipped
      (some MGGA functionals do not depend on the Laplacian).
    density_grid (Array): ``(spin, x, y, z)`` real-space density.
    g_vector_grid (Array): ``(x, y, z, 3)`` G-vector grid.

  Returns:
    tuple[Array, Array]:
      - ``v_xc_local``: ``(spin, x, y, z)`` local XC potential.
      - ``vtau_grid``: ``(spin, x, y, z)`` tau derivative for the non-local
        orbital operator.
  """
  # GGA part: vrho − ∇·(f_σ ∇ρ)
  v_xc_local = _gga_xc_potential(vrho, vsigma, density_grid, g_vector_grid)

  num_spin = density_grid.shape[0]

  # Laplacian correction: ∇²(vlapl)
  # In reciprocal space: ∇²f = IFFT(-|G|² · FFT(f))
  if vlapl is not None:
    if num_spin == 1:
      vl = vlapl[None, ...]          # (1, x, y, z)
    else:
      vl = jnp.moveaxis(vlapl, -1, 0)  # (2, x, y, z)

    g_sq = jnp.sum(g_vector_grid**2, axis=-1)  # (x, y, z)
    vl_recip = jnp.fft.fftn(vl, axes=range(-3, 0))
    lapl_vl = jnp.real(jnp.fft.ifftn(
      -g_sq[None, ...] * vl_recip, axes=range(-3, 0)
    ))
    v_xc_local = v_xc_local + lapl_vl

  # Reshape vtau to (spin, x, y, z) for the caller
  if num_spin == 1:
    vtau_grid = vtau[None, ...]          # (1, x, y, z)
  else:
    vtau_grid = jnp.moveaxis(vtau, -1, 0)  # (2, x, y, z)

  return v_xc_local, vtau_grid
