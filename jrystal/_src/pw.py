"""Planewave module."""
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Complex, Float

from .grid import g_vectors
from .typing import OccupationArray, ScalarGrid
from .unitary_module import unitary_matrix, unitary_matrix_param_init
from .utils import absolute_square, reshape_coefficient, volume


def param_init(
  key,
  num_bands: int,
  num_kpts: int,
  freq_mask: ScalarGrid[Bool, 3],
  restricted: bool = True,
):
  r"""Initialize the raw parameters.

  This function generates a random tensor of shape
  :code:`(num_spin, num_kpts, num_g, num_bands)`, where :code:`num_g` is the
  number of :code:`True` items in the :code:`freq_mask`.

  In planewave based calculation, a wave function is represented as a
  linear combination of the fourier series in 3D. Therefore to create one
  wave function we need a 3D shaped tensor to represent the mixing
  coefficients on each of the frequency component (denoted as :code:`G`).
  :code:`freq_mask` provides a 3D mask to decide which frequency components
  are selected, the number of selected components is denoted as :code:`num_g`.

  The :code:`num_bands` & :code:`num_kpts` is a bit hard to explain, intuitively,
  the wave functions consists of high frequency components that has a period
  smaller than the unit cell (denoted :math:`G`) and components that has a period
  larger than the unit cell (denoted :math:`k`).

  The form of wave function under solid state

  .. math::

    \psi(r) = e^{ikr}\sum_G c_{kG} e^{iGr}

  This function generates a raw parameter, which after processing by
  :py:func:`coeff` can be used as the :math:`c_{kG}` part of the above equation.

  Extension reads:
  1. Why and how to mask the frequency components.
  2. Bloch theorum.

  As far as this function is concerned, it simply just returns a randomly
<<<<<<< HEAD
<<<<<<< HEAD
  initialized parameter of shape :code:`(num_spin, num_kpts, num_g, num_bands)`.
=======
  initialized parameter of shape `(num_spin, num_kpts, num_g, num_bands)`.
>>>>>>> 67db202 (update)
=======
  initialized parameter of shape :code:`(num_spin, num_kpts, num_g, num_bands)`.
>>>>>>> 33691b5 (update)
  The input arguments to this function is only used to determine the shape.

  Note that this function returns the raw parameter that can not be used
  directly to weight the frequency components, as in quantum chemistry we
  require the wave functions to be orthogonal to each other.
  check :py:func:`coeff` for converting the raw parameter into an unitary
  tensor.

  Args:
<<<<<<< HEAD
    key: random key for initializing the parameters.
    num_bands: the number of bands.
    num_kpts: the number of k points.
    freq_mask: a 3D mask that denotes which frequency components are selected.
    restricted: if :code:`True`, :code:`num_spin=2` else :code:`num_spin=1`.

  Returns:
    A complex type raw parameter of shape
    :code:`(num_spin, num_kpts, num_g, num_bands)`.
=======
    key: ranodm key for initializing the parameters
    num_bands: the number of bands.
    num_kpts: the number of k points.
    freq_mask: a 3D mask that denotes which frequency components are selected.
    restricted: if :code:`True`, :code:`num_spin=2` else :code:`num_spin=1`.

  Returns:
    A complex type raw parameter of shape
<<<<<<< HEAD
    `(num_spin, num_kpts, num_g, num_bands)`.
>>>>>>> 67db202 (update)
=======
    :code:`(num_spin, num_kpts, num_g, num_bands)`.
>>>>>>> 33691b5 (update)
  """
  num_spin = 1 if restricted else 2
  num_g = np.sum(freq_mask).item()
  shape = (num_spin, num_kpts, num_g, num_bands)
  return unitary_matrix_param_init(key, shape, complex=True)


def coeff(
  pw_param: Union[Array, Tuple], freq_mask: ScalarGrid[Bool, 3]
) -> Complex[Array, "spin kpts band n1 n2 n3"]:
  r"""Create the linear coefficients to combine the frequency components.

<<<<<<< HEAD
<<<<<<< HEAD
  This function takes a raw parameter of shape
  :code:`(num_spin, num_kpts, num_g, num_bands)`, orthogonalize for the last
  two dimensions, so that the resulting tensor satisfies the unitary constraint
  :code:`einsum('kabc,labc->kl', ret[i, j], ret[i, j]) == eye(num_bands)`.

  The :code:`pw_param` should be created from :py:func:`param_init`, and the same
  :code:`freq_mask` used in :py:func:`param_init` should be used here. As mentioned
  in :py:func:`param_init`, we use linear combination over 3D fourier
  components for creating wave functions. Some extra requirements are

  1. The wave functions that has the same spin and same k component needs
     to be orthogonal to each other.
  2. We only activate some of the frequency components with the :code:`freq_mask`.

  As the raw parameter returned from :py:func:`param_init` has the shape
  :code:`(num_spin, num_kpts, num_g, num_bands)`, where :code:`num_g` is the number of
  activated frequencies flattend from the activated entries in the :code:`freq_mask`
  This function first orthogonalize over the last two dimensions and
  reorganize the orthogonalized parameter into a 3D grid the same shape of
  the frequency mask.

  Extension reads:
  1. Why and how to mask the frequency components.
  2. Bloch theorum.

  Args:
    pw_param: the raw parameter, maybe created from
    :py:func:`param_init`.
    freq_mask: a 3D mask to select the frequency components.

  Returns:
    Complex array of shape :code:`(num_spin, num_kpts, num_band, n1, n2, n3)`.
    It satisfies the unitary constraint that for any :code:`i,j`
    :code:`einsum('kabc,labc->kl', ret[i, j], ret[i, j])` is an identity matrix.
=======
  The `pw_param` should be created from :py:func:`param_init`, and the same
  `freq_mask` used in :py:func:`param_init` should be used here. As mentioned
=======
  The :code:`pw_param` should be created from :py:func:`param_init`, and the same
  :code:`freq_mask` used in :py:func:`param_init` should be used here. As mentioned
>>>>>>> 33691b5 (update)
  in :py:func:`param_init`, we use linear combination over 3D fourier
  components for creating wave functions. Some extra requirements are

  1. The wave functions that has the same spin and same k component needs
     to be orthogonal to each other.
  2. We only activate some of the frequency components with the :code:`freq_mask`.

  As the raw parameter returned from :py:func:`param_init` has the shape
  :code:`(num_spin, num_kpts, num_g, num_bands)`, where :code:`num_g` is the number of
  activated frequencies flattend from the activated entries in the :code:`freq_mask`
  This function first orthogonalize over the last two dimensions and
  reorganize the orthogonalized parameter into a 3D grid the same shape of
  the frequency mask.

  Extension reads:
  1. Why and how to mask the frequency components.
  2. Bloch theorum.

  Args:
    pw_param: the raw parameter, maybe created from
    :py:func:`param_init`.
    freq_mask: a 3D mask to select the frequency components.

  Returns:
<<<<<<< HEAD
    Complex array of shape `(num_spin, num_kpts, num_band, n1, n2, n3)`.
    It satisfies the unitary constraint that for any `i,j`
    `einsum('kabc,labc->kl', ret[i, j], ret[i, j])` is an identity matrix.
>>>>>>> 67db202 (update)
=======
    Complex array of shape :code:`(num_spin, num_kpts, num_band, n1, n2, n3)`.
    It satisfies the unitary constraint that for any :code:`i,j`
    :code:`einsum('kabc,labc->kl', ret[i, j], ret[i, j])` is an identity matrix.
>>>>>>> 33691b5 (update)
  """
  coeff = unitary_matrix(pw_param, complex=True)
  coeff = jnp.swapaxes(coeff, -1, -2)
  return reshape_coefficient(coeff, freq_mask)


def wave_grid(
  coeff: Complex[Array, "spin kpts band n1 n2 n3"],
  vol: Union[float, Array],
):
  r"""Wave function evaluated at a grid of spacial locations.

<<<<<<< HEAD
  This function implements the :math:`U(r)` part of the bloch wave function.

  .. math::

    U(r)=\frac{1}{\sqrt{\Omega_\text{cell}}} \sum_G c_{G} e^{iG^\top r}

  :math:`G` is the 3D frequency components, :math:`\Omega_\text{cell}` is the
  volume of the crystal unit cell, which is to make sure the wave function is
  normalized within the cell.

  where :math:`c` is the linear coefficient. It combines over different
  :math:`G` components that is generated with :py:func:`jrystal.grid.g_vectors`.
  We can evaluate the wave function at any spatial location :math:`r` which takes
  :math:`O(|G|)` computation. However, if we evaluate this function on a specific
  spatial grid of size :math:`|G|`, we can be faster than :math:`O(|G|^2)` by using
  fourier transform. IFFT gives us an :math:`O(|G|\log(|G|))` implementation of the
  above equation. The :math:`G` and :math:`R` grid can be obtained from
  :py:func:`jrystal.grid.g_vectors` and :py:func:`jrystal.grid.r_vectors`
  correspondingly.

  .. code:: python

    G = jrystal.grid.g_vectors(*args)  # (n1, n2, n3, 3)
    R = jrystal.grid.r_vectors(*args)  # (n1, n2, n3, 3)
    coefficients = ...  # (n1, n2, n3)
    vol = ...
=======
  Our wave functions lives in the 3D space, and we use linear combination of
  3D fourier components to parameterize them, the parameters are the
  linear coeffcients. A single wave function look like

  .. math::

    \psi(r)=\frac{1}{\sqrt{V}} \sum_G c_{G} e^{iG^\top r}

  :math:`G` is the 3D frequency components, :math:`V` is the volume of the crystal
  unit cell, which is to make sure the wave function is normalized within
  the cell.

  where :math:`c` is the linear coefficient. It combines over different :math:`G`
  components that is generated with :py:func:`jrystal.grid.g_vectors`.
  We can evaluate the wave function at any spatial location :math:`r` which takes
  :math:`O(|G|)` computation. However, if we evaluate this function on a specific
  spatial grid of size :math:`|G|`, we can be faster than :math:`O(|G|^2)` by using
  fourier transform. IFFT gives us an :math:`O(|G|\log(|G|))` implementation of the
  above equation. The :math:`G` and :math:`R` grid can be obtained from
  :py:func:`jrystal.grid.g_vectors` and :py:func:`jrystal.grid.r_vectors`
  correspondingly.

<<<<<<< HEAD
  ```python
  G = jrystal.grid.g_vectors(*args)  # (n1, n2, n3, 3)
  R = jrystal.grid.r_vectors(*args)  # (n1, n2, n3, 3)
  coefficients = ...  # (n1, n2, n3)
  vol = ...
>>>>>>> 67db202 (update)
=======
  .. code:: python
>>>>>>> 33691b5 (update)

    G = jrystal.grid.g_vectors(*args)  # (n1, n2, n3, 3)
    R = jrystal.grid.r_vectors(*args)  # (n1, n2, n3, 3)
    coefficients = ...  # (n1, n2, n3)
    vol = ...

<<<<<<< HEAD
<<<<<<< HEAD
=======
    def wave_function(r):
      return (coefficients * jnp.exp(1j * G @ r)).sum() / jnp.sqrt(vol)

>>>>>>> 33691b5 (update)
    # The following is O(|G|^2)
    wave_at_R_naive = jax.vmap(jax.vmap(jax.vmap(wave_function)))(R)
    # The following is O(|G|log|G|)
    wave_at_R_fft = wave_grid(coefficients, vol)
<<<<<<< HEAD

  As IFFT implements

  .. math::

    x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k e^{i\frac{2\pi}{N}kn}
=======
  # The following is O(|G|^2)
  wave_at_R_naive = jax.vmap(jax.vmap(jax.vmap(wave_function)))(R)
  # The following is O(|G|log|G|)
  wave_at_R_fft = wave_grid(coefficients, vol)
  ```

  As IFFT implements

  $$
  x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k e^{i\frac{2\pi}{N}kn}
  $$
>>>>>>> 67db202 (update)
=======

  As IFFT implements

  .. math::

    x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k e^{i\frac{2\pi}{N}kn}
>>>>>>> 33691b5 (update)

  It is a bit different from the definition of the wave function,
  if you check the code, we do two things to align them,

<<<<<<< HEAD
<<<<<<< HEAD
  1. we multiply back the :math:`N` to cancel the :math:`\frac{1}{N}`
  factor in the IFFT (in 3D the :code:`np.prod(grid_sizes)`).
  2. we divide by the :math:`\sqrt{\Omega_\text{cell}}`.

  The :code:`coeff` passed to this function has shape :code:`(..., n1, n2, n3)`,
=======
  1. we multiply back the $N$ to cancel the $\frac{1}{N}$
  factor in the IFFT (in 3D the `np.prod(grid_sizes)`).
  2. we divide by the $\sqrt{V}$.

  The `coeff` passed to this function has shape `(..., n1, n2, n3)`,
>>>>>>> 67db202 (update)
=======
  1. we multiply back the :math:`N` to cancel the :math:`\frac{1}{N}`
  factor in the IFFT (in 3D the :code:`np.prod(grid_sizes)`).
  2. we divide by the :math:`\sqrt{V}`.

  The :code:`coeff` passed to this function has shape :code:`(..., n1, n2, n3)`,
>>>>>>> 33691b5 (update)
  it can have any leading dimension.
  It can be created using :py:func:`param_init` and :py:func:`coeff`.
  :py:func:`param_init` creates a raw parameter and :py:func:`coeff` converts
  that parameter into coefficients that are used to linearly weight the 3D
  fourier components.

  Args:
    coeff: linear combination coefficients over the 3D fourier components.
<<<<<<< HEAD
<<<<<<< HEAD
      shape is :code:`(..., n1, n2, n3)` where :code:`(n1, n2, n3)` is the shape of the
=======
      shape is `(..., n1, n2, n3)` where `(n1, n2, n3)` is the shape of the
>>>>>>> 67db202 (update)
=======
      shape is :code:`(..., n1, n2, n3)` where :code:`(n1, n2, n3)` is the shape of the
>>>>>>> 33691b5 (update)
      3D frequency components generated
      from :py:func:`jrystal.grid.g_vectors`.
    vol: volume of the unit cell.

  Returns:
    The result of evaluating the wave function on the specific spatial grid.
  """
  grid_sizes = coeff.shape[-3:]
  wave_grid = jnp.fft.ifftn(coeff, axes=range(-3, 0))
  wave_grid *= np.prod(grid_sizes) / jnp.sqrt(vol)
  return wave_grid


def density_grid(
  coeff: Complex[Array, "spin kpts band n1 n2 n3"],
  vol: Union[float, Array],
  occupation: Optional[OccupationArray] = None
) -> ScalarGrid[Complex, 3]:
  r"""Compute the density at the spatial grid.

  In a system with electrons, the density of the electron is the result of
  multiple wave functions overlapping in the space. As we mention in
  :py:func:`wave_grid`, the wave function is a linear combination of 3D fourier
  components. To compute the density, usually we only need to take the absolute
  square of each wave function and sum them up.

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 33691b5 (update)
  .. math::

    \rho(r) = \sum_i |\psi_i(r)|^2

<<<<<<< HEAD
  This function evaluates the density :math:`\rho(r)` as the spatial grid
  generated from the :py:func:`jrystal.grid.r_vectors`.

  In crystals, this is a little bit more complicated. The form of the wave
  function is

  .. math::

    \psi(r) = \frac{1}{\sqrt{\Omega_\text{cell}}} e^{ikr} \sum_G c_{kG}
    e^{iGr}

  The :math:`c_{kG}` can be computed from :py:func:`param_init`
  and :py:func:`coeff`. For calculation of density, we only need
  the :math:`c_{kG}` and the occupation :math:`o_k` over :math:`k`.

  .. math::

    \rho(r) = \frac{1}{\Omega_\text{cell}} e^{ikr} \sum_G c_{kG}
    e^{iGr}

  Args:
    coeff: :math:`c_{kG}` part of the parameter. It can have a leading batch dimension
      which will be summed to get the overall density.
      Therefore the shape is :code:`(..., num_kpts, num_bands, n1, n2, n3)`.
    vol: volume of the unit cell, a real scalar.
    occupation: the occupation over different k frequencies.
      The shape is :code:`(..., num_kpts, num_bands)`, it should have the same leading dimension
      as :code:`coeff`.
      This is an option argument, when :code:`occupation=None`, we compute the density
      contribution from each :math:`k` without summing them. If :code:`occupation` is
      provided, we sum up all the density from each :math:`k` weighted by the
=======
=======
  This function evaluates the density :math:`\rho(r)` as the spatial grid generated
  from the :py:func:`jrystal.grid.r_vectors`.

>>>>>>> 33691b5 (update)
  In crystals, this is a little bit more complicated. If we consider the
  components whose period is smaller than the unit cell as :math:`G` and the
  components whose period is larger than the unit cell as :math:`k`, the most
  general form of wave function is

  .. math::

    \psi(r) = \frac{1}{\sqrt{V}} \sum_k \sum_G c_{kG} e^{i(k+G)r}

  However, according to bloch theorum, which uses extra periodic constraints,
  the parametric form of wave function can be reduced to

  .. math::

    \psi(r) = \frac{1}{\sqrt{V}} \sum_k d_k e^{ikr}\sum_G c_{kG} e^{iGr}

  The :math:`c_{kG}` part of the parameter can be computed from :py:func:`param_init`
  and :py:func:`coeff`. The :math:`d_k` part of the parameter we refer to
  :py:func:`jrystal.occupation`. For calculation of density, we only need
  the :math:`c_{kG}` and :math:`o_k=d_k^2`, which we also call :math:`o_k` the occupation over
  different :math:`k` frequencies. It is very intuitive because density is the
  absolute square of the wave function.

  Args:
    coeff: :math:`c_{kG}` part of the parameter. It can have a leading batch dimension
      which will be summed to get the overall density.
      Therefore the shape is :code:`(..., num_kpts, num_bands, n1, n2, n3)`.
    vol: volume of the unit cell, a real scalar.
    occupation: the occupation over different k frequencies.
<<<<<<< HEAD
      The shape is `(..., num_kpts)`, it should have the same leading dimension
      as `coeff`.
      This is an option argument, when `occupation=None`, we compute the density
      contribution from each $k$ without summing them. If `occupation` is
      provided, we sum up all the density from each $k$ weighted by the
>>>>>>> 67db202 (update)
=======
      The shape is :code:`(..., num_kpts, num_bands)`, it should have the same leading dimension
      as :code:`coeff`.
      This is an option argument, when :code:`occupation=None`, we compute the density
      contribution from each :math:`k` without summing them. If :code:`occupation` is
      provided, we sum up all the density from each :math:`k` weighted by the
>>>>>>> 33691b5 (update)
      occupation.

  Returns:
    A real valued tensor that represents the density at the spatial grid
    computed from :py:func:`jrystal.grid.r_vectors`.
<<<<<<< HEAD
<<<<<<< HEAD
    The shape is :code:`(n1, n2, n3)` if :code:`occupation` is provided,
    else the shape is :code:`(..., num_kpts, num_bands, n1, n2, n3)`.
=======
    The shape is `(n1, n2, n3)` if `occupation` is provided,
    else the shape is `(num_kpts, n1, n2, n3)`.
>>>>>>> 67db202 (update)
=======
    The shape is :code:`(n1, n2, n3)` if :code:`occupation` is provided,
    else the shape is :code:`(..., num_kpts, num_bands, n1, n2, n3)`.
>>>>>>> 33691b5 (update)
  """
  wave_grid_arr = wave_grid(coeff, vol)
  dens = absolute_square(wave_grid_arr)

  if occupation is not None:
<<<<<<< HEAD
<<<<<<< HEAD
    occ = jnp.expand_dims(occupation, range(-3, 0))
    try:
      occ = jnp.broadcast_to(occ, dens.shape)
      dens = jnp.sum(dens * occ, axis=range(occupation.ndim))
=======
    occupation = jnp.expand_dims(occupation, range(-3, 0))
    try:
      occupation = jnp.broadcast_to(occupation, dens.shape)
      dens = jnp.sum(dens * occupation, axis=range(3))
>>>>>>> 67db202 (update)
=======
    occ = jnp.expand_dims(occupation, range(-3, 0))
    try:
      occ = jnp.broadcast_to(occ, dens.shape)
      dens = jnp.sum(dens * occ, axis=range(occupation.ndim))
>>>>>>> 33691b5 (update)
    except ValueError:
      raise ValueError(
        "Occupation should have a leading dimension that is the same as coeff."
        f"Got occupation shape: {occupation.shape}, coeff shape: {coeff.shape}"
      )
  return dens


def density_grid_reciprocal(
  coeff: Complex[Array, "spin kpts band n1 n2 n3"],
  vol: Union[float, Array],
  occupation: Optional[OccupationArray] = None
) -> ScalarGrid[Complex, 3]:
  r"""Fourier transform of the density grid.

  In a system with electrons, the density of the electron is the result of
  multiple wave functions overlapping in the space. As we mention in
  :py:func:`wave_grid`, the wave function is a linear combination of 3D fourier
  components. To compute the density, usually we only need to take the absolute
  square of each wave function and sum them up.

  .. math::

    \rho(r) = \sum_i |\psi_i(r)|^2

  :py:func:`density_grid` computes the density :math:`\rho(r)` at the spatial grid
  generated from :py:func:`jrystal.grid.r_vectors`.

  The fourier transformation of :math:`\rho(r)` is

  .. math::

    \tilde{\rho}(G) = \int \rho(r) e^{-iGr} dr

  We can also evaluate the :math:`\tilde{\rho}(G)` at any :math:`G`, but this function
  computes the :math:`\tilde{\rho}(G)` evaluated at the grid generated from
  :py:func:`jrystal.grid.g_vectors`. It is equivalent to computing the :math:`\rho(r)`
  at :py:func:`jrystal.grid.r_vectors` and then do the discrete fourier
  transform.

  Since this function is just composing FFT with :py:func:`density_grid`,
  we refer you to :py:func:`density_grid` for more details.

  Args:
    coeff: coefficients of the wave functions.
    vol: volume of the unit cell.
    occupation: occupation over the different :math:`k` components, see more in
      :py:func:`density_grid`.

  Returns:
    A complex valued tensor representing :math:`\tilde{\rho}(G)` evaluated at :math:`G`
    generated from :py:func:`jrystal.grid.g_vectors`.
  """
  dens = density_grid(coeff, vol, occupation)
  return jnp.fft.fftn(dens, axes=range(-3, 0))


def wave_r(
  r: Float[Array, "3"],
  coeff: Complex[Array, "spin kpts band n1 n2 n3"],
  cell_vectors: Float[Array, "3 3"],
  g_vector_grid: Optional[ScalarGrid[Float, 3]] = None,
) -> Complex[Array, "spin kpts band *b"]:
  r"""Evaluate plane wave functions at location r.

  TODO(litb): I think the g_vector_grid is not necessary, we can compute it from
    cell_vectors. Otherwise, if we specify g_vector_grid and it is not
    compatible with cell_vectors, then we shouldn't use its vol. And also seem
    very weird.

  This function basically computes the :math:`\psi(r)` following the equation

  .. math::

<<<<<<< HEAD
    \psi(r) = \frac{1}{\sqrt{\Omega_\text{cell}}} \sum_G c_{G} e^{iGr}
=======
    \psi(r) = \frac{1}{\sqrt{V}} \sum_G c_{G} e^{iGr}
>>>>>>> 33691b5 (update)

  the :code:`coeff` provided is the :math:`c_{G}`, the :code:`cell_vectors` is
  used to generate the grid of frequency components $G$ by calling the function
  :py:func:`jrystal.grid.g_vectors`. The :code:`r` is the location where we
  evaluate the wave function.

  The :code:`coeff` passed to this function has shape :code:`(..., n1, n2, n3)`,
  it can have leading batch dimensions.

  Args:
<<<<<<< HEAD
    r: spatial location to evaluate the wave function, shape: (3,).
=======
    r: spatial location to evaluate the wave function, shape: (*batch, 3).
>>>>>>> 33691b5 (update)
    coeff: wave function coefficients, which has a shape of
      :code:`(..., n1, n2, n3)`.
      it can be created from :py:func:`param_init` followed by :py:func:`coeff`.
    cell_vectors: the cell vectors of the crystal unit cell.
    g_vector_grid: to be resolved.

  Returns:
    Complex tensor that represents wave functions evaluated at location r,
<<<<<<< HEAD
    with shape of the leading dimensions of the coeff (...).
=======
    with shape (*batch,).
>>>>>>> 33691b5 (update)
  """
  vol = volume(cell_vectors)
  n1, n2, n3 = coeff.shape[-3:]

  if g_vector_grid is None:
    g_vector_grid = g_vectors(cell_vectors, [n1, n2, n3])

<<<<<<< HEAD
  if r.shape != (3,):
    raise ValueError("r must have shape (3,)")
  leading_dims = coeff.shape[:-3]
  coeff_ = coeff.reshape((-1, n1, n2, n3))
  output = jnp.exp(1j * g_vector_grid @ r)
  output = jnp.einsum("lxyz,xyz->skb", coeff_, output)
  output = jnp.reshape(output, leading_dims)
=======
  batch_dims = r.shape[:-1]
  r_ = r.reshape((-1, r.shape[-1]))
  leading_dims = coeff.shape[:-3]
  coeff_ = coeff.reshape((-1, n1, n2, n3))
  output = jnp.exp(1j * jnp.einsum("lxyzd,d->lxyz", g_vector_grid, r))
  output = jnp.einsum("lxyz,xyz->skb", coeff_, output)
>>>>>>> 33691b5 (update)
  return output / jnp.sqrt(vol)


def density_r(
  r: Float[Array, "3"],
  coeff: Complex[Array, "spin kpts band n1 n2 n3"],
  cell_vectors: Float[Array, "3 3"],
  g_vector_grid: Optional[ScalarGrid[Float, 3]] = None,
  occupation: Optional[OccupationArray] = None,
):
  r"""Compute the electron density at location r.

  :py:func:`wave_r` computes the electron density at location :math:`r`.
  This function computes the density at location :math:`r`. If occupation is
  not provided, it simply returns the absolute square of the :py:func:`wave_r`.
  If occupation is provided, the :code:`coeff` needs to have a shape of
  :code:`(num_spin, num_kpts, num_bands, n1, n2, n3)`, the :code:`occupation`
  needs to have the dimension of the :code:`(num_spin, num_kpts, num_bands)`.

  Args:
    r: spatial location to evaluate the density, shape: (3,).
    coeff: wave function coefficients, which has a shape of
      :code:`(num_spin, num_kpts, num_bands, n1, n2, n3)`.
      it can be created from :py:func:`param_init` followed by :py:func:`coeff`.
    cell_vectors: the cell vectors of the crystal unit cell.
    g_vector_grid: the G vectors computed from
      :py:func:`jrystal.grid.g_vectors`.
    occupation: occupation over different k frequencies.
      Refer to :py:func:`density_grid` for more information.

  Returns:
    A real scalar that represents the density at location r.
  """
  density = absolute_square(wave_r(r, coeff, cell_vectors, g_vector_grid))
  if occupation is not None:
    density = jnp.sum(density * occupation)
  return density


def nabla_density_r(
  r: Float[Array, "3"],
  coeff: Complex[Array, "spin kpts band n1 n2 n3"],
  cell_vectors: Float[Array, "3 3"],
  g_vector_grid: Optional[ScalarGrid[Float, 3]] = None,
  occupation: Optional[OccupationArray] = None,
):
  r"""Compute the first order derivative of the density at location r.

  Refer the :py:func:`density_r` for more information.

  Args:
    r: spatial location to evaluate the density, shape: (3,).
    coeff: wave function coefficients, which has a shape of
      :code:`(num_spin, num_kpts, num_bands, n1, n2, n3)`.
      it can be created from :py:func:`param_init` followed by :py:func:`coeff`.
    cell_vectors: the cell vectors of the crystal unit cell.
    g_vector_grid: the G vectors computed from
      :py:func:`jrystal.grid.g_vectors`.
    occupation: occupation over different k frequencies.
      Refer to :py:func:`density_grid` for more information.

  Returns:
    A :code:`(3,)` real vector that represents the density derivative at
      location :code:`r`.
  """

  def den(r):
    return density_r(r, coeff, cell_vectors, g_vector_grid, occupation)

  return jax.grad(den)(r)


def nabla_density_grid(
  r: Float[Array, "3"],
  coeff: Complex[Array, "spin kpts band n1 n2 n3"],
  cell_vectors: Float[Array, "3 3"],
  g_vector_grid: Optional[ScalarGrid[Float, 3]] = None,
  occupation: Optional[OccupationArray] = None,
) -> ScalarGrid[Float, 3]:
  r"""Compute the first order derivative of the density at a specific grid of
  spatial locations.

  Refer the :py:func:`density_grid` for more information.

  Args:
    coeff: :math:`c_{kG}` part of the parameter. It can have a leading batch dimension
      which will be summed to get the overall density.
      Therefore the shape is :code:`(..., num_kpts, num_bands, n1, n2, n3)`.
    vol: volume of the unit cell, a real scalar.
    occupation: the occupation over different k frequencies.
      The shape is :code:`(..., num_kpts, num_bands)`, it should have the same leading dimension
      as :code:`coeff`.
      This is an option argument, when :code:`occupation=None`, we compute the density
      contribution from each :math:`k` without summing them. If :code:`occupation` is
      provided, we sum up all the density from each :math:`k` weighted by the
      occupation.

  Returns:
    A real valued tensor that represents the density derivative at the spatial
    grid computed from :py:func:`jrystal.grid.r_vectors`.
    The shape is :code:`(n1, n2, n3, 3)` if :code:`occupation` is provided,
    else the shape is :code:`(..., num_kpts, num_bands, n1, n2, n3, 3)`.
  """

  r = jnp.reshape(r, (-1))

  if r.shape[0] != 3:
    raise ValueError("r must have shape (3,)")

  vol = volume(cell_vectors)
  n1, n2, n3 = coeff.shape[-3:]

  if g_vector_grid is None:
    g_vector_grid = g_vectors(cell_vectors, [n1, n2, n3])

  gr = jnp.dot(g_vector_grid, r)

  cosgr = jnp.cos(gr)
  singr = jnp.sin(gr)
  cr = jnp.real(coeff)
  ci = jnp.imag(coeff)

  rcos = jnp.einsum("skbxyz,xyz->skbxyz", cr, cosgr)
  rsin = jnp.einsum("skbxyz,xyz->skbxyz", cr, singr)
  icos = jnp.einsum("skbxyz,xyz->skbxyz", ci, cosgr)
  isin = jnp.einsum("skbxyz,xyz->skbxyz", ci, singr)

  o1 = jnp.sum(rcos - isin, axis=range(-3, 0))
  o1 = jnp.expand_dims(o1, -1)
  o2 = -jnp.einsum("skbxyz,xyzd->skbd", rsin + icos, g_vector_grid)
  o3 = jnp.sum(rsin + icos, axis=range(-3, 0))
  o3 = jnp.expand_dims(o3, -1)
  o4 = jnp.einsum("skbxyz,xyzd->skbd", rcos - isin, g_vector_grid)

  output = 2 * (o1 * o2 + o3 * o4)

  if occupation is not None:
    occupation = jnp.expand_dims(occupation, -1)
    output = jnp.sum(output * occupation, axis=range(0, 3)) / vol

  return output
