# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy.special import gamma, spherical_jn
from scipy.fft import ifft, rfft, irfft

# usefull constants
PI = np.pi
TPI = 2.0 * np.pi
II = 1.0j


class pyNumSBT(object):
  r'''
    Numerically perform spherical Bessel transform (SBT) in :math:`O(Nln(N))` time based
    on the algorithm proposed by J. Talman.

        Talman, J. Computer Physics Communications 2009, 180, 332-338.

    For a function "f(r)" defined numerically on a LOGARITHMIC radial grid "r",
    the SBT for the function gives

    .. math::
        g(k) = \sqrt{2\over\pi} \int_0^\infty j_l(kr) f(r) r^2 dr           (c1)

    and the inverse SBT (iSBT) gives

    .. math::
        f(r) = \sqrt{2\over\pi} \int_0^\infty j_l(kr) g(k) k^2 dk           (c2)
  '''

  def __init__(self, rr, kmax: float = 500, lmax: int = 10):
    self.rr = np.asarray(rr, dtype=float)
    # the real-space logarithmic radial grid
    self.nr = self.rr.size  # No. of grid points
    self.nr2 = 2 * self.nr
    self.sbt_init(rr, kmax)

    self.lmax = lmax
    self.sbt_mltb(self.lmax)

  def sbt_init(self, rr, kmax: float):
    r'''
    Initialize the real-space grid (rr) and momentum-space grid (kk).

    .. math::
          \rho   = \ln(rr)
          \kappa = \ln(kk)

    The algorithm by Talman requries

    .. math::
          \Delta\kappa = \Delta\rho
    '''

    self.r_min = rr[0]
    self.r_max = rr[-1]

    self.rho_min = np.log(self.r_min)
    self.rho_max = np.log(self.r_max)
    # \Delta\rho = \ln(rr[1] - rr[0])
    self.drho = (self.rho_max - self.rho_min) / (self.nr - 1)

    self.kappa_max = np.log(kmax)
    self.kappa_min = self.kappa_max - (self.rho_max - self.rho_min)

    # the reciprocal-space (momentum-space) logarithmic grid
    self.kk = np.exp(self.kappa_min) * np.exp(np.arange(self.nr) * self.drho)
    self.k_min = self.kk[0]
    self.k_max = self.kk[-1]

    self.rr3 = self.rr**3
    self.kk3 = self.kk**3

    # r values for the extended mesh as discussed in Sec.4 of Talman paper.
    self.rr_ext = self.r_min * np.exp(np.arange(-self.nr, 0) * self.drho)

    # the multiply factor as in Talman paper after Eq.(32)
    self.rr15 = np.zeros((2, self.nr), dtype=float)
    self.rr15[0] = self.rr_ext**1.5 / self.r_min**1.5  # the extended r-mesh
    self.rr15[1] = self.rr**1.5 / self.r_min**1.5  # the normal r-mesh

    # \Delta\rho \times \Delta t = \frac{2\pi}{N}
    # as in Talman paper after Eq.(30), where N = 2 * nr as discussed in
    # Sec. 4
    self.dt = TPI / (self.nr2 * self.drho)

    # the post-division factor (1 / k^1.5) as in Eq. (32) of Talman paper.
    # Note that the factor is (1 / r^1.5) for inverse-SBT, as a result, we
    # omit the r_min^1.5 / k_min^1.5 here.
    self.post_div_fac = np.exp(-np.arange(self.nr) * self.drho * 1.5)

    # Simpson integration of a function on the logarithmic radial grid.
    #
    # Setup weights for simpson integration on radial grid any radial integral
    # can then be evaluated by just summing all radial grid points with the
    # weights SI
    #
    # \int dr = \sum_i w(i) * f(i)
    self.simp_wht_rr = np.zeros_like(self.rr)
    self.simp_wht_kk = np.zeros_like(self.kk)
    for ii in range(self.nr - 1, 1, -2):
      self.simp_wht_rr[ii] = self.drho * self.rr[ii] / 3. \
                             + self.simp_wht_rr[ii]
      self.simp_wht_rr[ii - 1] = self.drho * self.rr[ii - 1] * 4. / 3.
      self.simp_wht_rr[ii - 2] = self.drho * self.rr[ii - 2] / 3.

      self.simp_wht_kk[ii] = self.drho * self.kk[ii] / 3. + self.simp_wht_kk[ii]
      self.simp_wht_kk[ii - 1] = self.drho * self.kk[ii - 1] * 4. / 3.
      self.simp_wht_kk[ii - 2] = self.drho * self.kk[ii - 2] / 3.

  def sbt_mltb(self, lmax_in: int):
    '''
        construct the M_l(t) table according to Eq. (15), (16) and (24) of
        Talman paper.

        Note that Talman paper use Stirling's approximaton to evaluate the Gamma
        function, whereas I just call scipy.special.gamma here.
        '''

    if lmax_in > self.lmax:
      lmax = lmax_in
      self.lmax = lmax
    else:
      lmax = self.lmax

    tt = np.arange(self.nr) * self.dt

    # M_lt1 is quantity defined by Eq. (12)
    self.M_lt1 = np.zeros((lmax + 1, self.nr), dtype=complex)

    # Eq. (15) in Talman paper
    # self.M_lt1[0]=gamma(0.5 - II*tt) * np.sin(0.5*PI*(0.5 - II*tt)) / self.nr

    # Eq. (19) and Eq. (15) in Talman paper are equivalent, while the former
    # is more stable for larger tt
    self.M_lt1[0] = np.sqrt(np.pi / 2) * np.exp(
      II * (
        np.log(gamma(0.5 - II * tt)).imag
        # + np.log(np.sin(0.5*PI*(0.5 - II*tt))).imag
        - np.arctan(np.tanh(PI * tt / 2))
      )
    ) / self.nr

    self.M_lt1[0, 0] /= 2.0
    self.M_lt1[0] *= np.exp(II * tt * (self.kappa_min + self.rho_min))

    # Eq. (16) in Talman paper
    phi = np.arctan(np.tanh(PI * tt / 2)) - np.arctan(2 * tt)
    self.M_lt1[1] = self.M_lt1[0] * np.exp(2 * II * phi)

    # Eq. (24) in Talman paper
    for ll in range(1, lmax):
      phi_l = np.arctan(2 * tt / (2 * ll + 1))
      self.M_lt1[ll + 1] = np.exp(-2 * II * phi_l) * self.M_lt1[ll - 1]

    ll = np.arange(lmax + 1)
    # Eq. (35) in Talman paper
    xx = np.exp(self.rho_min + self.kappa_min + np.arange(self.nr2) * self.drho)

    # M_lt2 is just the Fourier transform of spherical Bessel function j_l
    self.M_lt2 = ifft(
      spherical_jn(ll[:, None], xx[None, :]), axis=1
    ).conj()[:, :self.nr + 1]

  def run(
    self,
    ff,
    l: int = 0,
    direction: int = 1,
    norm: bool = False,
    np_in: int = 0,
    return_rr: bool = False,
    include_zero: bool = False,
  ):
    '''
      Perform SBT or inverse-SBT.

      Input parapeters:
        ff: the function defined on the logarithmic radial grid
        l: the "l" as in the underscript of "j_l(kr)" of Eq. (c1) and (c2)
        direction: 1 for forward SBT and -1 for inverse SBT
        norm: whether to multiply the prefactor \sqrt{2\over\pi} in Eq. (c1) and
          (c2). If False, then subsequent applicaton of SBT and iSBT will yield
          the original data scaled by a factor of 2/pi.
        np_in: the asymptotic bahavior of ff when  r -> 0
          ff(r\to 0) \approx r^{np_in + l}
        include_zero: the SBT does not include the k = 0, i.e.
          self.kk.min() != 0, term by default.
    '''

    assert l <= self.lmax, \
           "lmax = {} smaller than l = {}! Increase lmax!".format(self.lmax, l)

    ff = np.asarray(ff, dtype=float)
    gg = np.zeros_like(ff)

    r2c_in = np.zeros(self.nr2, dtype=float)
    r2c_out = np.zeros(self.nr + 1, dtype=complex)

    c2r_in = np.zeros(self.nr + 1, dtype=complex)
    c2r_out = np.zeros(self.nr2, dtype=float)

    # The prefactor as in Eq. (c1) and (c2) of the Class docstring.
    sqrt_2_over_pi = np.sqrt(2 / PI) if norm else 1.0

    if direction == 1:
      rmin = self.r_min
      kmin = self.k_min
      C = ff[0] / self.r_min**(np_in + l)
    elif direction == -1:
      rmin = self.k_min
      kmin = self.r_min
      C = ff[0] / self.k_min**(np_in + l)
    else:
      raise ValueError("Use direction=1/-1 for forward- and inverse-SBT!")

    # SBT for LARGE k values extend the input to the doubled mesh,
    # extrapolating the input as C r^(np_in + l)

    # Step 1 in the procedure after Eq. (32) of Talman paper
    r2c_in[:self.nr] = C * self.rr_ext**(np_in + l) * self.rr15[0]
    r2c_in[self.nr:] = ff * self.rr15[1]

    # Step 2 in the procedure after Eq. (32) of Talman paper
    r2c_out = rfft(r2c_in)

    # Step 3 in the procedure after Eq. (32) of Talman paper
    tmp1 = np.zeros(self.nr2, dtype=complex)
    tmp1[:self.nr] = r2c_out[:self.nr].conj() * self.M_lt1[l]

    # Step 4 and 5 in the procedure after Eq. (32) of Talman paper
    tmp2 = ifft(tmp1) * self.nr2
    gg = (rmin /
          kmin)**1.5 * tmp2[self.nr:].real * self.post_div_fac * sqrt_2_over_pi

    # obtain the SMALL k results in the array c2r_out

    if direction == 1:
      r2c_in[:self.nr] = self.rr3 * ff
    else:
      r2c_in[:self.nr] = self.kk3 * ff
    r2c_in[self.nr:] = 0.0
    r2c_out = rfft(r2c_in)

    c2r_in = r2c_out.conj() * self.M_lt2[l] * sqrt_2_over_pi
    c2r_out = irfft(c2r_in) * self.nr2
    c2r_out[:self.nr] *= self.drho

    # compare the minimum difference between large and small k as described
    # in the paragraph above Eq. (39) of Talman paper.
    gdiff = np.abs(gg - c2r_out[:self.nr])
    minloc = np.argmin(gdiff)
    gg[:minloc + 1] = c2r_out[:minloc + 1]

    # include k = 0 in the SBT and r = 0 in the i-SBT.
    if include_zero:
      if direction == 1:
        gg0 = sqrt_2_over_pi * spherical_jn(l, 0) * np.sum(
          self.simp_wht_rr * self.rr**2 * ff
        )
      else:
        gg0 = sqrt_2_over_pi * spherical_jn(l, 0) * np.sum(
          self.simp_wht_kk * self.kk**2 * ff
        )

    if return_rr:
      if direction == 1:
        return (np.r_[0,
                      self.kk], np.r_[gg0,
                                      gg]) if include_zero else (self.kk, gg)
      else:
        return (np.r_[0,
                      self.rr], np.r_[gg0,
                                      gg]) if include_zero else (self.rr, gg)
    else:
      return np.r_[gg0, gg] if include_zero else gg

  def run_int(
    self,
    ff,
    l: int = 0,
    direction: int = 1,
    norm: bool = False,
    return_rr: bool = False,
    include_zero: bool = False,
  ):
    '''
        Perform SBT or inverse-SBT by numerically integrating Eq. (c1) and (c2).
        Input parapeters:
        ff: the function defined on the logarithmic radial grid
        l: the "l" as in the underscript of "j_l(kr)" of Eq. (c1) and (c2)
        direction: 1 for forward SBT and -1 for inverse SBT
        norm: whether to multiply the prefactor \sqrt{2\over\pi} in Eq. (c1)
        and (c2). If False, then subsequent applicaton of SBT and iSBT will
        yield the original data scaled by a factor of 2/pi.
        include_zero: the SBT does not include the k = 0, i.e. self.kk.min() !=
        0, term by default.
      '''

    kr = self.rr[:, None] * self.kk[None, :]
    jl_kr = spherical_jn(l, kr)
    ff = np.asarray(ff, dtype=float)

    # The prefactor as in Eq. (c1) and (c2) of the Class docstring.
    sqrt_2_over_pi = np.sqrt(2 / PI) if norm else 1.0

    if direction == 1:
      gg = sqrt_2_over_pi * np.sum(
        jl_kr * (self.rr**2 * ff * self.simp_wht_rr)[:, None], axis=0
      )
      # include k = 0 in the SBT and r = 0 in the i-SBT.
      if include_zero:
        gg0 = sqrt_2_over_pi * spherical_jn(l, 0) * np.sum(
          self.simp_wht_rr * self.rr**2 * ff
        )
    elif direction == -1:
      gg = sqrt_2_over_pi * np.sum(
        jl_kr * (self.kk**2 * ff * self.simp_wht_kk)[None, :], axis=1
      )
      # include k = 0 in the SBT and r = 0 in the i-SBT.
      if include_zero:
        gg0 = sqrt_2_over_pi * spherical_jn(l, 0) * np.sum(
          self.simp_wht_kk * self.kk**2 * ff
        )
    else:
      raise ValueError("Use direction=1/-1 for forward- and inverse-SBT!")

    if return_rr:
      if direction == 1:
        return (np.r_[0,
                      self.kk], np.r_[gg0,
                                      gg]) if include_zero else (self.kk, gg)
      else:
        return (np.r_[0,
                      self.rr], np.r_[gg0,
                                      gg]) if include_zero else (self.rr, gg)
    else:
      return np.r_[gg0, gg] if include_zero else gg
