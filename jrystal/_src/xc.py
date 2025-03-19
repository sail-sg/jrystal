import jax.numpy as jnp

def lda_x_spin(n, zeta, **kwargs):
    """Slater exchange functional (spin-polarized).

    Corresponds to the functional with the label LDA_X and ID 1 in Libxc.

    Reference: Phys. Rev. 81, 385.

    Args:
        n: Real-space electronic density.
        zeta: Relative spin polarization.

    Keyword Args:
        **kwargs: Throwaway arguments.

    Returns:
        Exchange energy density and potential.
    """
    f = -3 / 4 * (3 / jnp.pi) ** (1 / 3)

    rho13p = ((1 + zeta) * n) ** (1 / 3)
    rho13m = ((1 - zeta) * n) ** (1 / 3)

    ex_up = f * rho13p
    ex_dw = f * rho13m
    ex = 0.5 * ((1 + zeta) * ex_up + (1 - zeta) * ex_dw)

    vx_up = 4 / 3 * ex_up
    vx_dw = 4 / 3 * ex_dw
    return ex, jnp.array([vx_up, vx_dw]), None


def lda_c_pw(n, A=0.031091, a1=0.2137, b1=7.5957, b2=3.5876, b3=1.6382, b4=0.49294, **kwargs):
    """Perdew-Wang parametrization of the correlation functional (spin-paired).

    Corresponds to the functional with the label LDA_C_PW and ID 12 in Libxc.

    Reference: Phys. Rev. B 45, 13244.

    Args:
        n: Real-space electronic density.

    Keyword Args:
        A: Functional parameter.
        a1: Functional parameter.
        b1: Functional parameter.
        b2: Functional parameter.
        b3: Functional parameter.
        b4: Functional parameter.
        **kwargs: Throwaway arguments.

    Returns:
        PW correlation energy density and potential.
    """
    rs = (3 / (4 * jnp.pi * n)) ** (1 / 3)
    rs12 = jnp.sqrt(rs)
    rs32 = rs * rs12
    rs2 = rs**2

    om = 2 * A * (b1 * rs12 + b2 * rs + b3 * rs32 + b4 * rs2)
    olog = jnp.log(1 + 1 / om)
    ec = -2 * A * (1 + a1 * rs) * olog

    dom = 2 * A * (0.5 * b1 * rs12 + b2 * rs + 1.5 * b3 * rs32 + 2 * b4 * rs2)
    vc = -2 * A * (1 + 2 / 3 * a1 * rs) * olog - 2 / 3 * A * (1 + a1 * rs) * dom / (om * (om + 1))
    return ec, jnp.array([vc]), None


def lda_c_pw_spin(n, zeta, A=(0.031091, 0.015545, 0.016887), fzeta0=1.709921, **kwargs):
    """Perdew-Wang parametrization of the correlation functional (spin-polarized).

    Corresponds to the functional with the label LDA_C_PW and ID 12 in Libxc.

    Reference: Phys. Rev. B 45, 13244.

    Args:
        n: Real-space electronic density.
        zeta: Relative spin polarization.

    Keyword Args:
        A: Functional parameters.
        fzeta0: Functional parameter.
        **kwargs: Throwaway arguments.

    Returns:
        PW correlation energy density and potential.
    """
    a1 = (0.2137, 0.20548, 0.11125)
    b1 = (7.5957, 14.1189, 10.357)
    b2 = (3.5876, 6.1977, 3.6231)
    b3 = (1.6382, 3.3662, 0.88026)
    b4 = (0.49294, 0.62517, 0.49671)

    ec0, vc0, _ = lda_c_pw(n, A[0], a1[0], b1[0], b2[0], b3[0], b4[0])  # Unpolarized
    ec1, vc1, _ = lda_c_pw(n, A[1], a1[1], b1[1], b2[1], b3[1], b4[1])  # Polarized
    ac, dac, _ = lda_c_pw(n, A[2], a1[2], b1[2], b2[2], b3[2], b4[2])  # Spin stiffness
    ac = -ac  # The PW spin interpolation is parametrized with -ac instead of ac
    vc0, vc1, dac = vc0[0], vc1[0], dac[0]  # Remove spin dimension for the correct shape

    fzeta = ((1 + zeta) ** (4 / 3) + (1 - zeta) ** (4 / 3) - 2) / (2 ** (4 / 3) - 2)
    zeta3 = zeta**3
    zeta4 = zeta3 * zeta

    ec = ec0 + ac * fzeta * (1 - zeta4) / fzeta0 + (ec1 - ec0) * fzeta * zeta4

    dac = -dac  # Also change the sign for the derivative
    dfzeta = ((1 + zeta) ** (1 / 3) - (1 - zeta) ** (1 / 3)) * 4 / (3 * (2 ** (4 / 3) - 2))
    factor1 = vc0 + dac * fzeta * (1 - zeta4) / fzeta0 + (vc1 - vc0) * fzeta * zeta4
    factor2 = ac / fzeta0 * (dfzeta * (1 - zeta4) - 4 * fzeta * zeta3) + (ec1 - ec0) * (
        dfzeta * zeta4 + 4 * fzeta * zeta3
    )

    vc_up = factor1 + factor2 * (1 - zeta)
    vc_dw = factor1 - factor2 * (1 + zeta)
    return ec, jnp.array([vc_up, vc_dw]), None


def lda_c_pw_mod_spin(n, zeta, **kwargs):
    """Modified Perdew-Wang parametrization of the correlation functional (spin-polarized).

    Corresponds to the functional with the label LDA_C_PW_MOD and ID 13 in Libxc.

    Reference: Phys. Rev. B 45, 13244.

    Args:
        n: Real-space electronic density.
        zeta: Relative spin polarization.

    Keyword Args:
        **kwargs: Throwaway arguments.

    Returns:
        Modified PW correlation energy density and potential.
    """
    # fzeta0 = 4 / (9 * (2**(1 / 3) - 1))
    return lda_c_pw_spin(
        n,
        zeta,
        A=(0.0310907, 0.01554535, 0.0168869),
        fzeta0=1.709920934161365617563962776245,
        **kwargs,
    )


def pbe_x_base(n, dn=None):
  """Base PBE exchange functional to be used in the spin-paired and -polarized case.

  Reference: Phys. Rev. Lett. 77, 3865.
  https://github.com/wangenau/eminus/blob/main/eminus/xc/gga_x_pbe.py
      
  Args:
    n (Float[Array, '*d x y z']): Real-space electron density.
      May include batch dimensions.
    dn (Float[Array, '*d x y z']): Real-space gradient of density.
      May include batch dimensions.

  Returns:
    Float[Array, '*d x y z']: PBE exchange energy density, potential, and vsigma.
      Preserves any batch dimensions from the input.
  """
  kappa = 0.804
  mu=0.2195149727645171

  breakpoint()
  if jnp.any(jnp.isnan(dn)):
      print("NaN exists in dn...")
      breakpoint()
  norm_dn = jnp.linalg.norm(dn, axis=1)
  kf = (3 * jnp.pi**2 * n) ** (1 / 3)
  # Handle divisions by zero
  # divkf = 1 / kf
  # divkf = jnp.divide(1, kf, out=jnp.zeros_like(kf), where=kf > 0)
  kf_ = jnp.where(kf > 0, kf, 1)
  divkf = jnp.where(kf > 0, 1 / kf_, 0)
  # Handle divisions by zero
  # s = norm_dn * divkf / (2 * n)
  # s = jnp.divide(norm_dn * divkf, 2 * n, out=jnp.zeros_like(n), where=n > 0)
  n_ = jnp.where(n > 0, n, 1)
  s = jnp.where(n > 0, norm_dn * divkf / 2 / n_, 0)
  f1 = 1 + mu * s**2 / kappa
  Fx = kappa - kappa / f1
  exunif = -3 * kf / (4 * jnp.pi)
  # In Fx a "1 + " is missing, since n * exunif is the Slater exchange that is added later
  sx = exunif * Fx

  dsdn = -4 / 3 * s
  dFxds = 2 * mu * s / f1**2
  dexunif = exunif / 3
  exunifdFx = exunif * dFxds
  vx = sx + dexunif * Fx + exunifdFx * dsdn  # dFx/dn = dFx/ds * ds/dn

  # Handle divisions by zero
  # vsigmax = exunifdFx * divkf / (2 * norm_dn)
  # vsigmax = jnp.divide(
  #   exunifdFx * divkf, 2 * norm_dn, out=jnp.zeros_like(norm_dn), where=norm_dn > 0
  # )
  norm_dn_ = jnp.where(norm_dn > 0, norm_dn, 1)
  vsigmax = jnp.where(norm_dn > 0, exunifdFx * divkf / 2 / norm_dn_, 0)
  return sx * n, jnp.array([vx]), vsigmax


def gga_x_pbe_spin(n, zeta, dn_spin=None, **kwargs):
    """Perdew-Burke-Ernzerhof parametrization of the exchange functional (spin-polarized).

    Corresponds to the functional with the label GGA_X_PBE and ID 101 in Libxc.

    Reference: Phys. Rev. Lett. 77, 3865.

    Args:
        n: Real-space electronic density.
        zeta: Relative spin polarization.

    Keyword Args:
        mu: Functional parameter.
        dn_spin: Real-space gradient of densities per spin channel.
        **kwargs: Throwaway arguments.

    Returns:
        PBE exchange energy density, potential, and vsigma.
    """
    # Use the spin-scaling relationship Exc(n_up, n_down)=(Exc(2 n_up)+Exc(2 n_down))/2
    # zeta = zeta[0]  # Getting the non-zero values from zeta adds an extra dimension, remove it here
    n_up = zeta * n + n  # 2 * n_up
    n_dw = -zeta * n + n  # 2 * n_down
    ex_up, vx_up, vsigma_up = pbe_x_base(n_up, 2 * dn_spin[0], **kwargs)
    ex_dw, vx_dw, vsigma_dw = pbe_x_base(n_dw, 2 * dn_spin[1], **kwargs)
    vx_up, vx_dw = vx_up[0], vx_dw[0]  # Remove spin dimension for the correct shape

    ex, vx, _ = lda_x_spin(n, zeta, **kwargs)

    vsigmax = jnp.array([vsigma_up, jnp.zeros_like(ex), vsigma_dw])
    return ex + 0.5 * (ex_up + ex_dw) / n, jnp.array([vx[0] + vx_up, vx[1] + vx_dw]), vsigmax


def gga_c_pbe_spin(n, zeta, dn_spin=None, **kwargs):
    """Perdew-Burke-Ernzerhof parametrization of the correlation functional (spin-polarized).

    Corresponds to the functional with the label GGA_C_PBE and ID 130 in Libxc.

    Reference: Phys. Rev. Lett. 77, 3865.

    Args:
        n: Real-space electronic density.
        zeta: Relative spin polarization.

    Keyword Args:
        beta: Functional parameter.
        dn_spin: Real-space gradient of densities per spin channel.
        **kwargs: Throwaway arguments.

    Returns:
        PBE correlation energy density, potential, and vsigma.
    """
    zeta = 0
    beta=0.06672455060314922
    gamma = (1 - jnp.log(2)) / jnp.pi**2

    pi34 = (3 / (4 * jnp.pi)) ** (1 / 3)
    rs = pi34 * n ** (-1 / 3)
    norm_dn = jnp.linalg.norm(dn_spin[0] + dn_spin[1], axis=1)
    ec, vc, _ = lda_c_pw_mod_spin(n, zeta, **kwargs)
    vc_up, vc_dw = vc

    kf = (9 / 4 * jnp.pi) ** (1 / 3) / rs
    ks = jnp.sqrt(4 * kf / jnp.pi)
    phi = ((1 + zeta) ** (2 / 3) + (1 - zeta) ** (2 / 3)) / 2
    phi2 = phi**2
    phi3 = phi2 * phi
    t = norm_dn / (2 * phi * ks * n)
    expec = jnp.exp(-ec / (gamma * phi3))
    A = beta / (gamma * (expec - 1))
    t2 = t**2
    At2 = A * t2
    A2t4 = At2**2
    divsum = 1 + At2 + A2t4
    div = (1 + At2) / divsum
    nolog = 1 + beta / gamma * t2 * div
    gec = gamma * phi3 * jnp.log(nolog)

    # Handle divisions by zero
    with jnp.errstate(divide="ignore", invalid="ignore"):
        dfz = ((1 + zeta) ** (-1 / 3) - (1 - zeta) ** (-1 / 3)) / 3
    dfz = jnp.nan_to_num(dfz, nan=0, posinf=0, neginf=0)
    factor = A2t4 * (2 + At2) / divsum**2
    bfpre = expec / phi3
    bf_up = bfpre * (vc_up - ec)
    bf_dw = bfpre * (vc_dw - ec)
    dgecpre = beta * t2 * phi3 / nolog
    dgec_up = dgecpre * (-7 / 3 * div - factor * (A * bf_up / beta - 7 / 3))
    dgec_dw = dgecpre * (-7 / 3 * div - factor * (A * bf_dw / beta - 7 / 3))
    dgeczpre = (
        3 * gec / phi
        - beta * t2 * phi2 / nolog * (2 * div - factor * (3 * A * expec * ec / phi3 / beta + 2))
    ) * dfz
    dgecz_up = dgeczpre * (1 - zeta)
    dgecz_dw = -dgeczpre * (1 + zeta)
    gvc_up = gec + dgec_up + dgecz_up
    gvc_dw = gec + dgec_dw + dgecz_dw

    vsigma = beta * phi / (2 * ks * ks * n) * (div - factor) / nolog
    vsigmac = jnp.array([0.5 * vsigma, vsigma, 0.5 * vsigma])
    return ec + gec, jnp.array([vc_up + gvc_up, vc_dw + gvc_dw]), vsigmac