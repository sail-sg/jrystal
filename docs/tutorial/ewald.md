
# Table of Contents

1.  [Nucleus energy](#org970dedb)
2.  [Reciprocal space representation via the Yukawa kernel](#orgdc95a51)
3.  [Problem](#org4a23c9c)



<a id="org970dedb"></a>

# Nucleus energy

Denote the atomic point charge within the unit cell as

\begin{equation}
\rho ^{\text{atom}}(\vb{r})=-\sum_{\ell} Z_{\ell }\delta (\vb{r}-\vb*{\tau }_{\ell })
\end{equation}

The Coulombic potential generated from this charge is

\begin{equation}
\begin{split}
V_{\text{ext}}(\vb{r}) =& \rho ^{\text{atom}}  \star \frac{1}{r}
= \int_{\Omega + \vb{R} } \dd{\vb{r}'} \frac{1}{\norm{\vb{r} - \vb{r}'}} \left( -\sum_{\ell} Z_{\ell }\delta (\vb{r}'-\vb*{\tau }_{\ell }) \right) \\
=&  - \sum_{\vb{R}} \sum_{\ell} Z_{\ell } \frac{1}{\norm{\vb{r}  - \vb*{\tau }_{\ell } - \vb{R}}}
\end{split}
\end{equation}

The nucleus energy is then

\begin{equation}
E_{\text{nuc}}
= \braket{V_{\text{ext}}}{\rho ^{\text{atom}}}
=- \sum_{\vb{R}} \sum_{\ell'} \sum_{\ell} Z_{\ell '}Z_{\ell } \frac{1}{\norm{\vb*{\tau }_{\ell' }  - \vb*{\tau }_{\ell } - \vb{R}}}
\end{equation}

There are two problem with the above formula:

1.  In practical calculation the summation over $\vb{R}$ is truncated to a finite sized Bravais lattice, which need to be very big for the energy to converge
2.  A uniform negative background charge $\rho^{-}=\vb{Z}_{\text{tot}} / \Omega$ is required for the energy to be convergent, which is hard to apply in real space.


<a id="orgdc95a51"></a>

# Reciprocal space representation via the Yukawa kernel

The Fourier space representation of the electrostatic potential can then be determined by the Poisson equation in Fourier space:

\begin{equation}
  \nabla ^{2} V(\vb{r}) = -4\pi \rho (\vb{r}) \Rightarrow
  -\norm{\vb{G}}^{2} \tilde{V} (\vb{G}) = -4\pi \tilde{\rho} (\vb{G}) \Rightarrow
  \tilde{V} (\vb{G}) = \frac{4\pi}{\norm{\vb{G}}^{2}} \tilde{\rho}  (\vb{G}).
\end{equation}

Note that at $\vb{G}=\vb{0}$ we have a singularity, so this ill-defined.

To fix this, we introduce the Yukawa kernel $\nu_{\alpha }(r)=\frac{e^{-\alpha r}}{r}$, which represents a screened Coulomb interaction.
It&rsquo;s Fourier representation is given by

\begin{equation}
\tilde{\nu} ^{\alpha }(\vb{G})= \int_{\mathbb{R}^{3}} \dd{\vb{r}} \nu_{\alpha }(\vb{r}) e^{-\text{i} \vb{G} \cdot \vb{r}} = \dfrac{4\pi}{\norm{\vb{G}}^2 + \alpha^2}.
\end{equation}

This is equivalent to treat Fourier transform as the limit of the Laplace transform.

Now for any charge distribution $\rho$, we can define the Fourier transform of Coulombic potentials generated from $\rho$ as a limit, since $\lim_{\alpha  \to 0} -\frac{1}{4\pi} \nu _{\alpha }=G_{\text{coulomb}}$:

\begin{equation}
  \tilde{V} (\vb{G})=\lim_{\alpha  \to 0}  [\widetilde{-\frac{1}{4\pi}\nu_{\alpha } \star -4\pi \rho }] (\vb{G}) =\lim_{\alpha  \to 0}  \tilde{\nu}_{\alpha } (\vb{G}) \tilde{\rho} (\vb{G})  = \lim_{\alpha  \to 0}  \frac{4\pi \tilde{\rho} (\vb{G})}{\norm{\vb{G}}^{2}+ \alpha ^{2}}.
\end{equation}

$V_{\text{ext}}$ can now be represented in the reciprocal space via the Yukawa kernel:

\begin{equation}
  \tilde{V}_{\text{ext}} (\vb{G}) =\lim_{\alpha  \to 0} \frac{1}{4\pi} \tilde{\nu}_{\alpha } (\vb{G}) \tilde{n}^{\text{atom}} (\vb{G})
= \lim_{\alpha  \to 0}  \sum_{\ell} \frac{ Z_{\ell } e^{-\text{i} \vb{G} \cdot \bm{\tau }_{\ell}}}{\norm{\vb{G}}^{2}+ \alpha ^{2}}
= \lim_{\alpha  \to 0}  \sum_{\ell} \frac{ Z_{\ell } e^{-\text{i} \vb{G} \cdot \bm{\tau }_{\ell}}}{\norm{\vb{G}}^{2}}.
\end{equation}


<a id="org4a23c9c"></a>

# Problem

