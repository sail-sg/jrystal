# Band Structure

## Solid state basics
Solid-state physics mainly deals with crystalline solids, which has a periodic structure that can be described by a Bravais lattice. A crystal can be specified by the cell vectors $\vb{a}_{i}$ of the unit cell, and the atomic configuration $\{Z_\ell, \vb*{\tau}_\ell  \}_\ell$ within the unit cell, where $Z_\ell$ is the charge of atom $\ell $ and $\vb*{\tau}_\ell $ is the coordinate. To capture interaction between different translated copies of the unit cells, a finite Bravais lattice with periodic boundary condition (PBC) is typically used. The finite Bravais lattice is commonly referred to as the \emph{simulation cell}.

Different from molecular systems, in periodic systems the potential is periodic over the unit cell, since at each location $\vb{r}$ within the unit cell, potentials from all translated unit cells are felt. In other words, the tiling periodizes the non-periodic atomic Hartree and external potential. Bloch theorem [1] states that, for periodic potential $V(\vb{r} + \vb{R}_{\vb{m}})=V(\vb{r})$, the eigenstates of the Hamiltonian $\hat{H}$ takes the form
\begin{equation} \label{eqn:bloch}
  \psi _{i, \vb{k}}(\vb{r}) = \exp [\text{i} \vb{k} ^{\top} \vb{r}] u_{i, \vb{k}}(\vb{r}), \quad \vb{k} = \sum_{d=1}^3 k_{d} \vb{b}_{d}
\end{equation}
where $n$ is the band index and $u_{n, \vb{k}}$ is a function periodic over the unit cell.

To make sure that $\psi _{n, \vb{k}}$ is periodic over the simulation cell of size $M_1\times M_2\times M_3$, the k-points $\vb{k}$ can only take values from the lattice $k_{d}=m_{d} / M_{d}, m_{d}\in \mathbb{Z}$. Furthermore, $\vb{k}$ within the first Brillouin zone (FBZ), i.e. $m_{i}\in [-\lfloor(M_{i}-1) / 2 \rfloor, \lfloor M_{i} / 2 \rfloor]$, gives all unique eigenvalues due to the periodicity in the reciprocal space. All $\vb{k}$ within FBZ forms a reciprocal lattice with size $M_1\times M_2\times M_3$. Thus for periodic systems, the KS equation becomes
\begin{equation} \label{eqn:ks-eqn-periodic}
\hat{H}[\rho]\psi_{i,\vb{k}}  = \epsilon _{i,\vb{k}} \psi_{i,\vb{k}}.
\end{equation}
For each $i$, there are distinct energy levels for each $\vb{k}$, and the collection $\epsilon _{i,\vb{k}}$ for fixed $i$ forms a line that are commonly referred to as the $i$-th *band*. Analogous to the HOMO-LUMO gap in molecular systems, the narrowest gap between the highest occupied band and the lowest unoccupied band are referred to as the band gap, which is an important indicator of the electronic conductivity of the system.
The calculation of a collection of orbital energies within some selected bands $I$, $\{\epsilon _{i,\vb{k}}\}_{i\in I}$ is commonly refer to as *band structure* calculation. 

## K-space decoupling
In practical calculation, Galerkin approximation is usually applied to the periodic part of the orbital $u_{i, \vb{k}}$, and periodic basis are used to make sure that $u_{i , \vb{k}}$ is periodic. Multiplying $e^{-\text{i} \vb{k} \vb{r}}$ to both side of the periodic KS equation, we get
\begin{equation}
  \hat{H}_{\vb{k}}[\rho] u_{i, \vb{k}} = \epsilon _{i, \vb{k}} u_{i, \vb{k}}
\end{equation}
where 
\begin{equation}
\hat{H}_{\vb{k}}[\rho] := e^{-\text{i} \vb{k} \vb{r}}\hat{H}[\rho]e^{\text{i} \vb{k} \vb{r}}.
\end{equation}
and $\mathcal{K}$ is the k-path. The periodic KS equation can be solved at each k-point $\vb{k}$ independently by solving the above eigenvalue problem. This is because orbitals with different k are always orthogonal due to Bloch theorem, which makes the Hamiltonian matrix $\hat{H}[\rho]$ block diagonal. Therefore diagonalizing each block $\hat{H}_{\vb{k}}[\rho]$ solves a part of the original eigenvalue problem.

## Discretization and Direct optimization
Given a basis set $\{\phi_{\alpha}\}$, denote the coefficient as $c_{i, \vb{k}, \alpha }=\braket{\phi _{\alpha }}{u_{i,\vb{k}}}$, and the Hamiltonian matrix is discretized into the coefficient space as
\begin{equation}
  H_{\vb{k}, ij}[\rho]
  =\mel{u  _{i, \vb{k}}}{\hat{H}_{\vb{k}}[\rho]}{u _{j, \vb{k}}}
  = \sum_{\alpha \beta } c^{*}_{i, \vb{k}, \alpha } c_{j, \vb{k}, \beta  } H_{\vb{k}, \alpha \beta }[\rho].
\end{equation}

At this point, the band structure calculation becomes a collection of $\abs{\mathcal{K}}$ finite dimensional eigenvalue problems. One can solve them via direct optimization like in [2], where the trace of each $H_{\vb{k}, ij}[\rho]$ is minimized. That is, we solve the following optimization problem:
\begin{equation}
\min_{\vb{c}_{\vb{k}}} \tr H_{\vb{k}, ij}[\rho^*]
\end{equation}
where $\rho^*$ is the ground state density solved in a total energy calculation. In principle one need to update $\rho$ self-consistently since we are changing $\vb{c}$, however this is expensive and in practice the above approximation is usually applied. This approximation is commonly referred to as the Non-self-consistent-field (NSCF) step.

## Reference
[1] Bloch, Felix Über Die Quantenmechanik Der Elektronen in Kristallgittern, Zeitschrift für Physik 52 (1929).

[2] Tianbo Li and Min Lin and Zheyuan Hu and Kunhao Zheng and Giovanni Vignale and Kenji Kawaguchi and A. H. Castro Neto and Kostya S. Novoselov and Shuicheng Yan D4FT: A Deep Learning Approach to Kohn-Sham Density Functional Theory, The Eleventh International Conference on Learning Representations, {ICLR} 2023, Kigali, Rwanda, May 1-5, 2023  (2023).
 
 
