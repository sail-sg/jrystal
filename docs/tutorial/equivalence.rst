Equivalence Between Total Energy Minimization and Solving Kohn-Sham Equation
==============================================================================

In this tutorial, we demonstrate that minimizing the total energy in density functional theory (DFT) is mathematically equivalent to solving the Kohn-Sham equations. This equivalence underpins direct minimization methods, an alternative to conventional diagonalization-based approaches for DFT calculations.


.. note::
    For background on total energy minimization, refer to the tutorial :ref:`total_energy`.

.. note::
    Assumptions:
        - Occupation numbers are omitted (i.e., all orbitals are singly occupied).
        - Spin-spin_restricted formalism is used. For generalization with occupation numbers, see [Li2024]_.

Total Energy Functional
^^^^^^^^^^^^^^^^^^^^^^^
The total energy functional for a system of non-interacting electrons is expressed as:

.. math::
    E_{\text{total}}[\{\psi_i\}] = T_s[\{\psi_i\}] + E_{\text{Hxc}}[n] + \int V_{ext}(r) n(r) dr

where

1. Kinetic energy (:math:`T_s`): The kinetic energy of non-interacting electrons:

.. math::
    T_s[\{\psi_i\}] = \frac{1}{2} \sum_{i} \int \psi_i^*(r)  \nabla \psi_i(r) dr

2. Hartree-exchange-correlation energy (:math:`E_{\text{Hxc}}`): The Hartree-exchange-correlation energy of the system:

.. math::
    E_{\text{Hxc}}[n] = \dfrac12 \int \dfrac{n(r) n(r')}{|r - r'|} dr dr'  + \int \varepsilon_{\text{xc}}(\boldsymbol{r})  n(\boldsymbol{r}) dr

3. External potential (:math:`V_{ext}`): Represents interactions with ions or applied fields.

The electron density :math:`n(\mathbf{r})` is:

.. math::
    n(\mathbf{r}) = \sum_i |\psi_i(\mathbf{r})|^2

Constrained Minimization
^^^^^^^^^^^^^^^^^^^^^^^
The orbitals :math:`\psi_i(\mathbf{r})` must satisfy orthonormality:

.. math::
    \int \psi^*_i(\mathbf{r}) \psi_j(\mathbf{r}) d\mathbf{r} = \delta_{ij}

To enforce this, we use Lagrange multipliers :math:`\lambda_{ij}` and construct the Lagrangian:

.. math::
    L[{\psi_i}, {\lambda_{ij}}] = E_{\text{total}}[{\psi_i}] - \sum_{i,j} \lambda_{ij} \left( \int \psi^*_i(\mathbf{r}) \psi_j(\mathbf{r}) d\mathbf{r} - \delta_{ij} \right)

Functional Derivatives and Stationarity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Taking the functional derivative of the Lagrangian :math:`L` with respect to :math:`\psi_i^*(\mathbf{r})` (conjugate variable) gives:

.. math::
    \frac{\delta L}{\delta \psi^*_i (\mathbf{r})} = \frac{\delta T_s}{\delta \psi^*_i(\mathbf{r})} + \frac{\delta E_{\text{Hxc}}}{\delta \psi^*_i(\mathbf{r})} + \frac{\delta\int V_{\text{ext}} n(\mathbf{r})d\mathbf{r} }{\delta \psi^*_i(\mathbf{r})} - \sum_j \lambda_{ij} \psi_j(\mathbf{r}) = 0

Let's break down each term:

- Kinetic Term:

.. math::
    \frac{\delta T_s}{\delta \psi_i^*(\mathbf{r})} = -\frac{1}{2} \nabla^2 \psi_i(\mathbf{r})

- Hartree-XC Term:

.. math::
    \frac{\delta E_{\text{Hxc}}}{\delta \psi_i^*(\mathbf{r})} = \int \frac{n(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|}  d\mathbf{r}' + V_{\text{xc}}(\mathbf{r})

The first terms of the potential is also known as the Hartree potential:

.. math::
    \begin{equation}
        V_H(\mathbf{r}) := \int \frac{n(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|}  d\mathbf{r}'
    \end{equation}

- External Potential Term:
.. math::
    \begin{align}
    \dfrac{\delta E_{\text{ext}}}{\delta \psi_i^*(\mathbf{r})} \int V_{\text{ext}}(\mathbf{r}) n(\mathbf{r}) d\mathbf{r} &= V_{\text{ext}}(\mathbf{r}) \psi_i(\mathbf{r})
    \end{align}

Kohn-Sham Equation
^^^^^^^^^^^^^^^^^^

Combining all terms, the stationary condition becomes:

.. math::
    \left[ -\frac{1}{2} \nabla^2 + V_H(\mathbf{r}) + V_{\text{xc}}(\mathbf{r}) + V_{\text{ext}}(\mathbf{r}) \right] \psi_i(\mathbf{r}) = \sum_j \lambda_{ij} \psi_j(\mathbf{r})

The matrix :math:`\lambda_{ij}` is Hermitian due to orthonormality constraints. By choosing a unitary transformation that diagonalizes :math:`\lambda_{ij}`, we obtain the Kohn-Sham eigenvalue equation:

.. math::
    \hat{H}_{\text{KS}} \psi_i(\mathbf{r}) = \varepsilon_i \psi_i(\mathbf{r})

where the Kohn-Sham Hamiltonian is:

.. math::

    \hat{H}_{\text{KS}} = -\frac{1}{2} \nabla^2 + V_H(\mathbf{r}) + V_{\text{xc}}(\mathbf{r}) + V_{\text{ext}}(\mathbf{r}).


References
^^^^^^^^^^
.. [Li2024] Li, Tianbo, et al. "Diagonalization without Diagonalization: A Direct Optimization Approach for Solid-State Density Functional Theory." arXiv preprint arXiv:2411.05033 (2024).