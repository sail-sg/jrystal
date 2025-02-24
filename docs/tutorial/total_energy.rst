.. _total_energy:

Calculating Total Energy with Plane Waves
=========================================


In this tutorial, we derive the total energy functional using a **Plane Wave Basis Set** within the framework of **Density Functional Theory (DFT)**. This approach is widely used in computational materials science due to its efficiency for periodic systems.


The Plane Wave Basis Set
------------------------

For a periodic system, Bloch's theorem allows us to express the wavefunction as a product of a plane wave and a periodic function :math:`u_{i, \boldsymbol{k}}(\boldsymbol{r})`. Expanding :math:`u_{i, \boldsymbol{k}}(\boldsymbol{r})` in the plane wave basis set, we have

.. math::

  u_{i, \boldsymbol{k}}  (\boldsymbol{r}) = \dfrac{1}{  \sqrt{\Omega_\text{cell}}}    \sum_{\boldsymbol{G}} c_{i, \boldsymbol{k}, \boldsymbol{G} }	  \exp(\text{i} \boldsymbol{G}\cdot \boldsymbol{r} )

where
  * :math:`\boldsymbol{r}` Position vector in real space;
  * :math:`i`: Band index;
  * :math:`\boldsymbol{k}`: Wavevector in the Brillouin zone;
  * :math:`\boldsymbol{G}`: Reciprocal lattice vector (indexes plane waves); 
  * :math:`c_{i, \boldsymbol{k}, \boldsymbol{G} }`: Expansion coefficients (complex numbers);
  * :math:`\Omega_\text{cell}`: Volume of the unit cell.

The complete wave function can be then written as

.. math::

  \psi_{i, \boldsymbol{k}}  (\boldsymbol{r}) = \exp(\text{i} \boldsymbol{k}\cdot \boldsymbol{r} ) u_{i, \boldsymbol{k}}  (\boldsymbol{r}).


Unitarity Constraint
--------------------

Density Functional Theory (DFT) employs single-particle wave functions under the assumption of orthonormality. This requirement translates to a constraint on the plane wave expansion coefficients $c_{i, \boldsymbol{k}, \boldsymbol{G}}$, which must satisfy the orthogonality condition at eacheach $k$-point:

.. math::
  
  \sum_{\boldsymbol{G}} c_{i, \boldsymbol{k}, \boldsymbol{G}}^* c_{j, \boldsymbol{k}, \boldsymbol{G}} = \delta_{ij}, \quad \forall \ \boldsymbol{k}.

In our implementation, the orthonormality of the wave functions within the plane wave basis set is enforced via a QR decomposition. This is performed using the `jax.numpy.linalg.qr` function, which employs the Householder transformation—a numerically stable method for orthogonalization.


Total Energy as the Objective Function
-------------------------------------

The objective is to minimize the total energy of the system with respect to the plane wave coefficients.

.. math::
  
  \begin{align}
  \min_{\psi_{i, \boldsymbol{k}}} E_{\text{total}}[\left\{ \psi_{i, \boldsymbol{k}} \right\}]
  \end{align}

where the :math:`E_{\text{total}}` is the total energy functional of wave functions. The total energy functional in density functional theory (DFT) is constructed by the kinetic energy functional :math:`E_{\text{kinetic}}`, the electron-ion interaction energy functional, or the external potential energy functional :math:`E_{\text{external}}`, the electron-electron interaction energy functional, also know as the Hartree energy functional :math:`E_{\text{hartree}}`, and the exchange-correlation energy functional :math:`E_{\text{xc}}`, and the nuclear-nuclear interaction energy functional :math:`E_{\text{nuclear}}`:


.. math::
  E_{\text{total}} = E_{\text{kinetic}} + E_{\text{external}} + E_{\text{hartree}} + E_{\text{xc}} + E_{\text{nuclear}}.


The Kinetic Energy Functional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The kinetic energy functional is given by

.. math::
  \begin{align}
  E_{\text{kinetic}} &= \dfrac{1}{2} \sum_{i, \boldsymbol{k}} \int_{\Omega_\text{cell}} \psi_{i, \boldsymbol{k}} (\boldsymbol{r})  \nabla_{\boldsymbol{r}} \psi_{i, \boldsymbol{k}} (\boldsymbol{r}) d\boldsymbol{r}.  \\
  &= \dfrac{1}{2} \sum_{i, \boldsymbol{k}} \sum_{\boldsymbol{G}} \left\Vert \boldsymbol{k}+\boldsymbol{G} \right\Vert^2 \left\Vert c_{i, \boldsymbol{k}, \boldsymbol{G}} \right\Vert^2.  \\
  \end{align}

The External Potential Energy Functional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The external potential energy functional is given by

.. math::
  E_{\text{external}} =  

