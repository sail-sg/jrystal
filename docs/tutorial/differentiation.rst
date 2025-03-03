.. _tutorial-differentiation:

What Can Differentiation Do for Density Functional Theory
=======================================================


So what can Automatic Differentiation (AD) do for Density Functional Theory? Short answer is 


AD for Simply The Calculation of Kohn-Sham Hamiltonian matrix
-----------------------------------------------------------


Given a ground-state density :math:`\rho(\mathbf{r})`, the Kohn-Sham Hamiltonian matrix is defined as 

.. math::

  H_{ij} = < \psi_i | \hat{H}[\rho] | \psi_j >


where :math:`\psi_i` is the Kohn-Sham orbital. Although the analytical form of such a matrix is available, we don't have to implement it on our own, which may cost us mayby quite some time. As long as we have a kohn-sham total energy functional (sum of the kohn-sham eigenvalues), we can use AD to calculate the Hamiltonian matrix, as we know that it is simple the Hessian of the total energy functional with respect to the orbital coefficients.

Well, it is not as simple as it looks like. The most challenging part is that kohn-sham total energy functional is not a **holomorphic function** with respect to the orbital coefficients. 




















