.. _total_energy:
====================================
Total Energy Minimization
====================================

In this document, we start from the objective function of solid state DFT, starting from the abstract form and derive it until it is implementable. Solid state DFT solves the following energy minimization problem (we leave why the objective looks like this in the preliminaries of DFT and crystalgraphy)

.. math::

   \begin{align}
   &\min_{f,\psi_{ik}} E_\text{total}[\{\psi_{ik}\}, f] \\
   s.t.\quad & \psi_{ik}(r)=e^{ikr}u_{ik}(r); \\
   & \langle u_{ik}|u_{jk}\rangle=\delta_{ij}; \\
   & \sum_{k=1}^K\sum_{i=1}^I f_{ik} = N; \; f_{ik}\in [0, 1]
   \end{align}

The total energy functional is a functional of a set of wave functions :math:`\{\psi_{ik}\}` and the occupation vectors :math:`f`. :math:`k\in K` and :math:`i\in[1..I]`, we call :math:`I` the number of bands and :math:`K` the set of k points, and :math:`N` is the number of electrons in the system. :math:`f` specifies how the :math:`N` electrons are distributed over the k points and bands. With the wave function and occupation vectors, we can calculate the electron density function as :math:`\rho(r)=\sum_i^I\sum_{k\in K} f_{ik}\psi^2_{ik}(r)`. As for all KS DFT calculations, the energy functional contains four terms, kinetic energy, external energy, hartree energy, and exchange-correlation energy.

.. math::

   E_\text{total}[\{\psi_{ik}\},f] = E_\text{kin}[\{\psi_{ik}\},f] +
   E_\text{ext}[\rho] + E_\text{har}[\rho] + E_\text{xc}[\rho]

And notice that except that the kinetic energy term is a direct functional of the wave functions and occupation, all other energies are direct functionals of the electron density (which in term depends on the wave function and occupation). We write below the definition of each energy term except the exchange-correlation functional. As the crystal system is periodic over the entire :math:`R^3` space, we only calculate a fraction of the energy within a single unit cell :math:`\Omega`.

.. math::

   E_\text{kin}[\{\psi_{ik}\},f]=-\frac{1}{2}\sum_i\sum_k f_{ik}\int_{\Omega} \psi^*_{ik}(r)\nabla^2_r\psi_{ik}(r) dr

.. math::

   E_\text{ext}[\rho] = -\sum_a \int_{\Omega} \rho(r) \frac{Z_a}{r-R_a}dr

.. math::

   E_\text{har}[\rho] = \frac{1}{2}\int_\Omega dr \int dr'\rho(r)\frac{1}{r-r'}\rho(r')

To this point, we have introduced DFT as an optimization problem with constraints in the function space.

- The objective functional: :math:`E_\text{total}[\{\psi_{ik}\},f]`, which can be expanded into the above terms.
- The parameter: :math:`f` and :math:`\psi_{ik}`.
- The constraints:

  - :math:`\psi_{ik}(r)=e^{ikr}u_{ik}(r)`, where :math:`u_{ik}` is periodic over the unit cell, and :math:`\langle u_{ik}|u_{jk}\rangle=\delta_{ij}`.
  - :math:`f_{ik}\in [0,1]` and :math:`\sum_k \sum_i f_{ik}=N`.

To make the problem computable, we just need to parameterize :math:`u_{ik}` and :math:`f` in a way that satisfy the constraints, plugging them back into the objective function and then perform the optimization in the parameter space. This is what we will do in the rest of this document.


Parameterizing :math:`u_{ik}(r)` and :math:`f`
---------------------------------------------

In planewave calculations, :math:`u_{ik}` is parameterized as a linear combination over fourier components of different frequencies

.. math::

   u_{ik}(r)=\frac{1}{\sqrt{\Omega_\text{cell}}}\sum_{G} c_{ikG} e^{iGr}

where we limit the :math:`G` in :math:`e^{iGr}` to be frequency components that is periodic over the unit cell, making :math:`u_{ik}(r)` satisfy the periodic constraint. At the same time, plugging the planewave back to :math:`\langle u_{ik}|u_{jk}\rangle=\delta_{ij}`, we translate the orthogonality constraint into :math:`\sum_G c^*_{ikG}c_{jkG}=I_{ij}`. A orthogonal matrix can be easily generated via reparameterization, for example, with the QR decomposition

.. code:: python

   key = jax.random.PRNGKey(0)
   w = jax.random.normal(key, (num_K, num_G, I))
   c, _ = jnp.linalg.qr(w)


The other parameter :math:`f` needs to satisfy :math:`f_{ik}\in [0,1]` and :math:`\sum_k \sum_i f_{ik}=N`. Similarly, we can use reparameterize it as

.. code:: python

   key = jax.random.PRNGKey(0)
   v = jax.random.normal(key, (I*num_K, N*num_K))
   Q, _ = jnp.linalg.qr(v)  # Q has shape (I*num_K, N*num_K) and Q.T @ Q = I
   f = jnp.diag(Q @ Q.T).reshape(I, num_K)  # f has shape (I*num_K)

It is easily verifiable that :math:`\sum_{ik}f_{ik}=\Tr(QQ^\top)=\Tr(Q^\top Q)=N` and :math:`f_{ik}=\|Q_{i*|K|+k}\|^2\in[0,1]`


Casting into the parameter space
--------------------------------

Now that :math:`u_{ik}` becomes a function parameterized by :math:`c`, we can substitute it back to the energy terms to cast the energy into a function of the finite dimensional parameters :math:`c` and :math:`f`.

The Kinetic Energy
^^^^^^^^^^^^^^^^^^

Firstly, we apply the kinetic operator on the parameterized wave function

.. math::

   \nabla^2_r\psi_{ik}(r) = \nabla^2\left[\frac{1}{\sqrt{\Omega_\text{cell}}}e^{ikr}\sum_{G} c_{ikG}e^{iGr}\right] = \nabla^2\left[\frac{1}{\sqrt{\Omega_\text{cell}}}\sum_{G} c_{ikG}e^{i(k+G)r}\right] = -\|k+G\|^2\psi_{ik}(r)

The kinetic energy is then reduced to the following using the property that :math:`\int_{\Omega} \psi^*_{ik}(r)\psi_{jk}(r) dr=\delta_{ij}`.

.. math::

   \begin{align}
   E_\text{kin}[\{\psi_{ik}\},f]=&\frac{1}{2}\sum_i\sum_k f_{ik}\int_{\Omega} \psi^*_{ik}(r)\nabla^2_r\psi_{ik}(r) dr \\
   =& \frac{1}{2}\sum_i\sum_{k}\sum_G f_{ik} c_{ikG}^2\|k+G\|^2 \int_{\Omega} \psi^*_{ik}(r)\psi_{ik}(r) dr \\
   =& \frac{1}{2}\sum_i\sum_{k}\sum_G f_{ik} c_{ikG}^2\|k+G\|^2
   \end{align}


The External Energy
^^^^^^^^^^^^^^^^^^^

.. math::

   \begin{align}
   E_\text{ext}[\rho] &= -\sum_a \int_{\Omega} \rho(r) \frac{Z_a}{r-R_a}dr \\
   &= - 4\pi  \sum_{\boldsymbol{G}_\boldsymbol{n} \neq \boldsymbol{0}}  \tilde{\rho}  (\boldsymbol{G}_\boldsymbol{n}) \sum_\ell e^{ \text{i}\boldsymbol{G}_\boldsymbol{n} \tau_\ell}  \dfrac{q_\ell}{ \Vert \boldsymbol{G}_\boldsymbol{n} \Vert^2}
   \end{align}


The Hartree Energy
^^^^^^^^^^^^^^^^^^

.. math::

   E_\text{har}[\rho] = \frac{1}{2}\int_\Omega dr \int dr'\rho(r)\frac{1}{r-r'}\rho(r')
