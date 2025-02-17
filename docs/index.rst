jrystal
=======================================================================


.. automodule:: jrystal._src.braket
   :members:

.. automodule:: jrystal._src.hamiltonian
   :members:

.. automodule:: jrystal._src.energy
   :members:

.. automodule:: jrystal._src.grid
   :members:

.. automodule:: jrystal._src.occupation
   :members:

.. automodule:: jrystal._src.potential
   :members:

.. automodule:: jrystal._src.pw
   :members:

.. automodule:: jrystal._src.utils
   :members:

.. automodule:: jrystal.calc
   :members:

This project is a `JAX <https://github.com/google/jax/>`_-based package for differantiable density functional theory computation of solids.


.. _installation:

Installation
------------

To use :guilabel:`jrystal`, first install it using pip:

.. code-block:: console

   $ pip install jrystal

Getting Started
###############

Currently we support the following calculation methods:

* vanilla self-consistent field method.
* stochastic self-consistent field method.
* stochasic gradient-based optimizers supported by `Optax <https://optax.readthedocs.io/en/latest/>`_.


Examples
########


Theory
########
We include a detailed mathematical derivation of density functional theory from scratch, which is useful for those who do not have related background. 


Support 
-------


The Team
########

This project is developed by `SEA AI LAB (SAIL) <https://sail.sea.com/>`_. We are also grateful to researchers from `NUS I-FIM <https://ifim.nus.edu.sg/>`_ for contributing ideas and theoretical support.  

.. image:: images/sail_logo.png
   :width: 300
   :alt: Alternative text

.. image:: images/ifim_logo.png
   :width: 300
   :alt: Alternative text


Citation
########

.. code-block:: console

TODO: add citation here



License
-------



Contents
--------

.. toctree::
   :maxdepth: 2

   examples/index
