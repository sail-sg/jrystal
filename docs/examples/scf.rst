Self-Consistent Field Calculation
=================================


In this example, we will perform a self-consistent field (SCF) calculation using ``jrystal``.


1. Create a crystal structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  import jax
  import jax.numpy as jnp
  import jrystal as jr

  crystal = jr.Crystal.create_from_file("INSTALL_PATH/geometry/examples/diamond.xyz")
  

2. 
  



