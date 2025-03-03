.. _tutorial-array-shape:

What Are the Shapes of the Variable in ``jrystal``?
====================================================


The core idea behind the axis ordering is that axes which can be processed in parallel are placed at the beginning of the array shape. This ordering maximizes computational efficiency in two ways: it enables parallel calculations across these dimensions and leverages JAX's vectorization capabilities (``jax.vmap`` and ``jax.pmap``).


The shape of the variables in ``jrystal`` and the order of the axes follows  convention:

.. image:: images/axes.png
   :width: 400
   :align: center

- Spin axis: the dimenstion of this axis can be 1 or 2. If spin-spin_restricted calculation is performed, the dimension is 1, otherwise it is 2.
- K-point axis: the dimension of this axis is the number of k-points.
- Orbital (Band) axis: the dimension of this axis is the number of orbitals.
- the x, y, z axes in real or reciprocal space: the dimension of this axis is the number of grid points in the corresponding dimension.


The following is an example of the shape of the variables:

The 



