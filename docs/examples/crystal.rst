Create A Crystal Structure
==========================

The crystal structure is organized by the :class:`Crystal` class.

Suppose we want to create a crystal structure of carbon with two atoms in a cubic unit cell, we can do the following:

.. code-block:: python

  from jrystal import Crystal
  diamond = Crystal(
    charges=[6, 6],  # two carbon atoms
    positions=[[0, 0, 0], [1.5, 1.5, 1.5]],  # two atoms in the unit cell
    cell_vectors=[[3, 0, 0], [0, 3, 0], [0, 0, 3]]  # cubic unit cell
  )
  

.. note::
    ``jrystal`` uses atomic units, where:
    
    * Length is measured in Bohr radii (1 Bohr = 0.529177 Å)
    * Energy is measured in Hartree (1 Hartree = 27.211 eV)
    
