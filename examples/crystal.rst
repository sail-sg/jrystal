========================================
Create A Crystal Structure
========================================


The :class:`Crystal` class is the fundamental building block for representing crystal structures in jrystal. It contains essential attributes and derived properties that fully describe a crystal system.

Core Attributes
-------------

* ``charges``: Atomic numbers of each atom
* ``positions``: Cartesian coordinates of atoms 
* ``cell_vectors``: Unit cell vectors
* ``spin``: Number of unpaired electrons
* ``symbol``: Chemical symbols of atoms

Derived Properties
----------------

* ``scaled_positions``: Fractional coordinates of atoms
* ``vol``: Unit cell volume
* ``num_atom``: Total number of atoms
* ``num_electron``: Total number of electrons
* ``reciprocal_vectors``: Reciprocal lattice vectors
* ``A``: Alias for ``cell_vectors``
* ``B``: Alias for ``reciprocal_vectors``

.. note::
    All quantities in jrystal use atomic units:
    
    * Length: 1 Bohr = 0.529177 Å
    * Energy: 1 Hartree = 27.211 eV

Creating Crystal Structures
-------------------------

There are three ways to create a crystal structure:

1. Direct Construction
~~~~~~~~~~~~~~~~~~~~

Create a crystal by directly specifying its attributes:


.. code-block:: python

    from jrystal import Crystal

    crystal = Crystal(
        charges=[6, 6],  # Two carbon atoms
        positions=[[0, 0, 0], [1.5, 1.5, 1.5]],  # Positions in Bohr
        cell_vectors=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],  # Cubic cell in Bohr
        spin=0  # No unpaired electrons
    )
    print(crystal)


2. From Structure Files
~~~~~~~~~~~~~~~~~~~~~

Load a structure from common chemistry file formats:

.. code-block:: python

    crystal = Crystal.create_from_file("structure.xyz")

.. note::
    File loading uses ``ASE``'s IO capabilities. See supported formats in the 
    `ASE documentation <https://wiki.fysik.dtu.dk/ase/ase/io/io.html>`_.

3. From Chemical Symbols
~~~~~~~~~~~~~~~~~~~~~~

Create a crystal using chemical symbols and coordinates:

.. code-block:: python

    crystal = Crystal.create_from_symbols(
        symbols="C C",  # Two carbon atoms
        positions=[[0, 0, 0], [1.5, 1.5, 1.5]],  # In Bohr
        cell_vectors=[[3, 0, 0], [0, 3, 0], [0, 0, 3]]  # In Bohr
    )

.. note::
    ``jrysta.Crystal.create_from_symbols`` uses ``ASE``'s ``Atoms`` class internally and assumes periodic boundary conditions in all directions. Specifying periodic boundary conditions in a single direction is not supported.
