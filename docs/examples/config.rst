========================================
Command Line Interface and Configuration
========================================

Overview
--------

``jrystal`` is a quantum chemistry calculation package that can be executed via command line interface. The program uses a YAML configuration file to specify calculation parameters and settings.

Command Line Interface
--------------------

The command line interface accepts the following options:

- ``-m, --mode``: Specifies the calculation mode (``energy`` or ``band``). Default: ``energy``
- ``-c, --config``: Specifies the path to the configuration file. Default: ``config.yaml``

Example Usage
~~~~~~~~~~~~

For total energy minimization:

.. code-block:: bash

    jrystal -m energy -c config.yaml

For band structure calculation:

.. code-block:: bash

    jrystal -m band -c config.yaml

Configuration File Structure
--------------------------

The configuration file must be in YAML format and contain the following sections:

**Crystal Structure**

- ``crystal``: Identifier for the crystal structure. The program searches for ``$CRYSTAL.xyz`` in the geometry directory
- ``crystal_file_path_path``: Explicit path to the crystal structure file (takes precedence over ``crystal`` if both are specified)

**Exchange-Correlation Functional**

- ``xc``: Exchange-correlation functional specification (e.g., ``lda_x`` for Local Density Approximation exchange)

**Pseudopotential Configuration**

- ``use_pseudopotential``: Enables or disables pseudopotential calculations (``True``/``False``)
- ``pseudopotential_file_dir``: Path to pseudopotential files directory (uses system default if unspecified)

**Planewave Basis Settings**

- ``g_grid_mask_method``: Grid masking methodology:
    - ``cubic``: Employs cubic grid masking
    - ``spherical``: Employs spherical grid masking with user-defined cutoff
- ``cutoff_energy``: Planewave kinetic energy cutoff in Hartree (required for ``spherical`` method)
- ``grid_sizes``: Fast Fourier Transform (FFT) grid dimensions
- ``k_grid_sizes``: Monkhorst-Pack k-point grid dimensions for Brillouin zone sampling
- ``occupation``: Electronic state occupation methodology: can be chosen from ``fermi-dirac``, ``gamma``, or ``uniform``:
    - ``fermi-dirac``: Fermi-Dirac statistical distribution
    - ``gamma``: Gamma-point sampling scheme
    - ``uniform``: Uniform occupation distribution
- ``smearing``: Fermi-Dirac distribution temperature parameter in Hartree
- ``empty_bands``: Number of additional unoccupied bands to compute
- ``spin_restricted``: Enforces identical spatial orbitals for spin-up and spin-down electrons (``True``/``False``)

**Ewald Summation Parameters**

- ``ewald_args``: Ewald sum configuration:
    - ``ewald_eta``: Separation parameter for real/reciprocal space partitioning
    - ``ewald_cutoff``: Reciprocal space cutoff radius

**Optimization Parameters**

- ``epoch``: Maximum optimization iteration count
- ``optimizer``: Optimization algorithm selection (e.g., ``adam``)
- ``optimizer_args``: Algorithm-specific parameters:
    - ``learning_rate``: Optimization step size
- ``scheduler``: Learning rate scheduling specification (``null`` for constant rate)
- ``convergence_condition``: Energy variance threshold for convergence determination

**Band Structure Calculation Parameters**

- ``band_structure_empty_bands``: Number of unoccupied bands for band structure analysis
- ``k_path_special_points``: High-symmetry k-point sequence (e.g., ``LGXL``)
- ``num_kpoints``: Sampling point count per k-path segment
- ``k_path_file``: Path to NumPy (.npy) file containing custom k-point coordinates
- ``band_structure_epoch``: Maximum band structure optimization iterations
- ``k_path_fine_tuning``: Enables progressive k-path optimization using previous solutions
- ``k_path_fine_tuning_epoch``: Iteration count per k-point during fine-tuning

**System Configuration**

- ``seed``: Random number generator seed for reproducibility
- ``xla_preallocate``: Enables XLA memory preallocation for performance optimization
- ``jax_enable_x64``: Activates double-precision (64-bit) floating-point computation
- ``verbose``: Controls computation progress output detail
- ``eps``: Numerical stability threshold for division operations
