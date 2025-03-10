========================================
DFT Total Energy Calculation in Less than 50 Lines
========================================

This tutorial demonstrates how to perform a Density Functional Theory (DFT) total energy calculation for a diamond crystal using ``jrystal``. We'll walk through each step of the calculation, from structure setup to energy optimization.

.. note::
	In this example, we perform a **spin_restricted** **all-electron** calculation to find the ground state energy of a diamond crystal.


Prerequisites
^^^^^^^^^^^^^^

Before starting, ensure you have:

- Basic understanding of DFT concepts
- ``jrystal`` and ``jax`` installed


Step 1: Crystal Structure Setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, we need to create an object that contains the information about the crystal structure. ``jrystal`` manages this using a ``Crystal`` object:

.. code-block:: python

	import jax
	import jax.numpy as jnp
	import jrystal as jr

	# Create a diamond structure
	charges = jnp.array([6, 6])
	positions = jnp.array([[-0.84251071, -0.84251071, -0.84251071],
			[ 0.84251071,  0.84251071,  0.84251071]])

	cell_vectors = jnp.array([[0.        , 3.37004284, 3.37004284],
			[3.37004284, 0.        , 3.37004284],
			[3.37004284, 3.37004284, 0.        ]])

	crystal = jr.Crystal(charges=charges, positions=positions, cell_vectors=cell_vectors)


Step 2: Define Calculation Grids
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DFT calculations require two types of grids for calculating energy integrals:

- G-vectors (reciprocal space)
- K-points (Brillouin zone sampling)

We also need a Fourier transform frequency cutoff mask:

.. code-block:: python

	# Set grid parameters
	grid_size = [48, 48, 48]  # Real and reciprocal space grid
	kpt_grid = [3, 3, 3]      # k-point sampling
	
	# Generate grids
	g_vecs = jr.grid.g_vectors(crystal.cell_vectors, grid_sizes=grid_size)
	k_vecs = jr.grid.k_vectors(crystal.cell_vectors, grid_sizes=kpt_grid)
	
	# Create frequency cutoff mask (100 Ha cutoff energy)
	freq_mask = jr.grid.spherical_mask(
		cell_vectors=crystal.cell_vectors,
		grid_sizes=grid_size,
		cutoff_energy=100
	)

Step 3: Initialize Wavefunctions and Occupation Numbers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We need to initialize two sets of parameters:
- Plane wave coefficients for wavefunctions
- Occupation numbers for electron filling

.. code-block:: python

	# Set random seed for reproducibility
	key = jax.random.PRNGKey(0)
	key1, key2 = jax.random.split(key)
	
	# Initialize parameters
	num_bands = 12
	# Diamond has 12 electrons (2 atoms × 6 electrons)
	# We use 12 bands to include some empty states
	# For spin_restricted calculations, with 2 electrons per band, 12 bands are sufficient
	
	# Initialize plane wave coefficients
	param_pw = jr.pw.param_init(
		key1, 
		num_bands=num_bands,
		num_kpts=k_vecs.shape[0],
		freq_mask=freq_mask
	)

``jrystal`` provides three methods for handling occupation:

- ``idempotent``: Fermi-Dirac distributed occupation numbers
- ``gamma``: Occupation only at the Gamma point
- ``uniform``: Uniform occupation across all bands

Only ``idempotent`` is optimizable; the other two are fixed. Here we use ``idempotent``. For more details, see the :ref:`occupation` tutorial.

.. code-block:: python

	# Initialize occupation numbers
	param_occ = jr.occupation.idempotent_param_init(
		key=key2,
		num_bands=num_bands,
		num_kpts=k_vecs.shape[0]
	)

Step 4: Total Energy Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To find the ground state energy of the diamond crystal, we need to define a function that computes the total energy with respect to our optimizable parameters. We can construct this using the ``energy`` module:

.. code-block:: python

	def total_energy(param_pw, param_occ):
		# Calculate occupation numbers
		occ = jr.occupation.idempotent(
			param_occ, 
			num_electrons=crystal.num_electron, 
			num_kpts=k_vecs.shape[0]
		)

		# Generate coefficients
		coeff = jr.pw.coeff(param_pw, freq_mask)

		# Calculate total energy with LDA exchange-correlation
		return jr.energy.total_energy(
			coeff, crystal.positions, crystal.charges, 
			g_vecs, k_vecs, crystal.vol, occ, 
			xc="lda"
		)

Step 5: Energy Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we set up the optimizer using ``optax`` and create the optimization loop:

.. code-block:: python

	import optax

	# Initialize Adam optimizer
	optimizer = optax.adam(learning_rate=1e-3)
	opt_state = optimizer.init((param_pw, param_occ))

	# Define update step (JIT-compiled for speed)
	@jax.jit
	def update(param_pw, param_occ, opt_state):
		e_tot, grads = jax.value_and_grad(total_energy)((param_pw, param_occ))
		updates, opt_state = optimizer.update(grads, opt_state)
		param_pw, param_occ = optax.apply_updates(
			(param_pw, param_occ), updates
		)
		return e_tot, (param_pw, param_occ), opt_state

	# Run optimization
	print("Starting optimization...")
	for i in range(1000):
		e_tot, (param_pw, param_occ), opt_state = update(
			param_pw, param_occ, opt_state
		)
		
		if (i+1) % 100 == 0:
			print(f"Step {i+1:4d} | Total Energy: {e_tot:.6f} Ha")

The optimization will run for 1000 steps, printing the energy every 100 steps. You should see the total energy converge to a minimum value.

