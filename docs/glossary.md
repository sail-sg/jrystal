# Glossary

## Variable Names and Meanings
This section provides a summary of the variables and arguments used in the package, along with their meanings.

### Mesh Grid
| Math Symbol |   Variable Name   |           Meaning        |
|:-----------:|:------------------|:-------------------------|
|$\boldsymbol{r}$ |``r_vector_grid``       | Grid in real space of the unit cell. Shape: ``[x, y, z, 3]``  |
|$\boldsymbol{G}$ |``g_vector_grid`` <br /> (``g_vecs``)       | Grid in reciprocal space of the unit cell. Shape: ``[x, y, z, 3]`` |
|$\boldsymbol{k}$ |``kpts`` | k-point path sampling in reciprocal space. Shape: ``[num_kpts, 3]``|
|$\boldsymbol{w}_k$ |``kpts_weight`` | Weights for k-point integration. Shape: ``[num_kpts, ]``|

### Physical Quantities

The physical quantities in the table below are evaluated on the mesh grid in real or reciprocal space.

| Math Symbol |   Variable Name   |           Meaning        |
|:-----------:|:------------------|:-------------------------|
| $\bf{A}$         |``cell_vectors``  | Real-space unit cell translation vectors  |
| $\bf{B}$         |``cell_vectors_reciprocal``      | Reciprocal lattice vectors|
|$\rho(\bf{r})$     |``density_grid``       | Electronic charge density in real space    |
|$\tilde\rho(\bf{G})$ |``density_grid_reciprocal`` | Electronic charge density in reciprocal space |
|$\psi({\bf{r}})$     |``wave_grid``       | Electronic wavefunction on grid points in real space    |
|$\tilde\psi(\bf{G})$|``wave_grid_reciprocal`` | Electronic wavefunction on grid points in reciprocal space |
|$\boldsymbol{\varepsilon}_{i}$| ``eigenvalues`` | Single-particle eigenvalues of the Hamiltonian matrix |
|$h_{ij}$| ``hamiltonian`` | Single-particle Hamiltonian matrix |
|$c$  |``coefficient``,  <br /> (``coeff``) |  Plane wave expansion coefficients. <br /> Shape: ``[spin, kpts, band, x, y, z]``|
|$c$  |``coeff_dense``  |  Dense format of plane wave coefficients. <br /> Shape: ``[spin, kpts, g_vec, band]``|
|$w$ | ``weight_real``, <br /> \& ``weight_imaginary``      |  Trainable neural network parameters  |
|-|``mask``     | Binary mask for ``g_vector_grid`` that enforces the plane wave cutoff energy|
|$o$|``occupation``  | Electronic occupation numbers for each state|


**Notes**:
- Any variable ending with ``_vector_grid`` has the shape ``(x, y, z, 3)``, representing 3D vectors on a grid.
- Variables ending with ``_grid`` have the shape ``(x, y, z)``, representing scalar quantities on a grid.
- The postfix ``_real`` is always omitted. If a variable does not have the postfix ``_reciprocal``, it is in real space.


### Arguments and Intermediate Variables
|  Variable Name  |  Meaning     |
|:----------------|--------------|
| ``cutoff_energy`` | Energy cutoff for plane wave basis expansion |
| ``grid_sizes``  | Number of grid points along each direction for both $k$ and $G$ points|
| ``k_grid_sizes``| Number of k-points along each direction for Brillouin zone sampling|
| ``position``   | Atomic positions in real-space coordinates |
| ``charge``     | Atomic charge vector |
| ``ewald_eta``      | Ewald summation convergence parameter $\eta$|
| ``ewald_cut``      | Cutoff energy for Ewald summation. The higher the cutoff energy, the more accurate the Ewald summation, but the slower the calculation.|
| ``num_band`` | Number of electronic bands (usually equals the number of electrons)|
| ``num_k``    | Total number of k-points in the calculation |
| ``num_g``    | Total number of G-vectors in the plane wave basis |
| ``vol``    | Volume of the crystallographic unit cell|
| ``spin``    | Number of spin channels (1 for non-spin-polarized, 2 for spin-polarized calculations)|
