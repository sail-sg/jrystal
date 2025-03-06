# Glossary


## Variable names and meaning
A summary of the variables and arguments used in this package, along with their meanings, are shown below.


### Physical variables
| math_symbol |   variable_name   |           meaning        |
|:-----------:|:------------------|:-------------------------|
| $\bf{A}$         |``cell_vectors``            | cell translation vector  |
| $\bf{B}$         |``cell_vectors_reciprocal``      | reciprocal translation vector|
|$\rho(\bf{r})$     |``density_grid``       | density in real space    |
|$\tilde\rho(\bf{G})$ |``density_grid_reciprocal`` | density in reciprocal space |
|$\psi({\bf{r}})$     |``wave_grid``       | wave function on grid points in real space    |
|$\tilde\psi(\bf{G})$|``wave_grid_reciprocal`` | wave function on grid points in reciprocal space |
|$\boldsymbol{r}$ |``r_vector_grid``       | a grid in real space. Shape: ``[x, y, z, 3]``  |
|$\boldsymbol{G}$ |``g_vector_grid``       | a grid in reciprocal space. Shape: ``[x, y, z, 3]`` |
|$\boldsymbol{k}$ |``kpts`` | k point path sampling. Shape: ``[num_k, 3]``|
|$\boldsymbol{w}_k$ |``kpts_weight`` | k point weight. Shape: ``[num_k, ]``|
|$\boldsymbol{\varepsilon}_{ij}$| ``hamiltonian`` | hamiltonian matrix |
|$c$  |``coeff`` |  Plane wave coefficient. <br /> Shape: ``[spin, kpts, band, x, y, z]``|
|$c$  |``coeff_dense``  |  Plane wave coefficient. <br /> Shape: ``[spin, kpts, g_vec, band]``|
|$w$ | ``weight_real``, <br /> ``weight_imaginary``      |  trainable parameters  |
|-|``mask``     | the mask for ``g_vector_grid`` that enforces the cut-off energe|
|$o$|``occupation``  | occupation number|


**Note**:
- Any variable ending with ``_vector_grid`` has the shape ``(x, y, z, 3)``.
- Variables ending with ``_grid`` have the shape ``(x, y, z)``.
- The postfix ``_real`` is always omitted. If a variable does not have postfix ``_reciprocal``, it is in real space.


### Arguments and Intermediate Variable
|  variable_name  |  meaning     |
|:----------------|--------------|
| ``cutoff_energy`` |cut-off energy|
| ``grid_sizes``  | grid sizes for both $k$ and $G$ points|
| ``k_grid_sizes``| Brillouin zone sampling size|
| ``position``   | real-space coordiates |
| ``charge``     | the charges vector |
| ``ewald_eta``      |the hyperparameter $\eta$|
| ``ewwald_cut``      |cut-off for ewald lattice|
| ``num_band`` | number of bands (usually equals to the number of electorns)|
| ``num_k``    | number of $k$ points |
| ``num_g``    | number of $g$ points |
| ``vol``    | the volume of unit cell|
| ``spin``    | the number of unpaired electrons. Can be 0 or 1.|
