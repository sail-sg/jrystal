


## Variable names and meaning
To consolidate variable names and terminology for developers.

#### Physical variables 
| math_symbol |   variable_name   |           meaning        |
|:-----------:|:------------------|:-------------------------|
| $A$         |``cell_vectors``            | cell translation vector  |
| $B$         |``reciprocal_vectors``      | reciprocal translation vector|
|$\rho(r)$     |``density_grid``       | density in real space    |
|$\tilde\rho(g)$ |``reciprocal_density_grid`` | density in reciprocal space |
|$\psi(r)$     |``wave_grid``       | wave function in real space    |
|$\tilde\psi(g)$|``reciprocal_wave_grid`` | wave function in reciprocal space |
|$\boldsymbol{R}$ |``r_vector_grid``       | a grid in real space  |
|$\boldsymbol{G}$ |``g_vector_grid``       | a grid in reciprocal space |
|$\boldsymbol{k}$ |``k_vector_grid``       | Brillouin zone sampling |
|$\boldsymbol{w}_k$ |``k_weight_grid``     | the weight for each $k$ |
|$\boldsymbol{k}$ |``k_path`` | k point path sampling. shape: ``[num_k, 3]``|
|$\boldsymbol{w}_k$ |``k_weights`` | k point weight. shape: ``[num_k, ]``|


**Note**: 
Anything that ends with ``_vector_grid`` has shape ``(n1, n2, n3, 3)``;
Anything that ends with ``_grid`` has shape ``(n1, n2, n3)``.


#### Model variables
| math_symbol |   variable_name   |           meaning        |
|:-----------:|:------------------|:-------------------------|
|$c$  |``coeff_grid`` |  Hermitian coefficient. <br /> Shape: ``[2, nk, ni, n1, n2, n3]``|
|$c$  |``coeff_dense``  |  Hermitian coefficient. <br /> Shape: ``[2, nk, ng, ni]``|
|$w$ | ``weight_real``, <br /> ``weight_imaginary``      |  trainable parameters  |
||``mask``     | the mask for g_grid that enforces the cut-off energe|
|$o$|``occupation``  | occupation number|


#### Input variables
|  variable_name  |  meaning     |
|:----------------|--------------|
| ``cutoff_energy`` |cut-off energy|
| ``grid_sizes``  | grid sizes for both $k$ and $G$ points|
| ``k_grid_sizes``| Brillouin zone sampling size|
| ``positions``   | real-space coordiates |
| ``charges``     | the charges vector |

#### Ewald summation variables
|  variable_name  |  meaning     |
|:----------------|--------------|
| ``ewald_eta``      |the hyperparameter $\eta$|
| ``ewwald_cut``      |cut-off for ewald lattice|

#### Intermediate variables
| vriable_name    | meaning      |
|:----------------|--------------|
| ``num_band`` | number of bands (usually equals to the number of electorns)|
| ``num_k``    | number of $k$ points |
| ``num_g``    | number of $g$ points | 
| ``vol``    | the volume of unit cell|
| ``spin``    | the number of unpaired electrons. Can be 0 or 1.|


