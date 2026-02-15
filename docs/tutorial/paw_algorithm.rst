.. _paw_algorithm:
====================================
PAW Energy Formulation and Mapping
====================================

This page summarizes the PAW energy formulation used in Jrystal and maps
each term to its implementation. Mathematical derivations are intentionally
omitted; please refer to the PAW reference PDF in ``log/`` (Rostgaard, 2009)
for the full derivation.


1. Transformation and Energy Form
---------------------------------
PAW introduces a linear transformation between pseudo and all-electron
wavefunctions:

.. math::

   \ket{\psi} = \hat{T}\ket{\tilde{\psi}}
   = \ket{\tilde{\psi}} + \sum_a \sum_i
   \left( \ket{\phi_i^a} - \ket{\tilde{\phi}_i^a} \right)
   \braket{\tilde{p}_i^a | \tilde{\psi}}

Under this transformation, the total energy can be written as:

.. math::

   E = \tilde{E} + \sum_a \left( E^a - \tilde{E}^a \right)

where :math:`\tilde{E}` is the smooth (pseudo) part and the onsite
correction is the difference between all-electron and pseudo
atomic contributions.


2. Terms, Meanings, and Implementation Map
------------------------------------------
Below we list each PAW component, what it means, and where it is computed.

**Atomic density matrix**

.. math::

   D_{ij}^a =
   \sum_{s,k,n} f_{nks}
   \braket{\tilde{\psi}_{nks}|\tilde{p}_i^a}
   \braket{\tilde{p}_j^a|\tilde{\psi}_{nks}}

- Meaning: onsite occupancy in projector space (spin-summed).
- Code: `calc_atomic_density_matrix` in
  `jrystal/calc/calc_ground_state_energy_paw.py`.
- Packing: `pack(D_p)` in `jrystal/pseudopotential/utils.py`.

**Compensation charge**

.. math::

   \tilde{\rho}_\text{comp}(\mathbf{G}) =
   \sum_a e^{-i\mathbf{G}\cdot \mathbf{R}_a}
   \sum_L Q_L^a \, \hat{g}_L(\mathbf{G}),
   \quad
   Q_L^a = \sum_p D_p^a \Delta_{pL}^a + \Delta_0^a

- Meaning: restores correct multipole moments of the AE density.
- Code:
  - :math:`\Delta_{pL}^a, \Delta_0^a` in `jrystal/pseudopotential/paw_calc.py`
  - Compensation assembly in `_rho_comp_term` within
    `jrystal/calc/calc_ground_state_energy_paw.py`
  - :math:`\hat{g}_L(\mathbf{G})` from `build_paw_precompute` in
    `jrystal/pseudopotential/paw_setup.py`

**Core density**

- Meaning: frozen-core density :math:`\tilde{n}_c`.
- Source: PAW dataset (currently GPAW data).
- Code: parsed in `jrystal/pseudopotential/load_gpaw.py` and
  carried through `build_paw_precompute` in `jrystal/pseudopotential/paw_setup.py`.

**Pseudo energy**

- Meaning: smooth part of kinetic, Coulomb, XC, and local terms.
- Code: `total_energy` in `jrystal/calc/calc_ground_state_energy_paw.py`
  (variables `kinetic_pseudo`, `hartree_pseudo`, `exc_pseudo`, `e_zero_pseudo`).

**Onsite corrections**

- Meaning: AE–PS corrections for kinetic, Coulomb, XC, and local terms.
- Code: `_atomic_terms` in `jrystal/calc/calc_ground_state_energy_paw.py`.
- Tables:
  - :math:`K_p, K_c, M, M_p, M_{pp}, MB, MB_p` in
    `jrystal/pseudopotential/paw_calc.py`.


3. Precompute Outputs
---------------------
`build_paw_precompute` in `jrystal/pseudopotential/paw_setup.py` produces:

- `ghat_LG` (compensation basis),
- `phase_G` (structure factor),
- `nct_G` and `nct_g` (core density in G- and real-space),
- `vbar_G` (local potential),
- `e_zero0` (local zero-energy offset).

These are used directly in `calc_ground_state_energy_paw.py` to assemble
the pseudo part of the energy and the compensation density.

