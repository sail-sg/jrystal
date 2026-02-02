# Jrystal vs GPAW energy benchmark

Generated from config: `/home/aiops/zhaojx/paw-minimal/config.yaml`

Config notes (applies to all cases below):
- cutoff_energy: 40 Ha
- grid_sizes: 32

## Summary table

Notes:
- Estimated Jrystal runtime per case: ~3 minutes (single k-point).
- Jrystal values come from the **last optimizer line** (`Loss: ... | Energy ... | Kinetic ... | Hartree ... | XC ... | E_zero ... |`).

| system | jrystal_energy | jrystal_kinetic | jrystal_hartree | jrystal_xc | jrystal_e_zero | gpaw_e_total_free | gpaw_e_kinetic | gpaw_e_coulomb | gpaw_e_xc | gpaw_e_zero | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| diamond | -139.5397 | 2.1361 | -140.7159 | -0.9601 | 0.0002 | -139.5397 | 2.1359 | -140.7158 | -0.9601 | 0.0003 |  |
| diamond1 | -139.5396 | 2.1367 | -140.7164 | -0.9601 | 0.0002 | -139.5397 | 2.1359 | -140.7158 | -0.9601 | 0.0003 |  |
| diamond2 | -139.7175 | 1.2310 | -140.1668 | -0.7825 | 0.0008 | -139.7175 | 1.2320 | -140.1679 | -0.7826 | 0.0009 |  |
| si | -1116.6171 | 1.1956 | -1117.1715 | -0.6384 | -0.0029 | -1116.6171 | 1.1950 | -1117.1709 | -0.6384 | -0.0028 |  |
| na | -619.4681 | -0.1212 | -619.2460 | -0.0979 | -0.0029 | -619.4683 | -0.1211 | -619.2464 | -0.0979 | -0.0029 |  |
| mg | -767.1319 | -0.3394 | -766.5667 | -0.2203 | -0.0056 | -767.1378 | -0.4254 | -766.4760 | -0.2308 | -0.0057 |  |


INFO:absl:GPAW vs Jrystal split (pseudo / atomic) (Ha):
INFO:absl:  Kinetic: 4.041397/-2.704404 vs 3.685014/-3.390365
INFO:absl:  Coulomb: 10.745842/-1128.022309 vs 10.741894/-1127.912798
INFO:absl:  E_zero: 1.113868/-1.134170 vs 1.484160/-1.486999
INFO:absl:  XC: -2.736917/2.088970 vs -2.736919/2.098554
INFO:absl:GPAW vs Jrystal split deltas (Ha):
INFO:absl:  ΔKinetic (pseudo/atomic): 3.563829e-01 / 6.859604e-01
INFO:absl:  ΔCoulomb (pseudo/atomic): 3.947658e-03 / -1.095111e-01
INFO:absl:  ΔE_zero (pseudo/atomic): -3.702918e-01 / 3.528293e-01
INFO:absl:  ΔXC (pseudo/atomic): 2.135061e-06 / -9.583732e-03

Here are the results, let us have a close summary:
- XC energy aligns the best, the pseudo energy is almost the same, since it only depends on pseudo-density + nct_g, probably meaning
that our computation from the coeff to the pseudo-density is correct. The diff of the atomic part is proportional to that of the atomic
density matrix
- Coulomb also looks good, the diff of the pseudo-part is slightly greater than that of xc, partially because the pseudo part includes
the compensation charge, which is calculated from the atomic density matrix. The atomic part error is comparable to atomic density
matrix error
- The kinetic energy has some issues. Given that the pseudo-part totally depends on the coeff, there should not be such a big error:
4.041397 v.s.  3.685014 between jrystal and gpaw. We need to undertstand why jrystal and gpaw obtain diff pseudo kinetic energy, trace
their computation to the lowest level to understand this. The atomic part also contains unbearable difference, also compare the
difference of computation. One possible explanation is the gpaw's kinetic energy may involve some corrections if scf is used, you can
check the log/report.md and the source code for more details. But since in test_gpaw.py we are solving using the direct minimization method,
it calls the calculate_kinetic_energy_directly function inside hamiltonian.py to obtain the kinetic energy, which is consistent with the
kinetic energy calculation in jrystal
- The zero energy has the biggest relative error and must have some issue in their computation, please trace the computation of jrystal
and gpaw to the lowest level to understand the difference between

---

## Detailed component split (rows = components, columns = systems)

### diamond component split (rows = source)

Diff row is `jrystal - gpaw` (Ha, 4 d.p.).

| source | kinetic_pseudo | kinetic_atomic | kinetic | coulomb_pseudo | coulomb_atomic | coulomb | e_zero_pseudo | e_zero_atomic | e_zero | xc_pseudo | xc_atomic | xc | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diamond (gpaw) | 9.6031 | -7.4672 | 2.1359 | 15.5087 | -156.2244 | -140.7158 | 1.6402 | -1.6399 | 0.0003 | -3.5022 | 2.5421 | -0.9601 | -139.5397 |
| diamond (jrystal) | 9.6031 | -7.4141 | 2.1890 | 15.4902 | -156.2542 | -140.7640 | 1.6402 | -1.6656 | -0.0254 | -3.5022 | 2.5335 | -0.9687 | -139.5691 |
| diamond (diff) | 0.0000 | 0.0531 | 0.0531 | -0.0185 | -0.0298 | -0.0482 | 0.0000 | -0.0257 | -0.0257 | 0.0000 | -0.0086 | -0.0086 | -0.0294 |

### diamond1 component split (rows = source)

Diff row is `jrystal - gpaw` (Ha, 4 d.p.).

| source | kinetic_pseudo | kinetic_atomic | kinetic | coulomb_pseudo | coulomb_atomic | coulomb | e_zero_pseudo | e_zero_atomic | e_zero | xc_pseudo | xc_atomic | xc | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diamond1 (gpaw) | 9.6031 | -7.4672 | 2.1359 | 15.5087 | -156.2244 | -140.7158 | 1.6402 | -1.6399 | 0.0003 | -3.5022 | 2.5421 | -0.9601 | -139.5397 |
| diamond1 (jrystal) | 9.6031 | -7.4141 | 2.1890 | 15.4902 | -156.2542 | -140.7640 | 1.6402 | -1.6656 | -0.0254 | -3.5022 | 2.5335 | -0.9687 | -139.5691 |
| diamond1 (diff) | 0.0000 | 0.0531 | 0.0531 | -0.0185 | -0.0298 | -0.0482 | 0.0000 | -0.0257 | -0.0257 | 0.0000 | -0.0086 | -0.0086 | -0.0294 |

### diamond2 component split (rows = source)

Diff row is `jrystal - gpaw` (Ha, 4 d.p.).

| source | kinetic_pseudo | kinetic_atomic | kinetic | coulomb_pseudo | coulomb_atomic | coulomb | e_zero_pseudo | e_zero_atomic | e_zero | xc_pseudo | xc_atomic | xc | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diamond2 (gpaw) | 8.9605 | -7.7286 | 1.2320 | 15.9294 | -156.0973 | -140.1679 | 1.5435 | -1.5426 | 0.0009 | -3.3593 | 2.5768 | -0.7826 | -139.7175 |
| diamond2 (jrystal) | 8.9605 | -7.6793 | 1.2812 | 15.9116 | -156.1247 | -140.2131 | 1.5435 | -1.5665 | -0.0231 | -3.3593 | 2.5687 | -0.7906 | -139.7456 |
| diamond2 (diff) | 0.0000 | 0.0493 | 0.0492 | -0.0178 | -0.0274 | -0.0452 | 0.0000 | -0.0239 | -0.0240 | 0.0000 | -0.0081 | -0.0080 | -0.0281 |

### si component split (rows = source)

Diff row is `jrystal - gpaw` (Ha, 4 d.p.).

| source | kinetic_pseudo | kinetic_atomic | kinetic | coulomb_pseudo | coulomb_atomic | coulomb | e_zero_pseudo | e_zero_atomic | e_zero | xc_pseudo | xc_atomic | xc | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| si (gpaw) | 4.0414 | -2.8464 | 1.1950 | 10.7419 | -1127.9128 | -1117.1709 | 1.4842 | -1.4870 | -0.0028 | -2.7369 | 2.0986 | -0.6384 | -1116.6171 |
| si (jrystal) | 4.0414 | -2.7044 | 1.3370 | 10.7458 | -1128.0223 | -1117.2765 | 1.4842 | -1.5045 | -0.0203 | -2.7369 | 2.0890 | -0.6479 | -1116.6077 |
| si (diff) | 0.0000 | 0.1420 | 0.1420 | 0.0039 | -0.1095 | -0.1056 | 0.0000 | -0.0175 | -0.0175 | 0.0000 | -0.0096 | -0.0095 | 0.0094 |
