# Setup QE Conversion Guide

## Overview
This document explains the conversion factors needed when loading QE UPF files for use with GPAW's Setup class. The main challenges are handling different conventions for the r factor and √(4π) factor between QE and GPAW formats.

---

## Quick Reference Table

| Data Type | QE UPF Storage | GPAW Setup Expects | Required Conversion |
|-----------|----------------|-------------------|-------------------|
| **Projectors (β)** | β(r) × r | p(r) × √(4π) | Divide by r, multiply by √(4π) |
| **Wave functions (φ)** | φ(r) × r × √(4π) | u(r)/r × √(4π) | Divide by r (√(4π) cancels) |
| **Core density** | n(r) | n(r) × √(4π) | Multiply by √(4π) |
| **Local potential** | V(r) | V(r) | No conversion |
| **Grid (r)** | r | r | No conversion |
| **Grid weights** | dr × r | dr | Divide by r |

---

## Detailed Conversion Rules

### 1. Projector Functions (PP_BETA → pt_jg)

**QE Storage:**
- Stored as: `β(r) × r`
- No √(4π) factor included

**GPAW Expectation:**
- Expects: `p(r) × √(4π)`
- No r factor

**Conversion Code:**
```python
# Current code in align_qe.py (may be incorrect):
data.pt_jg = np.array([proj['values'] for proj in pp_dict['PP_NONLOCAL']['PP_BETA']])[:, :gcut] / r_g

# Correct conversion should be:
data.pt_jg = np.array([proj['values'] for proj in pp_dict['PP_NONLOCAL']['PP_BETA']])[:, :gcut] / r_g * np.sqrt(4 * np.pi)
```

**Why:** QE includes r but not √(4π), GPAW excludes r but includes √(4π). The current code may be missing the √(4π) multiplication.

---

### 2. Wave Functions (PP_AEWFC/PP_PSWFC → phi_jg/phit_jg)

**QE Storage:**
- Stored as: `φ(r) × r × √(4π)`
- Both factors included

**GPAW Expectation:**
- Expects: `u(r)/r × √(4π)` where u(r) = r×φ(r)
- No r factor, but √(4π) included

**Conversion Code:**
```python
# From align_qe.py
data.phi_jg = np.array([phi['values'] for phi in pp_dict['PP_FULL_WFC']['PP_AEWFC']])[:, :gcut] / r_g
data.phit_jg = np.array([phi['values'] for phi in pp_dict['PP_FULL_WFC']['PP_PSWFC']])[:, :gcut] / r_g
```

**Why:** Both have √(4π), so it cancels. Only need to remove the r factor from QE.

---

### 3. Core Densities (PP_AE_NLCC → nc_g, PP_NLCC → nct_g)

**QE Storage:**
- Stored as: `n(r)` (true physical density)
- No factors applied

**GPAW Expectation:**
- Expects: `n(r) × √(4π)`
- √(4π) factor included for spherical integration

**Conversion Code:**
```python
# From align_qe.py
data.nc_g = np.array(pp_dict['PP_PAW']['PP_AE_NLCC'])[:gcut] * np.sqrt(4 * np.pi)
data.nct_g = np.array(pp_dict['PP_NLCC'])[:gcut] * np.sqrt(4 * np.pi)
```

**Why:** GPAW uses √(4π) normalization for densities to simplify spherical integration.

---

### 4. Local Potential (PP_LOCAL → vbar_g)

**QE Storage:**
- Stored as: `V(r)`
- No special factors

**GPAW Expectation:**
- Expects: `V(r)`
- No special factors

**Conversion Code:**
```python
# From align_qe.py
data.vbar_g = np.array(pp_dict['PP_LOCAL'])[:gcut]
```

**Why:** Both use the same convention for potentials.

---

## Mathematical Background

### Spherical Harmonics Convention

**QE Convention:**
- Uses complex spherical harmonics
- Normalization: ∫ Y*_lm Y_l'm' dΩ = δ_ll' δ_mm'
- Angular integration: 4π steradians

**GPAW Convention:**
- Uses real spherical harmonics  
- Includes √(4π) in radial functions
- Simplifies to: ∫ f(r) r² dr for l=0 components

### Why √(4π)?

The factor √(4π) comes from the l=0 spherical harmonic:
- Y_00 = 1/√(4π)
- For spherically symmetric functions: ∫∫∫ f(r) dV = ∫ f(r) × 4π × r² dr

GPAW absorbs the √(4π) into radial functions to simplify calculations:
- Stored density: ñ(r) = n(r) × √(4π)
- Integration: ∫ ñ(r) r² dr = N (number of electrons)

---

## Common Pitfalls

1. **Missing √(4π) for projectors**: QE projectors don't include √(4π), but GPAW expects it.

2. **Double-counting √(4π)**: Wave functions in QE already include √(4π), so don't multiply again.

3. **Grid truncation**: Ensure consistent grid cutoff (gcut) for all arrays.

4. **Integration weights**: QE stores r×dr, GPAW uses dr directly.

5. **Augmentation functions**: QE includes r² factor, GPAW doesn't (not covered in align_qe.py).

---

## Verification

To verify conversions are correct:

1. **Check normalization:**
```python
# For GPAW wave functions
integral = np.sum(phi_jg[i]**2 * r_g**2 * dr_g)
assert np.abs(integral - 1.0) < 1e-6
```

2. **Check orthogonality:**
```python
# Projector-wave orthogonality
overlap = np.sum(pt_jg[i] * phit_jg[j] * r_g**2 * dr_g)
assert np.abs(overlap - delta_ij) < 1e-6
```

3. **Check charge conservation:**
```python
# Core charge
N_core = np.sum(nc_g * r_g**2 * dr_g)
assert np.abs(N_core - expected_core_electrons) < 1e-6
```

---

## Implementation in align_qe.py

The conversion happens in the `run_gpaw_setup()` function:

```python
def run_gpaw_setup():
    # Load UPF file
    pp_dict = parse_upf('path/to/file.UPF')
    
    # Convert projectors: divide by r AND multiply by √(4π)
    # WARNING: Current code may be missing the √(4π) factor!
    data.pt_jg = beta_values[:, :gcut] / r_g  # Current (possibly wrong)
    # data.pt_jg = beta_values[:, :gcut] / r_g * np.sqrt(4 * np.pi)  # Correct
    
    # Convert wave functions: divide by r (√(4π) already present in both)
    data.phi_jg = aewfc_values[:, :gcut] / r_g  # Correct
    data.phit_jg = pswfc_values[:, :gcut] / r_g  # Correct
    
    # Convert densities: multiply by √(4π)
    data.nc_g = ae_nlcc_values[:gcut] * np.sqrt(4 * np.pi)  # Correct
    data.nct_g = nlcc_values[:gcut] * np.sqrt(4 * np.pi)  # Correct
    
    # Local potential: no conversion
    data.vbar_g = local_values[:gcut]  # Correct
```

**⚠️ Note:** The projector conversion may need verification. If GPAW expects projectors with √(4π) factor but QE doesn't include it, the conversion should multiply by √(4π).

---

## References

- GPAW Setup class: `gpaw/setup.py`
- QE UPF format: https://pseudopotentials.quantum-espresso.org/
- PAW method: Blöchl, PRB 50, 17953 (1994)