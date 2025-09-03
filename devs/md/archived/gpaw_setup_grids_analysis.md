# GPAW Setup.py Grid Analysis

## Overview
The GPAW Setup class uses multiple radial grids for different components of the PAW method. This document analyzes all grids used, their purposes, and potential sources of discrepancies.

## Primary Grids

### 1. **rgd** - Main Radial Grid Descriptor
- **Definition**: Line 783: `rgd = self.rgd = data.rgd`
- **Source**: Comes from the pseudopotential data (UPF or PAW dataset)
- **Typical range**: Full radial grid from 0 to ~100 Bohr
- **Grid points**: Usually 1000-2000 points
- **Purpose**: Primary grid for atomic calculations

**Quantities calculated on rgd:**
- Projector functions (`pt_jg`)
- Local potential (`vbar_g`)
- Wave functions (`phi_jg`, `phit_jg`)
- Initial integrals and overlaps

### 2. **rgd2** - Truncated/Augmentation Grid
- **Definition**: Line 901: `rgd2 = self.local_corr.rgd2 = rgd.new(gcut2)`
- **Cutoff**: `gcut2 = rgd.ceil(2 * max(rcut_j))` (Line 801)
- **Purpose**: Grid for augmentation sphere calculations (twice the projector cutoff)
- **Relationship**: Subset of rgd, truncated at gcut2

**Quantities calculated on rgd2:**
- Core densities (`nc_g`, `nct_g`) - truncated versions
- Compensation charges (`g_lg`)
- Projector pair densities (`n_qg`, `nt_qg`)
- PAW correction integrals
- XC corrections
- Local corrections

### 3. **Implicit Grid for Basis Functions**
- **Definition**: Lines 1220-1221: `rcut3 = 8.0; gcut3 = rgd.ceil(rcut3)`
- **Cutoff**: Fixed at 8.0 Bohr
- **Purpose**: Extended grid for initial guess basis functions
- **Usage**: Only for generating initial atomic orbitals

## Grid Cutoffs and Their Meanings

### Critical Cutoff Radii:
1. **rcut_j**: Individual projector cutoff radii (from pseudopotential)
2. **rcut2 = 2 * max(rcut_j)**: Augmentation sphere radius (Line 800)
3. **rcut3 = 8.0**: Basis function cutoff (Line 1220)
4. **rcutfilter**: Filter cutoff for smoothing (Line 831-834)
5. **rcore**: Core density cutoff (Lines 875, 1211)

### Grid Point Indices:
- **gcut**: Max grid index for individual projectors
- **gcut2**: Grid index for augmentation sphere (rgd2 size)
- **gcut3**: Grid index for basis functions
- **gcutfilter**: Grid index for filtering

## Key Grid Transformations

### 1. Grid Creation/Truncation
```python
# Line 901: Create truncated grid for augmentation sphere
rgd2 = rgd.new(gcut2)  # Creates new grid descriptor up to gcut2
```

### 2. Spline Interpolation
Throughout the code, splines are used to interpolate between grids:
```python
# Line 334: Spline core density to augmentation sphere
nc = rgd.spline(nc_g, rcut2, points=1000)

# Line 868: Create spline for local potential
self.vbar = rgd.spline(vbar_g, rcutfilter)

# Line 993-994: Spline compensation charges
self.ghat_l = [rgd2.spline(g_g, rcut2, l, 50) for l, g_g in enumerate(self.g_lg)]
```

### 3. Grid Integration
Different integration domains are used:
```python
# Line 823: Full grid integration
A_nn = [[rgd.integrate(phit_jg[j1] * pt_jg[j2]) / 4 / pi ...]]

# Line 1075: Truncated integration up to gcut
N0_q[q] = sum(n_qg[q, :gcut] * r_g[:gcut]**2 * dr_g[:gcut])
```

## Sources of Grid-Related Discrepancies

### 1. **Grid Mismatch Between UPF and GPAW**
- UPF files define their own radial grid (r_g, dr_g)
- GPAW's `AbinitRadialGridDescriptor` may calculate different grid points
- Even with same parameters (a, d, N), numerical differences can occur

### 2. **Grid Truncation Effects**
- rgd2 is truncated at gcut2, which may not align with UPF expectations
- Quantities like Delta0 depend critically on integration limits

### 3. **Spline Interpolation Errors**
- Multiple spline operations can accumulate errors
- Different spline methods between implementations

### 4. **Integration Weight Differences**
- UPF: Uses provided `PP_RAB` as integration weights
- GPAW: Calculates dr_g from grid parameters
- Small differences in dr_g lead to large errors in integrals

## Critical Code Sections for Grid Issues

### Delta0 Calculation (Lines 1096-1104)
```python
# Involves integration over different grids:
Delta0 = np.dot(self.local_corr.nc_g - self.local_corr.nct_g,
                rgd2.r_g**2 * rgd2.dr_g)
```
**Issue**: If rgd2 grid differs from expected UPF grid, Delta0 will be wrong.

### Compensation Charge Normalization (Lines 988-994)
```python
# g_lg functions depend on grid normalization
g0_norm = np.dot(g_lg[0], r_g**2 * dr_g)  # Should equal 1/sqrt(4π)
```
**Issue**: Grid differences affect normalization constants.

## Recommendations for Grid Alignment

1. **Force Grid Consistency**: When loading UPF data, override GPAW's calculated grid with UPF's actual grid values:
   ```python
   data.rgd.r_g = upf_r_g
   data.rgd.dr_g = upf_dr_g
   ```

2. **Verify Grid Parameters**: Check that grid generation parameters (a, d, N) produce identical grids

3. **Careful with Truncation**: Ensure gcut2 calculation is consistent with UPF augmentation region

4. **Debug Integration**: Add checks to verify basic integrals (e.g., core charge = 2e for Carbon)

## Summary

The Setup class uses three main grids:
1. **rgd**: Full atomic grid from pseudopotential data
2. **rgd2**: Truncated grid for augmentation sphere (up to 2*max(rcut_j))
3. **Implicit basis grid**: Extended to 8.0 Bohr for initial orbitals

The primary source of discrepancies appears to be:
- GPAW recalculates grid points instead of using UPF-provided values
- This causes integration errors that propagate through all PAW corrections
- The fix requires ensuring GPAW uses the exact grid from the UPF file