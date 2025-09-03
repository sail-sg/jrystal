# PAW Pseudopotential File Documentation

## Overview
This document provides comprehensive documentation for PAW (Projector Augmented Wave) pseudopotential files used in GPAW and Quantum ESPRESSO (QE). The two formats store similar physical quantities but with different conventions and normalization factors.

---

## 1. GPAW PAW Setup Files (.gz format)

GPAW uses XML-formatted setup files, typically compressed with gzip. File naming: `{Element}.{XC}.gz`

### 1.1 Basic Atomic Information

| Element | Type | Description | Properties |
|---------|------|-------------|------------|
| **atom/@symbol** | string | Chemical symbol (e.g., 'C', 'N') | - |
| **atom/@Z** | float | Total atomic number | Integer value |
| **atom/@core** | float | Number of core electrons | Z_core = Z - Z_valence |
| **atom/@valence** | float | Number of valence electrons | Determines chemical properties |

### 1.2 Radial Grid

| Element | Shape | Description | Mathematical Form |
|---------|-------|-------------|-------------------|
| **radial_grid/@eq** | string | Grid equation type | e.g., "r=a*i/(n-i)" |
| **radial_grid/@a** | float | Grid parameter | Scale factor |
| **radial_grid/@n** | int | Number of grid points | Typically 300-1000 |
| **r_g** | (n,) | Radial grid points | r[i] = a*i/(n-i) |
| **dr_g** | (n,) | Integration weights | dr[i] = dr/di |

**Properties:**
- Grid starts at r=0 and extends to r_max (cutoff radius)
- Non-uniform spacing, denser near nucleus
- Integration: ∫f(r)dr ≈ Σ f[i] * dr[i]

### 1.3 Wave Functions

| Element | Shape | r factor | √(4π) factor | Normalization | Description |
|---------|-------|----------|--------------|---------------|-------------|
| **phi_jg** | (n_j, n_g) | NO | YES | ∫\|φ\|²r²dr = 1 | All-electron partial waves |
| **phit_jg** | (n_j, n_g) | NO | YES | NA | Pseudo partial waves |
| **pt_jg** | (n_j, n_g) | NO | YES | ⟨p̃\|φ̃⟩ = δᵢⱼ | Projector functions |

**GPAW Convention:**
```
Physical: φ(r) = radial(r) * Y_lm(θ,φ)
Stored: φ(r) * √(4π)  [includes √(4π) factor]
Normalization: ∫|stored|² r² dr = 1
```

**Properties:**
- φ_jg and φ̃_jg match outside core region (r > r_c)
- p̃_jg vanish outside cutoff radius
- Orthogonality: ∫ p̃ᵢ(r) φ̃ⱼ(r) r² dr = δᵢⱼ

### 1.4 Densities

| Element | Shape | r factor | √(4π) factor | Description | Properties |
|---------|-------|----------|--------------|-------------|------------|
| **ae_core_density** | (n_g,) | NO | YES (multiplied) | All-electron core density n_c | ∫n_stored r²dr * √(4π) = N_core |
| **pseudo_core_density** | (n_g,) | NO | YES (multiplied) | Smooth core density ñ_c | Matches n_c outside r_c |
| **pseudo_valence_density** | (n_g,) | NO | YES (multiplied) | Pseudo valence density | NA |

**GPAW Density Convention:**
```
Physical: n(r) [true density]
Stored: n(r) * √(4π)  [includes √(4π) factor]
Integration: ∫ n_stored r² dr * √(4π) = N_electrons
```

### 1.5 Potentials

| Element | Shape | Description | Properties |
|---------|-------|-------------|------------|
| **zero_potential** | (n_g,) | Zero potential v̄(r) | Local pseudopotential |
| **localpotential** | (n_g,) | Local potential | Alternative local part |

**Properties:**
- v̄(r) → -Z_val/r as r → ∞
- Smooth at origin (no singularity)

### 1.6 Shape Functions

| Element | Type | Description | Properties |
|---------|------|-------------|------------|
| **shape_function/@type** | string | Type (e.g., 'gauss') | Compensation charge shape |
| **shape_function/@rc** | float | Cutoff radius | g(r) = 0 for r > rc |

---

## 2. Quantum ESPRESSO UPF Files (.UPF format)

QE uses XML-based Unified Pseudopotential Format (UPF).

### 2.1 PP_HEADER Section

| Element | Type | Description |
|---------|------|-------------|
| **element** | string | Chemical symbol |
| **z_valence** | float | Valence charge |
| **l_max** | int | Maximum angular momentum |
| **mesh_size** | int | Radial grid size |

### 2.2 PP_MESH Section

| Element | Shape | Description | Difference from GPAW |
|---------|-------|-------------|----------------------|
| **PP_R** | (mesh_size,) | Radial grid r | Same concept |
| **PP_RAB** | (mesh_size,) | dr * r | QE stores r*dr, GPAW stores dr |

### 2.3 PP_NONLOCAL Section

#### Projector Functions (PP_BETA)
Note: In actual UPF files, beta tags include indices (e.g., PP_BETA.1, PP_BETA.2, etc.)

| Element | Shape | r factor | √(4π) factor | Storage Convention | GPAW Equivalent |
|---------|-------|----------|--------------|-------------------|-----------------|
| **PP_BETA.values** | (mesh_size,) | YES (multiplied) | NO | β(r) * r | pt_jg * r / √(4π) |
| **angular_momentum** | int | - | - | l quantum number | l_j |
| **cutoff_radius** | float | - | - | r_cut | rcut_j |
| **cutoff_radius_index** | int | - | - | Grid index at r_cut | gcut_j |

**QE Convention:**
```
Physical: β(r) [true projector]
Stored: β(r) * r  [includes r factor, NO √(4π)]
Conversion to GPAW: pt_gpaw = β_qe / r / √(4π)
```

#### Augmentation (PP_AUGMENTATION)

| Element | Shape | Description |
|---------|-------|-------------|
| **PP_Q** | (nqf,) | Q_ij augmentation charges |
| **PP_MULTIPOLES** | ((lmax+1)*nj*nj,) | Multipole moments Δ_lq |
| **PP_QIJ** | Multiple | Augmentation functions Q_ij(r) |

**PP_QIJL Structure:**
```xml
<PP_QIJL.i.j.l first_index="i" second_index="j" composite_index="idx" angular_momentum="l" size="mesh_size">
  values: Q_ij^l(r) * r²  [stored with r² factor]
</PP_QIJL.i.j.l>
```
Note: The tag name includes indices (e.g., PP_QIJL.1.2.0 for i=1, j=2, l=0)

### 2.4 PP_FULL_WFC Section

| Element | Shape | r factor | √(4π) factor | Storage | GPAW Equivalent |
|---------|-------|----------|--------------|---------|-----------------|
| **PP_AEWFC** | (n_wfc, mesh_size) | YES (multiplied) | YES (multiplied) | φ(r) * r * √(4π) | phi_jg * r |
| **PP_PSWFC** | (n_wfc, mesh_size) | YES (multiplied) | YES (multiplied) | φ̃(r) * r * √(4π) | phit_jg * r |

**QE Wave Function Convention:**
```
Physical: φ(r) [true wave function]
Stored: φ(r) * r * √(4π)  [includes BOTH r and √(4π)]
Conversion to GPAW: phi_gpaw = φ_qe / r / √(4π)
Normalization: ∫|φ_physical|² r² dr * 4π = 1
```

### 2.5 PP_PAW Section

| Element | Shape | r factor | √(4π) factor | Description | Convention |
|---------|-------|----------|--------------|-------------|------------|
| **PP_AE_NLCC** | (mesh_size,) | NO | NO | AE core charge | n_c(r) [true density] |
| **PP_AE_VLOC** | (mesh_size,) | NO | NO | AE local potential | V_ae(r) |

**QE Core Density Convention:**
```
Physical: n_c(r) [true charge density]
Stored: n_c(r)  [NO factors applied]
Conversion to GPAW: nc_gpaw = n_c_qe * √(4π)
Integration: ∫ n_c(r) * 4π * r² dr = N_core
```

### 2.6 Other Sections

| Element | Shape | Description |
|---------|-------|-------------|
| **PP_LOCAL** | (mesh_size,) | Local pseudopotential V_loc(r) |
| **PP_NLCC** | (mesh_size,) | Non-linear core correction |
| **PP_DIJ** | (nbeta*(nbeta+1)/2,) | D_ij matrix elements |

---

## 3. Key Differences Between GPAW and QE

### 3.1 Radial Functions Storage

| Quantity | GPAW Storage | QE Storage | r factor diff | √(4π) factor diff |
|----------|--------------|------------|---------------|-------------------|
| **Wave functions (phi, phit)** | u(r)/r × √(4π) | u(r) × √(4π) | QE has r, GPAW doesn't | Both have √(4π) |
| **Projectors (pt, beta)** | p(r) × √(4π) | p(r) × r | QE has r, GPAW doesn't | GPAW has √(4π), QE doesn't |
| **Core densities** | n(r) × √(4π) | n(r) | None | GPAW has √(4π), QE doesn't |
| **Augmentation (Q)** | Q(r) | Q(r) × r² | QE has r², GPAW doesn't | Neither has √(4π) |

### 3.2 Normalization Conventions

**GPAW:**
```python
# Wave functions (stored with √(4π))
∫ |φ_stored|² r² dr = 1
# Densities (stored with √(4π))  
∫ n_stored r² dr * √(4π) = N_electrons
```

**QE:**
```python
# Wave functions  
∫ |φ(r)|² r² dr * 4π = 1
# Densities
∫ n(r) 4π r² dr = N_electrons
```

### 3.3 Angular Momentum Coupling

| Aspect | GPAW | QE |
|--------|------|-----|
| **Gaunt coefficients** | Pre-calculated | On-the-fly |
| **Spherical harmonics** | Real | Complex |
| **Multipole storage** | Delta_pL matrix | Flattened array |

---

## 4. Mathematical Properties

### 4.1 Asymptotic Behavior

| Function | r → 0 | r → ∞ | r > r_cut |
|----------|-------|-------|-----------|
| **φ(r)** | ~ r^l | ~ exp(-αr) | Oscillatory |
| **φ̃(r)** | ~ r^l | ~ exp(-αr) | = φ(r) |
| **p̃(r)** | ~ r^l | 0 | = 0 |
| **n_c(r)** | Finite | ~ exp(-βr) | ~ 0 |
| **V_loc(r)** | Finite | ~ -Z/r | ~ -Z_val/r |

### 4.2 Orthogonality Relations

```
⟨p̃_i|φ̃_j⟩ = δ_ij                    (Projector-wave orthogonality)
⟨φ_i|φ_j⟩ = ⟨φ̃_i|φ̃_j⟩ + ⟨φ_i-φ̃_i|φ_j-φ̃_j⟩  (PAW transformation)
```

### 4.3 Charge Conservation

```
Q_total = ∫[n_v + n_c] dr = Z        (Total charge = atomic number)
Q_smooth = ∫[ñ_v + ñ_c + n_comp] dr = Z  (Smooth charge conservation)
```

### 4.4 Integration Rules

**Radial integration:**
```python
# GPAW
I = sum(f_g * r_g**2 * dr_g) * sqrt(4*pi)

# QE  
I = sum(f_g * r_g**2 * dr_g) * 4*pi
```

**Angular integration:**
```
∫ Y_lm* Y_l'm' dΩ = δ_ll' δ_mm'
```

---

## 5. Practical Conversion Formulas

### From QE to GPAW:
```python
phi_gpaw = phi_qe / r / sqrt(4*pi)
pt_gpaw = pt_qe / r / sqrt(4*pi)
nc_gpaw = nc_qe * sqrt(4*pi)
n_qg_gpaw = n_qg_qe / r**2 / (4*pi)
```

### From GPAW to QE:
```python
phi_qe = phi_gpaw * r * sqrt(4*pi)
pt_qe = pt_gpaw * r * sqrt(4*pi)
nc_qe = nc_gpaw / sqrt(4*pi)
n_qg_qe = n_qg_gpaw * r**2 * (4*pi)
```

---

## 6. Common Pitfalls

1. **Forgetting r factors**: QE stores r×φ, GPAW stores φ
2. **Missing 4π vs √(4π)**: Different spherical harmonic conventions
3. **Grid indices**: QE uses 1-based in docs, 0-based in code
4. **Cutoff handling**: QE may have explicit zeros, GPAW truncates array
5. **Density normalization**: Always check if 4π is included
6. **XML tag indexing**: UPF tags include indices (e.g., PP_BETA.1, PP_QIJL.1.2.0)
7. **Parser flexibility**: Use `tag.startswith()` to match indexed tags

---

## 7. Implementation Notes

### 7.1 QE UPF Loading (jrystal/pseudopotential/load.py)
- Uses `xml.etree.ElementTree` for parsing
- Handles indexed tags with `tag.startswith()` 
- Returns raw values from UPF file (no conversion)
- Preserves QE conventions (r×φ for projectors, n(r) for densities)

### 7.2 GPAW Setup Loading (gpaw/setup_data.py)
- Uses SAX parser for XML reading
- Stores values in GPAW convention during parsing
- Applies √(4π) factor to densities automatically
- Handles grid truncation based on cutoff radius

### 7.3 Conversion in align_qe.py
```python
# Convert projectors from QE to GPAW (divide by r)
data.pt_jg = upf_beta_values / r_g

# Convert densities from QE to GPAW (multiply by √(4π))
data.nc_g = upf_core_density * np.sqrt(4 * np.pi)
```

---

## References

- GPAW documentation: https://wiki.fysik.dtu.dk/gpaw/
- Quantum ESPRESSO UPF format: https://pseudopotentials.quantum-espresso.org/
- PAW method: P.E. Blöchl, Phys. Rev. B 50, 17953 (1994)