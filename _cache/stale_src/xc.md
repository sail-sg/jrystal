# `jxc`

[tianbo]
I would like to use `jxc` to calculate the XC potential.

there are several pleases that involves the xc functional, such as in energy, potential, the hamiltonian matrix, etc.

Please read the docs and the code under /_src/ and replace the existing xc related code with `jxc`.

`jxc` is installed in the `jrystal` environment (venv).

make sure that for different types of xc functionals, the code should be compatible with the existing code. support lda, gga, mgga, etc.

but some please may require the sigma, lapl, tau, etc arguments on grid. can you implement these arguments in pw.py?


[Claude]

## Questions

1. **jxc installation**: `pip show jxc` fails but the package exists locally at `/home/aiops/litb/projects/jxc/`. I can `import jxc` from energy.py already. Should I just rely on the existing import, or do you want me to add it to `pyproject.toml` dependencies?

2. **Compound functionals**: The config has a comment `# xc: "gga_x_pbe+gga_c_pbe"` suggesting exchange+correlation can be combined with `+`. Should I implement a parser that splits on `+` and sums contributions from each component? Or will you always pass a single functional name?

3. **XC potential strategy**: Currently `potential.py` uses a hand-coded `_lda_x_vrho()`. To generalize, I have two options:
   - (a) Use `jxc.get_xc_functional(xc_type, polarized, order='vxc')` to get analytical `vrho`, `vsigma`, `vtau` etc. directly.
   - (b) Use JAX `jax.grad` on the XC energy density from jxc.
   I lean toward **(a)** since jxc already provides derivatives. Does that sound right?

4. **Spin-polarized GGA/MGGA**: For polarized GGA, `sigma` has 3 components: (sigma_uu, sigma_ud, sigma_dd). Similarly `tau` has spin components. Should I support the full spin-polarized GGA/MGGA case, or is unpolarized sufficient for now?

5. **SCF compatibility**: `scf/scf.py` line 145 has a comment that jxc fails under nested `jit+grad`. Should I worry about this now, or is that a separate issue to solve later?

6. **GGA/MGGA potential terms**: For GGA, the XC potential is not just `vrho` — it also involves a divergence term: `v_xc = vrho - 2 * div(vsigma * nabla_rho)`. For MGGA there's an additional `vtau` contribution to the Hamiltonian. Should I implement these full GGA/MGGA potential expressions, or just wire up jxc and leave the potential corrections for later?

7. **Tests**: The existing tests in `energy_test.py` reference old API (`jr.energy.xc_lda`, `jr.potential.xc_lda`). Should I update these tests as part of this work?

---

## Implementation Plan

### Step 1: Add grid quantities to `pw.py`

Add the following functions:

- **`sigma_grid()`**: Compute sigma = |nabla rho|^2 from `nabla_density_grid()`.
  - Unpolarized: `sigma = sum(nabla_rho**2, axis=-1)` → shape `(x, y, z)`
  - Polarized: `sigma_uu, sigma_ud, sigma_dd` → shape `(3, x, y, z)`

- **`tau_grid()`**: Kinetic energy density tau = (1/2) sum_i |nabla psi_i|^2.
  - Requires computing nabla of each orbital on the grid (multiply G-vectors by coefficients, then IFFT).
  - Shape: `(spin, x, y, z)` or `(x, y, z)`

- **`lapl_grid()`**: Laplacian of density nabla^2 rho.
  - In reciprocal space: multiply rho(G) by -|G|^2, then IFFT.
  - Shape: `(spin, x, y, z)` or `(x, y, z)`

### Step 2: Refactor `xc.py` as the XC interface module

Create helper functions in `xc.py` that wrap jxc:

```python
def xc_energy_density(rho, xc_type, polarized, sigma=None, tau=None, lapl=None):
    """Get XC energy density using jxc. Dispatches LDA/GGA/MGGA based on xc_type."""

def xc_potential(rho, xc_type, polarized, sigma=None, tau=None, lapl=None):
    """Get XC potential (vrho, vsigma, vtau, etc.) using jxc order='vxc'."""
```

These functions:
- Detect functional level (LDA/GGA/MGGA) from the name or from jxc metadata
- Pass the appropriate kwargs (sigma, tau, lapl) to jxc
- Handle compound functionals (e.g., "gga_x_pbe+gga_c_pbe") by splitting and summing
- Return dictionaries with the relevant quantities

### Step 3: Update `energy.py`

Replace the current `xc_energy()` implementation:
- Use the new `xc.py` helper
- For GGA: compute sigma from `pw.sigma_grid()` and pass to jxc
- For MGGA: also compute tau from `pw.tau_grid()`
- Keep the integration logic: `E_xc = integral(exc * rho) * vol / num_grid`

### Step 4: Update `potential.py`

Replace `_lda_x_vrho()` and the hardcoded assertion in `effective()`:
- Use `xc.xc_potential()` to get `vrho` (and `vsigma`, `vtau` for GGA/MGGA)
- For LDA: `v_xc = vrho`
- For GGA: `v_xc = vrho - 2 * div(vsigma * nabla_rho)` (requires divergence on grid)
- For MGGA: additional `vtau` term contributes to kinetic part of Hamiltonian
- Remove the `assert xc_type == "lda_x"` guard

### Step 5: Update `hamiltonian.py` (if needed)

- For MGGA, the `vtau` term modifies the kinetic operator in the Hamiltonian matrix.
- This may require passing additional arguments through `potential.effective()` to the Hamiltonian.
- For LDA/GGA, no changes needed since the potential is still a local multiplicative operator.

### Step 6: Update tests

- Fix references to old API (`xc_lda`)
- Add tests for GGA energy/potential (e.g., PBE)
- Add tests for sigma/tau grid quantities vs. finite differences

---

## Summary of completed work

### Step 1: `pw.py` — new grid quantities

Added 4 functions:

- **`grad_density_grid(coeff, vol, g_vector_grid, occupation, k_weights)`** — computes ∇ρ on the full real-space grid via `iG·ρ(G)` in reciprocal space. Returns shape `(spin, x, y, z, 3)`.
- **`sigma_grid(...)`** — computes σ = |∇ρ|² by squaring and summing `grad_density_grid`. Returns shape `(spin, x, y, z)`.
- **`tau_grid(coeff, vol, g_vector_grid, kpts, occupation, k_weights)`** — computes kinetic energy density τ = ½ Σ_i f_i |∇ψ_i|² per orbital then contracts with occupation. Handles k-points by shifting G → G+k. Returns shape `(spin, x, y, z)`.
- **`lapl_grid(...)`** — computes ∇²ρ via `-|G|²ρ(G)` in reciprocal space. Returns shape `(spin, x, y, z)`.

Added 9 tests in `pw_test.py`: shape tests for all 4 functions, grad matches single-point analytical formula, σ = |∇ρ|², ∫τ dr = kinetic energy (Parseval), τ ≥ 0, ∫∇²ρ dr = 0. All pass.

### Step 2: `xc.py` — centralized XC interface

Created `jrystal/_src/xc.py` wrapping `jxc`:

- **`xc_level(xc_type)`** — detects highest functional level (LDA/GGA/MGGA) among compound components.
- **`_raw_exc(name, polarized, rho, sigma, tau, lapl)`** — bypasses buggy `jxc` public wrapper for `exc` order; calls raw internal functions (`jxc.functionals.<name>.unpol/pol`) directly, providing explicit zero arrays for optional MGGA args (`lapl`).
- **`compute_sigma(density_grid, g_vector_grid)`** — computes contracted density gradient from `(spin, x, y, z)` density. Unpolarized: `(x, y, z)`. Polarized: `(3, x, y, z)` = `[σ_uu, σ_ud, σ_dd]`.
- **`xc_energy_density(rho, xc_type, polarized, sigma, tau, lapl)`** — compound-functional-aware exc via `_raw_exc`, splits on `+` and sums.
- **`xc_potential(rho, xc_type, polarized, sigma, tau)`** — compound-functional-aware vxc via `jxc.get_xc_functional(..., order='vxc')` (the public API works for vxc order).
- **`gga_xc_potential(vrho, vsigma, density_grid, g_vector_grid)`** — computes full GGA local potential `v_xc = vrho − 2∇·(vsigma·∇ρ)` with divergence in reciprocal space. Handles both unpolarized and polarized cases.

Added `xc` to `__init__.py` imports/exports.

### Step 3: `energy.py` — updated XC energy

- Replaced `import jxc` with `from . import xc as _xc`.
- `xc_energy()` now accepts a `tau` parameter; uses `_xc.xc_level()` to detect level; computes `sigma` via `_xc.compute_sigma()` for GGA/MGGA; calls `_xc.xc_energy_density()`.
- `total_energy()` computes `tau` via `pw.tau_grid()` when MGGA is detected and passes it through.

### Step 4: `potential.py` — generalized XC potential

- Removed hardcoded `_lda_x_vrho()` and `assert xc_type == "lda_x"`.
- Added `from . import xc as _xc`.
- `effective()` now: detects level via `_xc.xc_level()`; computes `sigma` for GGA/MGGA via `_xc.compute_sigma()`; calls `_xc.xc_potential()` for `vrho`/`vsigma`; for LDA uses `vrho` directly; for GGA calls `_xc.gga_xc_potential()` for the full divergence-corrected potential.
- Handles jxc output convention (spin axis last) → jrystal convention (spin axis first).

### Step 5: `hamiltonian.py` — no changes needed

Delegates to `potential.effective()` which now transparently handles all XC levels.

### Known limitations

- **MGGA potential**: Only the local part (vrho + vsigma divergence) is implemented. The `vtau` term that modifies the kinetic operator in the Hamiltonian is not yet wired up.
- **Step 6 (tests)**: `energy_test.py` still references old API (`jr.energy.xc_lda`, `jr.potential.xc_lda`). Not yet updated.

### jxc bugs worked around

1. `jxc.get_xc_functional(..., order='exc')` is broken for polarized and MGGA functionals — passes `None` for optional grid args that generated Maple code tries to do arithmetic on. Workaround: `_raw_exc()` calls raw internal functions directly.
2. `jxc` doesn't support compound functional syntax (`'a+b'`). Workaround: split on `+` and sum results in all xc.py wrapper functions.



## Round 3

[Tianbo]
- why there is only _gga_xc_potential, does it repeats with xc_potential?

---

## Unified XC API Design

### Problem

The current XC interface leaks too much internal complexity to callers. Every call site
(`energy.py`, `potential.py`, and eventually `scf/`) must repeat the same boilerplate:

1. Detect the functional level with `xc_level()`
2. Strip / reshape the spin axis to match jxc convention (`rho` vs `density_grid`)
3. Compute `sigma` via `compute_sigma()` for GGA/MGGA
4. Prepare `tau` for MGGA
5. Call `xc_energy_density()` or `xc_potential()` with the right kwargs
6. Post-process the result based on level (LDA → use `vrho` directly;
   GGA → call `_gga_xc_potential`; MGGA → call `_mgga_xc_potential`)
7. Convert spin axis from jxc convention (last) back to jrystal convention (first)

There is also duplication: `xc.compute_sigma` and `pw.sigma_grid` overlap
(see discussion above — `compute_sigma` handles polarized cross-terms correctly,
`sigma_grid` does not).

### Design goals

- **One call** to get exc or vxc — no level detection or sigma/tau prep by the caller.
- **Consistent inputs**: always `(density_grid, g_vector_grid)` in jrystal convention
  `(spin, x, y, z)`, plus optional `tau` when the caller has it.
- **Consistent outputs**: always in jrystal convention `(spin, x, y, z)`.
- Keep the low-level functions (`_raw_exc`, `_raw_vxc`, `_gga_xc_potential`,
  `_mgga_xc_potential`) as private implementation details.

### Proposed public API

```python
def exc(density_grid, g_vector_grid, xc_type, tau=None):
    """Compute XC energy density per particle.

    Handles LDA/GGA/MGGA dispatch, sigma computation, and spin conventions
    internally.

    Args:
        density_grid: (spin, x, y, z) real-space density.
        g_vector_grid: (x, y, z, 3) G-vector grid.
        xc_type: functional name or compound ('+'–separated).
        tau: (spin, x, y, z) kinetic energy density. Required for MGGA;
            ignored for LDA/GGA.

    Returns:
        (x, y, z) energy density per particle (summed over functional
        components).
    """


def vxc(density_grid, g_vector_grid, xc_type, tau=None):
    """Compute XC potential on the real-space grid.

    Returns the local multiplicative potential for all levels. For MGGA,
    also returns vtau for the non-local per-orbital operator.

    Args:
        density_grid: (spin, x, y, z) real-space density.
        g_vector_grid: (x, y, z, 3) G-vector grid.
        xc_type: functional name or compound ('+'–separated).
        tau: (spin, x, y, z) kinetic energy density. Required for MGGA;
            ignored for LDA/GGA.

    Returns:
        (v_xc_local, vtau):
            v_xc_local: (spin, x, y, z) local XC potential.
            vtau: (spin, x, y, z) tau derivative for the non-local
                operator -1/2 div(vtau nabla psi_i). None for LDA/GGA.
    """
```

### Internal flow

```
caller
  │
  ├── exc(density_grid, g_vector_grid, xc_type, tau)
  │     ├── xc_level(xc_type)
  │     ├── polarized = density_grid.shape[0] == 2
  │     ├── rho = density_grid if polarized else density_grid[0]
  │     ├── sigma = compute_sigma(density_grid, g_vector_grid)  [GGA/MGGA]
  │     ├── tau_arg = tau if polarized else tau[0]               [MGGA]
  │     └── xc_energy_density(rho, xc_type, polarized, sigma, tau_arg)
  │
  └── vxc(density_grid, g_vector_grid, xc_type, tau)
        ├── xc_level(xc_type)
        ├── polarized, rho, sigma, tau_arg  [same prep as above]
        ├── vxc_dict = xc_potential(rho, xc_type, polarized, sigma, tau_arg)
        │
        ├── LDA:   v_local = vrho reshaped to (spin, x, y, z)
        │          vtau = None
        │
        ├── GGA:   v_local = _gga_xc_potential(vrho, vsigma, density_grid, g_vector_grid)
        │          vtau = None
        │
        └── MGGA:  v_local, vtau = _mgga_xc_potential(
                       vrho, vsigma, vxc_dict['vtau'], vxc_dict.get('vlapl'),
                       density_grid, g_vector_grid)
```

### What callers look like after

**energy.py** — `xc_energy()`:
```python
exc_grid = _xc.exc(density_grid, g_vector_grid, xc_type, tau)
e_xc = jnp.sum(exc_grid * density_grid) * vol / num_grid
```

**potential.py** — `effective()`:
```python
v_xc_local, vtau = _xc.vxc(density_grid, g_vector_grid, xc_type, tau)
# v_xc_local is ready to use; vtau forwarded to Hamiltonian for MGGA
```

### Cleanup

- `pw.sigma_grid` can be removed (or deprecated) — `compute_sigma` inside xc.py
  is the canonical implementation and correctly handles polarized cross-terms.
- `xc_energy_density` and `xc_potential` become internal helpers (rename to
  `_xc_energy_density`, `_xc_potential` or leave as-is — they're in `_src`
  already so not truly public).
- `xc_level` stays public — useful for callers that need to know whether to
  compute `tau` before calling `exc`/`vxc`.

### Open question: tau lifecycle

MGGA requires `tau` which depends on PW coefficients, not just density.
The XC module shouldn't compute tau itself (that's `pw.tau_grid`'s job).
So the caller must still do:

```python
tau = pw.tau_grid(coeff, vol, g_vector_grid, kpts, occupation) if level == 'mgga' else None
```

This is unavoidable — `exc`/`vxc` intentionally take `tau` as an optional input
rather than taking coefficients, to keep the XC module independent of the PW
representation. The alternative (passing coefficients into xc.py) would create
a circular dependency.