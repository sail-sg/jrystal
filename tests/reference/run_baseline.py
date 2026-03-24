"""Run baseline benchmarks and save reference values.

Usage:
    .venv/bin/python tests/reference/run_baseline.py
"""
import sys
import os
import yaml
import time

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jrystal._src import energy, pw
from jrystal._src.crystal import Crystal
from jrystal._src.grid import (
    g_vectors, r_vectors, proper_grid_size, spherical_mask
)
from jrystal._src.ewald import ewald_coulomb_repulsion
from jrystal._src.grid import translation_vectors
from jrystal._src.occupation import get_occupation_fn, params_init
from math import ceil


def run_all_electron_diamond(
    grid_size=24, k_grid=1, cutoff=50, epochs=2000,
    xc="lda_x", lr=0.01, smearing=0.001
):
    """Run a small all-electron diamond benchmark."""
    print("=" * 60)
    print("All-electron diamond baseline")
    print("=" * 60)

    pkg_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    crystal = Crystal.create_from_file(f"{pkg_path}/geometry/diamond.xyz", spin=0)
    print(f"Crystal: {crystal.symbols}, atoms: {crystal.num_atom}, "
          f"electrons: {crystal.num_electron}")

    grid_sizes = proper_grid_size(grid_size)
    g_vec = g_vectors(crystal.cell_vectors, grid_sizes)
    r_vec = r_vectors(crystal.cell_vectors, grid_sizes)

    # Single Gamma k-point for speed
    k_vec = np.zeros([1, 3])
    k_weights = np.ones([1]) / 1.0
    num_kpts = 1

    mask = spherical_mask(crystal.cell_vectors, grid_sizes, cutoff)
    print(f"Grid: {grid_sizes}, cutoff: {cutoff} Ha, "
          f"G-points: {np.sum(mask)}, mask%: {np.mean(mask)*100:.1f}%")

    # Ewald
    ewald_grid = translation_vectors(crystal.cell_vectors, 2e4)
    ew = ewald_coulomb_repulsion(
        crystal.positions, crystal.charges,
        g_vec, crystal.vol, ewald_eta=0.1, ewald_grid=ewald_grid
    )
    print(f"Ewald energy: {float(ew):.6f} Ha")

    num_electrons = int(crystal.num_electron)
    num_bands = ceil(num_electrons / 2) + 8

    # Init params
    key = jax.random.PRNGKey(123)
    params_pw = pw.param_init(key, num_bands, num_kpts, mask, spin_restricted=True)
    params_occ = params_init(num_bands, num_kpts, method="simplex-projector")
    occ_fn = get_occupation_fn(num_electrons, spin=0, spin_restricted=True,
                               method="simplex-projector")

    def total_energy_fn(params_pw, params_occ):
        coeff = pw.coeff(params_pw, mask)
        occ = occ_fn(params_occ)
        density = pw.density_grid(coeff, crystal.vol, occ)
        density_reciprocal = pw.density_grid_reciprocal(coeff, crystal.vol, occ)
        kin = energy.kinetic(coeff, g_vec, k_vec, occupation=occ)
        hart = energy.hartree(density_reciprocal, g_vec, crystal.vol)
        ext = energy.external(
            density_reciprocal, crystal.positions, crystal.charges,
            g_vec, crystal.vol
        )
        xc_e = energy.xc_energy(density, g_vec, crystal.vol, xc, kohn_sham=False)
        return kin + hart + ext + xc_e

    def free_energy_fn(params_pw, params_occ):
        te = total_energy_fn(params_pw, params_occ)
        return te, te  # no entropy term for simplex-projector

    import optax
    optimizer = optax.adam(learning_rate=lr)
    params = {"pw": params_pw, "occ": params_occ}
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state):
        loss_fn = lambda x: free_energy_fn(x["pw"], x["occ"])
        (loss_val, etot), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, etot

    print(f"Running {epochs} epochs...")
    start_time = time.time()
    etot_val = 0.0
    for i in range(epochs):
        params, opt_state, loss_val, etot_val = update(params, opt_state)
        if i % 500 == 0 or i == epochs - 1:
            e = float(jax.block_until_ready(etot_val))
            print(f"  epoch {i:5d}: total_energy = {e + float(ew):.6f} Ha")

    elapsed = time.time() - start_time
    print(f"  wall time: {elapsed:.1f}s")

    # Final energy decomposition
    coeff = pw.coeff(params["pw"], mask)
    occ = occ_fn(params["occ"])
    density = pw.density_grid(coeff, crystal.vol, occ)
    density_reciprocal = pw.density_grid_reciprocal(coeff, crystal.vol, occ)
    kin = float(energy.kinetic(coeff, g_vec, k_vec, occupation=occ))
    hart = float(energy.hartree(density_reciprocal, g_vec, crystal.vol))
    ext = float(energy.external(
        density_reciprocal, crystal.positions, crystal.charges, g_vec, crystal.vol
    ))
    xc_e = float(energy.xc_energy(density, g_vec, crystal.vol, xc, kohn_sham=False))
    etot_final = float(etot_val)

    result = {
        "system": "diamond (C2)",
        "method": "all-electron",
        "xc": xc,
        "grid_sizes": [int(x) for x in grid_sizes],
        "cutoff_energy_ha": cutoff,
        "k_grid": [k_grid, k_grid, k_grid],
        "num_kpts": num_kpts,
        "num_bands": num_bands,
        "optimizer": "adam",
        "learning_rate": lr,
        "smearing": smearing,
        "epochs": epochs,
        "seed": 123,
        "energy_ha": {
            "kinetic": round(kin, 8),
            "hartree": round(hart, 8),
            "external": round(ext, 8),
            "xc": round(xc_e, 8),
            "ewald": round(float(ew), 8),
            "electronic_total": round(etot_final, 8),
            "total": round(etot_final + float(ew), 8),
        },
        "wall_time_seconds": round(elapsed, 1),
        "notes": "Gamma-only, spin-restricted, uniform occupation, "
                 "no k-weights (single kpt so irrelevant)",
    }

    print("\nFinal energy decomposition:")
    for k, v in result["energy_ha"].items():
        print(f"  {k:25s}: {v:.8f} Ha")

    return result


def run_normcons_diamond(
    grid_size=24, k_grid=1, cutoff=50, epochs=2000,
    xc="lda_x", lr=0.01, smearing=0.001
):
    """Run a norm-conserving pseudopotential diamond benchmark."""
    print("\n" + "=" * 60)
    print("Norm-conserving pseudopotential diamond baseline")
    print("=" * 60)

    from jrystal.pseudopotential import NormConservingPseudopotential
    from jrystal.pseudopotential import normcons

    pkg_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    crystal = Crystal.create_from_file(f"{pkg_path}/geometry/diamond.xyz", spin=0)
    pp_dir = f"{pkg_path}/pseudopotential/normconserving/"
    pp = NormConservingPseudopotential.create(crystal, pp_dir)
    valence_charges = float(np.sum(pp.valence_charges))
    print(f"Crystal: {crystal.symbols}, valence electrons: {valence_charges}")

    grid_sizes = proper_grid_size(grid_size)
    g_vec = g_vectors(crystal.cell_vectors, grid_sizes)
    r_vec = r_vectors(crystal.cell_vectors, grid_sizes)

    k_vec = np.zeros([1, 3])
    k_weights = np.ones([1]) / 1.0
    num_kpts = 1

    mask = spherical_mask(crystal.cell_vectors, grid_sizes, cutoff)
    print(f"Grid: {grid_sizes}, cutoff: {cutoff} Ha, "
          f"G-points: {np.sum(mask)}, mask%: {np.mean(mask)*100:.1f}%")

    ewald_grid = translation_vectors(crystal.cell_vectors, 2e4)
    ew = ewald_coulomb_repulsion(
        crystal.positions, crystal.charges,
        g_vec, crystal.vol, ewald_eta=0.1, ewald_grid=ewald_grid
    )
    print(f"Ewald energy: {float(ew):.6f} Ha")

    num_electrons = int(valence_charges)
    num_bands = ceil(num_electrons / 2) + 8

    # Pseudopotential setup
    print("Computing local potential...")
    potential_loc = normcons.potential_local_reciprocal(
        crystal.positions, g_vec,
        pp.r_grid, pp.local_potential_grid,
        pp.local_potential_charge, crystal.vol
    )

    print("Computing nonlocal potential (SBT)...")
    from jrystal.pseudopotential.beta import _beta_sbt_single_atom
    beta_gk = []
    for r, b, l in zip(pp.r_grid, pp.nonlocal_beta_grid,
                        pp.nonlocal_angular_momentum):
        beta_gk.append(_beta_sbt_single_atom(r, b, l, np.array(g_vec),
                                              np.array(k_vec)))

    print("Computing nonlocal potential...")
    potential_nl = normcons.potential_nonlocal_psi_reciprocal(
        crystal.positions, g_vec, k_vec,
        pp.r_grid, pp.nonlocal_beta_grid,
        pp.nonlocal_angular_momentum,
        pp.nonlocal_d_matrix, beta_gk
    )

    # Init params
    key = jax.random.PRNGKey(123)
    params_pw = pw.param_init(key, num_bands, num_kpts, mask, spin_restricted=True)
    params_occ = params_init(num_bands, num_kpts, method="simplex-projector")
    occ_fn = get_occupation_fn(num_electrons, spin=0, spin_restricted=True,
                               method="simplex-projector")

    def total_energy_fn(params_pw, params_occ):
        coeff = pw.coeff(params_pw, mask)
        occ = occ_fn(params_occ)
        density = pw.density_grid(coeff, crystal.vol, occ)
        density_reciprocal = pw.density_grid_reciprocal(coeff, crystal.vol, occ)
        kin = energy.kinetic(coeff, g_vec, k_vec, occupation=occ)
        hart = energy.hartree(density_reciprocal, g_vec, crystal.vol)
        ext_loc = normcons.energy_local(
            density_reciprocal, potential_loc, vol=crystal.vol
        )
        ext_nloc = normcons.energy_nonlocal(
            coeff, potential_nl, vol=crystal.vol, occupation=occ
        )
        xc_e = energy.xc_energy(density, g_vec, crystal.vol, xc, kohn_sham=False)
        return kin + hart + ext_loc + ext_nloc + xc_e

    def free_energy_fn(params_pw, params_occ):
        te = total_energy_fn(params_pw, params_occ)
        return te, te

    import optax
    optimizer = optax.adam(learning_rate=lr)
    params = {"pw": params_pw, "occ": params_occ}
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state):
        loss_fn = lambda x: free_energy_fn(x["pw"], x["occ"])
        (loss_val, etot), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, etot

    print(f"Running {epochs} epochs...")
    start_time = time.time()
    etot_val = 0.0
    for i in range(epochs):
        params, opt_state, loss_val, etot_val = update(params, opt_state)
        if i % 500 == 0 or i == epochs - 1:
            e = float(jax.block_until_ready(etot_val))
            print(f"  epoch {i:5d}: total_energy = {e + float(ew):.6f} Ha")

    elapsed = time.time() - start_time
    print(f"  wall time: {elapsed:.1f}s")

    # Final energy decomposition
    coeff = pw.coeff(params["pw"], mask)
    occ = occ_fn(params["occ"])
    density = pw.density_grid(coeff, crystal.vol, occ)
    density_reciprocal = pw.density_grid_reciprocal(coeff, crystal.vol, occ)
    kin = float(energy.kinetic(coeff, g_vec, k_vec, occupation=occ))
    hart = float(energy.hartree(density_reciprocal, g_vec, crystal.vol))
    ext_loc = float(normcons.energy_local(
        density_reciprocal, potential_loc, vol=crystal.vol
    ))
    ext_nloc = float(normcons.energy_nonlocal(
        coeff, potential_nl, vol=crystal.vol, occupation=occ
    ))
    xc_e = float(energy.xc_energy(density, g_vec, crystal.vol, xc, kohn_sham=False))
    etot_final = float(etot_val)

    result = {
        "system": "diamond (C2)",
        "method": "norm-conserving pseudopotential",
        "xc": xc,
        "grid_sizes": [int(x) for x in grid_sizes],
        "cutoff_energy_ha": cutoff,
        "k_grid": [k_grid, k_grid, k_grid],
        "num_kpts": num_kpts,
        "num_bands": num_bands,
        "optimizer": "adam",
        "learning_rate": lr,
        "smearing": smearing,
        "epochs": epochs,
        "seed": 123,
        "pseudopotential": "C.pz-vbc.UPF",
        "energy_ha": {
            "kinetic": round(kin, 8),
            "hartree": round(hart, 8),
            "external_local": round(ext_loc, 8),
            "external_nonlocal": round(ext_nloc, 8),
            "xc": round(xc_e, 8),
            "ewald": round(float(ew), 8),
            "electronic_total": round(etot_final, 8),
            "total": round(etot_final + float(ew), 8),
        },
        "wall_time_seconds": round(elapsed, 1),
        "notes": "Gamma-only, spin-restricted, simplex-projector occupation, "
                 "no k-weights (single kpt so irrelevant)",
    }

    print("\nFinal energy decomposition:")
    for k, v in result["energy_ha"].items():
        print(f"  {k:25s}: {v:.8f} Ha")

    return result


def main():
    result_ae = run_all_electron_diamond()
    outpath = os.path.join(
        os.path.dirname(__file__), "baseline_ae_diamond.yaml"
    )
    with open(outpath, "w") as f:
        yaml.dump(result_ae, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved to {outpath}")

    result_nc = run_normcons_diamond()
    outpath = os.path.join(
        os.path.dirname(__file__), "baseline_nc_diamond.yaml"
    )
    with open(outpath, "w") as f:
        yaml.dump(result_nc, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
