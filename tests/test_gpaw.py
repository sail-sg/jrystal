"""Run GPAW PW calculations from config.yaml and print energy components."""

import ast
from pathlib import Path
import sys

from ase.io import read
import numpy as np
from gpaw import GPAW, PW

HA_TO_EV = 27.211386245988


def _parse_config(path: Path) -> dict:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        data = {}
        keys = {"crystal", "crystal_file_path_path",
                "cutoff_energy", "k_grid_sizes"}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                if key not in keys:
                    continue
                value = value.strip()
                if value in ("null", "None", ""):
                    data[key] = None
                elif value.startswith(("'", '"')) and value.endswith(("'", '"')):
                    data[key] = value[1:-1]
                elif value.startswith("["):
                    data[key] = ast.literal_eval(value)
                else:
                    try:
                        data[key] = float(value)
                    except ValueError:
                        data[key] = value
        return data


def _export_coefficients(calc: GPAW, output_path: Path) -> None:
    wfs = calc.wfs
    pd = wfs.pd
    shape = tuple(pd.gd.N_c)

    nspins = wfs.nspins
    nkpts = max(kpt.k for kpt in wfs.kpt_u) + 1 if wfs.kpt_u else 0

    # Use first kpt to infer band count
    nbands = 0
    if wfs.kpt_u:
        sample = wfs.kpt_u[0].psit_nG
        if hasattr(sample, "data"):
            sample = sample.data
        sample = np.asarray(sample)
        nbands = sample.shape[0]

    coeff = np.zeros((nspins, nkpts, nbands, *shape), dtype=np.complex128)
    occ = np.zeros((nspins, nkpts, nbands), dtype=float)
    d_asp = {}
    d_asp_calc = {}
    proj = {}
    proj_pw = {}
    d_calc_full = {}
    def pack(D_p: np.ndarray) -> np.ndarray:
        """Pack a Hermitian matrix for better efficiency

        The diagonal elements are halfed to calculate the inner product
        """

        n = D_p.shape[-1]
        tmp = D_p.copy()
        tmp[np.diag_indices(n)] = tmp[np.diag_indices(n)] / 2
        return tmp[np.triu_indices(n)].real * 2

    for kpt in wfs.kpt_u:
        s = kpt.s
        k = kpt.k
        q = kpt.q
        psit_nG = kpt.psit_nG
        if hasattr(psit_nG, "data"):
            psit_nG = psit_nG.data
        psit_nG = np.asarray(psit_nG)
        Q_G = pd.Q_qG[q]
        for b in range(psit_nG.shape[0]):
            flat = np.zeros(np.prod(shape), dtype=np.complex128)
            flat[Q_G] = psit_nG[b]
            coeff[s, k, b] = flat.reshape(shape)
        if kpt.f_n is not None:
            occ[s, k, :psit_nG.shape[0]] = np.asarray(kpt.f_n)
        if hasattr(kpt, "projections") and kpt.projections is not None:
            for a, P_ni in kpt.projections.items():
                proj[f"P_ani_{a}_s{s}_k{k}"] = np.asarray(P_ni)
                if kpt.f_n is not None:
                    f_n = np.asarray(kpt.f_n)
                    D_full = P_ni.conj().T @ (P_ni * f_n[:, None])
                    key = (a, s)
                    if key in d_calc_full:
                        d_calc_full[key] = d_calc_full[key] + D_full.real
                    else:
                        d_calc_full[key] = D_full.real
        # Export projector overlap matrix f_GI and indices once per k-point
        key_f = f"proj_f_GI_k{k}"
        if key_f not in proj_pw:
            f_GI = wfs.pt.expand(q=q, cc=False)
            if wfs.dtype == float:
                f_GI = f_GI[::2] + 1j * f_GI[1::2]
            proj_pw[key_f] = np.asarray(f_GI)
            proj_pw[f"proj_Q_G_k{k}"] = np.asarray(pd.Q_qG[q])
            proj_pw[f"proj_indices_k{k}"] = np.asarray(wfs.pt.my_indices, dtype=int)

    for a, D_sp in calc.density.D_asp.items():
        D_sp = np.asarray(D_sp)
        d_asp[f"D_asp_{a}"] = D_sp
        for s in range(D_sp.shape[0]):
            key = (a, s)
            if key not in d_calc_full:
                continue
            D_pack = pack(d_calc_full[key])
            d_asp_calc[f"D_asp_calc_{a}_s{s}"] = D_pack
            diff = np.max(np.abs(D_pack - D_sp[s]))
            print(f"[D_asp check] atom {a} s{s}: max|Δ| = {diff:.6e}")

    kpts_frac = getattr(wfs.kd, "bzk_kc", None)
    if kpts_frac is None:
        kpts_frac = getattr(wfs.kd, "ibzk_kc", None)

    np.savez(
        output_path,
        coeff=coeff,
        occupation=occ,
        grid_sizes=np.array(shape, dtype=int),
        gvec=np.asarray(pd.G_Qv),
        kpts_frac=None if kpts_frac is None else np.asarray(kpts_frac),
        cell=np.asarray(wfs.gd.cell_cv),
        **d_asp,
        **d_asp_calc,
        **proj,
        **proj_pw,
    )


def run_case(
    geom_path: Path,
    cutoff_ha: float,
    kpts: tuple[int, int, int],
    export_path: Path | None = None,
) -> None:
    atoms = read(geom_path)
    if not atoms.pbc.any():
        atoms.set_pbc(True)

    name = geom_path.stem
    cutoff_ev = cutoff_ha * HA_TO_EV

    calc = GPAW(
        mode=PW(cutoff_ev, force_complex_dtype=True),
        xc="LDA",
        setups="paw",
        kpts=kpts,
        txt=None,
        eigensolver={'name': 'etdm-fdpw'},
        mixer={'backend': 'no-mixing'},
        occupations={'name': 'fixed-uniform'},
        symmetry='off',
    )

    atoms.calc = calc
    energy = atoms.get_potential_energy()
    h = calc.hamiltonian

    print(f"\n=== {name} ===")
    print(energy)
    print("GPAW energy components (Ha):")
    print(f"  e_total_free: {h.e_total_free:.12f}")
    print(f"  e_total_extrapolated: {h.e_total_extrapolated:.12f}")
    print(f"  e_kinetic: {h.e_kinetic:.12f}")
    print(f"  e_coulomb: {h.e_coulomb:.12f}")
    print(f"  e_zero: {h.e_zero:.12f}")
    print(f"  e_external: {h.e_external:.12f}")
    print(f"  e_xc: {h.e_xc:.12f}")
    print(f"  e_entropy: {h.e_entropy:.12f}")
    print(f"  e_total_free (eV): {h.e_total_free * HA_TO_EV:.12f}")
    print("GPAW occupations:")
    wfs = calc.wfs
    for i, kpt in enumerate(wfs.kpt_u):
        kvec = getattr(kpt, "k", None)
        if kvec is None:
            kvec = getattr(kpt, "q", None)
        weight = getattr(kpt, "weight", None)
        spin = getattr(kpt, "s", None)
        f_n = getattr(kpt, "f_n", None)
        if f_n is None:
            print(f"  kpt {i}: occupations not available")
            continue
        f_str = " ".join(f"{float(x):.6f}" for x in f_n)
        print(f"  kpt {i} spin {spin} weight {weight} k={kvec}")
        print(f"    f_n: {f_str}")

    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        _export_coefficients(calc, export_path)
        print(f"GPAW coefficients saved to {export_path}")


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config.yaml")
    cfg = _parse_config(config_path)

    crystal = cfg.get("crystal")
    crystal_file = cfg.get("crystal_file_path_path")
    cutoff_ha = float(cfg.get("cutoff_energy", 40.0))
    k_grid = cfg.get("k_grid_sizes", [1, 1, 1])
    if isinstance(k_grid, (int, float)):
        k_grid = [int(k_grid)] * 3
    kpts = tuple(int(k) for k in k_grid)

    if crystal:
        geom_path = Path("geometry") / f"{crystal}.xyz"
    elif crystal_file:
        geom_path = Path(crystal_file)
    else:
        raise ValueError("No crystal or crystal_file_path_path in config.")

    export_path = cfg.get("gpaw_coeff_path")
    if export_path:
        export_path = Path(export_path)
    else:
        export_path = Path("log") / f"gpaw_coeff_{geom_path.stem}.npz"

    run_case(geom_path, cutoff_ha, kpts, export_path=export_path)


if __name__ == "__main__":
    main()
