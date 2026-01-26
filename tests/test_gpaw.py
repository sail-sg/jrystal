"""Run GPAW PW calculations from config.yaml and print energy components."""

import ast
from pathlib import Path
import sys

from ase.io import read
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


def run_case(geom_path: Path, cutoff_ha: float, kpts: tuple[int, int, int]) -> None:
    atoms = read(geom_path)
    if not atoms.pbc.any():
        atoms.set_pbc(True)

    name = geom_path.stem
    cutoff_ev = cutoff_ha * HA_TO_EV

    calc = GPAW(
        mode=PW(cutoff_ev),
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

    run_case(geom_path, cutoff_ha, kpts)


if __name__ == "__main__":
    main()
