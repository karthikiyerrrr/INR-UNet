"""One-shot generator for the bundled canonical CIF set (run manually, not at runtime).

Builds well-known structures from published lattice constants and space groups, writes
them as CIFs under src/inr_unet/data/cif/, and writes a starter manifest of
(cif, zone_axis) entries. Add more CIFs by dropping files in that directory and
extending the manifest. Run: `uv run python scripts/build_cif_set.py`.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

OUT_DIR = Path(__file__).resolve().parents[1] / "src" / "inr_unet" / "data" / "cif"

STRUCTURES = {
    "Pt": dict(sg="Fm-3m", lattice=Lattice.cubic(3.9242), species=["Pt"], coords=[[0, 0, 0]]),
    "NiO": dict(
        sg="Fm-3m",
        lattice=Lattice.cubic(4.177),
        species=["Ni", "O"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
    ),
    "SrTiO3": dict(
        sg="Pm-3m",
        lattice=Lattice.cubic(3.905),
        species=["Sr", "Ti", "O"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0]],
    ),
    "Si": dict(sg="Fd-3m", lattice=Lattice.cubic(5.431), species=["Si"], coords=[[0, 0, 0]]),
    "TiO2_rutile": dict(
        sg="P4_2/mnm",
        lattice=Lattice.tetragonal(4.593, 2.959),
        species=["Ti", "O"],
        coords=[[0, 0, 0], [0.305, 0.305, 0.0]],
    ),
}

MANIFEST_ENTRIES = [
    {"cif": "Pt.cif", "zone_axis": [0, 0, 1]},
    {"cif": "Pt.cif", "zone_axis": [1, 1, 0]},
    {"cif": "NiO.cif", "zone_axis": [0, 0, 1]},
    {"cif": "NiO.cif", "zone_axis": [1, 1, 0]},
    {"cif": "SrTiO3.cif", "zone_axis": [0, 0, 1]},
    {"cif": "SrTiO3.cif", "zone_axis": [1, 1, 0]},
    {"cif": "Si.cif", "zone_axis": [1, 1, 0]},
    {"cif": "TiO2_rutile.cif", "zone_axis": [0, 0, 1]},
    {"cif": "TiO2_rutile.cif", "zone_axis": [1, 0, 0]},
    {"cif": "TiO2_rutile.cif", "zone_axis": [1, 1, 0]},
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in STRUCTURES.items():
        structure = Structure.from_spacegroup(
            spec["sg"], spec["lattice"], spec["species"], spec["coords"]
        )
        CifWriter(structure).write_file(OUT_DIR / f"{name}.cif")
    with open(OUT_DIR / "manifest.yaml", "w") as fh:
        yaml.safe_dump({"entries": MANIFEST_ENTRIES}, fh, sort_keys=False)
    print(f"wrote {len(STRUCTURES)} CIFs + manifest to {OUT_DIR}")


if __name__ == "__main__":
    main()
