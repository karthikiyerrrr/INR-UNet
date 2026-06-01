"""The committed CIF bundle parses and projects."""

from pathlib import Path

import yaml
from pymatgen.core import Structure

from inr_unet.data.projection import project_structure

CIF_DIR = Path("src/inr_unet/data/cif")


def test_manifest_lists_existing_cifs():
    manifest = yaml.safe_load((CIF_DIR / "manifest.yaml").read_text())
    entries = manifest["entries"]
    assert len(entries) >= 8
    for entry in entries:
        assert (CIF_DIR / entry["cif"]).exists()
        assert len(entry["zone_axis"]) == 3


def test_each_manifest_entry_projects_to_columns():
    manifest = yaml.safe_load((CIF_DIR / "manifest.yaml").read_text())
    for entry in manifest["entries"]:
        structure = Structure.from_file(CIF_DIR / entry["cif"])
        pos, z, count, basis = project_structure(
            structure, entry["zone_axis"], fov_A=30.0, n_exponent=1.7, group_tol_A=0.4
        )
        assert pos.shape[0] > 0, f"no columns for {entry}"
        assert (count >= 1.0).all()
        assert tuple(basis.shape) == (2, 2)
