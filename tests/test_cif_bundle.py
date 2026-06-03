"""The committed CIF bundle parses and projects."""

from pathlib import Path

import torch
import yaml
from pymatgen.core import Structure

from inr_unet.data.projection import project_structure

CIF_DIR = Path("src/inr_unet/data/cif")


def _load(name):
    return Structure.from_file(CIF_DIR / name)


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


def test_graphene_projects_to_hex_columns():
    pos, z, count, basis = project_structure(
        _load("graphene.cif"), [0, 0, 1], fov_A=20.0, n_exponent=1.7, group_tol_A=0.3
    )
    assert pos.shape[0] > 20
    lengths = torch.linalg.norm(basis, dim=1)
    cosang = torch.dot(basis[0], basis[1]) / (lengths[0] * lengths[1])
    angle = torch.rad2deg(torch.arccos(cosang)).item()
    assert abs(angle - 120.0) < 1.0           # hexagonal frame
    assert torch.allclose(z, torch.full_like(z, 6.0), atol=1e-3)  # carbon only


def test_mos2_projects_with_mo_and_s_columns():
    pos, z, count, basis = project_structure(
        _load("MoS2.cif"), [0, 0, 1], fov_A=20.0, n_exponent=1.7, group_tol_A=0.3
    )
    assert pos.shape[0] > 20
    zset = {round(float(v)) for v in z.unique()}
    assert 42 in zset and 16 in zset         # Mo (Z=42) and S (Z=16) columns
    # the S columns stack two atoms along the beam
    assert float(count.max()) >= 2.0
