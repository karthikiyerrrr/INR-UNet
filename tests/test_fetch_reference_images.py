import importlib.util
from pathlib import Path

# Load the standalone script as a module (scripts/ is not a package).
_spec = importlib.util.spec_from_file_location(
    "fetch_reference_images",
    Path(__file__).resolve().parents[1] / "scripts" / "fetch_reference_images.py",
)
fetch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch)


def test_params_download_url_maps_image_to_params():
    img = (
        "https://raw.githubusercontent.com/xinhuolin/TEM-ImageNet-v1.3/"
        "master/image/00042.png"
    )
    assert fetch.params_download_url(img) == (
        "https://raw.githubusercontent.com/xinhuolin/TEM-ImageNet-v1.3/"
        "master/params/00042.txt"
    )


def test_params_download_url_handles_tif():
    img = "https://host/x/image/00098.tif"
    assert fetch.params_download_url(img) == "https://host/x/params/00098.txt"
