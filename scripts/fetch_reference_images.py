"""Fetch a sample of real STEM images from xinhuolin/TEM-ImageNet-v1.3 for calibration.

One-shot. Writes to data/reference/ (gitignored). Not imported at runtime.

Usage:
    uv run python scripts/fetch_reference_images.py --count 50
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

_REPO = "xinhuolin/TEM-ImageNet-v1.3"
_DIR = "image"
_API = f"https://api.github.com/repos/{_REPO}/contents/{_DIR}"


def _list_images() -> list[dict]:
    req = urllib.request.Request(_API, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted GitHub host)
        entries = json.load(resp)
    return [e for e in entries if e["name"].lower().endswith((".png", ".tif", ".tiff", ".jpg"))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50, help="number of images to fetch")
    parser.add_argument("--out", type=Path, default=Path("data/reference"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    images = _list_images()
    if not images:
        raise SystemExit(f"No images found at {_API}")
    selected = images[: args.count]
    print(f"Found {len(images)} images; downloading {len(selected)} to {args.out}/")
    for entry in selected:
        dest = args.out / entry["name"]
        urllib.request.urlretrieve(entry["download_url"], dest)  # noqa: S310
        print(f"  {entry['name']}")
    print(f"Done: {len(selected)} images in {args.out}/")


if __name__ == "__main__":
    main()
