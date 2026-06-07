"""Persistent render cache: materialize RenderedSamples once and serve them from RAM.

The cache stores only the rendered scene (image + atom positions/radii + pixel size), never
labels or query points -- those are cheap and idx-deterministic, so they stay derived on the fly
and changing label_kind / sample_q must not invalidate the cache.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.utils.data
from omegaconf import OmegaConf

from inr_unet.data.render_source import RenderedSample, SyntheticRenderSource

if TYPE_CHECKING:
    from omegaconf import DictConfig

_SCHEMA_VERSION = 1

# Top-level data.* knobs that do not affect the rendered bytes (loader/cache/storage settings).
_DATA_NONRENDER_KEYS = frozenset(
    {"root", "image_size", "batch_size", "num_workers", "cache_scenes",
     "persistent_workers", "pin_memory", "prefetch_factor", "render_cache"}
)
# data.synthetic.* knobs consumed only when deriving labels/queries (not by the renderer/crop).
_LABEL_QUERY_KEYS = frozenset(
    {"label_kind", "gaussian_fwhm_A", "gaussian_sigma_floor_px",
     "sample_q", "target_pixel_size_A_min", "target_pixel_size_A_max"}
)


def _git_sha() -> str:
    """HEAD SHA, suffixed '-dirty' if the working tree has uncommitted changes; 'nogit' if not a
    git repo. A code change to render logic flips this, so a stale cache is never reused."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def cache_key(cfg: DictConfig) -> str:
    """Stable 16-char key over render-affecting + split-determining config and the code SHA.

    Excludes loader/cache knobs and all label/query knobs (derived on the fly). Includes the split
    config because it determines which indices get materialized.
    """
    data = OmegaConf.to_container(cfg.data, resolve=True)
    assert isinstance(data, dict)
    for k in _DATA_NONRENDER_KEYS:
        data.pop(k, None)
    syn = data.get("synthetic")
    if isinstance(syn, dict):
        for k in _LABEL_QUERY_KEYS:
            syn.pop(k, None)
    keyed = {
        "data": data,
        "generation": OmegaConf.to_container(cfg.generation, resolve=True),
        "split": OmegaConf.to_container(cfg.train.split, resolve=True),
        "eval_draws_per_scene": int(cfg.train.eval.eval_draws_per_scene),
        "schema": _SCHEMA_VERSION,
        "git": _git_sha(),
    }
    blob = json.dumps(keyed, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _first(batch):
    """Identity collate for batch_size=1 (module-level so DataLoader workers can pickle it)."""
    return batch[0]


class _IndexItemDataset(torch.utils.data.Dataset):
    """Serves ``source.get(indices[i])`` so a DataLoader can render the pre-gen pass in parallel."""

    def __init__(self, source, indices: list[int]) -> None:
        self._source = source
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> RenderedSample:
        return self._source.get(self._indices[i])


def _progress(done: int, total: int) -> None:
    if total and (done == total or done % max(1, total // 10) == 0):
        print(f"[cache] rendered {done}/{total}", flush=True)


def _render_all(cfg: DictConfig, source, indices: list[int]) -> list[RenderedSample]:
    """Render every index once. Uses DataLoader workers when num_workers > 0 (single pass), else a
    plain in-process loop. Order matches ``indices`` (no shuffle)."""
    nw = int(cfg.data.num_workers)
    if nw <= 0:
        out = []
        for n, i in enumerate(indices):
            out.append(source.get(i))
            _progress(n + 1, len(indices))
        return out
    loader = torch.utils.data.DataLoader(
        _IndexItemDataset(source, indices), batch_size=1, num_workers=nw, collate_fn=_first
    )
    out = []
    for n, s in enumerate(loader):
        out.append(s)
        _progress(n + 1, len(indices))
    return out


@dataclass
class RenderCache:
    """In-RAM bundle of RenderedSamples with CSR-ragged atom columns."""

    key: str
    crop_size: int
    draws_per_scene: int
    idx: torch.Tensor             # [N] int64, materialized indices (sorted)
    image: torch.Tensor           # [N, S, S] float32
    offsets: torch.Tensor         # [N+1] int64, CSR offsets into positions/radii
    positions: torch.Tensor       # [sum M, 2] float32
    radii: torch.Tensor           # [sum M] float32
    pixel_size_A: torch.Tensor    # [N] float64 — preserves Python-float precision on round-trip
    valid_extent_A: torch.Tensor  # [N] float64

    @classmethod
    def build(cls, cfg: DictConfig, indices, *, source=None) -> RenderCache:
        idx_sorted = sorted(int(i) for i in indices)
        source = source if source is not None else SyntheticRenderSource(cfg)
        samples = _render_all(cfg, source, idx_sorted)
        counts = [int(s.positions_A.shape[0]) for s in samples]
        offsets = torch.zeros(len(samples) + 1, dtype=torch.int64)
        if counts:
            offsets[1:] = torch.tensor(counts, dtype=torch.int64).cumsum(0)
        positions = (
            torch.cat([s.positions_A for s in samples], dim=0)
            if samples else torch.zeros(0, 2)
        )
        radii = (
            torch.cat([s.radii_A for s in samples], dim=0)
            if samples else torch.zeros(0)
        )
        return cls(
            key=cache_key(cfg),
            crop_size=int(source.crop_size),
            draws_per_scene=int(source.draws_per_scene),
            idx=torch.tensor(idx_sorted, dtype=torch.int64),
            image=(torch.stack([s.image for s in samples]) if samples
                   else torch.empty(0, int(source.crop_size), int(source.crop_size))),
            offsets=offsets,
            positions=positions.to(torch.float32),
            radii=radii.to(torch.float32),
            pixel_size_A=torch.tensor(
                [s.input_pixel_size_A for s in samples], dtype=torch.float64
            ),
            valid_extent_A=torch.tensor(
                [s.valid_extent_A for s in samples], dtype=torch.float64
            ),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                # informational only; cache invalidation is enforced via `key` (which folds in
                # _SCHEMA_VERSION), not by reading this field back in load().
                "schema_version": _SCHEMA_VERSION,
                "key": self.key,
                "crop_size": self.crop_size,
                "draws_per_scene": self.draws_per_scene,
                "idx": self.idx,
                "image": self.image,
                "offsets": self.offsets,
                "positions": self.positions,
                "radii": self.radii,
                "pixel_size_A": self.pixel_size_A,
                "valid_extent_A": self.valid_extent_A,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> RenderCache:
        d = torch.load(Path(path), map_location="cpu", weights_only=True)
        return cls(
            key=d["key"], crop_size=d["crop_size"], draws_per_scene=d["draws_per_scene"],
            idx=d["idx"], image=d["image"], offsets=d["offsets"],
            positions=d["positions"], radii=d["radii"],
            pixel_size_A=d["pixel_size_A"], valid_extent_A=d["valid_extent_A"],
        )


class CachedRenderSource:
    """Serves RenderedSamples from a RenderCache. Duck-types SyntheticRenderSource for what the
    datasets, eval, and panels use: ``get(idx)``, ``crop_size``, ``draws_per_scene``, ``len``.

    ``__len__`` is the number of *cached* indices (the materialized split union), not
    scenes*draws; consumers index by the original idx via ``get`` and never rely on dense length.
    """

    def __init__(self, cache: RenderCache) -> None:
        self._c = cache
        self.crop_size = cache.crop_size
        self.draws_per_scene = cache.draws_per_scene
        self._row = {int(i): r for r, i in enumerate(cache.idx.tolist())}

    def __len__(self) -> int:
        return len(self._row)

    def get(self, idx: int) -> RenderedSample:
        r = self._row.get(int(idx))
        if r is None:
            raise KeyError(
                f"idx {idx} not in render cache ({len(self._row)} cached indices)"
            )
        c = self._c
        lo, hi = int(c.offsets[r]), int(c.offsets[r + 1])
        return RenderedSample(
            image=c.image[r],
            positions_A=c.positions[lo:hi],
            radii_A=c.radii[lo:hi],
            input_pixel_size_A=float(c.pixel_size_A[r]),
            valid_extent_A=float(c.valid_extent_A[r]),
        )
