"""Persistent render cache: materialize RenderedSamples once and serve them from RAM.

The cache stores only the rendered scene (image + atom positions/radii + pixel size), never
labels or query points -- those are cheap and idx-deterministic, so they stay derived on the fly
and changing label_kind / sample_q must not invalidate the cache.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

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


class RenderCache:  # placeholder: implemented in Task 2
    pass


class CachedRenderSource:  # placeholder: implemented in Task 2
    pass
