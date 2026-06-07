"""CachingProvider memoizes a wrapped provider's deterministic per-scene output."""

import torch

from inr_unet.config import load_config
from inr_unet.data import LIIFSegDataset
from inr_unet.data.generation.structures import ColumnList
from inr_unet.data.providers import CachingProvider

CONFIG = "configs/default.yaml"


class _CountingProvider:
    """Minimal ColumnListProvider that records every get() call."""

    def __init__(self, n: int) -> None:
        self._n = n
        self.calls: list[int] = []

    def __len__(self) -> int:
        return self._n

    def get(self, scene_idx: int) -> ColumnList:
        self.calls.append(scene_idx)
        return ColumnList(
            positions_A=torch.zeros(1, 2), z=torch.ones(1), count=torch.ones(1), fov_A=10.0
        )


def test_caching_provider_caches_repeated_get():
    inner = _CountingProvider(4)
    cp = CachingProvider(inner)
    first = cp.get(0)
    second = cp.get(0)
    assert inner.calls == [0]          # inner invoked once despite two gets
    assert second is first             # same cached object returned


def test_caching_provider_distinct_scenes_each_miss_once():
    inner = _CountingProvider(4)
    cp = CachingProvider(inner)
    cp.get(0)
    cp.get(1)
    cp.get(0)
    assert inner.calls == [0, 1]       # second get(0) is a hit


def test_caching_provider_len_delegates():
    inner = _CountingProvider(7)
    assert len(CachingProvider(inner)) == 7


def _synth_cfg():
    cfg = load_config(CONFIG)
    cfg.data.provider = "synthetic"
    cfg.data.synthetic.n_scenes = 4
    cfg.data.synthetic.draws_per_scene = 3
    cfg.data.synthetic.sample_q = 64
    return cfg


def test_ds_item_identical_with_and_without_cache():
    cfg_on = _synth_cfg()
    cfg_on.data.cache_scenes = True
    cfg_off = _synth_cfg()
    cfg_off.data.cache_scenes = False
    ds_on = LIIFSegDataset(cfg_on)
    ds_off = LIIFSegDataset(cfg_off)
    for i in range(min(len(ds_on), 6)):
        for ta, tb in zip(ds_on[i], ds_off[i], strict=False):
            assert torch.equal(ta, tb), f"mismatch at idx {i}"


def test_cache_scenes_default_wraps_provider():
    ds = LIIFSegDataset(_synth_cfg())  # cache_scenes default True
    assert isinstance(ds.source.provider, CachingProvider)
