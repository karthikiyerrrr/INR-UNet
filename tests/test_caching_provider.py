"""CachingProvider memoizes a wrapped provider's deterministic per-scene output."""

import torch

from inr_unet.data.generation.structures import ColumnList
from inr_unet.data.providers import CachingProvider


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
