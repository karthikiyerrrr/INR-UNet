"""Persistent render cache: keying, materialization, ragged reconstruction, alignment."""

import pytest  # noqa: F401
import torch  # noqa: F401
from omegaconf import OmegaConf

from inr_unet.config import ExperimentConfig
from inr_unet.data.cache import CachedRenderSource, RenderCache, cache_key  # noqa: F401
from inr_unet.data.dataset import LIIFSegDataset, STEMSegDataset  # noqa: F401
from inr_unet.data.render_source import SyntheticRenderSource  # noqa: F401


def _cfg(crop_size=32, n_scenes=2, draws_per_scene=2):
    cfg = OmegaConf.structured(ExperimentConfig)
    cfg.data.num_workers = 0            # single-process build keeps tests fast + fork-free
    cfg.data.synthetic.crop_size = crop_size
    cfg.data.synthetic.n_scenes = n_scenes
    cfg.data.synthetic.draws_per_scene = draws_per_scene
    cfg.generation.sampler.pixel_size_A_min = 0.20   # modest renders for test speed
    cfg.generation.sampler.pixel_size_A_max = 0.30
    return cfg


def test_key_excludes_label_and_query_knobs():
    a, b = _cfg(), _cfg()
    b.data.synthetic.label_kind = "circular"
    b.data.synthetic.sample_q = 128
    b.data.synthetic.target_pixel_size_A_max = 0.5
    b.data.synthetic.gaussian_fwhm_A = 0.9
    assert cache_key(a) == cache_key(b)


def test_key_sensitive_to_render_config():
    a, b = _cfg(), _cfg()
    b.data.synthetic.tile_fov_A_max = 99.0
    assert cache_key(a) != cache_key(b)


def test_key_sensitive_to_split():
    a, b = _cfg(), _cfg()
    b.train.split.val_frac = 0.25
    assert cache_key(a) != cache_key(b)


def test_cached_get_matches_live():
    cfg = _cfg()
    live = SyntheticRenderSource(cfg)
    src = CachedRenderSource(RenderCache.build(cfg, [0, 1, 2, 3]))
    for i in range(4):
        a, b = src.get(i), live.get(i)
        assert torch.equal(a.image, b.image)
        assert torch.equal(a.positions_A, b.positions_A)
        assert torch.equal(a.radii_A, b.radii_A)
        assert a.input_pixel_size_A == b.input_pixel_size_A
        assert a.valid_extent_A == b.valid_extent_A


def test_offsets_are_csr_monotonic():
    cache = RenderCache.build(_cfg(), [0, 1, 2, 3])
    offs = cache.offsets.tolist()
    assert offs[0] == 0
    assert offs == sorted(offs)
    assert offs[-1] == cache.positions.shape[0]
    assert cache.offsets.shape[0] == cache.idx.shape[0] + 1
    # per-row column slice length matches the stored radii slice length
    for r in range(cache.idx.shape[0]):
        lo, hi = offs[r], offs[r + 1]
        assert cache.positions[lo:hi].shape[0] == cache.radii[lo:hi].shape[0]


def test_unknown_idx_raises():
    src = CachedRenderSource(RenderCache.build(_cfg(), [0, 1]))
    with pytest.raises(KeyError):
        src.get(3)


def test_save_load_roundtrip(tmp_path):
    cfg = _cfg()
    cache = RenderCache.build(cfg, [0, 1, 2, 3])
    path = tmp_path / "render_cache.pt"
    cache.save(path)
    loaded = RenderCache.load(path)
    assert loaded.key == cache.key
    live = SyntheticRenderSource(cfg)
    src = CachedRenderSource(loaded)
    a, b = src.get(2), live.get(2)
    assert torch.equal(a.image, b.image)
    assert torch.equal(a.positions_A, b.positions_A)
    assert a.input_pixel_size_A == b.input_pixel_size_A
    assert loaded.pixel_size_A.dtype == torch.float64
    assert loaded.idx.dtype == torch.int64


def test_exports_available_from_package():
    from inr_unet.data import CachedRenderSource as CRS
    from inr_unet.data import RenderCache as RC
    from inr_unet.data import cache_key as ck
    assert RC is RenderCache and CRS is CachedRenderSource and ck is cache_key


def test_liif_and_stem_share_cached_image():
    # Tamper the cache so a live rebuild would differ: proves both datasets read the injected
    # source (not a fresh live one) AND that they share the same image for a given idx.
    cfg = _cfg()
    cache = RenderCache.build(cfg, [0, 1, 2, 3])
    cache.image[2].zero_()
    src = CachedRenderSource(cache)
    liif = LIIFSegDataset(cfg, source=src)
    stem = STEMSegDataset(cfg, source=src)
    assert torch.equal(liif[2][0], stem[2][0])      # identical image for both models
    assert float(liif[2][0].abs().sum()) == 0.0     # injected (zeroed) source is actually used


def test_liif_item_identical_live_vs_cached():
    cfg = _cfg()
    live_ds = LIIFSegDataset(cfg)
    cached_ds = LIIFSegDataset(cfg, source=CachedRenderSource(RenderCache.build(cfg, [0, 1, 2, 3])))
    for t_live, t_cached in zip(live_ds[1], cached_ds[1], strict=True):
        assert torch.equal(t_live, t_cached)


def test_stem_item_identical_live_vs_cached():
    cfg = _cfg()
    live_ds = STEMSegDataset(cfg)
    cached_ds = STEMSegDataset(cfg, source=CachedRenderSource(RenderCache.build(cfg, [0, 1, 2, 3])))
    for t_live, t_cached in zip(live_ds[1], cached_ds[1], strict=True):
        assert torch.equal(t_live, t_cached)


def test_render_cache_config_defaults():
    cfg = OmegaConf.structured(ExperimentConfig)
    assert cfg.data.render_cache.enabled is False   # opt-in; the Colab driver turns it on
    assert isinstance(cfg.data.render_cache.dir, str)
