"""AugmentationSampler: index-keyed, deterministic draws of the Table-1 grid."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from inr_unet.data.generation.psf import wavelength_A
from inr_unet.data.generation.renderer import _MARGIN_A  # keep in sync with renderer crop margin
from inr_unet.data.generation.structures import (
    IMAGING_CONDITIONS,
    BackgroundSpec,
    ImagingCondition,
    NoiseSpec,
    RenderParams,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from inr_unet.config import SamplerConfig

__all__ = ["AugmentationSampler"]


def _rot_factor(rotation_deg: float) -> float:
    """|cos theta| + |sin theta| -- the side-growth factor of a rotated square."""
    t = math.radians(rotation_deg)
    return abs(math.cos(t)) + abs(math.sin(t))


class AugmentationSampler:
    """Draws (ImagingCondition, RenderParams) for an externally supplied structure."""

    def __init__(self, cfg: SamplerConfig | DictConfig, master_seed: int) -> None:
        self.cfg = cfg
        self.master_seed = int(master_seed)
        self._bg_kinds = list(cfg.bg_weights.keys())
        w = np.array([float(cfg.bg_weights[k]) for k in self._bg_kinds])
        self._bg_weights = w / w.sum()

    def _streams(self, index: int) -> tuple[np.random.Generator, int, np.random.Generator]:
        """Derive a distribution RNG, a render seed, and an independent aberration RNG.

        spawn(3) leaves the first two children bitwise-identical to the previous spawn(2),
        so every existing draw value is preserved; the third child feeds aberration draws.
        """
        dist_ss, render_ss, aberr_ss = np.random.SeedSequence(
            [self.master_seed, int(index)]
        ).spawn(3)
        rng = np.random.default_rng(dist_ss)
        render_seed = int(render_ss.generate_state(1, dtype=np.uint32)[0])
        aberr_rng = np.random.default_rng(aberr_ss)
        return rng, render_seed, aberr_rng

    def _draw_condition(self, rng: np.random.Generator) -> ImagingCondition:
        name = self.cfg.conditions[int(rng.integers(len(self.cfg.conditions)))]
        return IMAGING_CONDITIONS[name]

    def _draw_noise(self, rng: np.random.Generator) -> NoiseSpec:
        n_peak = float(np.exp(rng.uniform(math.log(self.cfg.n_peak_min),
                                          math.log(self.cfg.n_peak_max))))
        return NoiseSpec(
            n_peak=n_peak,
            n_bg_frac=float(rng.uniform(0.0, self.cfg.n_bg_frac_max)),
            scan_freq_cyc_per_row=float(rng.uniform(self.cfg.scan_freq_min,
                                                    self.cfg.scan_freq_max)),
            scan_beta=float(rng.uniform(self.cfg.scan_beta_min, self.cfg.scan_beta_max)),
            scan_phi0=float(rng.uniform(0.0, 2.0 * math.pi)),
        )

    def _draw_aberration(self, rng: np.random.Generator) -> tuple[float, float, float]:
        """Draw (defocus_A, astig_a1_A, astig_a1_azimuth_rad) for one render.

        Defocus is signed (over/under focus); astigmatism is only visible with nonzero
        defocus. Azimuth spans [0, pi) because 2-fold astigmatism has period pi.
        """
        defocus = float(rng.uniform(-self.cfg.defocus_A_max, self.cfg.defocus_A_max))
        mag = float(rng.uniform(0.0, self.cfg.astig_a1_A_max))
        azimuth = float(rng.uniform(0.0, math.pi))
        return defocus, mag, azimuth

    def _draw_background(self, rng: np.random.Generator) -> BackgroundSpec:
        kind = self._bg_kinds[int(rng.choice(len(self._bg_kinds), p=self._bg_weights))]
        if kind == "constant":
            params = {"c": float(rng.uniform(self.cfg.bg_constant_c_min,
                                             self.cfg.bg_constant_c_max))}
        elif kind == "linear_ramp":
            params = {
                "c0": float(rng.uniform(self.cfg.bg_ramp_c0_min, self.cfg.bg_ramp_c0_max)),
                "g": float(rng.uniform(self.cfg.bg_ramp_g_min, self.cfg.bg_ramp_g_max)),
            }
        elif kind == "nonlinear":
            params = {"n_blobs": int(rng.integers(self.cfg.bg_nonlinear_blobs_min,
                                                  self.cfg.bg_nonlinear_blobs_max + 1))}
        elif kind == "perlin":
            params = {
                "amp": float(rng.uniform(self.cfg.bg_perlin_amp_min, self.cfg.bg_perlin_amp_max)),
                "cells": int(rng.integers(self.cfg.bg_perlin_cells_min,
                                          self.cfg.bg_perlin_cells_max + 1)),
            }
        else:
            raise ValueError(f"unknown background kind {kind!r}")
        return BackgroundSpec(kind=kind, params=params)

    def _rotation_choices(self) -> list[float]:
        n = round(self.cfg.rotation_max_deg / self.cfg.rotation_step_deg)
        return [i * self.cfg.rotation_step_deg for i in range(n + 1)]

    def _fov_fits(self, fov_A: float, rotation_deg: float, max_fov_A: float) -> bool:
        return fov_A * _rot_factor(rotation_deg) + 2.0 * _MARGIN_A <= max_fov_A

    def _draw_rotation(self, rng: np.random.Generator, max_fov_A: float) -> float:
        # feasibility vs the smallest FOV guarantees _draw_scale has >=1 candidate later
        smallest_fov = min(self.cfg.fov_set_A)
        feasible = [r for r in self._rotation_choices()
                    if self._fov_fits(smallest_fov, r, max_fov_A)]
        if not feasible:
            raise ValueError(
                f"max_fov_A={max_fov_A} too small for smallest FOV {smallest_fov} A "
                f"with margin {_MARGIN_A} A"
            )
        return float(feasible[int(rng.integers(len(feasible)))])

    def _draw_scale(
        self,
        rng: np.random.Generator,
        condition: ImagingCondition,
        rotation_deg: float,
        max_fov_A: float,
    ) -> tuple[float, float, int]:
        lam = wavelength_A(condition.energy_keV)
        k_cut = (condition.alpha_max_mrad * 1e-3) / lam
        px_alias = 1.0 / (3.0 * k_cut)  # 2/3-Nyquist guard
        if px_alias < self.cfg.pixel_size_A_min:
            raise ValueError(
                f"aliasing limit {px_alias:.4f} A < pixel_size_A_min "
                f"{self.cfg.pixel_size_A_min} A for condition {condition.name!r}; "
                f"reduce alpha_max_mrad or raise pixel_size_A_min"
            )
        px_max = max(min(self.cfg.pixel_size_A_max, px_alias), self.cfg.pixel_size_A_min)
        pixel_size_A = float(np.exp(rng.uniform(math.log(self.cfg.pixel_size_A_min),
                                                math.log(px_max))))
        candidates = [float(f) for f in self.cfg.fov_set_A
                      if self._fov_fits(float(f), rotation_deg, max_fov_A)]
        if not candidates:
            raise ValueError(
                f"no FOV in {list(self.cfg.fov_set_A)} fits max_fov_A={max_fov_A} "
                f"at rotation {rotation_deg} deg"
            )
        fov_A = candidates[int(rng.integers(len(candidates)))]
        output_size = max(1, int(fov_A / pixel_size_A))  # floor; >=1 guards sub-pixel FOVs
        return fov_A, pixel_size_A, output_size

    def _draw_offset(
        self, rng: np.random.Generator, fov_A: float, rotation_deg: float, max_fov_A: float
    ) -> torch.Tensor:
        slack = max((max_fov_A - fov_A * _rot_factor(rotation_deg)) / 2.0 - _MARGIN_A, 0.0)
        j = self.cfg.offset_jitter_frac * slack
        ox = float(rng.uniform(-j, j)) if j > 0 else 0.0
        oy = float(rng.uniform(-j, j)) if j > 0 else 0.0
        return torch.tensor([ox, oy], dtype=torch.float32)

    def sample(self, index: int, max_fov_A: float) -> tuple[ImagingCondition, RenderParams]:
        """Draw a renderable (ImagingCondition, RenderParams) for the given draw index.

        Raises ValueError if no FOV/rotation fits ``max_fov_A`` with margin.
        """
        rng, render_seed, aberr_rng = self._streams(index)
        condition = self._draw_condition(rng)
        defocus, astig_mag, astig_az = self._draw_aberration(aberr_rng)
        condition = replace(
            condition,
            defocus_A=defocus,
            astig_a1_A=astig_mag,
            astig_a1_azimuth_rad=astig_az,
        )
        rotation_deg = self._draw_rotation(rng, max_fov_A)
        fov_A, pixel_size_A, output_size = self._draw_scale(
            rng, condition, rotation_deg, max_fov_A
        )
        background = self._draw_background(rng)
        noise = self._draw_noise(rng)
        offset = self._draw_offset(rng, fov_A, rotation_deg, max_fov_A)
        z_exponent = float(rng.uniform(self.cfg.z_exponent_min, self.cfg.z_exponent_max))
        params = RenderParams(
            output_size=output_size,
            pixel_size_A=pixel_size_A,
            rotation_deg=rotation_deg,
            position_offset_A=offset,
            z_exponent=z_exponent,
            background=background,
            noise=noise,
            seed=render_seed,
            device=self.cfg.device,
        )
        return condition, params
