"""TEMRenderer: orchestrates the forward-model stages into a RenderOutput."""

from __future__ import annotations

from omegaconf import DictConfig

from inr_unet.data.generation.structures import (
    ColumnList,
    ImagingCondition,
    RenderOutput,
    RenderParams,
)


class TEMRenderer:
    """Render an ADF-STEM image and label rasters from projected columns."""

    def __init__(self, cfg: DictConfig) -> None:
        self.potential_backend = cfg.potential_backend
        self.sigma_potential_A = cfg.sigma_potential_A
        self.aperture_soft = cfg.aperture_soft

    def render(
        self,
        columns: ColumnList,
        condition: ImagingCondition,
        params: RenderParams,
    ) -> RenderOutput:
        raise NotImplementedError("TEMRenderer.render is implemented in Task 9.")
