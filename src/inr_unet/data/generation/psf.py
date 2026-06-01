"""Probe-forming optics: aperture, aberration, source-size blur -> effective PSF."""

from __future__ import annotations

import math

import torch

from inr_unet.data.generation.structures import Grid, ImagingCondition

_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))  # 2.3548


def wavelength_A(energy_keV: float) -> float:
    """Relativistic electron wavelength in angstroms."""
    e = energy_keV * 1e3  # eV
    return 12.2643 / math.sqrt(e * (1.0 + 0.97845e-6 * e))


def _aberration_chi(
    k: torch.Tensor, phi: torch.Tensor, lam: float, cond: ImagingCondition
) -> torch.Tensor:
    """Aberration phase chi(k, phi); radial terms plus 2-fold (A1) astigmatism.

    Zero when defocus = C3 = C5 = A1 = 0. The A1 term shares defocus's lam**2 * k**2
    radial dependence, azimuthally modulated by cos(2(phi - azimuth)); A1 is in angstroms.
    With A1 = 0 the term is a zero tensor, so chi is bitwise-identical to the radial-only
    form.
    """
    radial = (
        0.5 * cond.defocus_A * lam**2 * k**2
        + 0.25 * cond.c3_A * lam**4 * k**4
        + (1.0 / 6.0) * cond.c5_A * lam**6 * k**6
    )
    astig = (
        0.5 * cond.astig_a1_A * lam**2 * k**2
        * torch.cos(2.0 * (phi - cond.astig_a1_azimuth_rad))
    )
    return (2.0 * math.pi / lam) * (radial + astig)


def _smoothstep(t: torch.Tensor) -> torch.Tensor:
    t = torch.clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_psf(cond: ImagingCondition, grid: Grid, *, soft: bool = True) -> torch.Tensor:
    """Effective, sum-normalized, centered PSF [H, W] for the given condition."""
    lam = wavelength_A(cond.energy_keV)
    k_cut = (cond.alpha_max_mrad * 1e-3) / lam  # 1/A, since alpha = lam * k
    k_nyq = 1.0 / (2.0 * grid.pixel_size_A)
    if k_cut >= k_nyq:
        max_px = 1.0 / (2.0 * k_cut)
        raise ValueError(
            f"Aperture k_cut={k_cut:.4f}/A exceeds Nyquist {k_nyq:.4f}/A; "
            f"use pixel_size_A < {max_px:.4f}"
        )

    ky, kx = grid.freq_coords()
    k = torch.sqrt(ky**2 + kx**2)
    phi = torch.atan2(ky, kx)
    if soft:
        dk = 1.0 / grid.extent_A  # frequency sampling
        aperture = _smoothstep((k_cut - k) / (2.0 * dk) + 0.5)
    else:
        aperture = (k <= k_cut).to(torch.float32)

    chi = _aberration_chi(k, phi, lam, cond)
    probe_k = aperture * torch.exp(-1j * chi)
    probe_r = torch.fft.ifft2(probe_k)
    psf = probe_r.real**2 + probe_r.imag**2  # corner-origin

    # incoherent source-size blur: multiply PSF spectrum by a Gaussian MTF
    sigma_s_A = (cond.source_size_A / 2.0) / _FWHM_PER_SIGMA
    sigma_s_px = sigma_s_A / grid.pixel_size_A
    if sigma_s_px > 0:
        fy = torch.fft.fftfreq(grid.output_size, device=psf.device)  # cycles/px
        fx = torch.fft.fftfreq(grid.output_size, device=psf.device)
        gy, gx = torch.meshgrid(fy, fx, indexing="ij")
        mtf = torch.exp(-2.0 * math.pi**2 * sigma_s_px**2 * (gx**2 + gy**2))
        psf = torch.fft.ifft2(torch.fft.fft2(psf) * mtf).real

    psf = torch.clamp(psf, min=0.0)
    psf = torch.fft.fftshift(psf)  # center the peak
    return psf / psf.sum()


def fft_convolve(sigma: torch.Tensor, psf_centered: torch.Tensor) -> torch.Tensor:
    """Circular convolution sigma (x) PSF with no spatial shift (PSF is centered)."""
    otf = torch.fft.fft2(torch.fft.ifftshift(psf_centered))
    return torch.fft.ifft2(torch.fft.fft2(sigma) * otf).real
