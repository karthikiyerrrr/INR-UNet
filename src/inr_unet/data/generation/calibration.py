"""Pure visual-calibration statistics for comparing synthetic vs real STEM images.

No I/O, no plotting: each function maps a single [H, W] image tensor to summary arrays
the explorer notebook overlays against the real reference set.
"""

from __future__ import annotations

import torch


def intensity_histogram(
    image: torch.Tensor,
    bins: int = 64,
    value_range: tuple[float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Histogram of pixel intensities.

    With ``value_range=None`` (default) bins span the image's own [min, max]; with a
    fixed ``(lo, hi)`` they span that range so histograms are comparable across images
    (e.g. for cross-image KL). Returns (counts [bins], edges [bins + 1]).
    """
    flat = image.flatten()
    if value_range is None:
        lo, hi = float(flat.min()), float(flat.max())
    else:
        lo, hi = float(value_range[0]), float(value_range[1])
    if hi <= lo:  # degenerate range: put all mass in the first bin over a unit range
        hi = lo + 1.0
    counts = torch.histc(flat, bins=bins, min=lo, max=hi)
    edges = torch.linspace(lo, hi, bins + 1)
    return counts, edges


def radial_power_spectrum(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Azimuthally-averaged power spectrum |FFT|**2 vs integer radial frequency bin.

    Returns (freq [R + 1], power [R + 1]) where freq is the radius in cycles-per-image
    (pixels in the shifted spectrum from the DC center). The DC term sits at index 0.
    """
    spec = torch.fft.fftshift(torch.fft.fft2(image))
    power = spec.real**2 + spec.imag**2
    h, w = image.shape
    ax_y = torch.arange(h, dtype=torch.float32) - h // 2
    ax_x = torch.arange(w, dtype=torch.float32) - w // 2
    yy, xx = torch.meshgrid(ax_y, ax_x, indexing="ij")
    r = torch.sqrt(yy**2 + xx**2).round().to(torch.long).flatten()
    r_max = int(r.max())
    radial = torch.zeros(r_max + 1, dtype=power.dtype)
    counts = torch.zeros(r_max + 1, dtype=power.dtype)
    radial.index_add_(0, r, power.flatten())
    counts.index_add_(0, r, torch.ones_like(power.flatten()))
    radial = radial / counts.clamp(min=1.0)
    freq = torch.arange(r_max + 1, dtype=torch.float32)
    return freq, radial


def noise_autocorrelation(image: torch.Tensor) -> torch.Tensor:
    """2D autocorrelation of the mean-removed image, normalized to 1.0 at zero lag.

    Returns an [H, W] map with zero lag at the center (fftshifted). Row-correlated scan
    noise shows up as elevated correlation along the row (x-lag) axis -- the most
    under-simulated real-ADF artifact.
    """
    x = image - image.mean()
    spec = torch.fft.fft2(x)
    ac = torch.fft.ifft2(spec * spec.conj()).real
    ac = torch.fft.fftshift(ac)
    peak = ac.max()
    return ac / peak if peak > 0 else ac
