"""Image-quality metrics used by compression and denoising responses."""
from __future__ import annotations

import numpy as np


def mse(reference, comparison) -> float:
    """Mean squared error for equally shaped uint8-like image arrays."""
    reference = np.asarray(reference, dtype=np.float64)
    comparison = np.asarray(comparison, dtype=np.float64)
    if reference.shape != comparison.shape:
        raise ValueError("Images must have the same shape for quality metrics.")
    return float(np.mean((reference - comparison) ** 2))


def psnr(reference, comparison, data_range=255.0, cap=100.0) -> float:
    value = mse(reference, comparison)
    if value == 0:
        return cap
    return float(min(cap, 10 * np.log10(data_range ** 2 / value)))


def _windowed_ssim_channel(reference, comparison, window=7, data_range=255.0) -> float:
    """Uniform-window SSIM for one channel, averaged over valid windows."""
    height, width = reference.shape
    size = min(window, height, width)
    if size < 2:
        # SSIM is undefined for a one-pixel image; use the stable global form.
        size = 1
    windows_x = np.lib.stride_tricks.sliding_window_view(reference, (size, size))
    windows_y = np.lib.stride_tricks.sliding_window_view(comparison, (size, size))
    mu_x, mu_y = windows_x.mean(axis=(-2, -1)), windows_y.mean(axis=(-2, -1))
    var_x = ((windows_x - mu_x[..., None, None]) ** 2).mean(axis=(-2, -1))
    var_y = ((windows_y - mu_y[..., None, None]) ** 2).mean(axis=(-2, -1))
    cov = ((windows_x - mu_x[..., None, None]) * (windows_y - mu_y[..., None, None])).mean(axis=(-2, -1))
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return float(np.mean(score))


def ssim(reference, comparison, window=7, data_range=255.0) -> float:
    """Channel-averaged local-window structural similarity in the range [-1, 1]."""
    reference = np.asarray(reference, dtype=np.float64)
    comparison = np.asarray(comparison, dtype=np.float64)
    if reference.shape != comparison.shape:
        raise ValueError("Images must have the same shape for quality metrics.")
    if reference.ndim == 2:
        score = _windowed_ssim_channel(reference, comparison, window, data_range)
    elif reference.ndim == 3 and reference.shape[2] in (3, 4):
        score = float(np.mean([_windowed_ssim_channel(reference[:, :, c], comparison[:, :, c], window, data_range)
                               for c in range(reference.shape[2])]))
    else:
        raise ValueError("Expected grayscale, RGB, or RGBA images.")
    return float(np.clip(score, -1.0, 1.0))


def metrics(reference, comparison) -> dict[str, float]:
    """JSON-safe MSE, PSNR, and SSIM values for a labelled comparison."""
    return {
        "mse": round(mse(reference, comparison), 4),
        "psnr": round(psnr(reference, comparison), 2),
        "ssim": round(ssim(reference, comparison), 4),
    }
