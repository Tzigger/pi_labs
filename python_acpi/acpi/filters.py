"""Convolution utilities for ACPI image processing tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

def load_image(path: Path | str) -> np.ndarray:
    """Load an image as a float array in the range [0, 255]."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32)


def save_image(array: np.ndarray, path: Path | str) -> None:
    """Persist an RGB array as an image file."""
    out = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(path)


def _apply_kernel_to_channel(channel: np.ndarray, kernel: np.ndarray, factor: float, bias: float, wrap: bool) -> np.ndarray:
    height, width = channel.shape
    k_height, k_width = kernel.shape
    pad_y = k_height // 2
    pad_x = k_width // 2
    result = np.zeros_like(channel)

    for y in range(height):
        for x in range(width):
            acc = 0.0
            for ky in range(k_height):
                for kx in range(k_width):
                    yy = y + ky - pad_y
                    xx = x + kx - pad_x
                    if wrap:
                        yy %= height
                        xx %= width
                        sample = channel[yy, xx]
                    else:
                        if 0 <= yy < height and 0 <= xx < width:
                            sample = channel[yy, xx]
                        else:
                            sample = 0.0
                    acc += sample * kernel[ky, kx]
            acc = acc * factor + bias
            result[y, x] = np.clip(acc, 0.0, 255.0)
    return result


def convolve_image(image: np.ndarray, kernel: Iterable[Iterable[float]], factor: float = 1.0, bias: float = 0.0, wrap: bool = False) -> np.ndarray:
    """Apply a 2D convolution kernel to an RGB or grayscale image."""
    src = np.asarray(image, dtype=np.float32)
    kernel_arr = np.asarray(kernel, dtype=np.float32)

    if src.ndim == 2:
        convolved = _apply_kernel_to_channel(src, kernel_arr, factor, bias, wrap)
        return convolved

    if src.ndim == 3 and src.shape[2] in (3, 4):
        channels = []
        # We drop alpha if present to keep the output consistent with the assignment.
        for idx in range(3):
            channels.append(_apply_kernel_to_channel(src[..., idx], kernel_arr, factor, bias, wrap))
    return np.stack(channels, axis=2)

    msg = f"Unsupported image shape {src.shape}; expected HxW or HxWx3 array."
    raise ValueError(msg)


def apply_kernel(path_in: Path | str, path_out: Path | str, kernel: Iterable[Iterable[float]], factor: float = 1.0, bias: float = 0.0, wrap: bool = False) -> None:
    """Convenience helper to load, convolve, and save an image."""
    img = load_image(path_in)
    result = convolve_image(img, kernel, factor=factor, bias=bias, wrap=wrap)
    save_image(result, path_out)
