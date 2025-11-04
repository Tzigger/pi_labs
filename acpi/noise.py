"""Noise reduction filters built on top of the generic convolution utility."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .filters import convolve_image


def _validate_size(size: int) -> int:
    if size % 2 == 0 or size < 1:
        raise ValueError("Window size must be a positive odd integer.")
    return size


def apply_mean_filter(
    image: np.ndarray, size: int = 3, wrap: bool = False
) -> np.ndarray:
    size = _validate_size(size)
    kernel = np.ones((size, size), dtype=np.float32)
    factor = 1.0 / (size * size)
    return convolve_image(image, kernel, factor=factor, wrap=wrap)


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    size = _validate_size(size)
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def apply_gaussian_filter(
    image: np.ndarray, size: int = 5, sigma: float = 1.0, wrap: bool = False
) -> np.ndarray:
    kernel = gaussian_kernel(size, sigma)
    return convolve_image(image, kernel, factor=1.0, wrap=wrap)


def apply_median_filter(
    image: np.ndarray, size: int = 3, wrap: bool = False
) -> np.ndarray:
    size = _validate_size(size)
    radius = size // 2
    src = np.asarray(image, dtype=np.float32)

    if src.ndim == 2:
        return _median_channel(src, radius, wrap)
    if src.ndim == 3 and src.shape[2] in (3, 4):
        channels = []
        for idx in range(3):
            channels.append(_median_channel(src[..., idx], radius, wrap))
        return np.stack(channels, axis=2)
    msg = f"Unsupported image shape {src.shape}; expected HxW or HxWx3 array."
    raise ValueError(msg)


def _median_channel(channel: np.ndarray, radius: int, wrap: bool) -> np.ndarray:
    height, width = channel.shape
    result = np.zeros_like(channel)
    window_size = 2 * radius + 1

    for y in range(height):
        for x in range(width):
            window_values = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    yy = y + dy
                    xx = x + dx
                    if wrap:
                        yy %= height
                        xx %= width
                        window_values.append(channel[yy, xx])
                    else:
                        if 0 <= yy < height and 0 <= xx < width:
                            window_values.append(channel[yy, xx])
                        else:
                            window_values.append(0.0)
            result[y, x] = np.median(window_values)
    return result


def apply_huang_median_filter(
    image: np.ndarray, size: int = 3, wrap: bool = False
) -> np.ndarray:
    size = _validate_size(size)
    src = np.asarray(image, dtype=np.float32)

    if src.ndim == 2:
        return _huang_median_channel(src, size, wrap)
    if src.ndim == 3 and src.shape[2] in (3, 4):
        channels = []
        for idx in range(3):
            channels.append(_huang_median_channel(src[..., idx], size, wrap))
        return np.stack(channels, axis=2)
    msg = f"Unsupported image shape {src.shape}; expected HxW or HxWx3 array."
    raise ValueError(msg)


def _huang_median_channel(channel: np.ndarray, size: int, wrap: bool) -> np.ndarray:
    height, width = channel.shape
    result = np.zeros_like(channel)
    radius = size // 2
    median_pos = (size * size) // 2
    
    # Convertim la întregi 0-255 pentru histogramă
    img_int = np.clip(channel, 0, 255).astype(np.uint8)

    for y in range(height):
        # Histogramă pentru prima fereastră din rând
        hist = np.zeros(256, dtype=np.int32)
        
        # Calculăm histograma primei ferestre (x=0)
        y_start = max(0, y - radius)
        y_end = min(height, y + radius + 1)
        x_start = 0
        x_end = min(width, radius + 1)
        
        if wrap:
            # Pentru wrap trebuie să extragem manual
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    yy = (y + dy) % height
                    xx = dx % width
                    hist[img_int[yy, xx]] += 1
        else:
            # Extragem fereastra și calculăm histograma cu NumPy
            window = img_int[y_start:y_end, x_start:x_end]
            hist = np.bincount(window.ravel(), minlength=256).astype(np.int32)
        
        # Găsim mediana din histogramă
        result[y, 0] = _find_median_from_hist(hist, median_pos)
        
        # Glisăm fereastra pe restul rândului
        for x in range(1, width):
            # Scoatem coloana din stânga
            left_x = x - radius - 1
            if 0 <= left_x < width or wrap:
                for dy in range(-radius, radius + 1):
                    yy = (y + dy) if not wrap else (y + dy) % height
                    xx = left_x if not wrap else left_x % width
                    if 0 <= yy < height:
                        hist[img_int[yy, xx]] -= 1
            
            # Adăugăm coloana din dreapta
            right_x = x + radius
            if 0 <= right_x < width or wrap:
                for dy in range(-radius, radius + 1):
                    yy = (y + dy) if not wrap else (y + dy) % height
                    xx = right_x if not wrap else right_x % width
                    if 0 <= yy < height:
                        hist[img_int[yy, xx]] += 1
            
            # Găsim noua mediană
            result[y, x] = _find_median_from_hist(hist, median_pos)
    
    return result


def _find_median_from_hist(hist: np.ndarray, target: int) -> float:
    """Găsește mediana din histogramă numărând până la poziția target."""
    cumsum = 0
    for value in range(256):
        cumsum += hist[value]
        if cumsum > target:
            return float(value)
    return 255.0
