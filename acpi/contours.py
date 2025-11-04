"""Edge detection and image zooming operations.

This module implements:
1. Edge detection using Sobel and Prewitt gradient approximations
2. Image zooming operations (2x2 fixed and arbitrary scale factors)
"""

from __future__ import annotations

import numpy as np

from .filters import convolve_image


# =============================================================================
# Gradient/Edge Detection Kernels
# =============================================================================

# Sobel kernels for horizontal and vertical gradients
SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

SOBEL_Y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)

# Prewitt kernels for horizontal and vertical gradients
PREWITT_X = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

PREWITT_Y = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
], dtype=np.float32)

# Diagonal Sobel kernels (optional - pentru detectie diagonala)
SOBEL_XY = np.array([
    [-2, -1, 0],
    [-1, 0, 1],
    [0, 1, 2]
], dtype=np.float32)

SOBEL_YX = np.array([
    [0, -1, -2],
    [1, 0, -1],
    [2, 1, 0]
], dtype=np.float32)


# =============================================================================
# Edge Detection Functions
# =============================================================================

def apply_sobel(image: np.ndarray, wrap: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Sobel edge detection to an image.
    
    Computes horizontal gradient (Gx), vertical gradient (Gy), and 
    the combined gradient magnitude.
    
    Args:
        image: Input grayscale image as numpy array
        wrap: If True, use wrap-around padding for edges
        
    Returns:
        Tuple of (Gx, Gy, magnitude) where:
        - Gx: Horizontal gradient
        - Gy: Vertical gradient  
        - magnitude: Combined gradient magnitude (sqrt(Gx^2 + Gy^2))
    """
    # Use the convolve_image function from previous labs
    gx = convolve_image(image, SOBEL_X, factor=1.0, wrap=wrap)
    gy = convolve_image(image, SOBEL_Y, factor=1.0, wrap=wrap)
    
    # Compute magnitude: sqrt(Gx^2 + Gy^2)
    magnitude = np.sqrt(gx.astype(np.float64)**2 + gy.astype(np.float64)**2)
    
    # Clip to valid range [0, 255]
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    return gx, gy, magnitude


def apply_sobel_approximation(image: np.ndarray, wrap: bool = False, 
                               method: str = 'sum') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Sobel edge detection with magnitude approximation.
    
    Uses approximations instead of exact sqrt calculation for efficiency:
    - 'sum': |Gx| + |Gy|
    - 'max': max(|Gx|, |Gy|)
    - 'exact': sqrt(Gx^2 + Gy^2) [default in apply_sobel]
    
    Args:
        image: Input grayscale image
        wrap: If True, use wrap-around padding
        method: Approximation method ('sum' or 'max')
        
    Returns:
        Tuple of (Gx, Gy, magnitude)
    """
    gx = convolve_image(image, SOBEL_X, factor=1.0, wrap=wrap)
    gy = convolve_image(image, SOBEL_Y, factor=1.0, wrap=wrap)
    
    if method == 'sum':
        magnitude = np.abs(gx.astype(np.float64)) + np.abs(gy.astype(np.float64))
    elif method == 'max':
        magnitude = np.maximum(np.abs(gx.astype(np.float64)), 
                              np.abs(gy.astype(np.float64)))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sum' or 'max'.")
    
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    return gx, gy, magnitude


def apply_prewitt(image: np.ndarray, wrap: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Prewitt edge detection to an image.
    
    Computes horizontal gradient (Gx), vertical gradient (Gy), and 
    the combined gradient magnitude.
    
    Args:
        image: Input grayscale image as numpy array
        wrap: If True, use wrap-around padding for edges
        
    Returns:
        Tuple of (Gx, Gy, magnitude) where:
        - Gx: Horizontal gradient
        - Gy: Vertical gradient
        - magnitude: Combined gradient magnitude (sqrt(Gx^2 + Gy^2))
    """
    # Use the convolve_image function from previous labs
    gx = convolve_image(image, PREWITT_X, factor=1.0, wrap=wrap)
    gy = convolve_image(image, PREWITT_Y, factor=1.0, wrap=wrap)
    
    # Compute magnitude: sqrt(Gx^2 + Gy^2)
    magnitude = np.sqrt(gx.astype(np.float64)**2 + gy.astype(np.float64)**2)
    
    # Clip to valid range [0, 255]
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    return gx, gy, magnitude


def apply_prewitt_approximation(image: np.ndarray, wrap: bool = False,
                                 method: str = 'sum') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Prewitt edge detection with magnitude approximation.
    
    Uses approximations instead of exact sqrt calculation for efficiency.
    
    Args:
        image: Input grayscale image
        wrap: If True, use wrap-around padding
        method: Approximation method ('sum' or 'max')
        
    Returns:
        Tuple of (Gx, Gy, magnitude)
    """
    gx = convolve_image(image, PREWITT_X, factor=1.0, wrap=wrap)
    gy = convolve_image(image, PREWITT_Y, factor=1.0, wrap=wrap)
    
    if method == 'sum':
        magnitude = np.abs(gx.astype(np.float64)) + np.abs(gy.astype(np.float64))
    elif method == 'max':
        magnitude = np.maximum(np.abs(gx.astype(np.float64)),
                              np.abs(gy.astype(np.float64)))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sum' or 'max'.")
    
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    return gx, gy, magnitude


# =============================================================================
# Image Zooming Functions - Fixed 2x2 Scale
# =============================================================================

def zoom_by_duplication_2x(image: np.ndarray) -> np.ndarray:
    """Zoom image by 2x2 using pixel duplication.
    
    Each pixel is duplicated to create a 2x2 block, resulting in an image
    that is 4 times larger (2x width, 2x height).
    
    This is equivalent to:
    1. Interleaving the image with rows/columns of zeros
    2. Convolving with a 2x2 matrix of ones
    
    Args:
        image: Input image (M x N) grayscale or (M x N x C) RGB
        
    Returns:
        Zoomed image (2M x 2N) or (2M x 2N x C)
    """
    if image.ndim == 2:
        # Grayscale image
        height, width = image.shape
        zoomed = np.zeros((2 * height, 2 * width), dtype=image.dtype)
        
        for i in range(height):
            for j in range(width):
                zoomed[2*i:2*i+2, 2*j:2*j+2] = image[i, j]
    
    elif image.ndim == 3:
        # RGB image - process each channel
        height, width, channels = image.shape
        zoomed = np.zeros((2 * height, 2 * width, channels), dtype=image.dtype)
        
        for i in range(height):
            for j in range(width):
                zoomed[2*i:2*i+2, 2*j:2*j+2, :] = image[i, j, :]
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    return zoomed


def zoom_by_linear_interpolation_2x(image: np.ndarray) -> np.ndarray:
    """Zoom image by 2x2 using linear interpolation.
    
    Interpolates new pixel values as the average of neighboring pixels.
    Process is done in two stages: first along rows, then along columns.
    
    Args:
        image: Input image (M x N) grayscale or (M x N x C) RGB
        
    Returns:
        Zoomed image (2M x 2N) or (2M x 2N x C)
    """
    if image.ndim == 2:
        # Grayscale image
        height, width = image.shape
        
        # Stage 1: Interpolate along rows (M x 2N)
        intermediate = np.zeros((height, 2 * width), dtype=np.float32)
        
        for i in range(height):
            for j in range(width):
                intermediate[i, 2*j] = image[i, j]
                if j < width - 1:
                    intermediate[i, 2*j + 1] = (image[i, j] + image[i, j + 1]) / 2.0
                else:
                    intermediate[i, 2*j + 1] = image[i, j]
        
        # Stage 2: Interpolate along columns (2M x 2N)
        zoomed = np.zeros((2 * height, 2 * width), dtype=np.float32)
        
        for j in range(2 * width):
            for i in range(height):
                zoomed[2*i, j] = intermediate[i, j]
                if i < height - 1:
                    zoomed[2*i + 1, j] = (intermediate[i, j] + intermediate[i + 1, j]) / 2.0
                else:
                    zoomed[2*i + 1, j] = intermediate[i, j]
        
        return np.clip(zoomed, 0, 255).astype(np.uint8)
    
    elif image.ndim == 3:
        # RGB image - process each channel separately
        height, width, channels = image.shape
        zoomed_channels = []
        
        for c in range(channels):
            channel = image[:, :, c]
            
            # Stage 1: Interpolate along rows
            intermediate = np.zeros((height, 2 * width), dtype=np.float32)
            
            for i in range(height):
                for j in range(width):
                    intermediate[i, 2*j] = channel[i, j]
                    if j < width - 1:
                        intermediate[i, 2*j + 1] = (channel[i, j] + channel[i, j + 1]) / 2.0
                    else:
                        intermediate[i, 2*j + 1] = channel[i, j]
            
            # Stage 2: Interpolate along columns
            zoomed_channel = np.zeros((2 * height, 2 * width), dtype=np.float32)
            
            for j in range(2 * width):
                for i in range(height):
                    zoomed_channel[2*i, j] = intermediate[i, j]
                    if i < height - 1:
                        zoomed_channel[2*i + 1, j] = (intermediate[i, j] + intermediate[i + 1, j]) / 2.0
                    else:
                        zoomed_channel[2*i + 1, j] = intermediate[i, j]
            
            zoomed_channels.append(zoomed_channel)
        
        zoomed = np.stack(zoomed_channels, axis=2)
        return np.clip(zoomed, 0, 255).astype(np.uint8)
    
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


# =============================================================================
# Image Zooming Functions - Arbitrary Scale Factors
# =============================================================================

def zoom_nearest_neighbor(image: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """Zoom image using nearest neighbor interpolation.
    
    Args:
        image: Input image (M x N) grayscale or (M x N x C) RGB
        scale_x: Horizontal scale factor
        scale_y: Vertical scale factor
        
    Returns:
        Zoomed image (round(M*scale_y) x round(N*scale_x)) with same number of channels
    """
    if image.ndim == 2:
        # Grayscale image
        height, width = image.shape
        new_height = int(round(height * scale_y))
        new_width = int(round(width * scale_x))
        
        zoomed = np.zeros((new_height, new_width), dtype=image.dtype)
        
        for i in range(new_height):
            for j in range(new_width):
                src_i = int(i / scale_y)
                src_j = int(j / scale_x)
                src_i = min(src_i, height - 1)
                src_j = min(src_j, width - 1)
                zoomed[i, j] = image[src_i, src_j]
        
        return zoomed
    
    elif image.ndim == 3:
        # RGB image
        height, width, channels = image.shape
        new_height = int(round(height * scale_y))
        new_width = int(round(width * scale_x))
        
        zoomed = np.zeros((new_height, new_width, channels), dtype=image.dtype)
        
        for i in range(new_height):
            for j in range(new_width):
                src_i = int(i / scale_y)
                src_j = int(j / scale_x)
                src_i = min(src_i, height - 1)
                src_j = min(src_j, width - 1)
                zoomed[i, j, :] = image[src_i, src_j, :]
        
        return zoomed
    
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def zoom_bilinear(image: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """Zoom image using bilinear interpolation.
    
    Interpolates pixel values based on the 4 nearest neighbors in the
    original image.
    
    Args:
        image: Input image (M x N) grayscale or (M x N x C) RGB
        scale_x: Horizontal scale factor
        scale_y: Vertical scale factor
        
    Returns:
        Zoomed image (round(M*scale_y) x round(N*scale_x)) with same number of channels
    """
    if image.ndim == 2:
        # Grayscale image
        height, width = image.shape
        new_height = int(round(height * scale_y))
        new_width = int(round(width * scale_x))
        
        zoomed = np.zeros((new_height, new_width), dtype=np.float32)
        
        for i in range(new_height):
            for j in range(new_width):
                src_i = i / scale_y
                src_j = j / scale_x
                
                i0 = int(np.floor(src_i))
                j0 = int(np.floor(src_j))
                
                di = src_i - i0
                dj = src_j - j0
                
                i0 = min(i0, height - 1)
                j0 = min(j0, width - 1)
                i1 = min(i0 + 1, height - 1)
                j1 = min(j0 + 1, width - 1)
                
                p00 = image[i0, j0]
                p01 = image[i0, j1]
                p10 = image[i1, j0]
                p11 = image[i1, j1]
                
                p0 = p00 * (1 - dj) + p01 * dj
                p1 = p10 * (1 - dj) + p11 * dj
                value = p0 * (1 - di) + p1 * di
                
                zoomed[i, j] = value
        
        return np.clip(zoomed, 0, 255).astype(np.uint8)
    
    elif image.ndim == 3:
        # RGB image - process each channel
        height, width, channels = image.shape
        new_height = int(round(height * scale_y))
        new_width = int(round(width * scale_x))
        
        zoomed = np.zeros((new_height, new_width, channels), dtype=np.float32)
        
        for c in range(channels):
            channel = image[:, :, c]
            
            for i in range(new_height):
                for j in range(new_width):
                    src_i = i / scale_y
                    src_j = j / scale_x
                    
                    i0 = int(np.floor(src_i))
                    j0 = int(np.floor(src_j))
                    
                    di = src_i - i0
                    dj = src_j - j0
                    
                    i0 = min(i0, height - 1)
                    j0 = min(j0, width - 1)
                    i1 = min(i0 + 1, height - 1)
                    j1 = min(j0 + 1, width - 1)
                    
                    p00 = channel[i0, j0]
                    p01 = channel[i0, j1]
                    p10 = channel[i1, j0]
                    p11 = channel[i1, j1]
                    
                    p0 = p00 * (1 - dj) + p01 * dj
                    p1 = p10 * (1 - dj) + p11 * dj
                    value = p0 * (1 - di) + p1 * di
                    
                    zoomed[i, j, c] = value
        
        return np.clip(zoomed, 0, 255).astype(np.uint8)
    
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


# =============================================================================
# Helper Functions
# =============================================================================

def normalize_gradient_for_display(gradient: np.ndarray) -> np.ndarray:
    """Normalize gradient values to [0, 255] range for display.
    
    Since gradient values can be negative, this function:
    1. Takes absolute value
    2. Normalizes to [0, 255] range
    
    Args:
        gradient: Gradient image (can contain negative values)
        
    Returns:
        Normalized gradient in [0, 255] range as uint8
    """
    grad_abs = np.abs(gradient.astype(np.float64))
    
    # Normalize to [0, 255]
    if grad_abs.max() > 0:
        grad_norm = (grad_abs / grad_abs.max()) * 255.0
    else:
        grad_norm = grad_abs
    
    return np.clip(grad_norm, 0, 255).astype(np.uint8)
