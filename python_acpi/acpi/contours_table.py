"""Render edge detection and zooming results in a table layout."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from .filters import load_image
from .contours import (
    apply_prewitt,
    apply_sobel,
    normalize_gradient_for_display,
    zoom_bilinear,
    zoom_by_duplication_2x,
    zoom_by_linear_interpolation_2x,
    zoom_nearest_neighbor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show edge detection and zoom results in a table layout"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("Images"),
        help="Directory containing lab images",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="football.jpg",
        help="Image filename to process",
    )
    parser.add_argument(
        "--scale-x",
        type=float,
        default=1.5,
        help="Horizontal scale factor for arbitrary zoom",
    )
    parser.add_argument(
        "--scale-y",
        type=float,
        default=1.5,
        help="Vertical scale factor for arbitrary zoom",
    )
    parser.add_argument(
        "--wrap-edges",
        action="store_true",
        help="Use wrap-around padding instead of zeros",
    )
    return parser.parse_args()


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert image to uint8 format."""
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def ensure_exists(path: Path) -> None:
    """Check if file exists, raise error if not."""
    if not path.exists():
        raise FileNotFoundError(f"Required image '{path}' not found.")


def prepare_images(args: argparse.Namespace) -> Tuple[np.ndarray, ...]:
    """Prepare all images for the table display.
    
    Returns a tuple of images in the following order:
    Row 1 (Edge Detection - Sobel):
        0. Original
        1. Sobel Gx
        2. Sobel Gy
        3. Sobel Magnitude
    Row 2 (Edge Detection - Prewitt):
        4. Original
        5. Prewitt Gx
        6. Prewitt Gy
        7. Prewitt Magnitude
    Row 3 (Zoom 2x2):
        8. Original
        9. Zoom 2x Duplication
        10. Original
        11. Zoom 2x Linear Interpolation
    Row 4 (Zoom Arbitrary - shows smaller versions for display):
        12. Original
        13. Zoom Nearest Neighbor
        14. Original
        15. Zoom Bilinear
    """
    base = args.images_dir
    image_path = base / args.image
    ensure_exists(image_path)
    
    original = load_image(image_path)
    
    # ========== Edge Detection with Sobel ==========
    gx_sobel, gy_sobel, mag_sobel = apply_sobel(original, wrap=args.wrap_edges)
    gx_sobel_norm = normalize_gradient_for_display(gx_sobel)
    gy_sobel_norm = normalize_gradient_for_display(gy_sobel)
    
    # ========== Edge Detection with Prewitt ==========
    gx_prewitt, gy_prewitt, mag_prewitt = apply_prewitt(original, wrap=args.wrap_edges)
    gx_prewitt_norm = normalize_gradient_for_display(gx_prewitt)
    gy_prewitt_norm = normalize_gradient_for_display(gy_prewitt)
    
    # ========== Zoom 2x2 ==========
    zoom_dup = zoom_by_duplication_2x(original)
    zoom_interp = zoom_by_linear_interpolation_2x(original)
    
    # ========== Zoom Arbitrary ==========
    zoom_nn = zoom_nearest_neighbor(original, args.scale_x, args.scale_y)
    zoom_bilin = zoom_bilinear(original, args.scale_x, args.scale_y)
    
    return (
        # Row 1: Sobel
        original,
        gx_sobel_norm,
        gy_sobel_norm,
        mag_sobel,
        # Row 2: Prewitt
        original,
        gx_prewitt_norm,
        gy_prewitt_norm,
        mag_prewitt,
        # Row 3: Zoom 2x2
        original,
        zoom_dup,
        original,
        zoom_interp,
        # Row 4: Zoom Arbitrary
        original,
        zoom_nn,
        original,
        zoom_bilin,
    )


def show_table(images: Tuple[np.ndarray, ...], scale_x: float, scale_y: float) -> None:
    """Display all results in a 4x4 table layout."""
    fig, axes = plt.subplots(4, 4, figsize=(20, 22))
    
    titles = [
        # Row 1: Sobel
        ["Original", "Sobel Gx", "Sobel Gy", "Sobel Magnitude"],
        # Row 2: Prewitt
        ["Original", "Prewitt Gx", "Prewitt Gy", "Prewitt Magnitude"],
        # Row 3: Zoom 2x2
        ["Original", "Zoom 2x (Duplication)", "Original", "Zoom 2x (Linear Interp.)"],
        # Row 4: Zoom Arbitrary
        [
            "Original",
            f"Zoom {scale_x}x{scale_y} (Nearest)",
            "Original",
            f"Zoom {scale_x}x{scale_y} (Bilinear)",
        ],
    ]
    
    for row in range(4):
        for col in range(4):
            idx = row * 4 + col
            ax = axes[row, col]
            ax.axis("off")
            
            img = to_uint8(images[idx])
            ax.imshow(img)
            ax.set_title(titles[row][col], fontsize=12, pad=8)
    
    fig.suptitle(
        "Lab 6: Edge Detection and Image Zooming",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()
    fig.subplots_adjust(left=0.008, bottom=0.064, right=0.993, top=0.88, wspace=0.048, hspace=0.308)
    plt.show()


def main() -> None:
    args = parse_args()
    images = prepare_images(args)
    show_table(images, args.scale_x, args.scale_y)


if __name__ == "__main__":
    main()
