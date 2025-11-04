"""Render a grid of before/after comparisons in a table layout."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from .filters import convolve_image, load_image
from .noise import (
    apply_gaussian_filter,
    apply_huang_median_filter,
    apply_mean_filter,
    apply_median_filter,
)

BLUR_KERNEL_SMALL = np.array(
    [
        [0.0, 0.2, 0.0],
        [0.2, 0.2, 0.2],
        [0.0, 0.2, 0.0],
    ],
    dtype=np.float32,
)

BLUR_KERNEL_LARGE = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ],
    dtype=np.float32,
)

MOTION_KERNEL = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show ACPI filters in a table layout")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("Images"),
        help="Directory containing lab images",
    )
    parser.add_argument(
        "--football", type=str, default="football.jpg", help="Football image filename"
    )
    parser.add_argument(
        "--uniform",
        type=str,
        default="Fig5.03_zgomotUniform.jpg",
        help="Uniform noise image",
    )
    parser.add_argument(
        "--gaussian",
        type=str,
        default="Fig5.03_zgomotGaussian.jpg",
        help="Gaussian noise image",
    )
    parser.add_argument(
        "--saltpepper",
        type=str,
        default="Fig5.03_zgomotSarePiper.jpg",
        help="Salt and pepper noise image",
    )
    parser.add_argument(
        "--mean-size", type=int, default=3, help="Window size for mean filter"
    )
    parser.add_argument(
        "--gaussian-size", type=int, default=5, help="Window size for Gaussian filter"
    )
    parser.add_argument(
        "--gaussian-sigma", type=float, default=1.0, help="Sigma for Gaussian filter"
    )
    parser.add_argument(
        "--median-size", type=int, default=3, help="Window size for median filter"
    )
    parser.add_argument(
        "--huang-size", type=int, default=3, help="Window size for Huang median filter"
    )
    parser.add_argument(
        "--wrap-edges",
        action="store_true",
        help="Use wrap-around padding instead of zeros",
    )
    return parser.parse_args()


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required image '{path}' not found.")


def prepare_images(args: argparse.Namespace) -> Tuple[np.ndarray, ...]:
    base = args.images_dir

    football_path = base / args.football
    uniform_path = base / args.uniform
    gaussian_path = base / args.gaussian
    saltpepper_path = base / args.saltpepper

    for path in (football_path, uniform_path, gaussian_path, saltpepper_path):
        ensure_exists(path)

    football = load_image(football_path)
    # Compute both smoothing variants to follow the lab specification (small kernel result not displayed by default).
    _football_blur_small = convolve_image(
        football, BLUR_KERNEL_SMALL, factor=1.0, wrap=args.wrap_edges
    )
    football_blur_large = convolve_image(
        football, BLUR_KERNEL_LARGE, factor=1.0 / 13.0, wrap=args.wrap_edges
    )
    football_motion = convolve_image(
        football, MOTION_KERNEL, factor=1.0 / 9.0, wrap=args.wrap_edges
    )

    uniform = load_image(uniform_path)
    uniform_mean = apply_mean_filter(uniform, size=args.mean_size, wrap=args.wrap_edges)

    gaussian = load_image(gaussian_path)
    gaussian_filtered = apply_gaussian_filter(
        gaussian,
        size=args.gaussian_size,
        sigma=args.gaussian_sigma,
        wrap=args.wrap_edges,
    )

    saltpepper = load_image(saltpepper_path)
    saltpepper_median = apply_median_filter(
        saltpepper, size=args.median_size, wrap=args.wrap_edges
    )
    saltpepper_huang = apply_huang_median_filter(
        saltpepper, size=args.huang_size, wrap=args.wrap_edges
    )

    # Table order: original + filtered pairs for each filter sequence.
    return (
        football,
        football_blur_large,
        football,
        football_motion,
        uniform,
        uniform_mean,
        gaussian,
        gaussian_filtered,
        saltpepper,
        saltpepper_median,
        saltpepper,
        saltpepper_huang,
    )


def show_table(images: Tuple[np.ndarray, ...]) -> None:
    fig, axes = plt.subplots(2, 6, figsize=(26, 9))
    titles_top = [
        "Original (football)",
        "Blurred (smooth, 5x5)",
        "Original (football)",
        "Motion blur (9x9 diag)",
        "Uniform noise",
        "Mean filter",
    ]
    titles_bottom = [
        "Gaussian noise",
        "Gaussian filter",
        "Salt & pepper noise",
        "Median filter",
        "Salt & pepper noise",
        "Huang median filter",
    ]

    # First row: images 0-5, second row: images 6-11
    for idx in range(6):
        ax = axes[0, idx]
        ax.axis("off")
        ax.imshow(to_uint8(images[idx]))
        ax.set_title(titles_top[idx], fontsize=14)

    for idx in range(6, 12):
        ax = axes[1, idx - 6]
        ax.axis("off")
        ax.imshow(to_uint8(images[idx]))
        ax.set_title(titles_bottom[idx - 6], fontsize=14)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    plt.show()


def main() -> None:
    args = parse_args()
    images = prepare_images(args)
    show_table(images)


if __name__ == "__main__":
    main()
