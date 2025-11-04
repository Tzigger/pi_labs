"""Interactive viewer for comparing original and filtered images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, Iterable, Sequence

import numpy as np
from PIL import Image

from .filters import convolve_image, load_image
from .noise import (
    apply_gaussian_filter,
    apply_huang_median_filter,
    apply_mean_filter,
    apply_median_filter,
)

# Kernels for blur demonstrations, reused from the CLI presets.
KERNELS: Dict[str, tuple[np.ndarray, float]] = {
    "blur3": (
        np.array(
            [
                [0.0, 0.2, 0.0],
                [0.2, 0.2, 0.2],
                [0.0, 0.2, 0.0],
            ],
            dtype=np.float32,
        ),
        1.0,
    ),
    "blur5": (
        np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=np.float32,
        ),
        1.0 / 13.0,
    ),
    "motion": (
        np.array(
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
        ),
        1.0 / 9.0,
    ),
}

FILTER_MAP: Dict[str, Callable[[np.ndarray, argparse.Namespace], np.ndarray]] = {
    "mean": lambda img, args: apply_mean_filter(
        img, size=args.kernel_size, wrap=args.wrap_edges
    ),
    "gaussian": lambda img, args: apply_gaussian_filter(
        img, size=args.gaussian_size, sigma=args.gaussian_sigma, wrap=args.wrap_edges
    ),
    "median": lambda img, args: apply_median_filter(
        img, size=args.kernel_size, wrap=args.wrap_edges
    ),
    "huang": lambda img, args: apply_huang_median_filter(
        img, size=args.kernel_size, wrap=args.wrap_edges
    ),
    "blur3": lambda img, args: _apply_convolution(img, "blur3", args.wrap_edges),
    "blur5": lambda img, args: _apply_convolution(img, "blur5", args.wrap_edges),
    "motion": lambda img, args: _apply_convolution(img, "motion", args.wrap_edges),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display before/after comparisons for ACPI filters"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("Images/Fig5.03_zgomotSarePiper.jpg"),
        help="Image to process (default: salt-and-pepper example)",
    )
    parser.add_argument(
        "--filters",
        nargs="+",
        default=["median"],
        choices=sorted(FILTER_MAP.keys()),
        help="Filters to display (default: median)",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=3,
        help="Window size for mean and median filters",
    )
    parser.add_argument(
        "--gaussian-size",
        type=int,
        default=5,
        help="Window size for the Gaussian filter",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=1.0,
        help="Sigma for the Gaussian filter",
    )
    parser.add_argument(
        "--wrap-edges",
        action="store_true",
        help="Wrap image edges instead of zero padding",
    )
    return parser.parse_args()


def _apply_convolution(image: np.ndarray, kernel_name: str, wrap: bool) -> np.ndarray:
    kernel, factor = KERNELS[kernel_name]
    return convolve_image(image, kernel, factor=factor, wrap=wrap)


def _to_uint8(array: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


def show_side_by_side(original: np.ndarray, processed: np.ndarray, title: str) -> None:
    if original.shape != processed.shape:
        raise ValueError(
            "Original and processed images must share the same shape for visualization."
        )
    combined = np.hstack([_to_uint8(original), _to_uint8(processed)])
    Image.fromarray(combined).show(title=title)


def main() -> None:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image '{args.image}' does not exist.")

    source = load_image(args.image)

    for filter_name in args.filters:
        transform = FILTER_MAP[filter_name]
        result = transform(source, args)
        window_title = f"{filter_name} ({args.image.name})"
        show_side_by_side(source, result, window_title)


if __name__ == "__main__":
    main()
